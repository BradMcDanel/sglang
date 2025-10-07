"""
Adapted from https://github.com/chromecast56/sglang/blob/6f145d2eadb93a116134f703358ce76f15381045/benchmark/mtbench/bench_sglang.py

Benchmark SGLang EAGLE/EAGLE3 Speculative Decoding

Usage:
python3 benchmark/mtbench/bench_sglang_eagle.py --num-questions 80 --parallel 1
"""

import argparse
import json
import os
import time
import uuid

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def load_questions(filename):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions


def write_answers(filename, model_id, questions, answers):
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(answers)):
            ans_json = {
                "question_id": questions[i]["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": {
                    "index": 0,
                    "turns": [answers[i][0], answers[i][1]],
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


@sgl.function
def answer_mt_bench(s, question_1, question_2):
    s += sgl.system(
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    )
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


def main(args):
    # Construct prompts
    questions = load_questions(args.question_file)[: args.num_questions]
    arguments = [
        {"question_1": q["turns"][0], "question_2": q["turns"][1]} for q in questions
    ]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.perf_counter()
    rets = answer_mt_bench.run_batch(
        arguments,
        temperature=0,
        max_new_tokens=2048,
        num_threads=args.parallel,
        progress_bar=True,
    )
    answers = [[s["answer_1"], s["answer_2"]] for s in rets]

    latency = time.perf_counter() - tic
    num_output_tokens = sum(
        s.get_meta_info("answer_1")["completion_tokens"]
        + s.get_meta_info("answer_2")["completion_tokens"]
        for s in rets
    )

    # NOTE: acceptance length is just completion_tokens / spec_verify_ct
    # {'id': '3bb9c5ead109488d8ed5ee9cbecaec29', 'finish_reason': {'type': 'length', 'length': 256}, 'prompt_tokens': 37, 'spec_verify_ct': 101, 'completion_tokens': 256, 'cached_tokens': 0}

    output_throughput = num_output_tokens / latency

    has_verify = "spec_verify_ct" in rets[0].get_meta_info("answer_1")
    if has_verify:
        num_verify_tokens = sum(
            s.get_meta_info("answer_1")["spec_verify_ct"]
            + s.get_meta_info("answer_2")["spec_verify_ct"]
            for s in rets
        )

        accept_length = num_output_tokens / num_verify_tokens
    else:
        accept_length = 1.0

    print(
        f"#questions: {len(questions)}, Throughput: {output_throughput:.2f} token/s, Acceptance length: {accept_length:.2f}"
    )
    print(f"\nDEBUG: meta_info keys = {rets[0].get_meta_info('answer_1').keys()}")

    # Print expert activation statistics if available
    if "expert_layer_activations" in rets[0].get_meta_info("answer_1"):
        print("\nExpert Activation Statistics:")
        print("=" * 80)

        # Aggregate expert activations across all requests
        all_expert_activations = {}
        for s in rets:
            for answer_key in ["answer_1", "answer_2"]:
                expert_acts = s.get_meta_info(answer_key).get("expert_layer_activations")
                if expert_acts:
                    for layer_id, counts in expert_acts.items():
                        layer_id = int(layer_id)
                        if layer_id not in all_expert_activations:
                            all_expert_activations[layer_id] = [0] * len(counts)
                        for i, count in enumerate(counts):
                            all_expert_activations[layer_id][i] += count

        # Print per-layer statistics
        for layer_id in sorted(all_expert_activations.keys()):
            counts = all_expert_activations[layer_id]
            total = sum(counts)
            if total > 0:
                active_experts = sum(1 for c in counts if c > 0)
                max_count = max(counts)
                max_expert = counts.index(max_count)
                print(f"Layer {layer_id}: {active_experts}/{len(counts)} experts active, "
                      f"total={total}, max_expert={max_expert} (count={max_count})")

    # Optionally display per-verify expert routing history for the first sample
    sample_history = rets[0].get_meta_info("answer_1").get("expert_layer_history")
    if sample_history:
        print("\nExpert Activation History (first response):")
        print("=" * 80)
        for verify_idx, verify_snapshot in enumerate(sample_history):
            print(f"Verify pass {verify_idx}:")

            # Display tree structure metadata if available
            if "tree_structure" in verify_snapshot:
                tree = verify_snapshot["tree_structure"]
                print(f"  Tree: {tree['draft_token_num']} tokens, {tree['spec_steps']} steps")
                print(f"  Tree metadata: retrive_next_token shape={len(tree.get('retrive_next_token', []))}x{len(tree.get('retrive_next_token', [[]])[0]) if tree.get('retrive_next_token') else 0}")

            # Display layer-wise expert activations
            layer_map = verify_snapshot.get("layers", verify_snapshot)  # Support both old and new format
            items = list(layer_map.items())
            for layer_id_raw, layer_entry in items[:5]:
                layer_id = int(layer_id_raw)
                token_count = len(layer_entry.get("topk_ids", []))
                sample_topk = layer_entry.get("topk_ids", [])[:1]
                sample_weights = layer_entry.get("topk_weights")
                sample_weights = sample_weights[:1] if sample_weights else []
                preview = sample_topk[0] if sample_topk else []
                print(
                    f"  Layer {layer_id}: tokens={token_count}, sample_topk={preview}"
                    + (f", weights={sample_weights[0]}" if sample_weights else "")
                )
            if verify_idx >= 2:
                print("  ... (truncated)")
                break

    # Write expert activation logs if available
    if "expert_layer_activations" in rets[0].get_meta_info("answer_1"):
        expert_log_file = "expert_activations.json"
        expert_logs = []
        model_path = backend.model_info["model_path"]

        for i, s in enumerate(rets):
            for turn_idx, answer_key in enumerate(["answer_1", "answer_2"]):
                meta = s.get_meta_info(answer_key)
                log_entry = {
                    "question_id": questions[i]["question_id"],
                    "turn": turn_idx + 1,
                    "question": questions[i]["turns"][turn_idx],
                    "answer": s[answer_key],
                    "meta_info": {
                        "model_path": model_path,
                        "prompt_tokens": meta.get("prompt_tokens"),
                        "completion_tokens": meta.get("completion_tokens"),
                        "spec_verify_ct": meta.get("spec_verify_ct"),
                    },
                    "expert_layer_activations": meta.get("expert_layer_activations"),
                    "expert_layer_history": meta.get("expert_layer_history"),
                }
                expert_logs.append(log_entry)

        with open(expert_log_file, "w") as f:
            json.dump(expert_logs, f, indent=2)
        print(f"\nExpert activation logs saved to: {expert_log_file}")

    # Write results
    model_id = backend.model_info["model_path"]
    answer_file = args.answer_file or f"tmp_output_{args.backend}.txt"
    write_answers(answer_file, model_id, questions, answers)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "mtbench",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "throughput": round(output_throughput, 3),
            "accept_length": round(accept_length, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=80)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
