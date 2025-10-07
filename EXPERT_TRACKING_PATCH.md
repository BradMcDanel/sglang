# Expert Activation Tracking for MoE Models

## Summary

This patch adds optional expert activation tracking for MoE models (specifically Qwen3-30B-A3B) during EAGLE3 speculative decoding. When enabled via environment variable, it captures detailed information about which experts are activated during inference and exposes this data through the API response.

**Status**: ✅ Working

**Performance**: ~20-30 tok/s with tracking enabled vs ~140 tok/s baseline (5-7x slowdown)

---

## Quick Start

### Enable Expert Tracking
```bash
export SGLANG_TRACK_EXPERT_ACTIVATIONS=1
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path /path/to/Qwen3-30B-A3B \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /path/to/Qwen3-a3B_eagle3 \
  --speculative-num-steps 6 --speculative-eagle-topk 10 \
  --speculative-num-draft-tokens 32 --tp 2

cd benchmark/mtbench/
python3 bench_sglang_eagle.py --num-questions 2 --parallel 1
# Creates expert_activations.json automatically
```

### Disable Expert Tracking (Default)
```bash
# Don't set the environment variable (or set to 0)
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server ...
# Full speed: ~140 tok/s, zero overhead
```

---

## Implementation Overview

### Key Design Decisions

1. **Environment Variable Gating**: `SGLANG_TRACK_EXPERT_ACTIVATIONS=1` controls whether tracking is enabled. When disabled (default), `forward_batch.expert_activations = None` and all tracking code is skipped.

2. **CUDA Graph Compatibility**: Expert activations are collected **after** CUDA graph replay by reading GPU tensors that were updated during the graph execution. This allows tracking to work without disabling CUDA graph optimization.

3. **Post-Replay Collection**: MoE layers store routing tensors as instance variables (`last_topk_ids`, `last_topk_weights`). After the forward pass completes, `model.collect_expert_activations()` copies these GPU tensors to CPU.

4. **Complete Data Collection**: When enabled, captures 100% of verify passes (no sampling) for accurate analysis.

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Forward Pass (MoE Layer)                                 │
│    qwen3_moe.py: Store GPU routing tensors                  │
│    → self.last_topk_ids = topk_output.topk_ids             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. CUDA Graph Replay (if applicable)                        │
│    model_runner.py: Replay graph, then collect             │
│    → model.collect_expert_activations(forward_batch)       │
│    → Copies GPU tensors to CPU                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EAGLE Verify (per verify pass)                           │
│    eagle_info.py: Accumulate into request                   │
│    → req.expert_layer_activations[layer_id] += counts      │
│    → req.expert_layer_history.append(snapshot)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Output Pipeline                                           │
│    scheduler → tokenizer → API response                     │
│    → meta_info["expert_layer_activations"] = ...           │
│    → meta_info["expert_layer_history"] = ...               │
└─────────────────────────────────────────────────────────────┘
```

---

## Files Modified

### Core Tracking Infrastructure

**`python/sglang/srt/model_executor/forward_batch_info.py`**
- Conditionally initializes `expert_activations` dict based on `SGLANG_TRACK_EXPERT_ACTIVATIONS` env var
- When disabled: `expert_activations = None` (zero overhead)
- When enabled: `expert_activations = {}` (ready to collect data)

**`python/sglang/srt/models/qwen3_moe.py`**
- Stores routing outputs as instance variables: `self.last_topk_ids`, `self.last_topk_weights`
- Works for both `forward_normal` and `forward_deepep` paths
- Only stores when `TopKOutputFormat` is `STANDARD`
- Added `collect_expert_activations()` method to copy GPU tensors to CPU after forward

**`python/sglang/srt/model_executor/model_runner.py`**
- Calls `model.collect_expert_activations(forward_batch)` after CUDA graph replay
- Only collects for `TARGET_VERIFY` forward mode (verify passes)

### Request-Level Accumulation

**`python/sglang/srt/managers/schedule_batch.py`**
- Each `Req` tracks:
  - `expert_layer_activations`: Dict[int, List[int]] - cumulative expert counts per layer
  - `expert_layer_history`: List[Dict] - per-verify snapshots with token-level routing

**`python/sglang/srt/speculative/eagle_info.py`**
- Accumulates expert activations during verify
- Stores both aggregated counts and detailed per-pass history
- Only runs when `forward_batch.expert_activations is not None`

**`python/sglang/srt/speculative/eagle_worker.py`**
- Passes `forward_batch` to verify step

**`python/sglang/srt/managers/tp_worker.py`**
- Returns `forward_batch` in `ForwardBatchOutput`

### Output Pipeline

**`python/sglang/srt/managers/scheduler_output_processor_mixin.py`**
- Collects expert stats from completed requests

**`python/sglang/srt/managers/io_struct.py`**
- Extends batch output structs with expert fields

**`python/sglang/srt/managers/multi_tokenizer_mixin.py`**
- Preserves expert stats when batches are split

**`python/sglang/srt/managers/detokenizer_manager.py`**
- Preserves expert stats through detokenization

**`python/sglang/srt/managers/tokenizer_manager.py`**
- Attaches expert stats to API `meta_info`

### Benchmark Script

**`benchmark/mtbench/bench_sglang_eagle.py`**
- Prints aggregate expert statistics
- Automatically saves detailed logs to `expert_activations.json` when tracking is enabled
- Includes question, answer, and full expert routing data

---

## Output Format

### API Response Structure

```json
{
  "meta_info": {
    "expert_layer_activations": {
      "0": [1234, 567, 890, ...],  // 128 expert counts for layer 0
      "1": [456, 789, 123, ...],   // 128 expert counts for layer 1
      ...
      "47": [...]                   // Layer 47
    },
    "expert_layer_history": [
      {  // Verify pass 0
        "0": {
          "topk_ids": [[72, 33, 64, ...], ...],      // Per-token expert IDs
          "topk_weights": [[0.32, 0.12, 0.11, ...], ...]  // Routing weights
        },
        ...
      },
      ...  // Additional verify passes
    ]
  }
}
```

### JSON Log File

When running the benchmark with tracking enabled, creates `expert_activations.json`:

```json
[
  {
    "question_id": 81,
    "turn": 1,
    "question": "Compose an engaging travel blog post...",
    "answer": "Paris, the City of Light...",
    "meta_info": {
      "prompt_tokens": 37,
      "completion_tokens": 256,
      "spec_verify_ct": 101
    },
    "expert_layer_activations": { ... },
    "expert_layer_history": [ ... ]
  },
  ...
]
```

---

## Performance Characteristics

| Mode | Throughput | Overhead | Data Captured |
|------|------------|----------|---------------|
| **Disabled** (default) | ~140 tok/s | 0% | None |
| **Enabled** | ~20-30 tok/s | 5-7x slowdown | Complete expert routing |

**Bottleneck**: GPU→CPU tensor copies on every verify pass (48 layers × ~32 tokens × 2 tensors)

**Use Cases**:
- Disabled: Production inference
- Enabled: Research, debugging, expert usage analysis

---

## Technical Notes

- Compatible with CUDA graphs (uses post-replay collection)
- Compatible with TP/PP parallelism
- Only tracks `STANDARD` TopK output format
- Uses async GPU→CPU copies (`non_blocking=True`)
- Zero overhead when disabled (gated at initialization)
