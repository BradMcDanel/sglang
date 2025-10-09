# Expert Limiting for MoE Speculative Decoding - Debug Notes

## Goal

Reduce memory bandwidth pressure during EAGLE speculative decoding by limiting the number of unique experts loaded per layer during verification.

## Hypothesis

EAGLE achieves ~2.5 acceptance rate but shows no speedup over baseline (~155 tok/s for both). We suspect memory bandwidth is the bottleneck because:
- Each verify pass processes 32 draft tokens across 48 layers
- Each token routes to top-8 experts (of 128 total)
- This could require loading up to 256 unique experts per layer per verify
- Loading expert weights dominates over compute time

## Approach

Limit unique experts per layer during verification:
- Aggregate routing scores across all 32 tokens in the tree
- Select top-N experts globally (e.g., N=8, 16, 32)
- Each token still uses 8 experts, but drawn from limited pool
- **Same compute**, **less I/O**

Expected: If throughput improves with lower N, confirms I/O bottleneck.

## Implementation

### Environment Variables

Added two flags (similar to `SGLANG_TRACK_EXPERT_ACTIVATIONS`):

```bash
SGLANG_LIMIT_TREE_EXPERTS=1           # Enable/disable (0 or 1)
SGLANG_MAX_TREE_EXPERTS=32           # Max unique experts per layer
```

### Files Modified

1. **`python/sglang/srt/model_executor/forward_batch_info.py`**
   - Added fields `limit_tree_experts: bool`, `max_tree_experts: int`
   - Populate them from env vars in `ForwardBatch.init_new`

2. **`python/sglang/srt/models/qwen3_moe.py`**
   - Added limiting logic in all three routing paths:
     - `forward_normal()`
     - `forward_deepep()`
     - `op_select_experts()` (operations/TBO path)

### Algorithm (per layer, per verify pass)

```python
# Aggregate routing logits across all tokens
expert_scores = router_logits.sum(dim=0)  # [128]

# Select top-N experts
selected_experts = expert_scores.topk(max_tree_experts).indices  # [N]

# Mask non-selected experts
mask = torch.zeros(128, dtype=bool)
mask[selected_experts] = True
masked_logits = router_logits.clone()
masked_logits[:, ~mask] = float('-inf')

# TopK routing now picks from limited pool
topk_output = self.topk(hidden_states, masked_logits)
# Each token gets top-8 from the N available experts
# Renormalization happens automatically in TopK
```

## Current Status: NOT WORKING

### Observations

1. ✅ `ForwardBatch.init_new` correctly sets `limit_tree_experts` / `max_tree_experts` when the env vars are present.
2. ❌ Verify passes replay CUDA (or CPU) graphs that were captured without those fields, so the masking branch never runs.
3. ❌ Layer-10 debug prints never appear because graph replay bypasses the Python routing code.
4. ❌ Throughput remains ~155 tok/s even with `SGLANG_MAX_TREE_EXPERTS=8`.

### Root Cause

The limiting logic works for non-graph execution, but EAGLE verify passes execute under graph replay. During graph **capture** we instantiate fresh `ForwardBatch` objects inside various helpers (`cuda_graph_runner`, `cpu_graph_runner`, speculative draft runners, and the TBO splitter). Those constructors do not propagate the new fields, so the dataclass defaults (`False`, `0`) get recorded into the graph. Every replay therefore follows the original unmasked routing path even though the live `ForwardBatch` from `init_new` has limiting enabled. Expert tracking succeeds because it disables graph replay entirely; limiting must remain compatible with graphs, so the capture-time batches need the same flag values.

### Required Fixes

Propagate `limit_tree_experts` / `max_tree_experts` anywhere we build a `ForwardBatch` outside `ForwardBatch.init_new`:

- `python/sglang/srt/model_executor/cuda_graph_runner.py` – capture path for CUDA graphs.
- `python/sglang/srt/model_executor/cpu_graph_runner.py` – capture path for CPU graphs.
- `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` and `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` – draft graph capture/replay.
- `python/sglang/srt/two_batch_overlap.py` – TBO splitting creates child batches that must inherit the flags.
- (Tests instantiating `ForwardBatch` directly may require explicit defaults.)

Once these constructors mirror the env-driven initialization, capture will record the masking branch and replay will execute the limiting logic without sacrificing graph performance.

### Testing Commands

```bash
SGLANG_LIMIT_TREE_EXPERTS=1 SGLANG_MAX_TREE_EXPERTS=8 \
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path /data/models/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --reasoning-parser qwen3 --tool-call-parser qwen25 \
  --attention-backend flashinfer --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /data/models/hub/models--AngelSlim--Qwen3-a3B_eagle3/snapshots/8b3054f88692e0ad40b837079bc1585e242154d6 \
  --speculative-num-steps 8 --speculative-eagle-topk 10 \
  --speculative-num-draft-tokens 64 --tp 2

python3 bench_sglang_eagle.py --num-questions 10 --parallel 1
```

### Next Steps

1. Patch the capture/TBO paths above to copy the limiting fields.
2. Verify that layer-10 debug logs trigger under graph replay.
3. Sweep `SGLANG_MAX_TREE_EXPERTS` to evaluate throughput vs. expert budget.
