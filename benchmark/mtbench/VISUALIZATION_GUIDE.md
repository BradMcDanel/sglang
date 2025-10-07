# Expert Activation Visualization Scripts

Quick reference for visualizing expert activations during EAGLE3 speculative decoding.

## Prerequisites

1. Run SGLang server with expert tracking enabled:
   ```bash
   export SGLANG_TRACK_EXPERT_ACTIVATIONS=1
   CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
     --model-path /path/to/Qwen3-30B-A3B \
     --speculative-algorithm EAGLE3 \
     --speculative-draft-model-path /path/to/Qwen3-a3B_eagle3 \
     --speculative-num-steps 6 --speculative-eagle-topk 10 \
     --speculative-num-draft-tokens 32 --tp 2
   ```

2. Run benchmark to generate `expert_activations.json`:
   ```bash
   python3 bench_sglang_eagle.py --num-questions 2 --parallel 1
   ```

---

## Visualization Scripts

### 1. **Expert Usage Per Layer Across Verify Steps** (`viz_expert_usage_layer.py`)

**What it shows**: Line plot of unique experts used in a single layer across all verify cycles.

**Usage**:
```bash
python3 viz_expert_usage_layer.py --layer-id 20 --sample-idx 0 --output layer20.png
```

**Output**: Line graph with verify step on x-axis, number of unique experts on y-axis.

**Use case**: Track how expert diversity changes during generation for a specific layer.

---

### 2. **Expert Usage Statistics (All Layers)** (`viz_expert_usage_statistics.py`)

**What it shows**: Grouped bar chart showing mean, 90th percentile, and 99th percentile of unique experts used per layer across all verify cycles.

**Usage**:
```bash
python3 viz_expert_usage_statistics.py --sample-idx 0 --output stats.png
```

**Output**: Bar chart with layers on x-axis, 3 bars per layer (mean/P90/P99).

**Use case**: Compare expert usage patterns across all layers in the model.

---

### 3. **Expert Activation Tree** (`viz_expert_tree.py`)

**What it shows**: Tree visualization of draft tokens with:
- Decoded token text in each node
- Expert IDs activated for a specific layer
- **Green bold edges** for accepted path
- Gray thin edges for rejected speculative branches

**Usage**:
```bash
python3 viz_expert_tree.py --layer-id 20 --verify-step 3 --sample-idx 0 --output tree.png
```

**Output**: Hierarchical tree diagram showing the draft token structure and expert activations.

**Use case**: Understand which tokens/experts were used in accepted vs. rejected speculative paths.

---

## Common Parameters

- `--input`: Input JSON file (default: `expert_activations.json`)
- `--sample-idx`: Sample/question index to analyze (default: 0)
- `--layer-id`: Layer to visualize (0-47 for Qwen3-30B)
- `--verify-step`: Which verify cycle to visualize (tree script only)
- `--output`: Output PNG file path

---

## Finding Interesting Verify Steps

To find verify steps with high acceptance:

```bash
python3 -c "
import json
data = json.load(open('expert_activations.json'))
for i in range(min(20, len(data[0]['expert_layer_history']))):
    accept = data[0]['expert_layer_history'][i]['tree_structure']['accept_index']
    num_accepted = sum(1 for x in accept if x >= 0)
    if num_accepted > 3:
        print(f'Verify step {i}: {num_accepted} nodes accepted')
"
```

---

## Example Workflow

```bash
# 1. Generate expert activation data
export SGLANG_TRACK_EXPERT_ACTIVATIONS=1
# ... start server ...
python3 bench_sglang_eagle.py --num-questions 2 --parallel 1

# 2. Overview of all layers
python3 viz_expert_usage_statistics.py --sample-idx 0 --output overview.png

# 3. Deep dive into layer 20
python3 viz_expert_usage_layer.py --layer-id 20 --sample-idx 0 --output layer20_trend.png

# 4. Visualize specific verify step
python3 viz_expert_tree.py --layer-id 20 --verify-step 3 --sample-idx 0 --output tree_step3.png
```
