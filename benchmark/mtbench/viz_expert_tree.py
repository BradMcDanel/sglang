#!/usr/bin/env python3
"""
Visualize Expert Activation Tree for a Specific Layer

This script creates a tree visualization showing the draft token tree structure
with each node displaying the decoded token and the expert IDs activated for
a specific layer.

Usage:
    python viz_expert_tree.py [--input expert_activations.json] [--output tree.png] [--layer-id 0] [--verify-step 0]
"""

import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import defaultdict


def load_expert_data(filepath: str) -> dict:
    """Load expert activation data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from path. Raises exception if loading fails."""
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✓ Tokenizer loaded successfully")
    return tokenizer


def decode_tokens(tree_data: dict, tokenizer) -> list:
    """
    Decode draft tokens using tokenizer. Raises exception if draft_tokens not found.

    Returns:
        List of decoded token strings
    """
    draft_tokens = tree_data.get('draft_tokens')

    if not draft_tokens:
        raise ValueError("No draft_tokens found in tree structure. Run server with updated code to capture token IDs.")

    print(f"Found {len(draft_tokens)} draft tokens")

    # Decode each token individually for display
    decoded = []
    for token_id in draft_tokens:
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        # Escape newlines and limit length for display
        token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')
        if len(token_str) > 15:
            token_str = token_str[:12] + "..."
        decoded.append(token_str)

    print(f"Successfully decoded {len(decoded)} tokens")
    return decoded


def build_tree_structure(tree_data: dict) -> dict:
    """
    Build a tree structure from the tree_data.

    Returns:
        Dictionary mapping node_id to list of children
    """
    next_token = tree_data.get('retrive_next_token', [[]])[0]
    next_sibling = tree_data.get('retrive_next_sibling', [[]])[0]

    print(f"\n=== Tree Structure Debug ===")
    print(f"Draft token num: {tree_data.get('draft_token_num', 0)}")
    print(f"Spec steps: {tree_data.get('spec_steps', 0)}")
    print(f"next_token array (length {len(next_token)}): {next_token}")
    print(f"next_sibling array (length {len(next_sibling)}): {next_sibling}")

    # Build adjacency list
    tree = defaultdict(list)

    for node_id, child_id in enumerate(next_token):
        if child_id >= 0:
            tree[node_id].append(child_id)
            print(f"  Node {node_id} -> child {child_id}")

    # Also check siblings to build complete tree
    for node_id, sibling_id in enumerate(next_sibling):
        if sibling_id >= 0 and sibling_id not in tree[node_id]:
            # Find parent of this node to add sibling
            for parent_id, children in tree.items():
                if node_id in children and sibling_id not in children:
                    tree[parent_id].append(sibling_id)
                    print(f"  Node {parent_id} -> sibling child {sibling_id} (via node {node_id})")

    print(f"\nFinal tree structure:")
    for node_id in sorted(tree.keys()):
        print(f"  Node {node_id} has children: {tree[node_id]}")

    return tree


def compute_tree_layout(tree: dict, root: int = 0, all_nodes: set = None) -> dict:
    """
    Compute hierarchical tree layout positions.

    Returns:
        Dictionary mapping node_id to (x, y) position
    """
    positions = {}
    depths = {}
    widths = {}

    # First, find all reachable nodes from root
    def find_reachable(node, depth=0):
        if node in depths:
            return  # Already visited
        depths[node] = depth
        children = tree.get(node, [])
        for child in children:
            find_reachable(child, depth + 1)

    find_reachable(root)

    print(f"\nLayout computation:")
    print(f"  Nodes reachable from root {root}: {sorted(depths.keys())}")
    print(f"  Total nodes: {len(depths)}")

    # Calculate width of each node's subtree
    def get_width(node):
        if node in widths:
            return widths[node]
        children = tree.get(node, [])
        if not children:
            widths[node] = 1
            return 1
        else:
            total_width = sum(get_width(child) for child in children)
            widths[node] = total_width
            return total_width

    for node in depths.keys():
        get_width(node)

    # Assign x positions based on left-to-right traversal
    x_counter = [0]

    def assign_positions(node):
        if node in positions:
            return
        children = tree.get(node, [])
        if not children:
            x = x_counter[0]
            x_counter[0] += 1
            positions[node] = (x, -depths[node])
        else:
            # Position children first (left to right)
            child_x_positions = []
            for child in children:
                assign_positions(child)
                child_x_positions.append(positions[child][0])

            # Position parent at center of children
            x = np.mean(child_x_positions)
            positions[node] = (x, -depths[node])

    assign_positions(root)

    return positions


def extract_layer_experts(verify_snapshot: dict, layer_id: int) -> list:
    """
    Extract expert IDs for each token in the specified layer.

    Returns:
        List of expert ID lists (one per token), sorted numerically
    """
    layers = verify_snapshot.get('layers', {})
    layer_key = str(layer_id)

    print(f"\n=== Layer Data Debug ===")
    print(f"Available layers: {list(layers.keys())}")

    if layer_key not in layers:
        print(f"Layer {layer_key} not found!")
        return []

    layer_data = layers[layer_key]
    print(f"Layer {layer_key} keys: {list(layer_data.keys())}")

    topk_ids = layer_data.get('topk_ids', [])
    print(f"Number of tokens in topk_ids: {len(topk_ids)}")

    # Sort expert IDs for each token
    expert_lists = []
    for i, token_experts in enumerate(topk_ids):
        # Filter out invalid expert IDs and sort
        valid_experts = sorted([e for e in token_experts if e >= 0])
        expert_lists.append(valid_experts)
        if i < 5:  # Print first 5
            print(f"  Token {i} experts: {valid_experts}")

    return expert_lists


def create_tree_visualization(
    sample_data: dict,
    layer_id: int,
    verify_step: int,
    output_path: str = 'expert_tree.png',
    tokenizer = None
):
    """
    Create a tree visualization showing tokens and expert activations.
    """
    history = sample_data.get('expert_layer_history', [])

    if not history:
        print("No expert_layer_history found in sample")
        return

    if verify_step >= len(history):
        print(f"ERROR: Verify step {verify_step} out of range (max: {len(history)-1})")
        return

    verify_snapshot = history[verify_step]
    tree_data = verify_snapshot.get('tree_structure', {})

    # Build tree structure
    tree = build_tree_structure(tree_data)
    draft_token_num = tree_data.get('draft_token_num', 0)

    if draft_token_num == 0:
        print("No draft tokens in this verify step")
        return

    # Get expert activations for the layer
    expert_lists = extract_layer_experts(verify_snapshot, layer_id)

    if not expert_lists:
        print(f"No expert data found for layer {layer_id}")
        return

    # Compute tree layout
    pos = compute_tree_layout(tree, root=0)

    # Scale positions for better visualization
    scale_x = 1.5
    scale_y = 2.0
    pos = {node: (x * scale_x, y * scale_y) for node, (x, y) in pos.items()}

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # Get accepted nodes
    accept_index = tree_data.get('accept_index', [])
    accepted_nodes = set([idx for idx in accept_index if idx >= 0])
    print(f"Accepted nodes: {sorted(accepted_nodes)}")

    # Draw edges
    for parent, children in tree.items():
        if parent not in pos:
            continue
        x1, y1 = pos[parent]
        for child in children:
            if child not in pos:
                continue
            x2, y2 = pos[child]

            # Check if this edge is part of the accepted path
            if parent in accepted_nodes and child in accepted_nodes:
                # Accepted path: green and bold
                ax.plot([x1, x2], [y1, y2], 'g-', linewidth=3.5, alpha=0.9, zorder=2)
            else:
                # Not accepted: black and thin
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.4, zorder=1)

    # Get positions and decode tokens
    positions = tree_data.get('positions', [[]])[0] if tree_data.get('positions') else []
    print(f"\nPositions: {positions}")

    # Decode tokens
    decoded_tokens = decode_tokens(tree_data, tokenizer)

    # Draw nodes
    node_width = 1.2
    node_height = 0.8

    for node_id in range(min(draft_token_num, len(expert_lists))):
        if node_id not in pos:
            continue

        x, y = pos[node_id]

        # Get token text and experts
        token_text = decoded_tokens[node_id] if node_id < len(decoded_tokens) else f"tok_{node_id}"
        experts = expert_lists[node_id] if node_id < len(expert_lists) else []

        # Format expert IDs (limit to first 8 for readability)
        if len(experts) <= 8:
            expert_str = str(experts)[1:-1]  # Remove brackets
        else:
            expert_str = str(experts[:8])[1:-1] + "..."

        # Draw node box
        box = FancyBboxPatch(
            (x - node_width/2, y - node_height/2),
            node_width, node_height,
            boxstyle="round,pad=0.05",
            facecolor='lightblue',
            edgecolor='navy',
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)

        # Add text - token on top, experts below
        ax.text(x, y + 0.15, token_text, ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=3)

        # Split expert string into multiple lines if needed
        if len(expert_str) > 20:
            # Split by comma
            parts = expert_str.split(',')
            mid = len(parts) // 2
            line1 = ','.join(parts[:mid])
            line2 = ','.join(parts[mid:])
            ax.text(x, y - 0.1, line1, ha='center', va='center',
                    fontsize=6, color='darkred', zorder=3)
            ax.text(x, y - 0.25, line2, ha='center', va='center',
                    fontsize=6, color='darkred', zorder=3)
        else:
            ax.text(x, y - 0.15, expert_str, ha='center', va='center',
                    fontsize=6, color='darkred', zorder=3)

    # Set title
    meta = sample_data.get('meta_info', {})
    ax.set_title(
        f'Expert Activation Tree - Layer {layer_id} (Verify Step {verify_step})\n'
        f'Question ID: {sample_data.get("question_id", "N/A")} | '
        f'Draft Tokens: {draft_token_num}',
        fontsize=14, fontweight='bold', pad=20
    )

    # Auto-scale with margins
    if pos:
        x_coords = [p[0] for p in pos.values()]
        y_coords = [p[1] for p in pos.values()]
        margin_x = 2
        margin_y = 1
        ax.set_xlim(min(x_coords) - margin_x, max(x_coords) + margin_x)
        ax.set_ylim(min(y_coords) - margin_y, max(y_coords) + margin_y)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Tree visualization saved to: {output_path}")
    print(f"  Layer {layer_id}, Verify step {verify_step}")
    print(f"  Draft tokens: {draft_token_num}")
    print(f"  Tree nodes: {len(pos)}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize expert activation tree for a specific layer'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='expert_activations.json',
        help='Input JSON file with expert activations'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='expert_tree.png',
        help='Output PNG file for visualization'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Sample index to analyze (default: 0)'
    )
    parser.add_argument(
        '--layer-id',
        type=int,
        default=0,
        help='Layer ID to visualize (default: 0)'
    )
    parser.add_argument(
        '--verify-step',
        type=int,
        default=0,
        help='Verify step to visualize (default: 0)'
    )

    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    data = load_expert_data(args.input)

    if not data:
        print("ERROR: No data found in input file")
        return

    if args.sample_idx >= len(data):
        print(f"ERROR: Sample index {args.sample_idx} out of range (max: {len(data)-1})")
        return

    # Get model path from JSON metadata
    model_path = data[args.sample_idx].get('meta_info', {}).get('model_path')
    if not model_path:
        print("ERROR: No model_path found in JSON metadata")
        return

    print(f"Using model_path from JSON: {model_path}")

    # Load tokenizer
    tokenizer = load_tokenizer(model_path)

    sample = data[args.sample_idx]

    print(f"Analyzing sample {args.sample_idx}...")
    print(f"  Question ID: {sample.get('question_id', 'N/A')}")
    print(f"  Total verify passes: {sample.get('meta_info', {}).get('spec_verify_ct', 'N/A')}")

    create_tree_visualization(
        sample,
        layer_id=args.layer_id,
        verify_step=args.verify_step,
        output_path=args.output,
        tokenizer=tokenizer
    )

    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
