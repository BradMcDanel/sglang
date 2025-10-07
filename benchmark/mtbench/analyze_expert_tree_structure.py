#!/usr/bin/env python3
"""
Analyze Expert Usage Across EAGLE Verify Steps

This script tracks the number of unique experts used per layer across EAGLE verify steps.

Usage:
    python analyze_expert_tree_structure.py [--input expert_activations.json] [--output analysis.png]
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_expert_data(filepath: str) -> dict:
    """Load expert activation data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_verify_pass(verify_snapshot: dict) -> dict:
    """
    Analyze expert usage for a single verify pass.

    Returns:
        Dictionary with per-layer unique expert counts
    """
    layers = verify_snapshot.get('layers', verify_snapshot)

    layer_unique_experts = {}

    for layer_id_str, layer_data in layers.items():
        if layer_id_str == 'tree_structure':
            continue

        layer_id = int(layer_id_str)
        topk_ids = layer_data.get('topk_ids', [])

        if not topk_ids:
            continue

        # Collect all unique experts used in this layer
        all_experts = set()
        for token_experts in topk_ids:
            all_experts.update([e for e in token_experts if e >= 0])

        layer_unique_experts[layer_id] = len(all_experts)

    return layer_unique_experts


def create_visualization(
    sample_data: dict,
    layer_id: int,
    output_path: str = 'expert_tree_analysis.png'
):
    """
    Create a simple visualization showing unique experts per verify step for a single layer.

    X-axis: Verify step number
    Y-axis: Number of unique experts used in the specified layer
    """

    history = sample_data.get('expert_layer_history', [])
    if not history:
        print("No expert_layer_history found in sample")
        return

    num_passes = len(history)

    # Analyze all verify passes for the specified layer
    expert_counts = []
    for i in range(num_passes):
        layer_experts = analyze_verify_pass(history[i])

        if layer_id in layer_experts:
            expert_counts.append(layer_experts[layer_id])
        else:
            expert_counts.append(0)

    # Extract data for plotting
    verify_steps = list(range(num_passes))

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(verify_steps, expert_counts, 'o-', linewidth=2.5, markersize=4,
            color='steelblue', alpha=0.8)

    ax.set_xlabel('Verify Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Unique Experts Used', fontsize=14, fontweight='bold')

    meta = sample_data.get('meta_info', {})
    ax.set_title(
        f'Expert Usage Across Verify Steps (Layer {layer_id})\n'
        f'Question ID: {sample_data.get("question_id", "N/A")} | '
        f'Total Verify Passes: {meta.get("spec_verify_ct", "N/A")} | '
        f'Completion Tokens: {meta.get("completion_tokens", "N/A")}',
        fontsize=13, fontweight='bold', pad=15
    )

    # Add statistics
    avg_experts = np.mean(expert_counts)
    std_experts = np.std(expert_counts)
    ax.axhline(avg_experts, color='red', linestyle='--', linewidth=1.5,
               alpha=0.6, label=f'Mean: {avg_experts:.1f} ± {std_experts:.1f}')

    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-5, num_passes + 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"  Layer {layer_id} statistics:")
    print(f"    Mean unique experts: {avg_experts:.1f}")
    print(f"    Std deviation: {std_experts:.1f}")
    print(f"    Min: {min(expert_counts)}, Max: {max(expert_counts)}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze expert usage across EAGLE verify steps for a single layer'
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
        default='expert_tree_analysis.png',
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
        help='Layer ID to analyze (default: 0)'
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

    sample = data[args.sample_idx]

    print(f"Analyzing sample {args.sample_idx}, layer {args.layer_id}...")
    print(f"  Question ID: {sample.get('question_id', 'N/A')}")
    print(f"  Completion tokens: {sample.get('meta_info', {}).get('completion_tokens', 'N/A')}")
    print(f"  Verify passes: {sample.get('meta_info', {}).get('spec_verify_ct', 'N/A')}")

    create_visualization(
        sample,
        layer_id=args.layer_id,
        output_path=args.output
    )

    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
