#!/usr/bin/env python3
"""
Analyze Expert Usage Statistics Per Layer Across EAGLE Verify Steps

This script computes average, 90th percentile, and 99th percentile of unique
experts used per layer across all verify cycles for a specific sample.

Usage:
    python viz_expert_usage_statistics.py [--input expert_activations.json] [--output statistics.png]
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


def compute_layer_statistics(sample_data: dict) -> dict:
    """
    Compute statistics (mean, 90th percentile, 99th percentile) of unique
    experts used per layer across all verify cycles.

    Returns:
        Dictionary mapping layer_id to statistics dict
    """
    history = sample_data.get('expert_layer_history', [])
    if not history:
        return {}

    # Collect expert counts for each layer across all verify passes
    layer_counts = defaultdict(list)

    for verify_snapshot in history:
        layer_experts = analyze_verify_pass(verify_snapshot)
        for layer_id, count in layer_experts.items():
            layer_counts[layer_id].append(count)

    # Compute statistics for each layer
    statistics = {}
    for layer_id, counts in layer_counts.items():
        statistics[layer_id] = {
            'mean': np.mean(counts),
            'p90': np.percentile(counts, 90),
            'p99': np.percentile(counts, 99),
            'min': np.min(counts),
            'max': np.max(counts),
            'std': np.std(counts),
            'num_samples': len(counts)
        }

    return statistics


def create_visualization(
    sample_data: dict,
    output_path: str = 'expert_usage_statistics.png'
):
    """
    Create a grouped bar chart showing mean, 90th percentile, and 99th percentile
    of unique experts used per layer across verify cycles.
    """
    statistics = compute_layer_statistics(sample_data)

    if not statistics:
        print("No statistics to plot")
        return

    # Sort layers by layer_id
    layer_ids = sorted(statistics.keys())

    # Extract statistics
    means = [statistics[lid]['mean'] for lid in layer_ids]
    p90s = [statistics[lid]['p90'] for lid in layer_ids]
    p99s = [statistics[lid]['p99'] for lid in layer_ids]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(layer_ids))
    width = 0.25

    bars1 = ax.bar(x - width, means, width, label='Mean',
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, p90s, width, label='90th Percentile',
                   color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, p99s, width, label='99th Percentile',
                   color='mediumseagreen', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Customize plot
    ax.set_xlabel('Layer ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Unique Experts', fontsize=14, fontweight='bold')

    # Compute average draft tokens per cycle
    history = sample_data.get('expert_layer_history', [])
    draft_tokens = []
    for verify_snapshot in history:
        tree_struct = verify_snapshot.get('tree_structure', {})
        if 'draft_token_num' in tree_struct:
            draft_tokens.append(tree_struct['draft_token_num'])

    avg_draft_tokens = np.mean(draft_tokens) if draft_tokens else 0

    meta = sample_data.get('meta_info', {})
    ax.set_title(
        f'Expert Usage Statistics Per Layer Across Verify Cycles\n'
        f'Question ID: {sample_data.get("question_id", "N/A")} | '
        f'Total Verify Passes: {meta.get("spec_verify_ct", "N/A")} | '
        f'Completion Tokens: {meta.get("completion_tokens", "N/A")} | '
        f'Avg Draft Tokens/Cycle: {avg_draft_tokens:.1f}',
        fontsize=13, fontweight='bold', pad=15
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(lid) for lid in layer_ids])
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Print summary statistics
    print("\nPer-Layer Statistics:")
    print(f"{'Layer':<8} {'Mean':<8} {'P90':<8} {'P99':<8} {'Min':<8} {'Max':<8} {'Std':<8}")
    print("-" * 60)
    for layer_id in layer_ids:
        stats = statistics[layer_id]
        print(f"{layer_id:<8} {stats['mean']:<8.1f} {stats['p90']:<8.1f} "
              f"{stats['p99']:<8.1f} {stats['min']:<8} {stats['max']:<8} "
              f"{stats['std']:<8.2f}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute and visualize expert usage statistics per layer'
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
        default='expert_usage_statistics.png',
        help='Output PNG file for visualization'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Sample index to analyze (default: 0)'
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

    print(f"Analyzing sample {args.sample_idx}...")
    print(f"  Question ID: {sample.get('question_id', 'N/A')}")
    print(f"  Completion tokens: {sample.get('meta_info', {}).get('completion_tokens', 'N/A')}")
    print(f"  Verify passes: {sample.get('meta_info', {}).get('spec_verify_ct', 'N/A')}")

    create_visualization(sample, output_path=args.output)

    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
