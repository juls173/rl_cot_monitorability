import argparse
import json
import os
import re
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12


def load_readability_results(json_path: str) -> Dict:
    """Load readability results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_step_number(dirname: str) -> int:
    """Extract step number from directory name like 'step100'."""
    match = re.match(r'step(\d+)', dirname)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract step number from '{dirname}'")


def collect_model_data(model_dir: str) -> List[Tuple[int, Dict]]:
    """Collect readability data from all step subdirectories in a model directory.
    
    Returns list of (step_number, data_dict) tuples, sorted by step.
    """
    results = []
    
    for entry in os.listdir(model_dir):
        step_path = os.path.join(model_dir, entry)
        if not os.path.isdir(step_path):
            continue
        if not entry.startswith('step'):
            continue
        
        json_path = os.path.join(step_path, 'eval_readability_temp1.json')
        if not os.path.exists(json_path):
            continue
        
        step_num = extract_step_number(entry)
        data = load_readability_results(json_path)
        results.append((step_num, data))
    
    # Sort by step number
    results.sort(key=lambda x: x[0])
    return results


def extract_metrics(data: Dict) -> Tuple[float, float, float]:
    """Extract readability for correct, incorrect, and accuracy from data.
    
    Returns (readability_correct, readability_incorrect, accuracy).
    """
    correctness_stats = data['correctness_stats']
    
    readability_correct = None
    readability_incorrect = None
    count_correct = 0
    count_incorrect = 0
    
    for stat in correctness_stats:
        if stat['is_correct']:
            readability_correct = stat['mean']
            count_correct = stat['count']
        else:
            readability_incorrect = stat['mean']
            count_incorrect = stat['count']
    
    total = count_correct + count_incorrect
    accuracy = count_correct / total if total > 0 else 0.0
    
    return readability_correct, readability_incorrect, accuracy


def plot_readability_comparison(
    model_dirs: List[str],
    labels: List[str],
    output_path: str
):
    """Plot readability comparison across models over training steps.
    
    Args:
        model_dirs: List of paths to model directories
        labels: List of labels for each model
        output_path: Path to save the output plot
    """
    # Collect data for all models
    all_model_data = []
    for model_dir in model_dirs:
        model_data = collect_model_data(model_dir)
        all_model_data.append(model_data)
    
    # Set up distinct colors for models
    distinct_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(labels))]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax_acc = ax.twinx()
    
    # Track all values for dynamic y-axis
    all_readability = []
    all_accuracy = []
    
    for i, (model_data, label) in enumerate(zip(all_model_data, labels)):
        steps = [d[0] for d in model_data]
        readability_correct = []
        readability_incorrect = []
        accuracies = []
        
        for _, data in model_data:
            rc, ri, acc = extract_metrics(data)
            readability_correct.append(rc)
            readability_incorrect.append(ri)
            accuracies.append(acc * 100)  # Convert to percentage
        
        all_readability.extend(readability_correct)
        all_readability.extend(readability_incorrect)
        all_accuracy.extend(accuracies)
        
        color = colors[i]
        # Readability lines (solid for correct, dashed for incorrect)
        ax.plot(steps, readability_correct, 'o-', color=color, label=f'{label} (correct)', linewidth=2)
        ax.plot(steps, readability_incorrect, 's--', color=color, label=f'{label} (incorrect)', linewidth=2, alpha=0.7)
        # Accuracy line (dotted)
        ax_acc.plot(steps, accuracies, '^:', color=color, label=f'{label} (accuracy)', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Readability Score', fontsize=14, fontweight='bold')
    ax_acc.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='gray')
    ax.set_title('Readability and Accuracy Over Training', fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    # Dynamic y-axis limits with padding (same range for both axes)
    all_readability = [v for v in all_readability if v is not None]
    all_accuracy = [v for v in all_accuracy if v is not None]
    all_values = all_readability + all_accuracy
    
    if all_values:
        val_min, val_max = min(all_values), max(all_values)
        val_range = val_max - val_min
        ylim = (max(0, val_min - val_range * 0.1), min(100, val_max + val_range * 0.1))
        ax.set_ylim(ylim)
        ax_acc.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot readability comparison across models over training steps"
    )
    
    parser.add_argument(
        "--model-dirs",
        nargs='+',
        required=True,
        help="Paths to model directories containing step### subdirectories"
    )
    parser.add_argument(
        "--labels",
        nargs='+',
        required=True,
        help="Labels for each model (must match number of model-dirs)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output plot"
    )
    
    args = parser.parse_args()
    
    if len(args.model_dirs) != len(args.labels):
        parser.error("Number of --model-dirs must match number of --labels")
    
    plot_readability_comparison(
        model_dirs=args.model_dirs,
        labels=args.labels,
        output_path=args.output
    )

