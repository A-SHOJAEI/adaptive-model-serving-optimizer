#!/usr/bin/env python3
"""Evaluation script for adaptive model serving optimizer.

This script provides comprehensive evaluation of trained serving strategies,
including performance benchmarking, A/B testing analysis, and monitoring metrics.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adaptive_model_serving_optimizer import (
    Config,
    get_config,
    setup_environment_from_config,
    ModelLoader,
    BenchmarkDataset,
    ModelAdapterFactory,
    ServingStrategyOptimizer,
    MetricsCollector,
    ModelDriftDetector
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )


def load_experiment_results(results_path: Path) -> Dict[str, Any]:
    """Load experiment results from training.

    Args:
        results_path: Path to training results file

    Returns:
        Dictionary containing experiment results
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, 'r') as f:
        results = json.load(f)

    logger.info(f"Loaded experiment results from {results_path}")
    return results


def create_performance_comparison_plot(
    results: Dict[str, Any],
    output_dir: Path
) -> None:
    """Create performance comparison plots.

    Args:
        results: Experiment results
        output_dir: Output directory for plots
    """
    logger.info("Creating performance comparison plots")

    baseline_results = results['baseline_results']
    final_results = results['final_results']

    # Extract strategy names and metrics
    strategies = list(baseline_results.keys())
    metrics = ['mean_latency_ms', 'p99_latency_ms', 'mean_accuracy', 'mean_throughput']

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Comparison: Baseline vs Optimized', fontsize=16)

    metric_titles = {
        'mean_latency_ms': 'Mean Latency (ms)',
        'p99_latency_ms': 'P99 Latency (ms)',
        'mean_accuracy': 'Mean Accuracy',
        'mean_throughput': 'Mean Throughput (samples/sec)'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        baseline_values = [baseline_results[strategy][metric] for strategy in strategies]
        final_values = [final_results[strategy][metric] for strategy in strategies]

        x = np.arange(len(strategies))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width/2, final_values, width, label='Optimized', alpha=0.8)

        ax.set_xlabel('Serving Strategy')
        ax.set_ylabel(metric_titles[metric])
        ax.set_title(metric_titles[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()

        # Add value labels on bars
        for bar1, bar2 in zip(bars1, bars2):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{height1:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{height2:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "performance_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Performance comparison plot saved to {plot_path}")
    plt.close()


def create_optimization_progress_plot(
    experiment_data_path: Path,
    output_dir: Path
) -> None:
    """Create optimization progress plots.

    Args:
        experiment_data_path: Path to experiment data file
        output_dir: Output directory for plots
    """
    logger.info("Creating optimization progress plots")

    # Load experiment data
    with open(experiment_data_path, 'r') as f:
        data = json.load(f)

    experiment_history = data['experiment_history']

    if not experiment_history:
        logger.warning("No experiment history found, skipping progress plots")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(experiment_history)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Optimization Progress Over Time', fontsize=16)

    # Plot 1: Reward over time by strategy
    ax1 = axes[0, 0]
    strategies = df['strategy'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    for strategy, color in zip(strategies, colors):
        strategy_data = df[df['strategy'] == strategy]
        ax1.plot(strategy_data.index, strategy_data['reward'],
                label=strategy, color=color, alpha=0.7)

        # Add rolling average
        rolling_reward = strategy_data['reward'].rolling(window=50, min_periods=1).mean()
        ax1.plot(strategy_data.index, rolling_reward,
                color=color, linewidth=2)

    ax1.set_xlabel('Experiment Number')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Progress by Strategy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Latency over time
    ax2 = axes[0, 1]
    for strategy, color in zip(strategies, colors):
        strategy_data = df[df['strategy'] == strategy]
        rolling_latency = strategy_data['latency'].rolling(window=50, min_periods=1).mean()
        ax2.plot(strategy_data.index, rolling_latency,
                label=strategy, color=color)

    ax2.set_xlabel('Experiment Number')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Progress (50-point rolling average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Throughput over time
    ax3 = axes[1, 0]
    for strategy, color in zip(strategies, colors):
        strategy_data = df[df['strategy'] == strategy]
        rolling_throughput = strategy_data['throughput'].rolling(window=50, min_periods=1).mean()
        ax3.plot(strategy_data.index, rolling_throughput,
                label=strategy, color=color)

    ax3.set_xlabel('Experiment Number')
    ax3.set_ylabel('Throughput (samples/sec)')
    ax3.set_title('Throughput Progress (50-point rolling average)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Strategy selection frequency
    ax4 = axes[1, 1]
    window_size = 100
    strategy_counts = []
    x_positions = []

    for i in range(window_size, len(df), 10):
        window_data = df.iloc[i-window_size:i]
        strategy_freq = window_data['strategy'].value_counts(normalize=True)

        for strategy in strategies:
            freq = strategy_freq.get(strategy, 0)
            strategy_counts.append([i, strategy, freq])

    if strategy_counts:
        strategy_df = pd.DataFrame(strategy_counts, columns=['position', 'strategy', 'frequency'])

        for strategy, color in zip(strategies, colors):
            strategy_freq_data = strategy_df[strategy_df['strategy'] == strategy]
            ax4.plot(strategy_freq_data['position'], strategy_freq_data['frequency'],
                    label=strategy, color=color, linewidth=2)

    ax4.set_xlabel('Experiment Number')
    ax4.set_ylabel('Selection Frequency')
    ax4.set_title(f'Strategy Selection Frequency ({window_size}-experiment window)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "optimization_progress.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Optimization progress plot saved to {plot_path}")
    plt.close()


def perform_statistical_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Perform statistical analysis of results.

    Args:
        results: Experiment results

    Returns:
        Statistical analysis results
    """
    logger.info("Performing statistical analysis")

    baseline_results = results['baseline_results']
    final_results = results['final_results']
    improvements = results['improvements']

    analysis = {
        'target_metrics_achieved': {},
        'statistical_significance': {},
        'performance_rankings': {},
        'efficiency_analysis': {}
    }

    # Define target metrics from project specification
    target_metrics = {
        'p99_latency_reduction_percent': 40,
        'throughput_improvement_percent': 60,
        'model_accuracy_degradation_max_percent': 0.5,
        'serving_cost_reduction_percent': 35
    }

    # Check if target metrics were achieved
    for metric, target in target_metrics.items():
        achieved_value = improvements.get(metric, 0)

        if metric == 'model_accuracy_degradation_max_percent':
            # For accuracy degradation, lower is better
            achieved = abs(achieved_value) <= target
        else:
            # For other metrics, higher is better
            achieved = achieved_value >= target

        analysis['target_metrics_achieved'][metric] = {
            'target': target,
            'achieved': achieved_value,
            'met_target': achieved,
            'performance_ratio': achieved_value / target if target > 0 else float('inf')
        }

    # Performance rankings
    latency_ranking = sorted(
        final_results.items(),
        key=lambda x: x[1]['p99_latency_ms']
    )
    throughput_ranking = sorted(
        final_results.items(),
        key=lambda x: x[1]['mean_throughput'],
        reverse=True
    )
    accuracy_ranking = sorted(
        final_results.items(),
        key=lambda x: x[1]['mean_accuracy'],
        reverse=True
    )

    analysis['performance_rankings'] = {
        'latency_ranking': [{'strategy': name, 'p99_latency_ms': data['p99_latency_ms']}
                           for name, data in latency_ranking],
        'throughput_ranking': [{'strategy': name, 'mean_throughput': data['mean_throughput']}
                              for name, data in throughput_ranking],
        'accuracy_ranking': [{'strategy': name, 'mean_accuracy': data['mean_accuracy']}
                            for name, data in accuracy_ranking]
    }

    # Efficiency analysis (throughput per unit latency)
    efficiency_scores = {}
    for strategy, data in final_results.items():
        if data['p99_latency_ms'] > 0:
            efficiency = data['mean_throughput'] / data['p99_latency_ms']
            efficiency_scores[strategy] = efficiency

    efficiency_ranking = sorted(
        efficiency_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    analysis['efficiency_analysis'] = {
        'efficiency_scores': efficiency_scores,
        'efficiency_ranking': efficiency_ranking,
        'best_efficiency_strategy': efficiency_ranking[0][0] if efficiency_ranking else None
    }

    return analysis


def generate_evaluation_report(
    results: Dict[str, Any],
    analysis: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate comprehensive evaluation report.

    Args:
        results: Experiment results
        analysis: Statistical analysis results
        output_dir: Output directory for report
    """
    logger.info("Generating evaluation report")

    report_path = output_dir / "evaluation_report.md"

    with open(report_path, 'w') as f:
        f.write("# Adaptive Model Serving Optimizer - Evaluation Report\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        best_strategy = results['improvements']['best_strategy']
        f.write(f"**Best Strategy:** {best_strategy}\n\n")

        target_metrics = analysis['target_metrics_achieved']
        met_targets = sum(1 for metric_data in target_metrics.values() if metric_data['met_target'])
        total_targets = len(target_metrics)

        f.write(f"**Target Achievement:** {met_targets}/{total_targets} targets met\n\n")

        # Performance Improvements
        f.write("## Performance Improvements\n\n")
        improvements = results['improvements']

        improvements_table = [
            ["Metric", "Target", "Achieved", "Status"],
            ["---", "---", "---", "---"]
        ]

        for metric, data in target_metrics.items():
            status = "✅ Met" if data['met_target'] else "❌ Not Met"
            improvements_table.append([
                metric.replace('_', ' ').title(),
                f"{data['target']:.1f}%",
                f"{data['achieved']:.1f}%",
                status
            ])

        for row in improvements_table:
            f.write("| " + " | ".join(row) + " |\n")

        f.write("\n")

        # Strategy Performance
        f.write("## Strategy Performance Comparison\n\n")
        final_results = results['final_results']

        performance_table = [
            ["Strategy", "P99 Latency (ms)", "Mean Accuracy", "Throughput (samples/s)", "Efficiency Score"],
            ["---", "---", "---", "---", "---"]
        ]

        for strategy, data in final_results.items():
            efficiency = analysis['efficiency_analysis']['efficiency_scores'].get(strategy, 0)
            performance_table.append([
                strategy,
                f"{data['p99_latency_ms']:.2f}",
                f"{data['mean_accuracy']:.4f}",
                f"{data['mean_throughput']:.2f}",
                f"{efficiency:.3f}"
            ])

        for row in performance_table:
            f.write("| " + " | ".join(row) + " |\n")

        f.write("\n")

        # Rankings
        f.write("## Performance Rankings\n\n")
        rankings = analysis['performance_rankings']

        f.write("### Latency Ranking (Best to Worst)\n")
        for i, item in enumerate(rankings['latency_ranking'], 1):
            f.write(f"{i}. **{item['strategy']}** - {item['p99_latency_ms']:.2f} ms\n")

        f.write("\n### Throughput Ranking (Best to Worst)\n")
        for i, item in enumerate(rankings['throughput_ranking'], 1):
            f.write(f"{i}. **{item['strategy']}** - {item['mean_throughput']:.2f} samples/s\n")

        f.write("\n### Efficiency Ranking (Best to Worst)\n")
        for i, (strategy, score) in enumerate(analysis['efficiency_analysis']['efficiency_ranking'], 1):
            f.write(f"{i}. **{strategy}** - {score:.3f} throughput/latency\n")

        f.write("\n")

        # Optimization Details
        f.write("## Optimization Process Details\n\n")
        optimization_report = results['optimization_report']

        f.write(f"- **Algorithm Used:** {results['config']['bandit_algorithm']}\n")
        f.write(f"- **Total Experiments:** {optimization_report['total_experiments']}\n")
        f.write(f"- **Convergence Achieved:** {'Yes' if optimization_report['convergence_achieved'] else 'No'}\n")
        f.write(f"- **Optimization Duration:** {results['experiment_duration_minutes']:.1f} minutes\n")
        f.write(f"- **Best Strategy Reward:** {optimization_report['best_reward']:.4f}\n\n")

        # Strategy Statistics
        f.write("### Strategy Selection Statistics\n\n")
        strategy_stats = optimization_report['strategy_statistics']

        stats_table = [
            ["Strategy", "Total Pulls", "Avg Reward", "P95 Latency (ms)", "Error Rate"],
            ["---", "---", "---", "---", "---"]
        ]

        for strategy, stats in strategy_stats.items():
            stats_table.append([
                strategy,
                str(stats['pulls']),
                f"{stats['avg_reward']:.4f}",
                f"{stats['p95_latency_ms']:.2f}",
                f"{stats['error_rate']:.4f}"
            ])

        for row in stats_table:
            f.write("| " + " | ".join(row) + " |\n")

        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        best_efficiency_strategy = analysis['efficiency_analysis']['best_efficiency_strategy']
        if best_efficiency_strategy:
            f.write(f"1. **Deploy {best_efficiency_strategy}** as the primary serving strategy for optimal efficiency\n")

        unmet_targets = [metric for metric, data in target_metrics.items() if not data['met_target']]
        if unmet_targets:
            f.write("2. **Focus on improving:** ")
            f.write(", ".join(metric.replace('_', ' ').title() for metric in unmet_targets))
            f.write("\n")

        f.write("3. **Monitor continuously** for performance drift and adjust strategies accordingly\n")
        f.write("4. **Consider additional optimizations** such as model pruning or quantization\n")
        f.write("5. **Implement gradual rollout** of the optimized serving strategy in production\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        if met_targets == total_targets:
            f.write("✅ **All performance targets achieved!** The optimization was successful.\n")
        elif met_targets >= total_targets // 2:
            f.write("⚠️ **Most targets achieved.** The optimization shows good results with room for improvement.\n")
        else:
            f.write("❌ **Limited success.** Further optimization needed to meet performance targets.\n")

    logger.info(f"Evaluation report saved to {report_path}")


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate adaptive model serving optimizer")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to training results JSON file")
    parser.add_argument("--experiment-data", type=str,
                       help="Path to experiment data JSON file")
    parser.add_argument("--output-dir", type=str, default="./evaluation_outputs",
                       help="Output directory for evaluation results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting evaluation of adaptive model serving optimizer")

        # Load results
        results_path = Path(args.results)
        results = load_experiment_results(results_path)

        # Create performance comparison plots
        create_performance_comparison_plot(results, output_dir)

        # Create optimization progress plots if experiment data is provided
        if args.experiment_data:
            experiment_data_path = Path(args.experiment_data)
            if experiment_data_path.exists():
                create_optimization_progress_plot(experiment_data_path, output_dir)
            else:
                logger.warning(f"Experiment data file not found: {experiment_data_path}")

        # Perform statistical analysis
        analysis = perform_statistical_analysis(results)

        # Save analysis results
        analysis_path = output_dir / "statistical_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Generate comprehensive report
        generate_evaluation_report(results, analysis, output_dir)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)

        best_strategy = results['improvements']['best_strategy']
        target_metrics = analysis['target_metrics_achieved']
        met_targets = sum(1 for data in target_metrics.values() if data['met_target'])
        total_targets = len(target_metrics)

        logger.info(f"Best strategy: {best_strategy}")
        logger.info(f"Targets achieved: {met_targets}/{total_targets}")

        for metric, data in target_metrics.items():
            status = "✅" if data['met_target'] else "❌"
            logger.info(f"  {status} {metric}: {data['achieved']:.1f}% (target: {data['target']:.1f}%)")

        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()