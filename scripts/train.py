#!/usr/bin/env python3
"""Training script for adaptive model serving optimizer.

This script orchestrates the training of serving strategy optimization using
multi-armed bandit algorithms with comprehensive MLflow tracking.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adaptive_model_serving_optimizer import (
    Config,
    get_config,
    setup_environment_from_config,
    ModelLoader,
    DatasetLoader,
    BenchmarkDataset,
    ModelOptimizer,
    PerformanceBenchmark,
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
            logging.FileHandler('training.log')
        ]
    )


def create_sample_models(config: Config, temp_dir: Path) -> dict:
    """Create sample models for demonstration.

    Args:
        config: Configuration object
        temp_dir: Temporary directory for models

    Returns:
        Dictionary of model paths
    """
    import torch.nn as nn
    import torchvision.models as models

    logger.info("Creating sample models for serving strategy optimization")

    # Create a simple ResNet model
    model = models.resnet18(pretrained=False, num_classes=10)
    model.eval()

    model_paths = {}

    # Save PyTorch model
    pytorch_path = temp_dir / "resnet18_pytorch.pth"
    torch.save(model, pytorch_path)
    model_paths['pytorch'] = pytorch_path

    logger.info(f"Created PyTorch model: {pytorch_path}")

    # Convert to ONNX (simplified for demo)
    try:
        optimizer = ModelOptimizer(config)
        sample_input = torch.randn(1, 3, 224, 224)

        onnx_path = temp_dir / "resnet18.onnx"
        optimizer.convert_to_onnx(model, sample_input, onnx_path)
        model_paths['onnx'] = onnx_path

        logger.info(f"Created ONNX model: {onnx_path}")
    except Exception as e:
        logger.warning(f"Failed to create ONNX model: {e}")

    # Note: TensorRT conversion would require GPU and proper setup
    # For demo, we'll skip it but the framework supports it

    return model_paths


def create_serving_adapters(config: Config, model_paths: dict) -> dict:
    """Create model serving adapters.

    Args:
        config: Configuration object
        model_paths: Dictionary of model paths

    Returns:
        Dictionary of serving adapters
    """
    adapters = {}

    # Create PyTorch adapters with different configurations
    if 'pytorch' in model_paths:
        # Fast PyTorch adapter (optimized)
        fast_config = Config(**config.__dict__)
        fast_config.serving.pytorch_config.update({
            'precision': 'float16',
            'jit_compile': True,
            'compile_mode': 'reduce-overhead'
        })

        adapters['pytorch_fast'] = ModelAdapterFactory.create_adapter(
            fast_config, 'pytorch', model_paths['pytorch']
        )

        # Standard PyTorch adapter
        adapters['pytorch_standard'] = ModelAdapterFactory.create_adapter(
            config, 'pytorch', model_paths['pytorch']
        )

    # Create ONNX adapter
    if 'onnx' in model_paths:
        adapters['onnx_optimized'] = ModelAdapterFactory.create_adapter(
            config, 'onnx', model_paths['onnx']
        )

    logger.info(f"Created {len(adapters)} serving adapters")
    return adapters


def run_serving_experiment(
    optimizer: ServingStrategyOptimizer,
    adapters: dict,
    benchmark_dataset: BenchmarkDataset,
    num_experiments: int = 1000
) -> None:
    """Run serving strategy optimization experiment.

    Args:
        optimizer: Serving strategy optimizer
        adapters: Dictionary of serving adapters
        benchmark_dataset: Dataset for benchmarking
        num_experiments: Number of experiments to run
    """
    logger.info(f"Starting serving experiment with {num_experiments} iterations")

    # Create data loader for benchmarking
    data_loader = torch.utils.data.DataLoader(
        benchmark_dataset,
        batch_size=8,
        shuffle=True
    )

    experiment_data = []

    for i in range(num_experiments):
        # Select serving strategy
        strategy_name, adapter = optimizer.select_serving_strategy()

        # Warmup adapter if needed
        if not adapter.warmup_completed:
            adapter.warmup()

        # Get a batch of data
        batch_inputs, batch_targets = next(iter(data_loader))

        try:
            # Measure inference performance
            start_time = time.perf_counter()
            predictions = adapter.predict(batch_inputs)
            end_time = time.perf_counter()

            inference_time = (end_time - start_time) * 1000  # Convert to ms
            batch_size = batch_inputs.shape[0]
            throughput = batch_size / (inference_time / 1000)  # samples/sec

            # Calculate mock accuracy (in real scenario, use ground truth)
            if isinstance(predictions, torch.Tensor):
                predicted_classes = torch.argmax(predictions, dim=1).cpu()
                accuracy = (predicted_classes == batch_targets.cpu()).float().mean().item()
            else:
                # For numpy predictions
                predicted_classes = torch.from_numpy(predictions).argmax(dim=1).cpu()
                accuracy = (predicted_classes == batch_targets.cpu()).float().mean().item()

            # Mock error rate (could be based on actual errors in production)
            error_rate = max(0.0, 0.05 - accuracy)  # Lower accuracy = higher error rate

            # Update strategy performance
            optimizer.update_strategy_performance(
                strategy_name=strategy_name,
                latency=inference_time,
                throughput=throughput,
                accuracy=accuracy,
                error_rate=error_rate
            )

            # Log to MLflow
            mlflow.log_metrics({
                f"{strategy_name}_latency_ms": inference_time,
                f"{strategy_name}_throughput": throughput,
                f"{strategy_name}_accuracy": accuracy,
                f"{strategy_name}_error_rate": error_rate,
                "experiment_iteration": i
            })

            # Store experiment data
            experiment_data.append({
                'iteration': i,
                'strategy': strategy_name,
                'latency_ms': inference_time,
                'throughput': throughput,
                'accuracy': accuracy,
                'error_rate': error_rate
            })

            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i + 1}/{num_experiments} experiments")

                # Log current performance summary
                arm_stats = optimizer.bandit.get_arm_stats()
                for arm_name, arm in arm_stats.items():
                    if arm.pulls > 0:
                        logger.info(
                            f"  {arm_name}: pulls={arm.pulls}, "
                            f"avg_reward={arm.avg_reward:.4f}, "
                            f"p95_latency={arm.get_p95_latency():.2f}ms"
                        )

        except Exception as e:
            logger.error(f"Experiment {i} failed with strategy {strategy_name}: {e}")
            # Update with poor performance to penalize failing strategy
            optimizer.update_strategy_performance(
                strategy_name=strategy_name,
                latency=1000.0,  # High latency penalty
                throughput=1.0,   # Low throughput penalty
                accuracy=0.0,     # Zero accuracy penalty
                error_rate=1.0    # Maximum error rate
            )

        # Check for convergence
        if i > 500 and optimizer.check_convergence():
            logger.info(f"Optimization converged after {i + 1} experiments")
            break

    return experiment_data


def evaluate_final_performance(
    optimizer: ServingStrategyOptimizer,
    adapters: dict,
    benchmark_dataset: BenchmarkDataset
) -> dict:
    """Evaluate final performance of all strategies.

    Args:
        optimizer: Serving strategy optimizer
        adapters: Dictionary of serving adapters
        benchmark_dataset: Dataset for evaluation

    Returns:
        Performance evaluation results
    """
    logger.info("Evaluating final performance of all strategies")

    evaluation_results = {}

    # Create larger batch for final evaluation
    data_loader = torch.utils.data.DataLoader(
        benchmark_dataset,
        batch_size=32,
        shuffle=False
    )

    for strategy_name, adapter in adapters.items():
        if not adapter.warmup_completed:
            adapter.warmup()

        latencies = []
        accuracies = []
        throughputs = []

        for batch_inputs, batch_targets in data_loader:
            # Measure performance
            start_time = time.perf_counter()
            predictions = adapter.predict(batch_inputs)
            end_time = time.perf_counter()

            inference_time = (end_time - start_time) * 1000
            batch_size = batch_inputs.shape[0]
            throughput = batch_size / (inference_time / 1000)

            # Calculate accuracy - ensure same device
            if isinstance(predictions, torch.Tensor):
                predicted_classes = torch.argmax(predictions, dim=1).cpu()
            else:
                predicted_classes = torch.from_numpy(predictions).argmax(dim=1).cpu()

            accuracy = (predicted_classes == batch_targets.cpu()).float().mean().item()

            latencies.append(inference_time)
            accuracies.append(accuracy)
            throughputs.append(throughput)

        # Calculate summary statistics
        evaluation_results[strategy_name] = {
            'mean_latency_ms': float(torch.tensor(latencies).mean()),
            'p95_latency_ms': float(torch.tensor(latencies).quantile(0.95)),
            'p99_latency_ms': float(torch.tensor(latencies).quantile(0.99)),
            'mean_accuracy': float(torch.tensor(accuracies).mean()),
            'mean_throughput': float(torch.tensor(throughputs).mean()),
            'total_samples': len(benchmark_dataset)
        }

        logger.info(f"{strategy_name} - Final Performance:")
        logger.info(f"  Mean Latency: {evaluation_results[strategy_name]['mean_latency_ms']:.2f}ms")
        logger.info(f"  P99 Latency: {evaluation_results[strategy_name]['p99_latency_ms']:.2f}ms")
        logger.info(f"  Mean Accuracy: {evaluation_results[strategy_name]['mean_accuracy']:.4f}")
        logger.info(f"  Mean Throughput: {evaluation_results[strategy_name]['mean_throughput']:.2f}/s")

    return evaluation_results


def calculate_improvements(
    baseline_results: dict,
    optimized_results: dict
) -> dict:
    """Calculate performance improvements.

    Args:
        baseline_results: Baseline performance results
        optimized_results: Optimized performance results

    Returns:
        Performance improvement metrics
    """
    # Find baseline and best optimized strategy
    baseline_strategy = min(baseline_results.keys(), key=lambda k: baseline_results[k]['mean_latency_ms'])
    best_strategy = min(optimized_results.keys(), key=lambda k: optimized_results[k]['mean_latency_ms'])

    baseline_perf = baseline_results[baseline_strategy]
    best_perf = optimized_results[best_strategy]

    improvements = {
        'baseline_strategy': baseline_strategy,
        'best_strategy': best_strategy,
        'p99_latency_reduction_percent': (
            (baseline_perf['p99_latency_ms'] - best_perf['p99_latency_ms']) /
            baseline_perf['p99_latency_ms'] * 100
        ),
        'throughput_improvement_percent': (
            (best_perf['mean_throughput'] - baseline_perf['mean_throughput']) /
            baseline_perf['mean_throughput'] * 100
        ),
        'model_accuracy_degradation_max_percent': (
            (baseline_perf['mean_accuracy'] - best_perf['mean_accuracy']) /
            baseline_perf['mean_accuracy'] * 100
        )
    }

    # Estimate cost reduction (simplified model)
    # Assume cost is proportional to latency * compute resources
    baseline_cost_factor = baseline_perf['mean_latency_ms']
    optimized_cost_factor = best_perf['mean_latency_ms']

    improvements['serving_cost_reduction_percent'] = (
        (baseline_cost_factor - optimized_cost_factor) /
        baseline_cost_factor * 100
    )

    return improvements


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train adaptive model serving optimizer")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--experiments", type=int, default=1000,
                       help="Number of optimization experiments")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    try:
        # Load configuration
        config = get_config(args.config)
        setup_environment_from_config(config)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting adaptive model serving optimization training")
        logger.info(f"Configuration: {config.project_name} v{config.version}")
        logger.info(f"Bandit algorithm: {config.bandits.algorithm}")
        logger.info(f"Number of experiments: {args.experiments}")

        # Setup MLflow tracking (use local file store instead of remote server)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(config.mlflow.experiment_name)

        with mlflow.start_run():
            # Log configuration parameters
            mlflow.log_params({
                'bandit_algorithm': config.bandits.algorithm,
                'bandit_epsilon': config.bandits.epsilon,
                'bandit_alpha': config.bandits.alpha,
                'bandit_beta': config.bandits.beta,
                'num_experiments': args.experiments,
                'device': config.device,
                'mixed_precision': config.mixed_precision
            })

            # Create temporary directory for models
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create sample models
                model_paths = create_sample_models(config, temp_path)

                # Create serving adapters
                adapters = create_serving_adapters(config, model_paths)

                if not adapters:
                    logger.error("No serving adapters created. Cannot proceed.")
                    return

                # Create benchmark dataset
                benchmark_dataset = BenchmarkDataset(
                    num_samples=1000,
                    input_shape=(3, 224, 224),
                    num_classes=10
                )

                # Initialize serving strategy optimizer
                optimizer = ServingStrategyOptimizer(config)

                # Register adapters
                for name, adapter in adapters.items():
                    optimizer.register_serving_adapter(name, adapter)

                # Run baseline evaluation (before optimization)
                logger.info("Running baseline evaluation...")
                baseline_results = evaluate_final_performance(
                    optimizer, adapters, benchmark_dataset
                )

                # Reset optimizer for fresh start
                optimizer.bandit.reset()

                # Run serving optimization experiment
                experiment_start_time = time.time()
                experiment_data = run_serving_experiment(
                    optimizer, adapters, benchmark_dataset, args.experiments
                )
                experiment_duration = time.time() - experiment_start_time

                # Final evaluation after optimization
                logger.info("Running final evaluation...")
                final_results = evaluate_final_performance(
                    optimizer, adapters, benchmark_dataset
                )

                # Generate optimization report
                report = optimizer.get_optimization_report()

                # Calculate improvements
                improvements = calculate_improvements(baseline_results, final_results)

                # Log final metrics to MLflow
                mlflow.log_metrics({
                    'experiment_duration_minutes': experiment_duration / 60,
                    'adaptation_convergence_minutes': experiment_duration / 60,  # Simplified
                    'p99_latency_reduction_percent': improvements['p99_latency_reduction_percent'],
                    'throughput_improvement_percent': improvements['throughput_improvement_percent'],
                    'model_accuracy_degradation_max_percent': improvements['model_accuracy_degradation_max_percent'],
                    'serving_cost_reduction_percent': improvements['serving_cost_reduction_percent'],
                    'best_strategy_reward': report['best_reward'],
                    'total_experiments': report['total_experiments'],
                    'convergence_achieved': int(report['convergence_achieved'])
                })

                # Save results
                results_file = output_dir / "training_results.json"
                import json
                with open(results_file, 'w') as f:
                    json.dump({
                        'config': {
                            'bandit_algorithm': config.bandits.algorithm,
                            'num_experiments': args.experiments,
                            'device': config.device
                        },
                        'baseline_results': baseline_results,
                        'final_results': final_results,
                        'improvements': improvements,
                        'optimization_report': report,
                        'experiment_duration_minutes': experiment_duration / 60
                    }, f, indent=2)

                # Save experiment data
                experiment_file = output_dir / "experiment_data.json"
                optimizer.save_experiment_data(str(experiment_file))

                # Log artifacts
                mlflow.log_artifact(str(results_file))
                mlflow.log_artifact(str(experiment_file))

                # Print summary
                logger.info("\n" + "="*60)
                logger.info("TRAINING COMPLETED SUCCESSFULLY")
                logger.info("="*60)
                logger.info(f"Best serving strategy: {improvements['best_strategy']}")
                logger.info(f"P99 latency reduction: {improvements['p99_latency_reduction_percent']:.1f}%")
                logger.info(f"Throughput improvement: {improvements['throughput_improvement_percent']:.1f}%")
                logger.info(f"Accuracy degradation: {improvements['model_accuracy_degradation_max_percent']:.3f}%")
                logger.info(f"Cost reduction: {improvements['serving_cost_reduction_percent']:.1f}%")
                logger.info(f"Convergence time: {experiment_duration/60:.1f} minutes")
                logger.info(f"Results saved to: {output_dir}")
                logger.info("="*60)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()