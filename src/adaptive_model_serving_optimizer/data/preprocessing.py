"""Data preprocessing utilities for adaptive model serving optimizer.

This module provides comprehensive data preprocessing capabilities including
model optimization, format conversion, and performance benchmarking.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
import onnxruntime as ort

# Optional dependencies for TensorRT
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

from ..utils.config import Config

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Utility class for optimizing models for different serving backends."""

    def __init__(self, config: Config):
        """Initialize model optimizer.

        Args:
            config: Configuration object
        """
        self.config = config

    def optimize_pytorch_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        optimization_level: str = 'default'
    ) -> nn.Module:
        """Optimize PyTorch model for serving.

        Args:
            model: PyTorch model to optimize
            sample_input: Sample input for tracing
            optimization_level: Optimization level ('basic', 'default', 'aggressive')

        Returns:
            Optimized PyTorch model
        """
        try:
            model.eval()

            # Move to appropriate device
            device = torch.device(self.config.device)
            model = model.to(device)
            sample_input = sample_input.to(device)

            optimized_model = model

            if optimization_level in ['default', 'aggressive']:
                # Apply torch.jit compilation
                if self.config.serving.pytorch_config.get('jit_compile', True):
                    try:
                        optimized_model = torch.jit.trace(model, sample_input)
                        logger.info("Applied JIT compilation")
                    except Exception as e:
                        logger.warning(f"JIT compilation failed: {e}")
                        optimized_model = model

            if optimization_level == 'aggressive':
                # Apply torch.compile for PyTorch 2.0+
                if hasattr(torch, 'compile') and self.config.serving.pytorch_config.get('compile_mode'):
                    try:
                        compile_mode = self.config.serving.pytorch_config['compile_mode']
                        optimized_model = torch.compile(optimized_model, mode=compile_mode)
                        logger.info(f"Applied torch.compile with mode: {compile_mode}")
                    except Exception as e:
                        logger.warning(f"torch.compile failed: {e}")

                # Enable mixed precision if configured
                if self.config.mixed_precision:
                    optimized_model = optimized_model.half()
                    logger.info("Applied mixed precision (FP16)")

            # Warm up the model
            self._warmup_model(optimized_model, sample_input)

            logger.info(f"PyTorch model optimization completed (level: {optimization_level})")
            return optimized_model

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"PyTorch model optimization failed: {e}", exc_info=True)
            logger.warning("Falling back to unoptimized model")
            return model
        except Exception as e:
            logger.error(f"Unexpected error during PyTorch optimization: {e}", exc_info=True)
            logger.warning("Falling back to unoptimized model due to unexpected error")
            return model

    def convert_to_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: Union[str, Path],
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> Path:
        """Convert PyTorch model to ONNX format.

        Args:
            model: PyTorch model to convert
            sample_input: Sample input for tracing
            output_path: Path to save ONNX model
            dynamic_axes: Dynamic axes specification for variable input sizes

        Returns:
            Path to saved ONNX model
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            model.eval()

            # Default dynamic axes for batch dimension
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }

            # Convert to ONNX
            torch.onnx.export(
                model,
                sample_input,
                str(output_path),
                export_params=True,
                opset_version=17,  # Latest stable opset
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )

            # Validate the exported model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)

            # Optimize the ONNX model
            optimized_path = self._optimize_onnx_model(output_path)

            logger.info(f"ONNX conversion completed: {optimized_path}")
            return optimized_path

        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            raise

    def _optimize_onnx_model(self, onnx_path: Path) -> Path:
        """Optimize ONNX model for better performance.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Path to optimized ONNX model
        """
        try:
            from onnxruntime.tools import optimizer

            # Create optimized model path
            optimized_path = onnx_path.with_suffix('.optimized.onnx')

            # Load and optimize
            optimization_levels = [
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ]

            optimization_succeeded = False
            last_error = None

            for level in optimization_levels:
                try:
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = level
                    sess_options.optimized_model_filepath = str(optimized_path)

                    # Create session to trigger optimization
                    providers = ['CPUExecutionProvider']  # Use CPU for optimization
                    session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)

                    logger.info(f"ONNX optimization completed with level: {level}")
                    optimization_succeeded = True
                    break

                except Exception as e:
                    logger.debug(f"ONNX optimization level {level} failed: {e}")
                    last_error = e
                    continue

            if not optimization_succeeded:
                logger.warning(
                    f"All ONNX optimization levels failed. Using original model. "
                    f"Last error: {last_error}"
                )

            return optimized_path if optimized_path.exists() else onnx_path

        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"ONNX file access error: {e}")
            return onnx_path
        except (RuntimeError, ValueError, ImportError) as e:
            logger.warning(f"ONNX optimization failed: {e}")
            return onnx_path
        except Exception as e:
            logger.error(f"Unexpected error during ONNX optimization: {e}", exc_info=True)
            return onnx_path

    def convert_to_tensorrt(
        self,
        onnx_path: Union[str, Path],
        output_path: Union[str, Path],
        precision: str = 'fp16'
    ) -> Path:
        """Convert ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            precision: Precision mode ('fp32', 'fp16', 'int8')

        Returns:
            Path to TensorRT engine

        Raises:
            ImportError: If TensorRT dependencies are not available
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT dependencies not available. Install tensorrt and pycuda to use TensorRT conversion."
            )

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize TensorRT
            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)
            config = builder.create_builder_config()

            # Set workspace size
            workspace_size = self.config.serving.tensorrt_config.get('max_workspace_size', 1 << 30)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

            # Set precision
            if precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
                # Note: INT8 calibration would be needed for production use

            # Parse ONNX model
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)

            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error_idx in range(parser.num_errors):
                        logger.error(f"ONNX parsing error: {parser.get_error(error_idx)}")
                    raise RuntimeError("Failed to parse ONNX model")

            # Configure optimization profiles for dynamic batching
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            input_shape = input_tensor.shape

            # Set dynamic batch size if configured
            max_batch_size = self.config.serving.tensorrt_config.get('max_batch_size', 32)
            opt_batch_size = self.config.serving.tensorrt_config.get('opt_batch_size', 8)

            min_shape = tuple([1] + list(input_shape[1:]))
            opt_shape = tuple([opt_batch_size] + list(input_shape[1:]))
            max_shape = tuple([max_batch_size] + list(input_shape[1:]))

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            # Build engine
            logger.info("Building TensorRT engine (this may take several minutes)...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")

            # Save engine
            with open(output_path, 'wb') as f:
                f.write(serialized_engine)

            logger.info(f"TensorRT conversion completed: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            raise

    def _warmup_model(self, model: nn.Module, sample_input: torch.Tensor, num_iterations: int = 5) -> None:
        """Warm up model for consistent timing measurements.

        Args:
            model: Model to warm up
            sample_input: Sample input for warmup
            num_iterations: Number of warmup iterations
        """
        try:
            model.eval()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(sample_input)

            # Synchronize CUDA if available
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")


class PerformanceBenchmark:
    """Utility class for benchmarking model performance."""

    def __init__(self, config: Config):
        """Initialize performance benchmark.

        Args:
            config: Configuration object
        """
        self.config = config

    def benchmark_latency(
        self,
        model: Any,
        inputs: List[torch.Tensor],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark model latency.

        Args:
            model: Model to benchmark
            inputs: List of input tensors for testing
            num_iterations: Number of timing iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary of latency statistics
        """
        try:
            # Warmup
            for i in range(warmup_iterations):
                input_tensor = inputs[i % len(inputs)]
                self._run_inference(model, input_tensor)

            # Synchronize before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Timing runs
            latencies = []
            for i in range(num_iterations):
                input_tensor = inputs[i % len(inputs)]

                start_time = time.perf_counter()
                self._run_inference(model, input_tensor)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            # Calculate statistics
            latencies = np.array(latencies)
            stats = {
                'mean_latency_ms': float(np.mean(latencies)),
                'median_latency_ms': float(np.median(latencies)),
                'p95_latency_ms': float(np.percentile(latencies, 95)),
                'p99_latency_ms': float(np.percentile(latencies, 99)),
                'min_latency_ms': float(np.min(latencies)),
                'max_latency_ms': float(np.max(latencies)),
                'std_latency_ms': float(np.std(latencies))
            }

            logger.info(f"Latency benchmark completed: mean={stats['mean_latency_ms']:.2f}ms")
            return stats

        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}")
            return {}

    def benchmark_throughput(
        self,
        model: Any,
        batch_sizes: List[int],
        input_shape: Tuple[int, ...],
        duration_seconds: float = 10.0
    ) -> Dict[int, float]:
        """Benchmark model throughput for different batch sizes.

        Args:
            model: Model to benchmark
            batch_sizes: List of batch sizes to test
            input_shape: Shape of individual inputs
            duration_seconds: Duration of each throughput test

        Returns:
            Dictionary mapping batch sizes to throughput (samples/sec)
        """
        try:
            throughput_results = {}

            for batch_size in batch_sizes:
                logger.info(f"Benchmarking throughput for batch size: {batch_size}")

                # Create batch
                batch = torch.randn(batch_size, *input_shape, device=self.config.device)

                # Warmup
                for _ in range(5):
                    self._run_inference(model, batch)

                # Timing
                start_time = time.perf_counter()
                total_samples = 0

                while time.perf_counter() - start_time < duration_seconds:
                    self._run_inference(model, batch)
                    total_samples += batch_size

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                elapsed_time = time.perf_counter() - start_time
                throughput = total_samples / elapsed_time

                throughput_results[batch_size] = throughput
                logger.info(f"Batch size {batch_size}: {throughput:.2f} samples/sec")

            return throughput_results

        except Exception as e:
            logger.error(f"Throughput benchmark failed: {e}")
            return {}

    def _run_inference(self, model: Any, input_tensor: torch.Tensor) -> Any:
        """Run inference on model with input tensor.

        Safely executes forward pass on various model types with proper
        exception handling. Currently supports PyTorch models with torch.no_grad()
        context for inference optimization.

        Args:
            model: Model to run inference on. Must implement __call__ method
                (e.g., nn.Module subclasses)
            input_tensor: Input tensor with shape matching model's expected input

        Returns:
            Model output tensor(s) from forward pass

        Raises:
            ValueError: If model type is not supported (no __call__ method)
            RuntimeError: If inference fails due to model or input incompatibility

        Example:
            >>> optimizer = ModelOptimizer(config)
            >>> output = optimizer._run_inference(pytorch_model, input_batch)
        """
        if hasattr(model, '__call__'):
            # PyTorch model
            with torch.no_grad():
                return model(input_tensor)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    def get_model_memory_usage(self, model: nn.Module) -> Dict[str, int]:
        """Get memory usage statistics for PyTorch model.

        Args:
            model: PyTorch model

        Returns:
            Dictionary of memory usage statistics
        """
        try:
            if not torch.cuda.is_available():
                return {}

            # Get parameter memory
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())

            # Get GPU memory
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            max_memory_allocated = torch.cuda.max_memory_allocated()

            memory_stats = {
                'parameter_memory_bytes': param_memory,
                'buffer_memory_bytes': buffer_memory,
                'total_model_memory_bytes': param_memory + buffer_memory,
                'gpu_memory_allocated_bytes': memory_allocated,
                'gpu_memory_reserved_bytes': memory_reserved,
                'gpu_max_memory_allocated_bytes': max_memory_allocated
            }

            return memory_stats

        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}


def profile_model_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_iterations: int = 10
) -> Dict[str, Any]:
    """Profile model inference with detailed timing breakdown.

    Uses PyTorch's built-in profiler to collect detailed performance metrics
    including CPU/CUDA execution times, memory usage, and operation-level
    timing breakdowns across multiple inference iterations.

    Args:
        model: PyTorch model to profile (nn.Module)
        input_tensor: Input tensor with shape matching model's expected input
        num_iterations: Number of profiling iterations to average over (default: 10)
            Higher values provide more stable timing measurements

    Returns:
        Dictionary containing profiling results with keys:
        - 'cpu_time_total_us': Total CPU execution time in microseconds
        - 'cuda_time_total_us': Total CUDA execution time in microseconds
        - 'num_iterations': Number of iterations actually profiled
        - 'profiler_trace': Formatted table string with operation-level timing
            breakdown sorted by CUDA time. Contains function names, CPU/CUDA times,
            and memory usage for each operation in the model forward pass.
        Returns empty dict if profiling fails or PyTorch profiler unavailable.

    Example:
        >>> results = profile_model_inference(resnet_model, input_batch, 20)
        >>> print(f"Average CUDA time: {results['cuda_time_total_us'] / results['num_iterations']:.2f}μs")
        >>> print(results['profiler_trace'])  # Detailed operation breakdown
    """
    try:
        # Use PyTorch profiler if available
        if hasattr(torch, 'profiler'):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = model(input_tensor)

            # Extract key metrics
            events = prof.events()
            cpu_time_total = sum(event.cpu_time_total for event in events)
            cuda_time_total = sum(event.cuda_time_total for event in events)

            return {
                'cpu_time_total_us': cpu_time_total,
                'cuda_time_total_us': cuda_time_total,
                'num_iterations': num_iterations,
                'profiler_trace': prof.key_averages().table(sort_by="cuda_time_total")
            }

        else:
            logger.warning("PyTorch profiler not available")
            return {}

    except Exception as e:
        logger.error(f"Model profiling failed: {e}")
        return {}