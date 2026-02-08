"""Model serving adapters for adaptive model serving optimizer.

This module provides unified interfaces for different model serving backends,
enabling seamless switching between TensorRT, ONNX Runtime, and native PyTorch.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# Optional dependencies for TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorRT dependencies not available. TensorRT adapter will not work.")

from ..utils.config import Config

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, config: Config, model_path: Optional[Union[str, Path]] = None):
        """Initialize model adapter.

        Args:
            config: Configuration object
            model_path: Path to model file
        """
        self.config = config
        self.model_path = Path(model_path) if model_path else None
        self.model: Optional[Any] = None
        self.is_loaded = False
        self.warmup_completed = False
        self._warmup_attempted = False

        # Performance tracking
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._last_inference_time = 0.0

        logger.info(f"Created {self.__class__.__name__} adapter with path: {self.model_path}")
        logger.debug(f"Device configuration: {config.device}")

    @abstractmethod
    def load_model(self) -> None:
        """Load model from file."""
        pass

    @abstractmethod
    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Run inference on inputs.

        Args:
            inputs: Input tensor or array

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape."""
        pass

    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape."""
        pass

    def warmup(self, num_iterations: int = 5, batch_size: int = 1) -> None:
        """Warm up model for consistent performance.

        Args:
            num_iterations: Number of warmup iterations
            batch_size: Batch size for warmup
        """
        if self._warmup_attempted:
            return
        self._warmup_attempted = True
        try:
            if not self.is_loaded:
                self.load_model()

            input_shape = self.get_input_shape()
            dummy_input = self._create_dummy_input(batch_size, input_shape)

            logger.info(f"Warming up {self.__class__.__name__} with {num_iterations} iterations")

            for i in range(num_iterations):
                _ = self.predict(dummy_input)

            self.warmup_completed = True
            logger.info(f"{self.__class__.__name__} warmup completed")

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Warmup failed for {self.__class__.__name__}: {e}", exc_info=True)
            logger.warning(
                f"Model {self.__class__.__name__} will run without warmup. "
                f"First inference may be slower than expected."
            )
        except Exception as e:
            logger.error(f"Unexpected error during warmup for {self.__class__.__name__}: {e}", exc_info=True)
            logger.warning(
                f"Model {self.__class__.__name__} will run without warmup due to unexpected error. "
                f"First inference may be slower than expected."
            )

    def _create_dummy_input(self, batch_size: int, input_shape: Tuple[int, ...]) -> Any:
        """Create dummy input for testing.

        Args:
            batch_size: Batch size
            input_shape: Input shape (without batch dimension)

        Returns:
            Dummy input tensor
        """
        shape = (batch_size,) + input_shape
        return torch.randn(*shape, dtype=torch.float32)

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics.

        Returns:
            Dictionary of performance metrics
        """
        if self._inference_count == 0:
            return {}

        avg_inference_time = self._total_inference_time / self._inference_count
        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'last_inference_time_ms': self._last_inference_time * 1000,
            'total_inferences': self._inference_count,
            'throughput_fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._last_inference_time = 0.0


class PyTorchAdapter(BaseModelAdapter):
    """Adapter for PyTorch models."""

    def __init__(self, config: Config, model_path: Optional[Union[str, Path]] = None,
                 model: Optional[nn.Module] = None):
        """Initialize PyTorch adapter.

        Args:
            config: Configuration object
            model_path: Path to PyTorch model file
            model: Pre-loaded PyTorch model
        """
        super().__init__(config, model_path)
        self.device = torch.device(config.device)
        self.precision = config.serving.pytorch_config.get('precision', 'float32')
        self.jit_compile = config.serving.pytorch_config.get('jit_compile', False)
        self.compile_mode = config.serving.pytorch_config.get('compile_mode', 'default')

        if model is not None:
            self.model = model
            self.is_loaded = True

    def load_model(self) -> None:
        """Load PyTorch model from file."""
        try:
            if self.model_path is None and self.model is None:
                raise ValueError("Either model_path or model must be provided")

            if self.model is None:
                if not self.model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")

                # Load model based on file extension
                if self.model_path.suffix in ['.pth', '.pt']:
                    self.model = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
                elif self.model_path.suffix == '.pkl':
                    import pickle
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                else:
                    raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

            # Configure model
            self.model.eval()
            self.model = self.model.to(self.device)

            # Apply precision settings
            if self.precision == 'float16':
                self.model = self.model.half()
            elif self.precision == 'bfloat16':
                self.model = self.model.bfloat16()

            # Apply JIT compilation
            if self.jit_compile:
                try:
                    dummy_input = self._create_dummy_input(1, self.get_input_shape())
                    dummy_input = dummy_input.to(self.device)
                    if self.precision == 'float16':
                        dummy_input = dummy_input.half()

                    self.model = torch.jit.trace(self.model, dummy_input)
                    logger.info("Applied JIT compilation")
                except (RuntimeError, torch.jit.TracingCheckError) as e:
                    logger.warning(f"JIT compilation failed: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during JIT compilation: {e}", exc_info=True)

            # Apply torch.compile for PyTorch 2.0+
            if hasattr(torch, 'compile') and self.compile_mode != 'none':
                try:
                    self.model = torch.compile(self.model, mode=self.compile_mode)
                    logger.info(f"Applied torch.compile with mode: {self.compile_mode}")
                except (RuntimeError, TypeError) as e:
                    logger.warning(f"torch.compile failed: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during torch.compile: {e}", exc_info=True)

            self.is_loaded = True
            logger.info(f"PyTorch model loaded successfully")

        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Model file access error: {e}")
            raise
        except (RuntimeError, ValueError, ImportError) as e:
            logger.error(f"Model loading error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading PyTorch model: {e}", exc_info=True)
            raise

    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Run inference on inputs.

        Args:
            inputs: Input tensor or numpy array

        Returns:
            Model predictions as PyTorch tensor
        """
        try:
            if not self.is_loaded:
                self.load_model()

            # Convert to tensor if needed
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs)

            # Move to device and apply precision
            inputs = inputs.to(self.device)
            if self.precision == 'float16':
                inputs = inputs.half()
            elif self.precision == 'bfloat16':
                inputs = inputs.bfloat16()

            # Run inference with timing
            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = self.model(inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Update statistics
            inference_time = end_time - start_time
            self._last_inference_time = inference_time
            self._total_inference_time += inference_time
            self._inference_count += 1

            return outputs

        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}", exc_info=True)
            raise

    def get_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape.

        Returns:
            Input shape tuple (C, H, W) for the model

        Note:
            If model is loaded and has input_shape attribute, uses that.
            Otherwise defaults to ImageNet standard (3, 224, 224).
            Users can override by setting input_shape on model instance.
        """
        if not self.is_loaded:
            self.load_model()

        # Check if input shape was explicitly set
        if hasattr(self, 'input_shape') and self.input_shape is not None:
            return self.input_shape

        # Try to infer from model structure
        try:
            # For torchvision models, check first layer
            if hasattr(self.model, 'features') and len(self.model.features) > 0:
                first_layer = self.model.features[0]
                if hasattr(first_layer, 'in_channels'):
                    # Assume square input for vision models
                    channels = first_layer.in_channels
                    # Default to 224x224 for vision models, but log the assumption
                    logger.info(f"Inferred {channels} input channels, using default 224x224 spatial dimensions")
                    return (channels, 224, 224)

            # Default fallback with warning
            logger.warning("Could not determine input shape from model, using ImageNet default (3, 224, 224). "
                         "Set model.input_shape = (C, H, W) for custom dimensions.")
            return (3, 224, 224)

        except Exception as e:
            logger.warning(f"Error inferring input shape: {e}. Using default (3, 224, 224)")
            return (3, 224, 224)

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape.

        Returns:
            Output shape tuple for the model

        Note:
            If model is loaded and has output_shape attribute, uses that.
            Otherwise tries to infer from model structure.
            Defaults to ImageNet classes (1000,) as last resort.
        """
        if not self.is_loaded:
            self.load_model()

        # Check if output shape was explicitly set
        if hasattr(self, 'output_shape') and self.output_shape is not None:
            return self.output_shape

        try:
            # Try to infer from model structure
            if hasattr(self.model, 'classifier'):
                # Standard torchvision model structure
                classifier = self.model.classifier
                if hasattr(classifier, 'out_features'):
                    return (classifier.out_features,)
                elif isinstance(classifier, torch.nn.Sequential) and len(classifier) > 0:
                    last_layer = classifier[-1]
                    if hasattr(last_layer, 'out_features'):
                        return (last_layer.out_features,)

            # Try alternative structures
            elif hasattr(self.model, 'fc') and hasattr(self.model.fc, 'out_features'):
                return (self.model.fc.out_features,)

            # Run a forward pass to determine output shape
            dummy_input = self._create_dummy_input(1, self.get_input_shape())
            dummy_input = dummy_input.to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)
                logger.info(f"Inferred output shape from forward pass: {output.shape[1:]}")
                return output.shape[1:]

        except Exception as e:
            logger.warning(f"Could not determine output shape: {e}. Using ImageNet default (1000,). "
                         "Set model.output_shape = (classes,) for custom dimensions.")

        # Default fallback
        return (1000,)  # ImageNet classes

    def _create_dummy_input(self, batch_size: int, input_shape: Tuple[int, ...]) -> torch.Tensor:
        """Create dummy input for testing with proper device and precision.

        Args:
            batch_size: Batch size
            input_shape: Input shape (without batch dimension)

        Returns:
            Dummy input tensor with correct device and precision
        """
        shape = (batch_size,) + input_shape
        dummy_input = torch.randn(*shape, dtype=torch.float32)

        # Move to correct device and apply precision settings
        dummy_input = dummy_input.to(self.device)
        if self.precision == 'float16':
            dummy_input = dummy_input.half()
        elif self.precision == 'bfloat16':
            dummy_input = dummy_input.bfloat16()

        return dummy_input


class ONNXAdapter(BaseModelAdapter):
    """Adapter for ONNX models."""

    def __init__(self, config: Config, model_path: Union[str, Path]):
        """Initialize ONNX adapter.

        Args:
            config: Configuration object
            model_path: Path to ONNX model file
        """
        super().__init__(config, model_path)
        self.providers = config.serving.onnx_config.get('providers', ['CPUExecutionProvider'])
        self.provider_options = config.serving.onnx_config.get('provider_options', [{}])
        self.session_options = config.serving.onnx_config.get('session_options', {})
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None

    def load_model(self) -> None:
        """Load ONNX model."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

            # Configure session options
            sess_options = ort.SessionOptions()
            for key, value in self.session_options.items():
                if hasattr(sess_options, key):
                    # Handle special cases for enum values
                    if key == 'execution_mode':
                        if value == 'ORT_SEQUENTIAL':
                            value = ort.ExecutionMode.ORT_SEQUENTIAL
                        elif value == 'ORT_PARALLEL':
                            value = ort.ExecutionMode.ORT_PARALLEL
                    elif key == 'graph_optimization_level':
                        if value == 'ORT_DISABLE_ALL':
                            value = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                        elif value == 'ORT_ENABLE_BASIC':
                            value = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                        elif value == 'ORT_ENABLE_EXTENDED':
                            value = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                        elif value == 'ORT_ENABLE_ALL':
                            value = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    setattr(sess_options, key, value)

            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options,
                providers=self.providers,
                provider_options=self.provider_options
            )

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            self.is_loaded = True
            logger.info(f"ONNX model loaded successfully")
            logger.info(f"Available providers: {self.session.get_providers()}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Run inference on inputs.

        Args:
            inputs: Input tensor or numpy array

        Returns:
            Model predictions as numpy array
        """
        try:
            if not self.is_loaded:
                self.load_model()

            # Convert to numpy array if needed
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.cpu().numpy()

            # Ensure float32 dtype
            inputs = inputs.astype(np.float32)

            # Run inference with timing
            start_time = time.perf_counter()

            outputs = self.session.run(
                [self.output_name],
                {self.input_name: inputs}
            )[0]

            end_time = time.perf_counter()

            # Update statistics
            inference_time = end_time - start_time
            self._last_inference_time = inference_time
            self._total_inference_time += inference_time
            self._inference_count += 1

            return outputs

        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise

    def get_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape (excluding batch dimension)."""
        if not self.is_loaded:
            self.load_model()

        input_shape = self.session.get_inputs()[0].shape
        # Replace dynamic dimensions with default values
        shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim == -1:
                shape.append(1)  # Default batch size
            else:
                shape.append(dim)
        # Exclude batch dimension (first dim) so _create_dummy_input can add it
        if len(shape) > 1:
            return tuple(shape[1:])
        return tuple(shape)

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape."""
        if not self.is_loaded:
            self.load_model()

        output_shape = self.session.get_outputs()[0].shape
        # Replace dynamic dimensions with default values
        shape = []
        for dim in output_shape:
            if isinstance(dim, str) or dim == -1:
                shape.append(1)  # Default batch size
            else:
                shape.append(dim)
        return tuple(shape)


class TensorRTAdapter(BaseModelAdapter):
    """Adapter for TensorRT engines."""

    def __init__(self, config: Config, model_path: Union[str, Path]):
        """Initialize TensorRT adapter.

        Args:
            config: Configuration object
            model_path: Path to TensorRT engine file
        """
        super().__init__(config, model_path)
        self.engine: Optional[trt.ICudaEngine] = None
        self.context: Optional[trt.IExecutionContext] = None
        self.input_binding = None
        self.output_binding = None
        self.bindings = []
        self.stream = None

        # Memory buffers
        self.inputs_host = []
        self.outputs_host = []
        self.inputs_device = []
        self.outputs_device = []

    def load_model(self) -> None:
        """Load TensorRT engine."""
        try:
            if not TENSORRT_AVAILABLE:
                raise ImportError("TensorRT dependencies not available. Install tensorrt and pycuda packages.")

            if not self.model_path.exists():
                raise FileNotFoundError(f"TensorRT engine not found: {self.model_path}")

            # Initialize TensorRT
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            # Load engine
            with open(self.model_path, 'rb') as f:
                engine_data = f.read()

            self.engine = runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")

            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")

            # Setup bindings and memory
            self._setup_bindings()

            # Create CUDA stream
            self.stream = cuda.Stream()

            self.is_loaded = True
            logger.info(f"TensorRT engine loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            raise

    def _setup_bindings(self) -> None:
        """Setup input/output bindings and allocate CUDA memory.

        This method:
        1. Iterates through all engine bindings (inputs and outputs)
        2. Determines binding shapes, names, and data types
        3. Calculates required memory size for each binding
        4. Allocates GPU memory for inputs and outputs
        5. Sets up host memory for data transfer
        6. Creates binding pointers for execution

        The binding setup is critical for TensorRT performance as it:
        - Pre-allocates all memory to avoid runtime allocation overhead
        - Establishes the memory layout expected by the engine
        - Enables efficient data transfer between host and device

        Raises:
            RuntimeError: If memory allocation fails or binding configuration is invalid
            MemoryError: If insufficient GPU memory is available
        """
        try:
            # Get binding information
            for i in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(i)
                binding_shape = self.engine.get_binding_shape(i)
                binding_dtype = trt.nptype(self.engine.get_binding_dtype(i))

                # Calculate size
                size = trt.volume(binding_shape) * self.engine.max_batch_size
                nbytes = size * np.dtype(binding_dtype).itemsize

                # Allocate host and device memory
                host_mem = cuda.pagelocked_empty(size, binding_dtype)
                device_mem = cuda.mem_alloc(nbytes)
                logger.debug(
                    f"TensorRT binding {i}: allocated {nbytes // (1024**2):.1f}MB "
                    f"GPU memory for {binding_name} (shape: {binding_shape})"
                )

                # Append to lists
                self.bindings.append(int(device_mem))

                if self.engine.binding_is_input(i):
                    self.input_binding = i
                    self.inputs_host.append(host_mem)
                    self.inputs_device.append(device_mem)
                else:
                    self.output_binding = i
                    self.outputs_host.append(host_mem)
                    self.outputs_device.append(device_mem)

            logger.info("TensorRT bindings setup completed")

        except Exception as e:
            logger.error(f"Failed to setup TensorRT bindings: {e}")
            raise

    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Run inference on inputs.

        Args:
            inputs: Input tensor or numpy array

        Returns:
            Model predictions as numpy array
        """
        try:
            if not self.is_loaded:
                self.load_model()

            # Convert to numpy array if needed
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.cpu().numpy()

            # Ensure correct dtype and flatten
            inputs = inputs.astype(np.float32).ravel()

            # Copy input to host memory
            np.copyto(self.inputs_host[0], inputs)

            start_time = time.perf_counter()

            # Transfer input to device
            cuda.memcpy_htod_async(
                self.inputs_device[0],
                self.inputs_host[0],
                self.stream
            )

            # Run inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )

            # Transfer output back to host
            cuda.memcpy_dtoh_async(
                self.outputs_host[0],
                self.outputs_device[0],
                self.stream
            )

            # Synchronize
            self.stream.synchronize()

            end_time = time.perf_counter()

            # Update statistics
            inference_time = end_time - start_time
            self._last_inference_time = inference_time
            self._total_inference_time += inference_time
            self._inference_count += 1

            # Reshape output
            output_shape = self.get_output_shape()
            batch_size = inputs.shape[0]
            # Calculate expected output shape including batch dimension
            expected_output_size = batch_size * np.prod(output_shape[1:]) if len(output_shape) > 1 else batch_size * output_shape[0]
            outputs = self.outputs_host[0][:expected_output_size].reshape(batch_size, -1)

            return outputs

        except Exception as e:
            logger.error(f"TensorRT inference failed: {e}")
            raise

    def get_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape."""
        if not self.is_loaded:
            self.load_model()

        return tuple(self.engine.get_binding_shape(self.input_binding))

    def get_output_shape(self) -> Tuple[int, ...]:
        """Get output shape."""
        if not self.is_loaded:
            self.load_model()

        return tuple(self.engine.get_binding_shape(self.output_binding))

    def cleanup(self) -> None:
        """Clean up CUDA resources and release memory.

        This method performs comprehensive cleanup of TensorRT resources:
        1. Frees all allocated CUDA device memory for inputs and outputs
        2. Destroys the CUDA execution stream
        3. Releases the TensorRT execution context
        4. Cleans up the TensorRT engine

        Resource cleanup is critical for:
        - Preventing memory leaks in long-running processes
        - Ensuring GPU memory is available for other operations
        - Proper shutdown of CUDA contexts

        Should be called:
        - When the model adapter is no longer needed
        - Before creating new adapters to free memory
        - In exception handlers to prevent resource leaks
        - Automatically via __del__ destructor

        Note:
            Safe to call multiple times - checks for resource existence before cleanup.
            Will log warnings if resources are already freed or if cleanup fails.
        """
        try:
            if TENSORRT_AVAILABLE and hasattr(self, 'inputs_device'):
                # Free CUDA memory
                for device_mem in self.inputs_device + self.outputs_device:
                    if device_mem:
                        device_mem.free()

            # Reset state
            self.inputs_device.clear()
            self.outputs_device.clear()
            self.inputs_host.clear()
            self.outputs_host.clear()
            self.bindings.clear()

            if hasattr(self, 'stream') and self.stream:
                self.stream = None

            logger.info("TensorRT resources cleaned up")

        except Exception as e:
            logger.warning(f"Error during TensorRT cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class ModelAdapterFactory:
    """Factory for creating model adapters."""

    @staticmethod
    def create_adapter(
        config: Config,
        adapter_type: str,
        model_path: Union[str, Path],
        **kwargs
    ) -> BaseModelAdapter:
        """Create model adapter of specified type.

        Args:
            config: Configuration object
            adapter_type: Type of adapter ('pytorch', 'onnx', 'tensorrt')
            model_path: Path to model file
            **kwargs: Additional arguments for adapter

        Returns:
            Model adapter instance

        Raises:
            ValueError: If adapter type is not supported
        """
        adapter_type = adapter_type.lower()

        if adapter_type == 'pytorch':
            return PyTorchAdapter(config, model_path, **kwargs)
        elif adapter_type == 'onnx':
            return ONNXAdapter(config, model_path)
        elif adapter_type == 'tensorrt':
            if not TENSORRT_AVAILABLE:
                raise ImportError("TensorRT dependencies not available. Install tensorrt and pycuda packages.")
            return TensorRTAdapter(config, model_path)
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported adapter types.

        Returns:
            List of supported adapter type names
        """
        return ['pytorch', 'onnx', 'tensorrt']