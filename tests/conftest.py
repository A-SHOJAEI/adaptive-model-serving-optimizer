"""Pytest configuration and fixtures for adaptive model serving optimizer tests."""

import pytest
import tempfile
from pathlib import Path
from typing import Generator

import torch
import torch.nn as nn
import numpy as np

from adaptive_model_serving_optimizer.utils.config import Config
from adaptive_model_serving_optimizer.data.loader import BenchmarkDataset


class SimpleModel(nn.Module):
    """Simple model for testing purposes."""

    def __init__(self, input_size: int = 3*224*224, num_classes: int = 10):
        """Initialize simple model.

        Args:
            input_size: Size of flattened input
            num_classes: Number of output classes
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.flatten(x)
        return self.classifier(x)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def config(temp_dir: Path) -> Config:
    """Create test configuration.

    Args:
        temp_dir: Temporary directory path

    Returns:
        Test configuration
    """
    config = Config()
    config.data.data_dir = str(temp_dir / "data")
    config.data.cache_dir = str(temp_dir / "cache")
    config.device = 'cpu'  # Use CPU for tests
    config.training.batch_size = 4
    config.training.epochs = 2
    config.seed = 42

    # Configure PyTorch settings for tests
    config.serving.pytorch_config['precision'] = 'float32'  # Use float32 for stability in tests
    config.serving.pytorch_config['jit_compile'] = False  # Disable JIT compilation for tests
    config.serving.pytorch_config['compile_mode'] = 'none'  # Disable torch.compile for tests

    return config


@pytest.fixture
def simple_model() -> nn.Module:
    """Create simple PyTorch model for testing.

    Returns:
        Simple PyTorch model
    """
    model = SimpleModel(input_size=3*224*224, num_classes=10)
    model.eval()
    return model


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Create sample input tensor.

    Returns:
        Sample input tensor
    """
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """Create sample batch tensor.

    Returns:
        Sample batch tensor
    """
    torch.manual_seed(42)
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_targets() -> torch.Tensor:
    """Create sample target labels.

    Returns:
        Sample target tensor
    """
    torch.manual_seed(42)
    return torch.randint(0, 10, (4,))


@pytest.fixture
def benchmark_dataset() -> BenchmarkDataset:
    """Create benchmark dataset for testing.

    Returns:
        Benchmark dataset
    """
    return BenchmarkDataset(
        num_samples=100,
        input_shape=(3, 224, 224),
        num_classes=10
    )


@pytest.fixture
def saved_model_path(temp_dir: Path, simple_model: nn.Module) -> Path:
    """Save model and return path.

    Args:
        temp_dir: Temporary directory
        simple_model: Model to save

    Returns:
        Path to saved model
    """
    model_path = temp_dir / "test_model.pth"
    torch.save(simple_model, model_path)
    return model_path


@pytest.fixture
def sample_performance_data() -> dict:
    """Create sample performance data.

    Returns:
        Dictionary with sample performance metrics
    """
    return {
        'latency_ms': 10.5,
        'throughput_fps': 95.2,
        'memory_usage_mb': 512.0,
        'cpu_usage_percent': 45.0,
        'gpu_usage_percent': 75.0,
        'gpu_memory_usage_mb': 1024.0,
        'error_rate': 0.01
    }


@pytest.fixture
def sample_accuracy_data() -> dict:
    """Create sample accuracy data.

    Returns:
        Dictionary with sample accuracy metrics
    """
    np.random.seed(42)
    predictions = np.random.randint(0, 10, 100)
    targets = np.random.randint(0, 10, 100)
    confidence_scores = np.random.uniform(0.5, 1.0, 100)

    return {
        'predictions': predictions,
        'targets': targets,
        'confidence_scores': confidence_scores
    }


class MockModelAdapter:
    """Mock model adapter for testing."""

    def __init__(self, name: str, latency_ms: float = 10.0):
        """Initialize mock adapter.

        Args:
            name: Adapter name
            latency_ms: Simulated latency
        """
        self.name = name
        self.latency_ms = latency_ms
        self.is_loaded = False
        self.warmup_completed = False
        self._inference_count = 0

    def load_model(self) -> None:
        """Mock model loading."""
        self.is_loaded = True

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mock prediction.

        Args:
            inputs: Input tensor

        Returns:
            Mock predictions
        """
        import time
        time.sleep(self.latency_ms / 1000.0)  # Simulate latency
        self._inference_count += 1

        batch_size = inputs.shape[0]
        return torch.randn(batch_size, 10)  # Mock output

    def warmup(self, num_iterations: int = 5, batch_size: int = 1) -> None:
        """Mock warmup."""
        self.warmup_completed = True

    def get_input_shape(self) -> tuple:
        """Get mock input shape."""
        return (3, 224, 224)

    def get_output_shape(self) -> tuple:
        """Get mock output shape."""
        return (10,)

    def get_performance_stats(self) -> dict:
        """Get mock performance stats."""
        return {
            'avg_inference_time_ms': self.latency_ms,
            'total_inferences': self._inference_count,
            'throughput_fps': 1000.0 / self.latency_ms if self.latency_ms > 0 else 0.0
        }


@pytest.fixture
def mock_adapters() -> dict:
    """Create mock model adapters.

    Returns:
        Dictionary of mock adapters
    """
    return {
        'pytorch_fast': MockModelAdapter('pytorch_fast', latency_ms=8.0),
        'pytorch_slow': MockModelAdapter('pytorch_slow', latency_ms=15.0),
        'onnx_medium': MockModelAdapter('onnx_medium', latency_ms=12.0),
        'tensorrt_fast': MockModelAdapter('tensorrt_fast', latency_ms=5.0)
    }


# Test data generators
def generate_random_tensor(
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate random tensor for testing.

    Args:
        shape: Tensor shape
        dtype: Data type
        device: Device

    Returns:
        Random tensor
    """
    torch.manual_seed(42)
    return torch.randn(*shape, dtype=dtype, device=device)


def generate_classification_data(
    num_samples: int = 100,
    num_classes: int = 10,
    input_shape: tuple = (3, 224, 224)
) -> tuple:
    """Generate classification data for testing.

    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        input_shape: Input shape

    Returns:
        Tuple of (inputs, targets)
    """
    torch.manual_seed(42)
    inputs = torch.randn(num_samples, *input_shape)
    targets = torch.randint(0, num_classes, (num_samples,))
    return inputs, targets


# Assertion helpers
def assert_tensor_equal(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> None:
    """Assert tensors are approximately equal.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    assert tensor1.shape == tensor2.shape, f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), "Tensors are not approximately equal"


def assert_performance_metrics_valid(metrics: dict) -> None:
    """Assert performance metrics are valid.

    Args:
        metrics: Performance metrics dictionary
    """
    required_keys = ['latency_ms', 'throughput_fps', 'error_rate']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert metrics[key] >= 0, f"Negative metric value: {key}={metrics[key]}"

    # Latency should be reasonable (< 10 seconds)
    assert metrics['latency_ms'] < 10000, f"Unreasonable latency: {metrics['latency_ms']}ms"

    # Error rate should be between 0 and 1
    assert 0 <= metrics['error_rate'] <= 1, f"Invalid error rate: {metrics['error_rate']}"


def assert_accuracy_metrics_valid(metrics: dict) -> None:
    """Assert accuracy metrics are valid.

    Args:
        metrics: Accuracy metrics dictionary
    """
    required_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert 0 <= metrics[key] <= 1, f"Invalid metric range: {key}={metrics[key]}"