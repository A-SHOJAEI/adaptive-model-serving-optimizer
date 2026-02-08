"""Adaptive Model Serving Optimizer.

A comprehensive MLOps system that dynamically selects between multiple serving
strategies (TensorRT-optimized, ONNX Runtime, and native PyTorch) based on
real-time latency requirements, batch sizes, and hardware utilization.
"""

__version__ = "1.0.0"
__author__ = "Adaptive Model Serving Optimizer Team"
__email__ = "contact@adaptive-serving.ai"

from .utils import Config, ConfigManager, get_config, setup_environment_from_config
from .data import (
    ModelLoader,
    DatasetLoader,
    BenchmarkDataset,
    ModelOptimizer,
    PerformanceBenchmark
)
from .models import (
    BaseModelAdapter,
    PyTorchAdapter,
    ONNXAdapter,
    TensorRTAdapter,
    ModelAdapterFactory
)
from .training import (
    ServingStrategyOptimizer,
    UCBBandit,
    ThompsonSamplingBandit,
    EpsilonGreedyBandit
)
from .evaluation import (
    MetricsCollector,
    ModelDriftDetector,
    PerformanceMetrics,
    AccuracyMetrics
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',

    # Configuration
    'Config',
    'ConfigManager',
    'get_config',
    'setup_environment_from_config',

    # Data utilities
    'ModelLoader',
    'DatasetLoader',
    'BenchmarkDataset',
    'ModelOptimizer',
    'PerformanceBenchmark',

    # Model adapters
    'BaseModelAdapter',
    'PyTorchAdapter',
    'ONNXAdapter',
    'TensorRTAdapter',
    'ModelAdapterFactory',

    # Training and optimization
    'ServingStrategyOptimizer',
    'UCBBandit',
    'ThompsonSamplingBandit',
    'EpsilonGreedyBandit',

    # Evaluation and monitoring
    'MetricsCollector',
    'ModelDriftDetector',
    'PerformanceMetrics',
    'AccuracyMetrics'
]