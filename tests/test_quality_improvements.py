"""Tests for quality improvements and edge cases."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import torch.nn as nn

from adaptive_model_serving_optimizer.models.model import (
    PyTorchAdapter,
    BaseModelAdapter
)
from adaptive_model_serving_optimizer.training.trainer import (
    UCBBandit,
    ThompsonSamplingBandit,
    EpsilonGreedyBandit
)
from adaptive_model_serving_optimizer.evaluation.metrics import calculate_top_k_accuracy
from adaptive_model_serving_optimizer.utils.config import Config
from adaptive_model_serving_optimizer.data.loader import ModelLoader
from adaptive_model_serving_optimizer.data.preprocessing import ModelOptimizer


class TestTensorRTImportHandling:
    """Test optional TensorRT dependency handling."""

    @patch('src.adaptive_model_serving_optimizer.data.loader.TENSORRT_AVAILABLE', False)
    def test_tensorrt_unavailable_in_loader(self, config: Config):
        """Test that loader handles TensorRT unavailability gracefully."""
        loader = ModelLoader(config)

        with pytest.raises(ImportError, match="TensorRT dependencies not available"):
            loader.load_tensorrt_model("dummy_path.engine")

    @patch('src.adaptive_model_serving_optimizer.data.preprocessing.TENSORRT_AVAILABLE', False)
    def test_tensorrt_unavailable_in_preprocessing(self, config: Config):
        """Test that preprocessing handles TensorRT unavailability gracefully."""
        optimizer = ModelOptimizer(config)

        with pytest.raises(ImportError, match="TensorRT dependencies not available"):
            optimizer.convert_to_tensorrt("dummy.onnx", "dummy.engine")


class TestBanditMetricValidation:
    """Test bandit algorithm metric validation."""

    def test_ucb_missing_metrics_warning(self, config: Config, caplog):
        """Test UCB bandit logs warnings for missing metrics."""
        bandit = UCBBandit()
        bandit.add_arm("test_arm")

        # Test missing metrics
        bandit.update_arm("test_arm", 0.8)  # No latency, throughput, error_rate

        assert "Missing latency metric" in caplog.text
        assert "Missing throughput metric" in caplog.text
        assert "Missing error_rate metric" in caplog.text

    def test_thompson_sampling_missing_metrics_warning(self, config: Config, caplog):
        """Test Thompson Sampling bandit logs warnings for missing metrics."""
        bandit = ThompsonSamplingBandit()
        bandit.add_arm("test_arm")

        # Test missing metrics
        bandit.update_arm("test_arm", 0.8)

        assert "Missing latency metric" in caplog.text
        assert "Missing throughput metric" in caplog.text
        assert "Missing error_rate metric" in caplog.text

    def test_epsilon_greedy_missing_metrics_warning(self, config: Config, caplog):
        """Test Epsilon-Greedy bandit logs warnings for missing metrics."""
        bandit = EpsilonGreedyBandit()
        bandit.add_arm("test_arm")

        # Test missing metrics
        bandit.update_arm("test_arm", 0.8)

        assert "Missing latency metric" in caplog.text
        assert "Missing throughput metric" in caplog.text
        assert "Missing error_rate metric" in caplog.text


class TestErrorHandlingImprovements:
    """Test improved error handling and logging."""

    def test_warmup_failure_logging(self, config: Config, caplog):
        """Test that warmup failures are properly logged."""
        # Create adapter with broken model
        adapter = PyTorchAdapter(config)
        adapter.model = MagicMock(side_effect=Exception("Simulated warmup failure"))
        adapter.is_loaded = True

        adapter.warmup()

        assert "Warmup failed" in caplog.text
        assert "will run without warmup" in caplog.text
        assert adapter.warmup_completed is False

    @patch('src.adaptive_model_serving_optimizer.evaluation.metrics.accuracy_score')
    def test_top_k_accuracy_calculation_failure(self, mock_accuracy, caplog):
        """Test top-k accuracy calculation handles errors gracefully."""
        mock_accuracy.side_effect = Exception("Simulated calculation failure")

        predictions = np.array([[0.8, 0.2], [0.3, 0.7]])
        targets = np.array([0, 1])

        result = calculate_top_k_accuracy(predictions, targets, k=5)

        assert np.isnan(result)
        assert "Failed to calculate top-5 accuracy" in caplog.text
        assert "Returning NaN to indicate calculation failure" in caplog.text


class TestONNXOptimizationHandling:
    """Test ONNX optimization fallback behavior."""

    @patch('onnxruntime.InferenceSession')
    def test_onnx_optimization_all_levels_fail(self, mock_session, config: Config, caplog):
        """Test ONNX optimization when all levels fail."""
        mock_session.side_effect = Exception("Optimization failed")

        optimizer = ModelOptimizer(config)

        # Create dummy ONNX file for testing
        onnx_path = Path("dummy.onnx")
        onnx_path.touch()

        try:
            result = optimizer.optimize_onnx_model(onnx_path, "optimized.onnx")

            # Should fallback to original path
            assert result == onnx_path
            assert "All ONNX optimization levels failed" in caplog.text
            assert "Using original model" in caplog.text
        finally:
            onnx_path.unlink()  # Clean up

    @patch('onnxruntime.InferenceSession')
    def test_onnx_optimization_partial_success(self, mock_session, config: Config, caplog):
        """Test ONNX optimization with partial success."""
        # First level fails, second succeeds
        mock_session.side_effect = [Exception("Level 1 failed"), MagicMock()]

        optimizer = ModelOptimizer(config)

        # Create dummy ONNX file
        onnx_path = Path("dummy.onnx")
        onnx_path.touch()

        try:
            optimizer.optimize_onnx_model(onnx_path, "optimized.onnx")

            # Should log first failure and then success
            assert "ONNX optimization completed with level" in caplog.text
        finally:
            onnx_path.unlink()


class TestConfigurationValidation:
    """Test configuration validation improvements."""

    def test_learning_rate_edge_cases(self):
        """Test learning rate validation with edge cases."""
        config = Config()

        # Test boundary values
        config.training.learning_rate = 1.0  # Should be valid now
        config._validate_config()  # Should not raise

        config.training.learning_rate = 0.999  # Should be valid
        config._validate_config()  # Should not raise

        config.training.learning_rate = 1.1  # Should be invalid
        with pytest.raises(ValueError, match="Learning rate must be between"):
            config._validate_config()

    def test_learning_rate_zero(self):
        """Test learning rate cannot be zero."""
        config = Config()
        config.training.learning_rate = 0.0

        with pytest.raises(ValueError, match="Learning rate must be between"):
            config._validate_config()


class TestLoggingEnhancements:
    """Test logging improvements."""

    def test_model_adapter_creation_logging(self, config: Config, caplog):
        """Test that model adapter creation is logged."""
        adapter = PyTorchAdapter(config, model_path="test.pth")

        assert "Created PyTorchAdapter adapter" in caplog.text
        assert "Device configuration" in caplog.text

    def test_bandit_selection_logging(self, config: Config, caplog):
        """Test that bandit selections are logged at debug level."""
        bandit = UCBBandit()
        bandit.add_arm("arm1")
        bandit.add_arm("arm2")

        # Update arms to give them some history
        bandit.update_arm("arm1", 0.8, latency=100, throughput=50, error_rate=0.1)
        bandit.update_arm("arm2", 0.7, latency=120, throughput=45, error_rate=0.15)

        with caplog.at_level(10):  # Debug level
            selected = bandit.select_arm()
            assert f"UCB selection: chose {selected}" in caplog.text

    def test_cache_logging(self, config: Config, caplog):
        """Test cache hit/miss logging."""
        loader = ModelLoader(config)

        # Create a dummy file to simulate cache hit
        cache_path = loader.cache_dir / "test_file.bin"
        cache_path.touch()

        try:
            result = loader.download_model("http://example.com/model.bin", "test_file.bin")

            assert "Cache HIT" in caplog.text
            assert result == cache_path
        finally:
            cache_path.unlink()


class TestPerformanceMetricsLogging:
    """Test performance metrics and timing logging."""

    def test_performance_summary_includes_timestamp(self, config: Config, caplog):
        """Test that performance summaries include timestamps."""
        from adaptive_model_serving_optimizer.training.trainer import ServingStrategyOptimizer

        optimizer = ServingStrategyOptimizer(config)
        optimizer.bandit.add_arm("test_arm")
        optimizer.bandit.update_arm("test_arm", 0.8, latency=100, throughput=50, error_rate=0.1)

        optimizer._log_performance_summary()

        assert "Performance Summary" in caplog.text
        assert "Timestamp:" in caplog.text
        assert "test_arm:" in caplog.text


class TestStackTraceLogging:
    """Test that stack traces are included in error logs."""

    def test_pytorch_adapter_load_error_with_stack_trace(self, config: Config, caplog):
        """Test that model loading errors include stack traces."""
        adapter = PyTorchAdapter(config, model_path="non_existent.pth")

        with pytest.raises(FileNotFoundError):
            adapter.load_model()

        # Check that error was logged with stack trace info
        assert "Failed to load PyTorch model" in caplog.text

    def test_pytorch_adapter_inference_error_with_stack_trace(self, config: Config, caplog):
        """Test that inference errors include stack traces."""
        adapter = PyTorchAdapter(config)
        adapter.model = MagicMock(side_effect=Exception("Simulated inference error"))
        adapter.is_loaded = True

        with pytest.raises(Exception):
            adapter.predict(torch.randn(1, 3, 224, 224))

        assert "PyTorch inference failed" in caplog.text