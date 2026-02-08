"""Tests for model serving adapters."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import numpy as np

from adaptive_model_serving_optimizer.models.model import (
    BaseModelAdapter,
    PyTorchAdapter,
    ONNXAdapter,
    TensorRTAdapter,
    ModelAdapterFactory
)
from adaptive_model_serving_optimizer.utils.config import Config


class TestPyTorchAdapter:
    """Tests for PyTorchAdapter class."""

    def test_pytorch_adapter_init_with_model(self, config: Config, simple_model: nn.Module):
        """Test PyTorchAdapter initialization with pre-loaded model."""
        adapter = PyTorchAdapter(config, model=simple_model)

        assert adapter.config == config
        assert adapter.model == simple_model
        assert adapter.is_loaded is True

    def test_pytorch_adapter_init_with_path(self, config: Config, saved_model_path: Path):
        """Test PyTorchAdapter initialization with model path."""
        adapter = PyTorchAdapter(config, model_path=saved_model_path)

        assert adapter.config == config
        assert adapter.model_path == saved_model_path
        assert adapter.is_loaded is False

    def test_load_model_success(self, config: Config, saved_model_path: Path):
        """Test successful model loading."""
        adapter = PyTorchAdapter(config, model_path=saved_model_path)
        adapter.load_model()

        assert adapter.is_loaded is True
        assert isinstance(adapter.model, nn.Module)
        assert not adapter.model.training  # Should be in eval mode

    def test_load_model_file_not_found(self, config: Config):
        """Test model loading with non-existent file."""
        non_existent_path = Path("non_existent.pth")
        adapter = PyTorchAdapter(config, model_path=non_existent_path)

        with pytest.raises(FileNotFoundError):
            adapter.load_model()

    def test_predict_success(self, config: Config, simple_model: nn.Module, sample_input: torch.Tensor):
        """Test successful prediction."""
        adapter = PyTorchAdapter(config, model=simple_model)
        adapter.load_model()

        output = adapter.predict(sample_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == sample_input.shape[0]  # Same batch size

    def test_predict_with_numpy_input(self, config: Config, simple_model: nn.Module):
        """Test prediction with numpy input."""
        adapter = PyTorchAdapter(config, model=simple_model)
        adapter.load_model()

        numpy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = adapter.predict(numpy_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1

    def test_warmup(self, config: Config, simple_model: nn.Module):
        """Test model warmup."""
        adapter = PyTorchAdapter(config, model=simple_model)
        adapter.warmup(num_iterations=3, batch_size=2)

        assert adapter.warmup_completed is True

    def test_get_performance_stats(self, config: Config, simple_model: nn.Module, sample_input: torch.Tensor):
        """Test performance statistics."""
        adapter = PyTorchAdapter(config, model=simple_model)
        adapter.load_model()

        # Run some predictions to generate stats
        adapter.predict(sample_input)
        adapter.predict(sample_input)

        stats = adapter.get_performance_stats()

        assert 'avg_inference_time_ms' in stats
        assert 'total_inferences' in stats
        assert 'throughput_fps' in stats
        assert stats['total_inferences'] == 2

    def test_reset_stats(self, config: Config, simple_model: nn.Module, sample_input: torch.Tensor):
        """Test statistics reset."""
        adapter = PyTorchAdapter(config, model=simple_model)
        adapter.load_model()

        # Generate some stats
        adapter.predict(sample_input)

        # Reset and check
        adapter.reset_stats()
        stats = adapter.get_performance_stats()

        assert stats == {}  # Should be empty after reset

    def test_precision_configuration(self, config: Config, simple_model: nn.Module):
        """Test precision configuration."""
        config.serving.pytorch_config['precision'] = 'float16'

        adapter = PyTorchAdapter(config, model=simple_model)
        adapter.load_model()

        # Check if model parameters are in half precision
        first_param = next(adapter.model.parameters())
        assert first_param.dtype == torch.float16


class TestONNXAdapter:
    """Tests for ONNXAdapter class."""

    def test_onnx_adapter_init(self, config: Config, temp_dir: Path):
        """Test ONNXAdapter initialization."""
        model_path = temp_dir / "test_model.onnx"
        model_path.touch()  # Create empty file

        adapter = ONNXAdapter(config, model_path)

        assert adapter.config == config
        assert adapter.model_path == model_path
        assert adapter.is_loaded is False

    @patch('onnxruntime.InferenceSession')
    def test_load_model_success(self, mock_session, config: Config, temp_dir: Path):
        """Test successful ONNX model loading."""
        model_path = temp_dir / "test_model.onnx"
        model_path.touch()

        # Mock ONNX session
        mock_session_instance = MagicMock()
        mock_session_instance.get_inputs.return_value = [MagicMock(name='input')]
        mock_session_instance.get_outputs.return_value = [MagicMock(name='output')]
        mock_session_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_session.return_value = mock_session_instance

        adapter = ONNXAdapter(config, model_path)
        adapter.load_model()

        assert adapter.is_loaded is True
        assert adapter.session is not None

    def test_load_model_file_not_found(self, config: Config):
        """Test ONNX model loading with non-existent file."""
        non_existent_path = Path("non_existent.onnx")
        adapter = ONNXAdapter(config, non_existent_path)

        with pytest.raises(FileNotFoundError):
            adapter.load_model()

    @patch('onnxruntime.InferenceSession')
    def test_predict_success(self, mock_session, config: Config, temp_dir: Path, sample_input: torch.Tensor):
        """Test successful ONNX prediction."""
        model_path = temp_dir / "test_model.onnx"
        model_path.touch()

        # Mock session and prediction
        mock_session_instance = MagicMock()
        mock_session_instance.get_inputs.return_value = [MagicMock(name='input')]
        mock_session_instance.get_outputs.return_value = [MagicMock(name='output')]
        mock_session_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_session_instance.run.return_value = [np.random.randn(1, 10).astype(np.float32)]
        mock_session.return_value = mock_session_instance

        adapter = ONNXAdapter(config, model_path)
        adapter.load_model()

        output = adapter.predict(sample_input)

        assert isinstance(output, np.ndarray)
        assert output.shape[0] == sample_input.shape[0]

    @patch('onnxruntime.InferenceSession')
    def test_get_input_shape(self, mock_session, config: Config, temp_dir: Path):
        """Test getting input shape from ONNX model."""
        model_path = temp_dir / "test_model.onnx"
        model_path.touch()

        # Mock input shape
        mock_input = MagicMock()
        mock_input.shape = [1, 3, 224, 224]

        mock_session_instance = MagicMock()
        mock_session_instance.get_inputs.return_value = [mock_input]
        mock_session_instance.get_outputs.return_value = [MagicMock()]
        mock_session_instance.get_providers.return_value = ['CPUExecutionProvider']
        mock_session.return_value = mock_session_instance

        adapter = ONNXAdapter(config, model_path)
        adapter.load_model()

        input_shape = adapter.get_input_shape()

        assert input_shape == (1, 3, 224, 224)


class TestTensorRTAdapter:
    """Tests for TensorRTAdapter class."""

    def test_tensorrt_adapter_init(self, config: Config, temp_dir: Path):
        """Test TensorRTAdapter initialization."""
        engine_path = temp_dir / "test_model.engine"
        engine_path.touch()

        adapter = TensorRTAdapter(config, engine_path)

        assert adapter.config == config
        assert adapter.model_path == engine_path
        assert adapter.is_loaded is False

    def test_load_model_file_not_found(self, config: Config):
        """Test TensorRT model loading with non-existent file."""
        non_existent_path = Path("non_existent.engine")
        adapter = TensorRTAdapter(config, non_existent_path)

        with pytest.raises((FileNotFoundError, ImportError)):
            adapter.load_model()

    # Note: Full TensorRT testing would require actual TensorRT installation
    # and GPU hardware, so we focus on initialization and error handling


class TestModelAdapterFactory:
    """Tests for ModelAdapterFactory class."""

    def test_create_pytorch_adapter(self, config: Config, saved_model_path: Path):
        """Test creating PyTorch adapter via factory."""
        adapter = ModelAdapterFactory.create_adapter(
            config,
            'pytorch',
            saved_model_path
        )

        assert isinstance(adapter, PyTorchAdapter)
        assert adapter.model_path == saved_model_path

    def test_create_onnx_adapter(self, config: Config, temp_dir: Path):
        """Test creating ONNX adapter via factory."""
        model_path = temp_dir / "test.onnx"
        model_path.touch()

        adapter = ModelAdapterFactory.create_adapter(
            config,
            'onnx',
            model_path
        )

        assert isinstance(adapter, ONNXAdapter)
        assert adapter.model_path == model_path

    def test_create_tensorrt_adapter(self, config: Config, temp_dir: Path):
        """Test creating TensorRT adapter via factory."""
        model_path = temp_dir / "test.engine"
        model_path.touch()

        adapter = ModelAdapterFactory.create_adapter(
            config,
            'tensorrt',
            model_path
        )

        assert isinstance(adapter, TensorRTAdapter)
        assert adapter.model_path == model_path

    def test_create_adapter_invalid_type(self, config: Config, temp_dir: Path):
        """Test creating adapter with invalid type."""
        model_path = temp_dir / "test.model"
        model_path.touch()

        with pytest.raises(ValueError, match="Unsupported adapter type"):
            ModelAdapterFactory.create_adapter(
                config,
                'invalid_type',
                model_path
            )

    def test_get_supported_types(self):
        """Test getting supported adapter types."""
        supported_types = ModelAdapterFactory.get_supported_types()

        assert 'pytorch' in supported_types
        assert 'onnx' in supported_types
        assert 'tensorrt' in supported_types
        assert len(supported_types) == 3


class TestBaseModelAdapter:
    """Tests for BaseModelAdapter base class functionality."""

    def test_base_adapter_abstract(self):
        """Test that BaseModelAdapter cannot be instantiated directly."""
        from adaptive_model_serving_optimizer.utils.config import Config

        config = Config()

        with pytest.raises(TypeError):
            BaseModelAdapter(config)

    def test_create_dummy_input(self, config: Config, simple_model: nn.Module):
        """Test dummy input creation in concrete adapter."""
        adapter = PyTorchAdapter(config, model=simple_model)
        batch_size = 4
        input_shape = (3, 224, 224)

        dummy_input = adapter._create_dummy_input(batch_size, input_shape)

        assert dummy_input.shape == (batch_size, 3, 224, 224)
        assert dummy_input.dtype == torch.float32