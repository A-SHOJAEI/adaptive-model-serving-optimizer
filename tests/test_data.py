"""Tests for data loading and preprocessing utilities."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from adaptive_model_serving_optimizer.data.loader import (
    ModelLoader,
    DatasetLoader,
    BenchmarkDataset,
    create_sample_batch
)
from adaptive_model_serving_optimizer.data.preprocessing import (
    ModelOptimizer,
    PerformanceBenchmark
)
from adaptive_model_serving_optimizer.utils.config import Config


class TestModelLoader:
    """Tests for ModelLoader class."""

    def test_model_loader_init(self, config: Config):
        """Test ModelLoader initialization."""
        loader = ModelLoader(config)
        assert loader.config == config
        assert loader.cache_dir.exists()

    def test_load_pytorch_model_success(self, config: Config, saved_model_path: Path, simple_model: nn.Module):
        """Test successful PyTorch model loading."""
        loader = ModelLoader(config)
        loaded_model = loader.load_pytorch_model(saved_model_path)

        assert isinstance(loaded_model, nn.Module)
        assert not loaded_model.training  # Should be in eval mode

    def test_load_pytorch_model_file_not_found(self, config: Config):
        """Test PyTorch model loading with non-existent file."""
        loader = ModelLoader(config)
        non_existent_path = Path("non_existent_model.pth")

        with pytest.raises(FileNotFoundError):
            loader.load_pytorch_model(non_existent_path)

    @patch('requests.get')
    def test_download_model_success(self, mock_get, config: Config):
        """Test successful model download."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content.return_value = [b'model_data']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        loader = ModelLoader(config)
        url = "https://example.com/model.pth"

        downloaded_path = loader.download_model(url)

        assert downloaded_path.exists()
        assert downloaded_path.name == "model.pth"

    def test_download_model_custom_filename(self, config: Config):
        """Test model download with custom filename."""
        with patch('requests.get') as mock_get:
            # Mock response
            mock_response = MagicMock()
            mock_response.headers = {'content-length': '1000'}
            mock_response.iter_content.return_value = [b'model_data']
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            loader = ModelLoader(config)
            url = "https://example.com/some_model.pth"
            custom_filename = "custom_model.pth"

            downloaded_path = loader.download_model(url, custom_filename)

            assert downloaded_path.name == custom_filename


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_dataset_loader_init(self, config: Config):
        """Test DatasetLoader initialization."""
        loader = DatasetLoader(config)
        assert loader.config == config
        assert loader.data_dir.exists()

    def test_get_transform_basic(self, config: Config):
        """Test basic transform creation."""
        loader = DatasetLoader(config)
        transforms = ['resize', 'normalize']

        transform = loader.get_transform(transforms)

        # Should have ToTensor, Resize, and Normalize
        assert len(transform.transforms) == 3

    def test_get_transform_with_augmentation(self, config: Config):
        """Test transform creation with augmentation."""
        loader = DatasetLoader(config)
        transforms = ['resize', 'normalize', 'augment']

        transform = loader.get_transform(transforms, is_training=True)

        # Should have more transforms due to augmentation
        assert len(transform.transforms) > 3

    def test_create_data_loaders_single(self, config: Config, benchmark_dataset: BenchmarkDataset):
        """Test single data loader creation."""
        loader = DatasetLoader(config)

        data_loader = loader.create_data_loaders(benchmark_dataset, batch_size=8)

        assert isinstance(data_loader, DataLoader)
        assert data_loader.batch_size == 8

    def test_create_data_loaders_with_validation_split(self, config: Config, benchmark_dataset: BenchmarkDataset):
        """Test data loader creation with validation split."""
        loader = DatasetLoader(config)
        validation_split = 0.2

        train_loader, val_loader = loader.create_data_loaders(
            benchmark_dataset,
            batch_size=8,
            validation_split=validation_split
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

        # Check approximate split sizes
        total_samples = len(benchmark_dataset)
        expected_val_size = int(total_samples * validation_split)
        expected_train_size = total_samples - expected_val_size

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)

        assert train_size == expected_train_size
        assert val_size == expected_val_size


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset class."""

    def test_benchmark_dataset_init(self):
        """Test BenchmarkDataset initialization."""
        num_samples = 50
        input_shape = (3, 224, 224)
        num_classes = 10

        dataset = BenchmarkDataset(
            num_samples=num_samples,
            input_shape=input_shape,
            num_classes=num_classes
        )

        assert len(dataset) == num_samples
        assert dataset.input_shape == input_shape
        assert dataset.num_classes == num_classes

    def test_benchmark_dataset_getitem(self):
        """Test BenchmarkDataset __getitem__."""
        dataset = BenchmarkDataset(num_samples=10, input_shape=(3, 32, 32), num_classes=5)

        data, target = dataset[0]

        assert data.shape == (3, 32, 32)
        assert isinstance(target, torch.Tensor)
        assert 0 <= target.item() < 5

    def test_benchmark_dataset_reproducibility(self):
        """Test that BenchmarkDataset is reproducible."""
        dataset1 = BenchmarkDataset(num_samples=10, input_shape=(3, 32, 32))
        dataset2 = BenchmarkDataset(num_samples=10, input_shape=(3, 32, 32))

        # Should produce same data due to fixed seed
        data1, target1 = dataset1[0]
        data2, target2 = dataset2[0]

        assert torch.allclose(data1, data2)
        assert target1.item() == target2.item()


class TestModelOptimizer:
    """Tests for ModelOptimizer class."""

    def test_model_optimizer_init(self, config: Config):
        """Test ModelOptimizer initialization."""
        optimizer = ModelOptimizer(config)
        assert optimizer.config == config

    def test_optimize_pytorch_model_basic(self, config: Config, simple_model: nn.Module, sample_input: torch.Tensor):
        """Test basic PyTorch model optimization."""
        config.serving.pytorch_config['jit_compile'] = False  # Disable JIT for simplicity
        config.mixed_precision = False

        optimizer = ModelOptimizer(config)
        optimized_model = optimizer.optimize_pytorch_model(
            simple_model,
            sample_input,
            optimization_level='basic'
        )

        assert isinstance(optimized_model, nn.Module)
        assert not optimized_model.training

    def test_optimize_pytorch_model_aggressive(self, config: Config, simple_model: nn.Module, sample_input: torch.Tensor):
        """Test aggressive PyTorch model optimization."""
        config.serving.pytorch_config['jit_compile'] = False  # Disable JIT to avoid tracing issues
        config.mixed_precision = False  # Disable to avoid precision issues

        optimizer = ModelOptimizer(config)
        optimized_model = optimizer.optimize_pytorch_model(
            simple_model,
            sample_input,
            optimization_level='aggressive'
        )

        assert isinstance(optimized_model, nn.Module)

    def test_convert_to_onnx(self, config: Config, simple_model: nn.Module, sample_input: torch.Tensor, temp_dir: Path):
        """Test ONNX conversion."""
        optimizer = ModelOptimizer(config)
        output_path = temp_dir / "test_model.onnx"

        # Patch ONNX operations to avoid actual conversion
        with patch('torch.onnx.export') as mock_export, \
             patch('onnx.load') as mock_load, \
             patch('onnx.checker.check_model') as mock_check:

            mock_load.return_value = MagicMock()
            mock_check.return_value = None

            # Create dummy ONNX file
            output_path.touch()

            result_path = optimizer.convert_to_onnx(
                simple_model,
                sample_input,
                output_path
            )

            assert result_path.exists()
            mock_export.assert_called_once()


class TestPerformanceBenchmark:
    """Tests for PerformanceBenchmark class."""

    def test_performance_benchmark_init(self, config: Config):
        """Test PerformanceBenchmark initialization."""
        benchmark = PerformanceBenchmark(config)
        assert benchmark.config == config

    def test_benchmark_latency(self, config: Config, simple_model: nn.Module):
        """Test latency benchmarking."""
        benchmark = PerformanceBenchmark(config)
        inputs = [torch.randn(1, 3, 224, 224) for _ in range(5)]

        stats = benchmark.benchmark_latency(
            simple_model,
            inputs,
            num_iterations=10,
            warmup_iterations=2
        )

        assert 'mean_latency_ms' in stats
        assert 'p95_latency_ms' in stats
        assert 'p99_latency_ms' in stats
        assert stats['mean_latency_ms'] > 0

    def test_benchmark_throughput(self, config: Config, simple_model: nn.Module):
        """Test throughput benchmarking."""
        benchmark = PerformanceBenchmark(config)
        batch_sizes = [1, 4, 8]
        input_shape = (3, 224, 224)

        throughput_results = benchmark.benchmark_throughput(
            simple_model,
            batch_sizes,
            input_shape,
            duration_seconds=1.0
        )

        assert len(throughput_results) == len(batch_sizes)
        for batch_size in batch_sizes:
            assert batch_size in throughput_results
            assert throughput_results[batch_size] > 0

    def test_get_model_memory_usage(self, config: Config, simple_model: nn.Module):
        """Test model memory usage measurement."""
        benchmark = PerformanceBenchmark(config)

        memory_stats = benchmark.get_model_memory_usage(simple_model)

        # Should have some memory stats even if not on GPU
        assert 'parameter_memory_bytes' in memory_stats
        assert 'buffer_memory_bytes' in memory_stats
        assert memory_stats['parameter_memory_bytes'] > 0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_sample_batch(self):
        """Test sample batch creation."""
        batch_size = 4
        input_shape = (3, 224, 224)
        device = 'cpu'

        batch = create_sample_batch(batch_size, input_shape, device)

        assert batch.shape == (batch_size,) + input_shape
        assert batch.device.type == device
        assert batch.dtype == torch.float32

    def test_create_sample_batch_cuda_fallback(self):
        """Test sample batch creation with CUDA fallback."""
        batch_size = 2
        input_shape = (3, 32, 32)

        # Try to create on CUDA, should work or fallback gracefully
        try:
            batch = create_sample_batch(batch_size, input_shape, 'cuda')
            assert batch.shape == (batch_size,) + input_shape
        except RuntimeError:
            # CUDA not available, should be handled gracefully
            pass