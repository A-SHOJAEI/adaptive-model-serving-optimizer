"""Data loading utilities for adaptive model serving optimizer.

This module provides comprehensive data loading capabilities for various model
formats and datasets, supporting both local and remote data sources.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100
import onnx

# Optional dependencies for TensorRT
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

from ..utils.config import Config

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading models in different formats."""

    def __init__(self, config: Config):
        """Initialize model loader.

        Args:
            config: Configuration object
        """
        self.config = config
        self.cache_dir = Path(config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_pytorch_model(self, model_path: Union[str, Path]) -> torch.nn.Module:
        """Load PyTorch model from file.

        Args:
            model_path: Path to PyTorch model file

        Returns:
            Loaded PyTorch model

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model based on file extension
            if model_path.suffix == '.pth' or model_path.suffix == '.pt':
                model = torch.load(model_path, map_location=self.config.device, weights_only=False)
            elif model_path.suffix == '.pkl':
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")

            # Set model to evaluation mode
            model.eval()

            # Move to configured device
            if hasattr(model, 'to'):
                model = model.to(self.config.device)

            logger.info(f"Successfully loaded PyTorch model from {model_path}")
            return model

        except FileNotFoundError:
            raise  # Re-raise FileNotFoundError as-is
        except Exception as e:
            logger.error(f"Failed to load PyTorch model from {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def load_onnx_model(self, model_path: Union[str, Path]) -> onnx.ModelProto:
        """Load ONNX model from file.

        Args:
            model_path: Path to ONNX model file

        Returns:
            Loaded ONNX model

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load and validate ONNX model
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)

            logger.info(f"Successfully loaded ONNX model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load ONNX model from {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def load_tensorrt_model(self, model_path: Union[str, Path]) -> 'trt.ICudaEngine':
        """Load TensorRT engine from file.

        Args:
            model_path: Path to TensorRT engine file

        Returns:
            Loaded TensorRT engine

        Raises:
            ImportError: If TensorRT dependencies are not available
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT dependencies not available. Install tensorrt and pycuda to use TensorRT models."
            )

        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Initialize TensorRT runtime
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            # Load engine from file
            with open(model_path, 'rb') as f:
                engine_data = f.read()

            engine = runtime.deserialize_cuda_engine(engine_data)

            if engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")

            logger.info(f"Successfully loaded TensorRT engine from {model_path}")
            return engine

        except Exception as e:
            logger.error(f"Failed to load TensorRT engine from {model_path}: {e}")
            raise RuntimeError(f"Engine loading failed: {e}")

    def download_model(self, url: str, filename: Optional[str] = None) -> Path:
        """Download model from URL.

        Args:
            url: Model download URL
            filename: Optional filename for downloaded model

        Returns:
            Path to downloaded model file

        Raises:
            RuntimeError: If download fails
        """
        try:
            parsed_url = urlparse(url)
            if not filename:
                filename = Path(parsed_url.path).name or 'model'

            download_path = self.cache_dir / filename

            # Check if file already exists
            if download_path.exists():
                logger.info(f"Cache HIT: Model found at {download_path}")
                logger.debug(f"Cache statistics: Avoided downloading {filename}")
                return download_path

            # Download with progress tracking
            logger.info(f"Cache MISS: Downloading model from {url}")
            logger.debug(f"Download destination: {download_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")

            logger.info(f"Model downloaded successfully to {download_path}")
            return download_path

        except Exception as e:
            logger.error(f"Failed to download model from {url}: {e}")
            raise RuntimeError(f"Model download failed: {e}")


class DatasetLoader:
    """Utility class for loading datasets."""

    def __init__(self, config: Config):
        """Initialize dataset loader.

        Args:
            config: Configuration object
        """
        self.config = config
        self.data_dir = Path(config.data.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_transform(self, transform_names: List[str], is_training: bool = False) -> transforms.Compose:
        """Get torchvision transforms based on configuration.

        Args:
            transform_names: List of transform names to apply. Supported transforms:
                - 'resize': Resize images to config.data.image_size
                - 'normalize': Normalize with config.data.normalize_mean and normalize_std
                - 'augment': Apply data augmentation (only when is_training=True)
                    Includes RandomHorizontalFlip(0.5), RandomRotation(10°),
                    and ColorJitter for brightness, contrast, saturation, and hue
                - 'to_tensor': Convert PIL Image to tensor and scale to [0, 1]
            is_training: Whether this is for training data. When True, enables
                augmentation transforms for better generalization

        Returns:
            Composed transforms that can be applied to dataset images

        Example:
            >>> loader = DatasetLoader(config)
            >>> transforms = loader.get_transform(['resize', 'normalize', 'augment'], is_training=True)
            >>> transformed_image = transforms(pil_image)
        """
        transform_list = []

        for transform_name in transform_names:
            if transform_name == 'resize':
                transform_list.append(
                    transforms.Resize(self.config.data.image_size)
                )
            elif transform_name == 'normalize':
                transform_list.append(
                    transforms.Normalize(
                        mean=self.config.data.normalize_mean,
                        std=self.config.data.normalize_std
                    )
                )
            elif transform_name == 'augment' and is_training:
                transform_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.05
                    )
                ])
            elif transform_name == 'to_tensor':
                transform_list.append(transforms.ToTensor())

        # Always add ToTensor if not explicitly specified
        if not any(isinstance(t, transforms.ToTensor) for t in transform_list):
            transform_list.insert(-1 if transform_list else 0, transforms.ToTensor())

        return transforms.Compose(transform_list)

    def load_cifar10(self, train: bool = True) -> Dataset:
        """Load CIFAR-10 dataset.

        Args:
            train: Whether to load training or test split

        Returns:
            CIFAR-10 dataset
        """
        transform_names = (
            self.config.data.train_transforms if train
            else self.config.data.val_transforms
        )
        transform = self.get_transform(transform_names, is_training=train)

        dataset = CIFAR10(
            root=str(self.data_dir),
            train=train,
            transform=transform,
            download=True
        )

        logger.info(f"Loaded CIFAR-10 {'training' if train else 'test'} dataset")
        return dataset

    def load_cifar100(self, train: bool = True) -> Dataset:
        """Load CIFAR-100 dataset.

        Args:
            train: Whether to load training or test split

        Returns:
            CIFAR-100 dataset
        """
        transform_names = (
            self.config.data.train_transforms if train
            else self.config.data.val_transforms
        )
        transform = self.get_transform(transform_names, is_training=train)

        dataset = CIFAR100(
            root=str(self.data_dir),
            train=train,
            transform=transform,
            download=True
        )

        logger.info(f"Loaded CIFAR-100 {'training' if train else 'test'} dataset")
        return dataset

    def load_imagenet(self, split: str = 'train') -> Dataset:
        """Load ImageNet dataset.

        Args:
            split: Dataset split ('train' or 'val')

        Returns:
            ImageNet dataset
        """
        is_training = split == 'train'
        transform_names = (
            self.config.data.train_transforms if is_training
            else self.config.data.val_transforms
        )
        transform = self.get_transform(transform_names, is_training=is_training)

        dataset = ImageNet(
            root=str(self.data_dir),
            split=split,
            transform=transform
        )

        logger.info(f"Loaded ImageNet {split} dataset")
        return dataset

    def create_data_loaders(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        validation_split: Optional[float] = None,
        shuffle: bool = True
    ) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
        """Create data loaders from dataset.

        Args:
            dataset: Input dataset
            batch_size: Batch size (uses config default if None)
            validation_split: Fraction for validation split
            shuffle: Whether to shuffle data

        Returns:
            Data loader or tuple of (train_loader, val_loader) if validation_split specified
        """
        batch_size = batch_size or self.config.training.batch_size

        if validation_split is None:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory,
                prefetch_factor=self.config.data.prefetch_factor
            )
            return loader

        # Split dataset for validation
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor
        )

        logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}")
        return train_loader, val_loader


class BenchmarkDataset(Dataset):
    """Synthetic dataset for benchmarking serving performance."""

    def __init__(
        self,
        num_samples: int = 1000,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        num_classes: int = 1000,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize benchmark dataset.

        Args:
            num_samples: Number of samples to generate
            input_shape: Shape of input tensors
            num_classes: Number of output classes
            dtype: Data type for tensors
        """
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dtype = dtype

        # Pre-generate random data for consistency
        torch.manual_seed(42)  # Fixed seed for reproducible benchmarks
        self.data = torch.randn(num_samples, *input_shape, dtype=dtype)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input, target)
        """
        return self.data[idx], self.targets[idx]


def load_pretrained_models() -> Dict[str, Path]:
    """Load information about available pretrained models from registry.

    Scans a predefined model registry for available pretrained models in different
    formats (PyTorch, ONNX, TensorRT). Only returns models that exist on disk.

    The model registry contains mappings for common vision models:
    - ResNet50 (PyTorch, ONNX, TensorRT formats)
    - MobileNet (PyTorch, ONNX formats)
    - EfficientNet (PyTorch format)

    Models not found at their expected paths are logged as warnings but excluded
    from the returned dictionary.

    Returns:
        Dictionary mapping model names to their actual file paths for models
        that exist on disk. Keys are formatted as '{architecture}_{format}'
        (e.g., 'resnet50_pytorch', 'mobilenet_onnx').

    Example:
        >>> available = load_pretrained_models()
        >>> if 'resnet50_pytorch' in available:
        ...     model_path = available['resnet50_pytorch']
        ...     # Use model_path to load the model
    """
    # This would typically load from a model registry or config file
    model_registry = {
        'resnet50_pytorch': 'models/resnet50.pth',
        'resnet50_onnx': 'models/resnet50.onnx',
        'resnet50_tensorrt': 'models/resnet50.engine',
        'mobilenet_pytorch': 'models/mobilenet.pth',
        'mobilenet_onnx': 'models/mobilenet.onnx',
        'efficientnet_pytorch': 'models/efficientnet.pth',
    }

    available_models = {}
    for name, path in model_registry.items():
        full_path = Path(path)
        if full_path.exists():
            available_models[name] = full_path
        else:
            logger.warning(f"Model {name} not found at {path}")

    return available_models


def create_sample_batch(
    batch_size: int,
    input_shape: Tuple[int, ...] = (3, 224, 224),
    device: str = 'cuda'
) -> torch.Tensor:
    """Create a sample batch for testing.

    Args:
        batch_size: Size of batch to create
        input_shape: Shape of individual inputs
        device: Device to place tensor on

    Returns:
        Sample batch tensor
    """
    return torch.randn(batch_size, *input_shape, device=device)