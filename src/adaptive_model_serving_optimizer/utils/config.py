"""Configuration management for adaptive model serving optimizer.

This module provides comprehensive configuration management for the MLOps system,
supporting environment-based configurations, validation, and dynamic updates.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    """Configuration for model serving strategies."""

    tensorrt_config: Dict[str, Any] = field(default_factory=lambda: {
        'precision': 'fp16',
        'max_workspace_size': 1 << 30,  # 1GB
        'min_timing_iterations': 2,
        'avg_timing_iterations': 1,
        'max_batch_size': 32,
        'opt_batch_size': 8
    })

    onnx_config: Dict[str, Any] = field(default_factory=lambda: {
        'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'provider_options': [{'device_id': 0}, {}],
        'session_options': {
            'intra_op_num_threads': 4,
            'inter_op_num_threads': 4,
            'execution_mode': 'ORT_SEQUENTIAL',
            'graph_optimization_level': 'ORT_ENABLE_ALL'
        }
    })

    pytorch_config: Dict[str, Any] = field(default_factory=lambda: {
        'device': 'cuda',
        'precision': 'float16',
        'compile_mode': 'default',
        'dynamic_batching': True,
        'jit_compile': True
    })


@dataclass
class BanditsConfig:
    """Configuration for multi-armed bandit algorithms."""

    algorithm: str = 'ucb'  # ucb, thompson, epsilon_greedy
    epsilon: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    confidence_interval: float = 0.95
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    window_size: int = 1000
    update_frequency: int = 10


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring and metrics."""

    latency_percentiles: List[float] = field(default_factory=lambda: [50, 90, 95, 99])
    accuracy_threshold: float = 0.005  # Max degradation allowed
    throughput_window: int = 60  # seconds
    metric_collection_interval: int = 5  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'p99_latency_ms': 100,
        'accuracy_drop': 0.01,
        'error_rate': 0.05,
        'memory_usage_percent': 90
    })


@dataclass
class TrainingConfig:
    """Configuration for model training and optimization."""

    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    checkpoint_frequency: int = 10

    # Model compression settings
    pruning_sparsity: float = 0.1
    quantization_bits: int = 8
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    data_dir: str = './data'
    cache_dir: str = './cache'
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Dataset settings
    train_transforms: List[str] = field(default_factory=lambda: [
        'resize', 'normalize', 'augment'
    ])
    val_transforms: List[str] = field(default_factory=lambda: [
        'resize', 'normalize'
    ])

    # Preprocessing
    image_size: tuple = (224, 224)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class RewardConfig:
    """Configuration for reward calculation in bandit algorithms."""

    # Component weights (must sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'latency': 0.3,
        'throughput': 0.3,
        'accuracy': 0.3,
        'error': 0.1
    })

    # Normalization constants for scoring
    normalization: Dict[str, float] = field(default_factory=lambda: {
        'max_latency_ms': 1000.0,  # Worst case latency for scoring
        'ideal_throughput': 1000.0,  # Ideal throughput samples/sec
        'min_accuracy': 0.0,  # Minimum acceptable accuracy
        'max_error_rate': 1.0  # Maximum error rate for scoring
    })

    # Convergence detection parameters
    convergence_threshold: float = 0.01  # 1% change threshold
    baseline_window_size: int = 100  # Number of experiments for baseline

    def __post_init__(self):
        """Validate reward configuration."""
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Reward weights must sum to 1.0, got {weight_sum}")

        # Validate required weight keys
        required_keys = {'latency', 'throughput', 'accuracy', 'error'}
        if set(self.weights.keys()) != required_keys:
            raise ValueError(f"Reward weights must contain keys {required_keys}")

        # Validate normalization values are non-negative
        for key, value in self.normalization.items():
            if key == 'min_accuracy' and not (0 <= value <= 1):
                raise ValueError(f"min_accuracy must be between 0 and 1, got {value}")
            elif key in ['max_latency_ms', 'ideal_throughput', 'max_error_rate'] and value <= 0:
                raise ValueError(f"Normalization constant {key} must be positive, got {value}")


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking."""

    tracking_uri: str = 'http://localhost:5000'
    experiment_name: str = 'adaptive_serving_optimizer'
    artifact_location: Optional[str] = None
    registry_uri: Optional[str] = None

    # Logging settings
    log_artifacts: bool = True
    log_models: bool = True
    log_metrics_step: int = 1
    log_params: bool = True
    auto_log: bool = True


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""

    serving: ServingConfig = field(default_factory=ServingConfig)
    bandits: BanditsConfig = field(default_factory=BanditsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Global settings
    seed: int = 42
    device: str = 'cuda'
    debug: bool = False
    log_level: str = 'INFO'
    project_name: str = 'adaptive-model-serving-optimizer'
    version: str = '1.0.0'

    # Performance settings
    mixed_precision: bool = True
    compile_model: bool = True
    optimize_memory: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_logging()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if not 0 < self.training.learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0 and 1 (inclusive)")

        if not 0 <= self.training.validation_split <= 1:
            raise ValueError("Validation split must be between 0 and 1")

        if not 0 <= self.bandits.epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")

        if self.monitoring.accuracy_threshold < 0:
            raise ValueError("Accuracy threshold must be non-negative")

        if self.training.epochs <= 0:
            raise ValueError("Training epochs must be positive")

        if self.training.early_stopping_patience <= 0:
            raise ValueError("Early stopping patience must be positive")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('adaptive_serving_optimizer.log')
            ]
        )


class ConfigManager:
    """Manager for loading and updating configurations."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize config manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self._config: Optional[Config] = None

    def _get_default_config_path(self) -> Path:
        """Get default configuration file path based on environment."""
        # Check for environment-specific config first
        env = os.getenv('ENVIRONMENT', os.getenv('ENV', 'default')).lower()

        # Environment-specific config paths
        env_specific_paths = [
            Path(f'./configs/{env}.yaml'),
            Path('./configs/default.yaml'),
            Path('./config.yaml'),
            Path('~/.adaptive_serving/config.yaml').expanduser()
        ]

        for path in env_specific_paths:
            if path.exists():
                logger.info(f"Using configuration file: {path}")
                return path

        # Return first path as default
        logger.warning(f"No configuration file found, will create default at: {env_specific_paths[1]}")
        return env_specific_paths[1]

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Config:
        """Load configuration from file.

        Args:
            config_path: Optional path to config file

        Returns:
            Loaded configuration object
        """
        if config_path:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self._config = Config()
            return self._config

        try:
            with open(self.config_path, 'r') as f:
                config_content = f.read()

            # Substitute environment variables
            config_content = self._substitute_env_vars(config_content)
            config_dict = yaml.safe_load(config_content)

            # Load environment-specific overrides if available
            config_dict = self._load_env_overrides(config_dict)

            # Merge with defaults using OmegaConf
            default_config = OmegaConf.structured(Config())
            loaded_config = OmegaConf.create(config_dict)
            merged_config = OmegaConf.merge(default_config, loaded_config)

            # Convert back to dataclass
            self._config = OmegaConf.to_object(merged_config)

            logger.info(f"Configuration loaded from {self.config_path}")
            return self._config

        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
            self._config = Config()
            return self._config

    def save_config(self, config: Config, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.

        Args:
            config: Configuration object to save
            config_path: Optional path to save config file
        """
        save_path = Path(config_path) if config_path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert to OmegaConf for better serialization
            omega_config = OmegaConf.structured(config)
            config_dict = OmegaConf.to_yaml(omega_config)

            with open(save_path, 'w') as f:
                f.write(config_dict)

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
            raise

    def update_config(self, updates: Dict[str, Any]) -> Config:
        """Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates

        Returns:
            Updated configuration object
        """
        if self._config is None:
            self._config = self.load_config()

        # Convert to OmegaConf for easier updates
        omega_config = OmegaConf.structured(self._config)
        omega_updates = OmegaConf.create(updates)

        merged_config = OmegaConf.merge(omega_config, omega_updates)
        self._config = OmegaConf.to_object(merged_config)

        return self._config

    def get_config(self) -> Config:
        """Get current configuration.

        Returns:
            Current configuration object
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def reload_config(self) -> Config:
        """Reload configuration from file.

        Returns:
            Reloaded configuration object
        """
        self._config = None
        return self.load_config()

    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration content.

        Args:
            content: Configuration file content

        Returns:
            Content with environment variables substituted
        """
        import re

        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

        def replace_env_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""

            env_value = os.getenv(var_name, default_value)
            if not env_value and not default_value:
                logger.warning(f"Environment variable {var_name} not found and no default provided")
            return env_value

        substituted = pattern.sub(replace_env_var, content)
        return substituted

    def _load_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load environment-specific configuration overrides.

        Args:
            config_dict: Base configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        env = os.getenv('ENVIRONMENT', os.getenv('ENV', 'default')).lower()

        if env != 'default':
            env_config_path = self.config_path.parent / f"{env}.yaml"

            if env_config_path.exists():
                try:
                    with open(env_config_path, 'r') as f:
                        env_content = f.read()

                    # Substitute environment variables in override file too
                    env_content = self._substitute_env_vars(env_content)
                    env_config = yaml.safe_load(env_content)

                    if env_config:
                        # Deep merge the environment config
                        env_conf = OmegaConf.create(env_config)
                        base_conf = OmegaConf.create(config_dict)
                        merged = OmegaConf.merge(base_conf, env_conf)
                        config_dict = OmegaConf.to_container(merged, resolve=True)
                        logger.info(f"Applied environment overrides from {env_config_path}")

                except Exception as e:
                    logger.warning(f"Failed to load environment config {env_config_path}: {e}")

        return config_dict


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Convenience function to get configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configuration object
    """
    manager = ConfigManager(config_path)
    return manager.load_config()


def setup_environment_from_config(config: Config) -> None:
    """Setup environment variables and settings from configuration.

    Args:
        config: Configuration object
    """
    # Set random seeds
    import random
    import numpy as np
    import torch

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    if config.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        logger.debug("GPU detection failed - no CUDA-capable devices found")
        config.device = 'cpu'
    elif config.device == 'cuda':
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        logger.info(f"CUDA available: using GPU {current_device}/{gpu_count-1} ({gpu_name})")

        # Log GPU memory info
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        logger.debug(f"GPU memory: {total_memory // (1024**3)} GB total")

    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' if config.debug else '0'

    # Memory optimization
    if config.optimize_memory and config.device == 'cuda':
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared for optimization")

    logger.info(f"Environment configured with device: {config.device}, seed: {config.seed}")