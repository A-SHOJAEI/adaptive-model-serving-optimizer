# Configuration Management

This directory contains configuration files for the Adaptive Model Serving Optimizer. The system supports environment-based configuration with automatic environment variable substitution.

## Configuration Files

### Core Configuration Files

- **`default.yaml`** - Base configuration with all default settings
- **`development.yaml`** - Development environment overrides
- **`production.yaml`** - Production environment optimizations
- **`testing.yaml`** - Testing environment settings

### Environment Selection

The system automatically selects the appropriate configuration based on the `ENVIRONMENT` or `ENV` environment variable:

```bash
# Use development configuration
export ENVIRONMENT=development
python your_script.py

# Use production configuration
export ENV=production
python your_script.py

# Use default configuration (if no environment is set)
python your_script.py
```

### Environment Variable Substitution

Configuration files support environment variable substitution using the syntax:
- `${VAR_NAME}` - Required environment variable
- `${VAR_NAME:default_value}` - Environment variable with default fallback

#### Example Usage

```yaml
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI:http://localhost:5000}"
  experiment_name: "${EXPERIMENT_NAME:default_experiment}"

data:
  data_dir: "${DATA_DIR:./data}"
  cache_dir: "${CACHE_DIR:./cache}"
```

### Configuration Sections

#### Global Settings
```yaml
device: "cuda"                    # Computing device (cuda/cpu)
debug: false                      # Debug mode
log_level: "INFO"                 # Logging level
mixed_precision: true             # Enable mixed precision training
```

#### Serving Configuration
```yaml
serving:
  tensorrt_config:
    precision: "fp16"             # TensorRT precision (fp32/fp16/int8)
    max_workspace_size: 1073741824 # Memory limit for TensorRT

  pytorch_config:
    device: "cuda"                # PyTorch device
    jit_compile: true            # Enable JIT compilation
```

#### Bandit Algorithm Settings
```yaml
bandits:
  algorithm: "ucb"               # Algorithm type (ucb/thompson/epsilon_greedy)
  epsilon: 0.1                   # Exploration rate
  update_frequency: 10           # Updates between performance summaries
```

#### Monitoring Configuration
```yaml
monitoring:
  alert_thresholds:
    p99_latency_ms: 100.0        # Latency alert threshold
    accuracy_drop: 0.01          # Accuracy degradation threshold
    error_rate: 0.05             # Error rate threshold
```

### Environment-Specific Optimizations

#### Development Environment
- CPU-only execution for compatibility
- Debug logging enabled
- Smaller batch sizes for faster iteration
- Reduced monitoring intervals

#### Production Environment
- GPU optimization enabled
- Minimal logging for performance
- Large batch sizes for throughput
- Strict monitoring thresholds
- Environment variable integration for secrets

#### Testing Environment
- Deterministic settings for reproducibility
- Minimal resource usage
- Single-threaded execution
- Lenient thresholds for test stability

### Usage in Code

```python
from src.adaptive_model_serving_optimizer.utils.config import get_config

# Load configuration (automatically selects environment)
config = get_config()

# Load specific configuration file
config = get_config("configs/production.yaml")

# Use configuration manager for advanced features
from src.adaptive_model_serving_optimizer.utils.config import ConfigManager

manager = ConfigManager()
config = manager.load_config()

# Update configuration dynamically
updated_config = manager.update_config({
    "training.learning_rate": 0.001,
    "serving.tensorrt_config.precision": "fp16"
})
```

### Best Practices

1. **Environment Variables**: Use environment variables for sensitive data and deployment-specific settings
2. **Defaults**: Always provide sensible defaults in the base configuration
3. **Validation**: The system automatically validates configuration parameters
4. **Documentation**: Document any new configuration parameters you add
5. **Testing**: Test configuration changes across all environments

### Adding New Configuration Options

1. Add the parameter to the appropriate dataclass in `src/adaptive_model_serving_optimizer/utils/config.py`
2. Add validation logic if needed in the `_validate_config` method
3. Update the `default.yaml` file with the default value
4. Add environment-specific overrides as needed
5. Update this README with documentation for the new parameter

### Environment Variables Reference

Common environment variables used in production:

```bash
# MLflow Configuration
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
export MLFLOW_ARTIFACT_LOCATION="s3://mlflow-artifacts/"
export MLFLOW_REGISTRY_URI="http://mlflow-server:5000"

# Data Configuration
export DATA_DIR="/mnt/shared/data"
export CACHE_DIR="/tmp/model_cache"

# Environment Selection
export ENVIRONMENT="production"

# GPU Configuration
export CUDA_VISIBLE_DEVICES="0,1"

# Logging
export LOG_LEVEL="WARNING"
```