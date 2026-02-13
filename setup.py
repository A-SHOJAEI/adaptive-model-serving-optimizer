#!/usr/bin/env python3
"""Setup script for Adaptive Model Serving Optimizer.

This script provides backward compatibility for environments that don't support
pyproject.toml. The main configuration is in pyproject.toml.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "A comprehensive MLOps system for adaptive model serving optimization"

# Core dependencies
install_requires = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
    "mlflow>=2.5.0",
    "ray>=2.5.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "omegaconf>=2.3.0",
    "pyyaml>=6.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "requests>=2.31.0",
    "psutil>=5.9.0",
    "tqdm>=4.65.0",
    "click>=8.1.0",
]

# Extra dependencies for different use cases
extras_require = {
    "gpu": [
        "onnxruntime-gpu>=1.15.0",
        "py3nvml>=0.2.7",
        "gpustat>=1.1.0",
    ],
    "tensorrt": [
        "pycuda>=2022.2",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "black>=23.3.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.4.0",
        "pre-commit>=3.3.0",
        "bandit>=1.7.5",
        "safety>=2.3.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "ipykernel>=6.23.0",
        "jupyterlab>=4.0.0",
        "notebook>=6.5.0",
        "nbconvert>=7.6.0",
        "plotly>=5.15.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=2.0.0",
    ],
    "api": [
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
    ],
}

# Add 'all' extra that includes everything
extras_require["all"] = sum(extras_require.values(), [])

# Entry points for command-line scripts
entry_points = {
    "console_scripts": [
        "adaptive-serving-train=adaptive_model_serving_optimizer.scripts.train:main",
        "adaptive-serving-evaluate=adaptive_model_serving_optimizer.scripts.evaluate:main",
        "adaptive-serving-serve=adaptive_model_serving_optimizer.scripts.serve:main",
    ]
}

setup(
    name="adaptive-model-serving-optimizer",
    version="1.0.0",
    author="Adaptive Model Serving Team",
    author_email="team@adaptive-serving.ai",
    description="A comprehensive MLOps system for adaptive model serving optimization using multi-armed bandits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adaptive-serving/adaptive-model-serving-optimizer",
    project_urls={
        "Bug Tracker": "https://github.com/adaptive-serving/adaptive-model-serving-optimizer/issues",
        "Documentation": "https://adaptive-model-serving-optimizer.readthedocs.io",
        "Source Code": "https://github.com/adaptive-serving/adaptive-model-serving-optimizer",
        "Change Log": "https://github.com/adaptive-serving/adaptive-model-serving-optimizer/blob/main/CHANGELOG.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "adaptive_model_serving_optimizer": [
            "configs/*.yaml",
            "data/*.json",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    keywords=[
        "machine-learning",
        "mlops",
        "model-serving",
        "optimization",
        "multi-armed-bandits",
        "tensorrt",
        "onnx",
        "pytorch",
        "triton",
        "performance-optimization",
    ],
    license="MIT",
    zip_safe=False,
    include_package_data=True,
)