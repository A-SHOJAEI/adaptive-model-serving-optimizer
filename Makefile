# Makefile for Adaptive Model Serving Optimizer

.PHONY: help install install-dev test lint format clean docs docker run-training run-evaluation
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PACKAGE_NAME := adaptive-model-serving-optimizer
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
DOCKER_IMAGE := adaptive-serving-optimizer
VERSION := 1.0.0

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)Adaptive Model Serving Optimizer - Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package and dependencies
	@echo "$(BLUE)Installing package and dependencies...$(RESET)"
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing package with development dependencies...$(RESET)"
	$(PIP) install -e ".[dev,jupyter,docs]"
	pre-commit install

install-gpu: ## Install package with GPU support
	@echo "$(BLUE)Installing package with GPU support...$(RESET)"
	$(PIP) install -e ".[gpu,dev,jupyter]"

install-all: ## Install package with all optional dependencies
	@echo "$(BLUE)Installing package with all dependencies...$(RESET)"
	$(PIP) install -e ".[all]"

test: ## Run tests
	@echo "$(BLUE)Running tests...$(RESET)"
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR)/adaptive_model_serving_optimizer --cov-report=html --cov-report=term

test-fast: ## Run fast tests only
	@echo "$(BLUE)Running fast tests...$(RESET)"
	pytest $(TEST_DIR) -v -m "not slow" --cov=$(SRC_DIR)/adaptive_model_serving_optimizer --cov-report=term

test-gpu: ## Run GPU tests
	@echo "$(BLUE)Running GPU tests...$(RESET)"
	pytest $(TEST_DIR) -v -m "gpu" --cov=$(SRC_DIR)/adaptive_model_serving_optimizer --cov-report=term

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest $(TEST_DIR) -v -m "integration" --cov=$(SRC_DIR)/adaptive_model_serving_optimizer --cov-report=term

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	flake8 $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)
	bandit -r $(SRC_DIR) -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black $(SRC_DIR) $(TEST_DIR) scripts/
	isort $(SRC_DIR) $(TEST_DIR) scripts/

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	black --check $(SRC_DIR) $(TEST_DIR) scripts/
	isort --check-only $(SRC_DIR) $(TEST_DIR) scripts/

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	cd $(DOCS_DIR) && make html

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

clean: ## Clean up build artifacts
	@echo "$(BLUE)Cleaning up build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	rm -rf .tox/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-all: clean ## Clean everything including virtual environments
	@echo "$(BLUE)Cleaning everything...$(RESET)"
	rm -rf .venv/
	rm -rf venv/
	rm -rf env/

build: clean ## Build package
	@echo "$(BLUE)Building package...$(RESET)"
	$(PYTHON) -m build

build-wheel: clean ## Build wheel package
	@echo "$(BLUE)Building wheel package...$(RESET)"
	$(PYTHON) -m build --wheel

upload-test: build ## Upload to test PyPI
	@echo "$(BLUE)Uploading to test PyPI...$(RESET)"
	twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	@echo "$(BLUE)Uploading to PyPI...$(RESET)"
	twine upload dist/*

# Development commands
setup-dev: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(PYTHON) -m venv .venv
	@echo "$(YELLOW)Activate virtual environment with: source .venv/bin/activate$(RESET)"
	@echo "$(YELLOW)Then run: make install-dev$(RESET)"

check: format-check lint test ## Run all checks (format, lint, test)

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

# Training and evaluation commands
run-training: ## Run training with default configuration
	@echo "$(BLUE)Running training with default configuration...$(RESET)"
	$(PYTHON) scripts/train.py --config configs/default.yaml --experiments 1000 --output-dir ./outputs/training

run-training-quick: ## Run quick training for testing
	@echo "$(BLUE)Running quick training for testing...$(RESET)"
	$(PYTHON) scripts/train.py --config configs/default.yaml --experiments 100 --output-dir ./outputs/training_quick

run-evaluation: ## Run evaluation with default results
	@echo "$(BLUE)Running evaluation...$(RESET)"
	$(PYTHON) scripts/evaluate.py --results ./outputs/training/training_results.json --experiment-data ./outputs/training/experiment_data.json --output-dir ./outputs/evaluation

# Jupyter commands
jupyter: ## Start Jupyter lab
	@echo "$(BLUE)Starting Jupyter lab...$(RESET)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

jupyter-notebook: ## Start Jupyter notebook
	@echo "$(BLUE)Starting Jupyter notebook...$(RESET)"
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Docker commands
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE):$(VERSION) .
	docker tag $(DOCKER_IMAGE):$(VERSION) $(DOCKER_IMAGE):latest

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run -it --rm --gpus all -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE):latest

docker-train: ## Run training in Docker
	@echo "$(BLUE)Running training in Docker...$(RESET)"
	docker run --rm --gpus all -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE):latest \
		python scripts/train.py --config configs/default.yaml --experiments 500

# MLflow commands
mlflow-ui: ## Start MLflow UI
	@echo "$(BLUE)Starting MLflow UI at http://localhost:5000$(RESET)"
	mlflow ui --host 0.0.0.0 --port 5000

# Monitoring commands
monitor: ## Start monitoring dashboard (placeholder)
	@echo "$(BLUE)Starting monitoring dashboard...$(RESET)"
	@echo "$(YELLOW)Monitoring dashboard would be implemented here$(RESET)"

# Benchmarking commands
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(PYTHON) -c "from scripts.train import main; import sys; sys.argv=['', '--experiments', '50', '--output-dir', './outputs/benchmark']; main()"

# Security checks
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	bandit -r $(SRC_DIR) -f json -o bandit-report.json
	safety check --json --output safety-report.json
	@echo "$(GREEN)Security reports generated: bandit-report.json, safety-report.json$(RESET)"

# Quality assurance
qa: format lint test security ## Run full quality assurance pipeline

# Installation verification
verify: ## Verify installation
	@echo "$(BLUE)Verifying installation...$(RESET)"
	$(PYTHON) -c "import adaptive_model_serving_optimizer; print('✓ Package imported successfully')"
	$(PYTHON) -c "from adaptive_model_serving_optimizer import Config; print('✓ Config module works')"
	$(PYTHON) -c "from adaptive_model_serving_optimizer import ServingStrategyOptimizer; print('✓ Main optimizer works')"
	@echo "$(GREEN)Installation verified successfully!$(RESET)"

# Performance profiling
profile: ## Run performance profiling
	@echo "$(BLUE)Running performance profiling...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats scripts/train.py --experiments 10 --output-dir ./outputs/profile
	@echo "$(GREEN)Profile saved to profile.stats$(RESET)"

# Data preparation
prepare-data: ## Prepare sample data for training
	@echo "$(BLUE)Preparing sample data...$(RESET)"
	mkdir -p data/models
	mkdir -p data/datasets
	$(PYTHON) -c "from adaptive_model_serving_optimizer.data import BenchmarkDataset; ds = BenchmarkDataset(1000); print('✓ Sample data prepared')"

# Project statistics
stats: ## Show project statistics
	@echo "$(BLUE)Project Statistics:$(RESET)"
	@echo "$(GREEN)Python files:$(RESET) $$(find $(SRC_DIR) $(TEST_DIR) -name '*.py' | wc -l)"
	@echo "$(GREEN)Lines of code:$(RESET) $$(find $(SRC_DIR) -name '*.py' -exec cat {} + | wc -l)"
	@echo "$(GREEN)Test files:$(RESET) $$(find $(TEST_DIR) -name 'test_*.py' | wc -l)"
	@echo "$(GREEN)Documentation files:$(RESET) $$(find . -name '*.md' -o -name '*.rst' | wc -l)"

# Release preparation
release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(RESET)"
	@echo "$(GREEN)Running full test suite...$(RESET)"
	@$(MAKE) test
	@echo "$(GREEN)Checking code quality...$(RESET)"
	@$(MAKE) lint
	@echo "$(GREEN)Verifying installation...$(RESET)"
	@$(MAKE) verify
	@echo "$(GREEN)Building package...$(RESET)"
	@$(MAKE) build
	@echo "$(GREEN)✓ Ready for release!$(RESET)"