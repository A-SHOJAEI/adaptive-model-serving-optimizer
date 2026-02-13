# Dockerfile for Adaptive Model Serving Optimizer
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support (specific version for reproducibility)
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install ONNX Runtime with GPU support
RUN pip install onnxruntime-gpu==1.15.1

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/

# Install the package in editable mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/cache /app/logs

# Set permissions
RUN chmod -R 755 /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import adaptive_model_serving_optimizer; print('OK')" || exit 1

# Default command
CMD ["python3", "scripts/train.py", "--help"]

# Build arguments for customization
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Labels for metadata
LABEL maintainer="team@adaptive-serving.ai"
LABEL version="${VERSION}"
LABEL description="Adaptive Model Serving Optimizer - A comprehensive MLOps system"
LABEL org.label-schema.build-date="${BUILD_DATE}"
LABEL org.label-schema.name="adaptive-model-serving-optimizer"
LABEL org.label-schema.description="A comprehensive MLOps system for adaptive model serving optimization"
LABEL org.label-schema.url="https://github.com/adaptive-serving/adaptive-model-serving-optimizer"
LABEL org.label-schema.vcs-ref="${VCS_REF}"
LABEL org.label-schema.vcs-url="https://github.com/adaptive-serving/adaptive-model-serving-optimizer.git"
LABEL org.label-schema.vendor="Adaptive Serving Team"
LABEL org.label-schema.version="${VERSION}"
LABEL org.label-schema.schema-version="1.0"