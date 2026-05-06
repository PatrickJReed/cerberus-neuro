# cerberus-neuro reproducible training image
#
# Usage:
#   Local CPU:    docker build -t cerberus-neuro:latest .
#                 docker run --rm cerberus-neuro:latest python -c "import cerberus_neuro; print(cerberus_neuro.__version__)"
#
#   GPU host:     docker run --gpus all -e HF_TOKEN=$HF_TOKEN \
#                            -v $(pwd)/checkpoints:/workspace/checkpoints \
#                            cerberus-neuro:latest python -m cerberus_neuro.train ...
#
#   Jupyter dev: docker run --rm -it -p 8888:8888 cerberus-neuro:latest \
#                            jupyter lab --ip=0.0.0.0 --no-browser --allow-root
#
# Build is ~5-8 GB due to PyTorch CUDA base image. First build takes 10-20 min;
# subsequent builds reuse cached layers.

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# OCI metadata
LABEL org.opencontainers.image.title="cerberus-neuro"
LABEL org.opencontainers.image.description="Multi-task ResNet34 for the Broad NeuroPainting Cell Painting dataset"
LABEL org.opencontainers.image.source="https://github.com/PatrickJReed/cerberus-neuro"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# System deps for image processing + git for editable installs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY pyproject.toml README.md LICENSE /workspace/
COPY src/ /workspace/src/

RUN pip install --upgrade pip \
    && pip install -e ".[dev,training]"

# Default ports for jupyter lab
EXPOSE 8888

# Default command starts a Python REPL for sanity testing.
# Override at run time for training (`python -m cerberus_neuro.train ...`)
# or jupyter (`jupyter lab --ip=0.0.0.0 --no-browser --allow-root`).
CMD ["python"]
