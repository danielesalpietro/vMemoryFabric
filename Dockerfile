# OSX-PoC — Development Environment
# Target: Docker on Windows (WSL2 + NVIDIA Container Toolkit)
# GPU: RTX 3090 24 GB (single GPU, no pinned memory, no io_uring)
# Scope: M1 (EAT) + M2 (Tier Manager) + M3 (Expert Scheduler)
# M4 (RecursiveMAS) deferred.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# ── system deps ────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    git \
    curl \
    wget \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# make python3.12 the default and bootstrap pip for it
# (the apt python3-pip package targets Ubuntu's default python3, not deadsnakes' 3.12)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && python3.12 -m ensurepip --upgrade \
 && python -m pip install --upgrade pip

# ── Python deps ────────────────────────────────────────────────────────────────
# Pin versions for reproducibility.
# NOTE: no libpmem2 (PMEM deferred), no liburing (io_uring not available on WSL2/Windows Docker).
# pinned memory (cudaMallocHost) is intentionally NOT used; cudaMemcpy standard path instead.

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── workspace ─────────────────────────────────────────────────────────────────
WORKDIR /workspace
ENV PYTHONPATH=/workspace/src

# ── runtime defaults ──────────────────────────────────────────────────────────
# CUDA visible devices passed at runtime via docker-compose / docker run --gpus
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=8

CMD ["/bin/bash"]
