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
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# ── SSH (Sprint 4 / Tekniska: RunPod pod access) ────────────────────────────────
# The base image + this Dockerfile never ran as a persistent service before —
# `docker compose run` always passes an explicit command, which masked two real
# gaps: no sshd installed, and CMD was bare `/bin/bash` (exits immediately with
# no TTY attached, e.g. under a cloud provider's container supervisor). Neither
# gap mattered for local interactive dev; both are fatal for a RunPod Pod, which
# needs the container to (a) stay running and (b) accept SSH.
# RunPod's own convention: injects the account's registered public key into the
# $PUBLIC_KEY env var at pod boot — docker-entrypoint.sh below writes it to
# authorized_keys before starting sshd, rather than baking any key into the image.
RUN mkdir -p /var/run/sshd /root/.ssh \
 && chmod 700 /root/.ssh \
 && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

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

# vLLM + coupled torch/transformers overlay (Sprint 3) — see requirements-vllm.txt
# header for why this is a separate install rather than merged into the file above.
COPY requirements-vllm.txt /tmp/requirements-vllm.txt
RUN pip install --no-cache-dir -r /tmp/requirements-vllm.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

# ── workspace ─────────────────────────────────────────────────────────────────
WORKDIR /workspace
# /workspace/osx-poc/src, not /workspace/src: docker-compose.yml's local
# bind mount is `.:/workspace` (repo root), so the code has only ever
# really lived at /workspace/osx-poc/src, even locally — this ENV disagreed
# with that from before Sprint 4, just never mattered because `make`/CI
# always override PYTHONPATH explicitly at invocation time instead of
# relying on it. Caught 2026-08-12 via scripts/smoke_test.py's own internal
# contradiction (docstring said /workspace/src, its _warn on failure said
# /workspace/osx-poc/src) and confirmed against what check_osx_src_importable()
# actually imports (eat/tier/scheduler as top-level packages — only real
# under osx-poc/src, never under a top-level src/).
ENV PYTHONPATH=/workspace/osx-poc/src

# Bake the project source into the image (Sprint 4 / Tekniska). Locally this
# is shadowed by docker-compose.yml's `.:/workspace` bind mount — live code,
# no rebuild needed to pick up a change — so nothing changes for local dev.
# On a RunPod Pod there is no bind mount: without this, the image only had
# the runtime dependencies, no project code at all (found 2026-08-12 —
# GCSGWorker/TierManager both ImportError on a real pod).
COPY osx-poc/src /workspace/osx-poc/src
COPY osx-poc/scripts /workspace/osx-poc/scripts
COPY osx-poc/configs /workspace/osx-poc/configs
COPY osx-poc/tests /workspace/osx-poc/tests

# ── runtime defaults ──────────────────────────────────────────────────────────
# CUDA visible devices passed at runtime via docker-compose / docker run --gpus
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=8

# `ENV` above sets these for the container's main process and anything
# exec'd from it — NOT for a separate SSH login session, which gets its own
# environment via PAM, not from Docker. Confirmed 2026-08-12: PYTHONPATH
# read as empty over SSH on a real pod despite being set here. Writing the
# same values to /etc/environment makes PAM's pam_env pick them up for SSH
# logins too (harmless for local dev, which never goes through sshd).
RUN { \
      echo "PYTHONPATH=/workspace/osx-poc/src"; \
      echo "CUDA_VISIBLE_DEVICES=0"; \
      echo "TOKENIZERS_PARALLELISM=false"; \
      echo "OMP_NUM_THREADS=8"; \
    } >> /etc/environment

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Local dev (`docker compose run` / `make shell`) always passes an explicit
# command, which replaces this CMD entirely — unaffected by the change below.
# With no override (a RunPod Pod), this now keeps the container alive and
# SSH-reachable instead of exiting immediately (see docker-entrypoint.sh).
CMD ["/usr/local/bin/docker-entrypoint.sh"]
