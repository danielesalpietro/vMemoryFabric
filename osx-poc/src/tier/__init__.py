"""M2 — EMH Tier Manager.

Promozione/evizione shard tra tier EMH su setup dev corrente:
    NVMe (EMH-3) → DDR4 (EMH-1c) → VRAM 3090 (EMH-1a)

Adattamenti rispetto a spec OSX v1.0:
    - PMEM (EMH-2) deferred — sarà inserito tra DDR4 e NVMe.
    - io_uring non disponibile su WSL2/Docker Windows → asyncio + aiofiles.
    - Pinned memory (cudaMallocHost) non disponibile → cudaMemcpy standard.
      Il delta di latenza verrà misurato e documentato come baseline.
    - Dual-GPU deferred (RTX 5080 non disponibile).

Policy eviction: SEE (Semantic Expert Eviction) — stub, fallback LRU.
"""
from .manager import TierManager
from .policies import SEEPolicy, LRUPolicy, EvictionCandidate
from .io import AsyncNVMeIO
from .gpu import GPUTransfer

__all__ = ["TierManager", "SEEPolicy", "LRUPolicy", "EvictionCandidate",
           "AsyncNVMeIO", "GPUTransfer"]
