"""M2 — EMH Tier Manager.

Promozione/evizione shard tra tier EMH su setup dev corrente:
    NVMe (EMH-3) → DDR4 (EMH-1c) → VRAM 3090 (EMH-1a)

Adattamenti rispetto a spec OSX v1.0:
    - PMEM (EMH-2): disponibile solo se TierManager riceve pmem_path
      (host con mount DAX reale — vedi tier/pmem.py e
      osx-poc/LOGBOOK_NEW_Z8.md "passo 4"). None su Docker-on-Windows/
      WSL2/RunPod: nessun cambio di comportamento per quegli ambienti.
    - io_uring non disponibile su WSL2/Docker Windows → asyncio + aiofiles.
    - Pinned memory (cudaMallocHost) non disponibile → cudaMemcpy standard.
      Il delta di latenza verrà misurato e documentato come baseline.
    - Dual-GPU deferred (RTX 5080 non disponibile).

Policy eviction: SEE (Semantic Expert Eviction) — stub, fallback LRU.
"""
from .gpu import GPUTransfer
from .io import AsyncNVMeIO
from .manager import TierManager
from .pmem import PMEMTransfer
from .policies import EvictionCandidate, LRUPolicy, SEEPolicy

__all__ = ["TierManager", "SEEPolicy", "LRUPolicy", "EvictionCandidate",
           "AsyncNVMeIO", "GPUTransfer", "PMEMTransfer"]
