"""M2 — Async NVMe I/O.

Sostituisce io_uring (non disponibile su WSL2/Docker Windows) con
asyncio + aiofiles. L'interfaccia è identica a quella che userà
io_uring su Linux bare-metal: il cambio di backend sarà trasparente.

Target latenza NVMe → DDR4 (256 MB shard):
    Teorico   : 256 MB / 5.5 GB/s ≈ 46 ms transfer + ~100 µs seek
    Dev target : < 100 ms (asyncio overhead accettabile in dev)
    Prod target: < 50 ms (con io_uring su Linux)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from eat.types import SHARD_SIZE_MB, ExpertID, ShardID

SHARD_SIZE_BYTES = SHARD_SIZE_MB * 1024 * 1024


class AsyncNVMeIO:
    """I/O asincrono su volume NVMe (simulato via filesystem nel container).

    Args:
        base_path: Directory root per cold storage degli shard.
                   In dev: volume Docker montato su /data/nvme.
                   In prod: mount point NVMe PCIe Gen3.
    """

    def __init__(self, base_path: str = "/data/nvme") -> None:
        self._base = Path(base_path)

    def _shard_path(self, expert_id: ExpertID, shard_idx: ShardID) -> Path:
        """Path su filesystem per uno shard: <base>/<expert_id>/<shard_idx>.bin"""
        return self._base / str(expert_id) / f"{shard_idx}.bin"

    # ── read (NVMe → buffer DDR4) ──────────────────────────────────────────────

    async def read_shard(self, expert_id: ExpertID, shard_idx: ShardID,
                         out: np.ndarray | None = None) -> np.ndarray:
        """Legge uno shard da NVMe in un buffer numpy DDR4.

        Args:
            expert_id: ID expert.
            shard_idx: Indice shard.
            out:       Buffer pre-allocato (riutilizzo Slab). Se None, alloca.

        Returns:
            numpy array uint8 con il contenuto dello shard.
        """
        raise NotImplementedError("TODO Sprint 2")

    # ── write (DDR4 → NVMe) ────────────────────────────────────────────────────

    async def write_shard(self, expert_id: ExpertID, shard_idx: ShardID,
                          data: np.ndarray) -> None:
        """Scrive uno shard da buffer DDR4 su NVMe (eviction write-back).

        Args:
            expert_id: ID expert.
            shard_idx: Indice shard.
            data:      numpy array uint8 sorgente.
        """
        raise NotImplementedError("TODO Sprint 2")

    # ── utils ──────────────────────────────────────────────────────────────────

    async def exists(self, expert_id: ExpertID, shard_idx: ShardID) -> bool:
        """Verifica esistenza shard su NVMe (non-blocking)."""
        raise NotImplementedError("TODO Sprint 2")

    async def delete(self, expert_id: ExpertID, shard_idx: ShardID) -> None:
        """Cancella shard da NVMe dopo promozione completata."""
        raise NotImplementedError("TODO Sprint 2")
