"""M1 — Slab Allocator per EAT.

Gestisce un pool pre-allocato di slot da SHARD_SIZE bytes su DDR4 (numpy).
PMEM deferred: quando disponibile, il backend numpy sarà sostituito
con un mmap su /dev/pmem0 via libpmem2.

Caratteristiche target:
- Zero frammentazione esterna (slot fissi da 256 MB).
- Variable-tail: l'ultimo shard di ogni expert ha dimensione variabile (0–256 MB).
- Metadata in DDR4; payload in numpy array (placeholder per PMEM).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

SHARD_SIZE_MB: int = 256
SHARD_SIZE_BYTES: int = SHARD_SIZE_MB * 1024 * 1024


@dataclass
class SlotMetadata:
    expert_id:  int
    shard_idx:  int
    size_bytes: int        # effettivo (≤ SHARD_SIZE_BYTES per variable-tail)
    is_tail:    bool = False


class SlabAllocator:
    """Pool allocator DDR4 per shard neurali.

    Args:
        n_slots:    Numero di slot pre-allocati nel pool.
        shard_size: Dimensione fissa per slot non-tail (bytes).
    """

    def __init__(self, n_slots: int = 4, shard_size: int = SHARD_SIZE_BYTES) -> None:
        self._n_slots    = n_slots
        self._shard_size = shard_size
        # Placeholder: in produzione sarà mmap PMEM / numpy su DDR4
        self._pool: Optional[np.ndarray] = None
        self._free_slots: list[int] = []
        self._alloc_map: Dict[int, SlotMetadata] = {}  # slot_idx → metadata

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Pre-alloca il pool in DDR4. Chiamare prima di qualsiasi alloc/free."""
        raise NotImplementedError("TODO Sprint 1")

    def shutdown(self) -> None:
        """Rilascia la memoria del pool."""
        raise NotImplementedError("TODO Sprint 1")

    # ── alloc / free ───────────────────────────────────────────────────────────

    def alloc(self, expert_id: int, shard_idx: int,
              size_bytes: int, is_tail: bool = False) -> int:
        """Alloca uno slot dal pool.

        Returns:
            slot_idx: indice dello slot allocato.
        Raises:
            MemoryError: pool esaurito.
        """
        raise NotImplementedError("TODO Sprint 1")

    def free(self, slot_idx: int) -> None:
        """Restituisce uno slot al pool."""
        raise NotImplementedError("TODO Sprint 1")

    def get_buffer(self, slot_idx: int) -> np.ndarray:
        """Restituisce la view numpy del buffer associato allo slot."""
        raise NotImplementedError("TODO Sprint 1")

    # ── stats ──────────────────────────────────────────────────────────────────

    @property
    def free_slots(self) -> int:
        raise NotImplementedError("TODO Sprint 1")

    @property
    def used_slots(self) -> int:
        raise NotImplementedError("TODO Sprint 1")
