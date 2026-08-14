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

from dataclasses import dataclass

import numpy as np

from .types import SHARD_SIZE_BYTES


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
        self._pool: np.ndarray | None = None
        self._free_slots: list[int] = list(range(n_slots))
        self._alloc_map: dict[int, SlotMetadata] = {}  # slot_idx → metadata

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Pre-alloca il pool in DDR4. Chiamare prima di qualsiasi alloc/free."""
        self._pool = np.empty((self._n_slots, self._shard_size), dtype=np.uint8)

    def shutdown(self) -> None:
        """Rilascia la memoria del pool."""
        self._pool = None
        self._free_slots = list(range(self._n_slots))
        self._alloc_map = {}

    # ── alloc / free ───────────────────────────────────────────────────────────

    def alloc(self, expert_id: int, shard_idx: int,
              size_bytes: int, is_tail: bool = False) -> int:
        """Alloca uno slot dal pool.

        Returns:
            slot_idx: indice dello slot allocato.
        Raises:
            MemoryError: pool esaurito.
        """
        if self._pool is None:
            raise RuntimeError("SlabAllocator non inizializzato — chiamare initialize()")
        if not (0 <= size_bytes <= self._shard_size):
            raise ValueError(f"size_bytes {size_bytes} fuori range [0, {self._shard_size}]")
        if not self._free_slots:
            raise MemoryError("slab pool esaurito")
        slot_idx = self._free_slots.pop()
        self._alloc_map[slot_idx] = SlotMetadata(expert_id, shard_idx, size_bytes, is_tail)
        return slot_idx

    def free(self, slot_idx: int) -> None:
        """Restituisce uno slot al pool."""
        if slot_idx not in self._alloc_map:
            raise KeyError(f"slot {slot_idx} non allocato")
        del self._alloc_map[slot_idx]
        self._free_slots.append(slot_idx)

    def get_buffer(self, slot_idx: int) -> np.ndarray:
        """Restituisce la view numpy del buffer associato allo slot."""
        if self._pool is None:
            raise RuntimeError("SlabAllocator non inizializzato — chiamare initialize()")
        if slot_idx not in self._alloc_map:
            raise KeyError(f"slot {slot_idx} non allocato")
        return self._pool[slot_idx]

    # ── stats ──────────────────────────────────────────────────────────────────

    @property
    def free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def used_slots(self) -> int:
        return len(self._alloc_map)
