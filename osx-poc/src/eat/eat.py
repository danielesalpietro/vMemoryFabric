"""M1 — Expert Access Table (EAT) — core.

Struttura centrale di OSX: mappa (expert_id, shard_idx) → EATEntry.
Bloom filter 2-livelli per lookup O(1) fast-path.
RW lock per thread safety; version counter per CAS ottimistico.

Latenza target:
    Bloom hit  → EATEntry : < 100 ns
    Bloom miss → None      : < 500 ns (confermato miss)
"""
from __future__ import annotations
import threading
import time
from typing import Dict, Iterator, Optional, Tuple

from .bloom import BloomFilter
from .slab import SlabAllocator
from .types import EATEntry, ExpertID, SHARD_SIZE_BYTES, ShardID, Tier


_Key = Tuple[ExpertID, ShardID]


class ExpertAccessTable:
    """Thread-safe Expert Access Table con Bloom filter 2-livelli.

    Args:
        capacity:   Capacità Bloom (default: 256 expert × 64 shard = 16 384).
        n_slots:    Numero di slot Slab Allocator.
    """

    def __init__(self, capacity: int = 16_384, n_slots: int = 4) -> None:
        self._bloom  = BloomFilter(capacity=capacity)
        self._slab   = SlabAllocator(n_slots=n_slots)
        self._table: Dict[_Key, EATEntry] = {}
        self._lock   = threading.RLock()

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def insert(self, expert_id: ExpertID, shard_idx: ShardID,
               tier: Tier = Tier.NVME, size_bytes: int = 0) -> EATEntry:
        """Inserisce un nuovo shard nella EAT.

        Args:
            expert_id:   ID dell'expert.
            shard_idx:   Indice dello shard all'interno dell'expert.
            tier:        Tier corrente dello shard.
            size_bytes:  Dimensione effettiva (0 = SHARD_SIZE_BYTES per non-tail).

        Returns:
            EATEntry appena creata.

        Raises:
            KeyError: shard già presente.
        """
        if not (0 <= size_bytes <= SHARD_SIZE_BYTES):
            raise ValueError(f"size_bytes {size_bytes} fuori range [0, {SHARD_SIZE_BYTES}]")
        with self._lock:
            key = (expert_id, shard_idx)
            if key in self._table:
                raise KeyError(f"shard già presente: {key}")
            entry = EATEntry(expert_id=expert_id, shard_idx=shard_idx, tier=tier)
            self._table[key] = entry
            self._bloom.add(expert_id, shard_idx)
            return entry

    def lookup(self, expert_id: ExpertID, shard_idx: ShardID) -> Optional[EATEntry]:
        """Recupera un EATEntry — fast path via Bloom filter.

        Returns:
            EATEntry se presente, None altrimenti.
        """
        with self._lock:
            if not self._bloom.may_contain_shard(expert_id, shard_idx):
                return None
            return self._table.get((expert_id, shard_idx))

    def update_tier(self, expert_id: ExpertID, shard_idx: ShardID, new_tier: Tier) -> None:
        """Aggiorna il tier di uno shard (chiamato dal Tier Manager post-promozione/evizione).

        Thread-safe tramite RW lock + version bump.
        """
        with self._lock:
            entry = self._table.get((expert_id, shard_idx))
            if entry is None:
                raise KeyError(f"shard non presente: {(expert_id, shard_idx)}")
            entry.tier = new_tier
            entry.version += 1

    def evict(self, expert_id: ExpertID, shard_idx: ShardID) -> Optional[EATEntry]:
        """Rimuove uno shard dalla EAT (eviction dal Tier Manager).

        NOTE: il Bloom filter non supporta cancellazione — la entry rimane
        nel BF come falso positivo fino al prossimo rebuild.
        """
        with self._lock:
            return self._table.pop((expert_id, shard_idx), None)

    def access(self, expert_id: ExpertID, shard_idx: ShardID) -> Optional[EATEntry]:
        """Registra un accesso (touch) e restituisce la entry aggiornata."""
        with self._lock:
            entry = self._table.get((expert_id, shard_idx))
            if entry is None:
                return None
            entry.touch()
            return entry

    # ── bulk ops ───────────────────────────────────────────────────────────────

    def get_tier(self, tier: Tier) -> list[EATEntry]:
        """Restituisce tutte le entry in un dato tier (per Tier Manager)."""
        with self._lock:
            return [e for e in self._table.values() if e.tier == tier]

    def eviction_candidates(self, tier: Tier, n: int) -> list[EATEntry]:
        """Top-n candidati all'eviction nel tier dato (SEE score — vedi Tier Manager).

        Fallback LRU se SEE non disponibile.
        """
        with self._lock:
            candidates = [e for e in self._table.values() if e.tier == tier]
            candidates.sort(key=lambda e: e.last_access_ts)
            return candidates[:n]

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Inizializza Slab Allocator e strutture interne."""
        self._slab.initialize()

    def shutdown(self) -> None:
        """Shutdown graceful — rilascia Slab Allocator."""
        self._slab.shutdown()
        with self._lock:
            self._table.clear()

    # ── stats / iteration ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._table)

    def __iter__(self) -> Iterator[EATEntry]:
        with self._lock:
            return iter(list(self._table.values()))

    def stats(self) -> dict:
        """Metriche per Prometheus / Grafana."""
        with self._lock:
            by_tier: Dict[str, int] = {}
            for entry in self._table.values():
                by_tier[entry.tier.name] = by_tier.get(entry.tier.name, 0) + 1
            return {
                "total_entries": len(self._table),
                "by_tier": by_tier,
                "bloom_shard_count": len(self._bloom),
            }
