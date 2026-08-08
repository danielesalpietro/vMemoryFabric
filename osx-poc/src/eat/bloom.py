"""M1 — Bloom Filter 2-livelli per EAT.

Livello 1: expert-level  (expert_id presente nell'EAT?)
Livello 2: shard-level   (shard (expert_id, shard_idx) presente nell'EAT?)

False positive rate target: 1% per livello.
Implementazione: pybloom-live (wrapper Bloom standard).
Lookup O(1) in ~80 ns target (DDR4 resident).

TODO (Sprint 1): sostituire con implementazione custom numpy
      se pybloom non raggiunge il target di latenza.
"""
from __future__ import annotations
from pybloom_live import BloomFilter as _BF


class BloomFilter:
    """Bloom filter 2-livelli wrappato per EAT."""

    def __init__(self, capacity: int = 16_384, error_rate: float = 0.01) -> None:
        """
        Args:
            capacity:   Numero massimo di elementi attesi (default: 256 expert × 64 shard).
            error_rate: False positive rate target per livello.
        """
        self._expert_bf: _BF = _BF(capacity=capacity, error_rate=error_rate)
        self._shard_bf:  _BF = _BF(capacity=capacity, error_rate=error_rate)

    # ── write ──────────────────────────────────────────────────────────────────

    def add(self, expert_id: int, shard_idx: int) -> None:
        """Registra (expert_id, shard_idx) in entrambi i livelli."""
        self._expert_bf.add(f"e:{expert_id}")
        self._shard_bf.add(f"s:{expert_id}:{shard_idx}")

    def remove_expert(self, expert_id: int) -> None:
        """Marca expert come non presente (bloom non supporta delete — richiede rebuild).

        NOTE: BloomFilter standard non supporta cancellazione. Questa firma è
        definita per chiarezza dell'interfaccia; l'implementazione reale richiederà
        un Counting Bloom Filter o un rebuild periodico.
        """
        raise NotImplementedError("TODO Sprint 1 — valutare Counting BF")

    # ── read ───────────────────────────────────────────────────────────────────

    def may_contain_expert(self, expert_id: int) -> bool:
        """True se expert_id è *probabilmente* presente (falsi positivi possibili)."""
        return f"e:{expert_id}" in self._expert_bf

    def may_contain_shard(self, expert_id: int, shard_idx: int) -> bool:
        """True se (expert_id, shard_idx) è *probabilmente* presente."""
        return f"s:{expert_id}:{shard_idx}" in self._shard_bf

    # ── stats ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Numero di elementi nel shard-level BF."""
        return len(self._shard_bf)
