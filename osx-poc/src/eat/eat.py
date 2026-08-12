"""M1 — Expert Access Table (EAT) — core.

Struttura centrale di OSX: mappa (expert_id, shard_idx) → EATEntry.
RW lock per thread safety; version counter per CAS ottimistico.

Bloom filter 2-livelli RIMOSSO (2026-08-12, issue #1, decisione presa
non solo misurata): a questa scala (capacity ~16k, la struttura sotto è
già un dict in-memory O(1)) il fast-negative path del Bloom filter
misurava consistentemente PIÙ LENTO di un lookup diretto sul dict — vedi
LOGBOOK.md 2026-08-12 per i numeri (~6.8-8.1x più lento, ri-misurato su
più run) e la sequenza di decisione. Il Bloom filter proteggeva una
struttura che non ne aveva bisogno: costava latenza invece di
risparmiarla. `src/eat/bloom.py` è stato rimosso per intero, non solo
scollegato — nessun altro punto del codice lo usava (verificato via grep
prima della rimozione), tenerlo in giro come codice morto "per sicurezza"
non avrebbe protetto nulla.

Latenza target: lookup diretto su dict sotto RLock, nessun livello
fast-negative intermedio.
"""
from __future__ import annotations
import threading
import time
from typing import Dict, Iterator, Optional, Tuple

from .slab import SlabAllocator
from .types import EATEntry, ExpertID, SHARD_SIZE_BYTES, ShardID, Tier


_Key = Tuple[ExpertID, ShardID]


class ExpertAccessTable:
    """Thread-safe Expert Access Table — mappa (expert_id, shard_idx) -> EATEntry.

    Args:
        capacity:   Mantenuto per compatibilità di firma con le versioni
                    precedenti (era la capacità del Bloom filter, rimosso
                    2026-08-12 — vedi docstring di modulo) — non usato
                    internamente, nessuna struttura dimensionata su di esso.
        n_slots:    Numero di slot Slab Allocator.
    """

    def __init__(self, capacity: int = 16_384, n_slots: int = 4) -> None:
        self._slab   = SlabAllocator(n_slots=n_slots)
        self._table: Dict[_Key, EATEntry] = {}
        self._lock   = threading.RLock()

    @property
    def slab(self) -> SlabAllocator:
        """SlabAllocator la cui lifecycle è già gestita da initialize()/shutdown().

        Esposto per il Tier Manager (M2): alloc/free dei DDR4 slot restano
        di sua competenza, ma devono operare sulla stessa istanza il cui
        lifecycle EAT già guida — non su una seconda istanza scollegata.
        """
        return self._slab

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
            return entry

    def lookup(self, expert_id: ExpertID, shard_idx: ShardID) -> Optional[EATEntry]:
        """Recupera un EATEntry — lookup diretto sul dict (2026-08-12: nessun
        fast-negative path Bloom davanti, vedi docstring di modulo).

        Returns:
            EATEntry se presente, None altrimenti.
        """
        with self._lock:
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

        Nota storica: fino al 2026-08-12 issue #4 tracciava il fatto che
        il Bloom filter non supportava cancellazione (entry evicted
        restavano falsi positivi permanenti) — risolto quel giorno con un
        Counting Bloom Filter, poi il Bloom filter stesso è stato rimosso
        del tutto poche ore dopo (issue #1: misurato più lento di un
        lookup diretto a questa scala). Questo metodo ora fa solo
        `dict.pop()`, senza alcuna struttura ausiliaria da tenere in sync.
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

    def hottest_candidates(self, tier: Tier, n: int) -> list[EATEntry]:
        """Top-n entry più "calde" nel tier dato — complemento di
        eviction_candidates() (2026-08-12, issue #17).

        eviction_candidates() ordina per last_access_ts crescente (più
        vecchio prima → chi merita di essere evictato). Questo ordina per
        (access_count, last_access_ts) decrescente (più acceduto, e tra
        pari il più recente, prima → chi merita di essere promosso/tenuto).
        Nessun peso SEE/semantico qui — quello vive nella policy del Tier
        Manager; questo è il segnale grezzo recency+frequency di EAT su
        cui una policy più sofisticata può costruire, stesso livello di
        "primitiva" di eviction_candidates().

        Usata da GCSGWorker (M3) per selezionare quali expert entrano
        nello shadow pool quando è wired a un TierManager, al posto del
        placeholder round-robin — vedi scheduler.gcsg._select_shadow_expert_ids.
        """
        with self._lock:
            candidates = [e for e in self._table.values() if e.tier == tier]
            candidates.sort(key=lambda e: (e.access_count, e.last_access_ts), reverse=True)
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
            }
