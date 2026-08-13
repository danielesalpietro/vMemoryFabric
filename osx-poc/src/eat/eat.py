"""M1 — Expert Access Table (EAT) — core.

Struttura centrale di OSX: mappa (expert_id, shard_idx) → EATEntry.
Locking a strategia selezionabile per thread safety (vedi sotto);
version counter per CAS ottimistico (non ancora usato per controllo,
solo bump — issue #23 Opzione D).

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

Latenza target: lookup diretto su dict, nessun livello fast-negative
intermedio.

Locking strategy (issue #23, 2026-08-13 — tail latency ~1360x sotto
contesa concorrente, issue #2): tre strategie selezionabili via
`locking_strategy` al costruttore, stessa API pubblica per tutte:

    "single"        (default, invariato per i chiamanti esistenti) —
                     un solo threading.Lock su tutta la tabella
                     (Opzione A: era RLock, ma nessun metodo EAT
                     richiama un altro metodo lockato mentre tiene già
                     il lock — nessuna rientranza usata — quindi Lock
                     semplice basta, senza il bookkeeping di RLock).
    "striped"        (Opzione B) — tabella partizionata in `n_shards`
                     shard indipendenti, ciascuno col proprio Lock;
                     chiave -> shard via hash. I metodi bulk (get_tier,
                     eviction_candidates, hottest_candidates, stats,
                     __len__, __iter__) leggono uno shard alla volta
                     sotto il suo lock, senza tenerli tutti insieme:
                     snapshot rilassato, potenzialmente incoerente tra
                     shard diversi se letto durante una mutazione
                     concorrente — scelta deliberata (vedi issue #23),
                     non un bug.
    "lockfree_read"  (Opzione C) — lookup() legge senza lock,
                     sfruttando l'atomicità di dict.get() sotto il GIL;
                     touch() (chiamato da access()) resta sotto lock
                     perché fa due scritture non atomiche
                     (access_count, poi last_access_ts).

Internamente "single" e "lockfree_read" sono un caso speciale di
"striped" con un solo shard: stesso comportamento di prima della issue
#23 (lock singolo, snapshot bulk coerente), nessuna migrazione richiesta
per i chiamanti esistenti.
"""
from __future__ import annotations
import threading
import time
from typing import Dict, Iterator, List, Literal, Optional, Tuple

from .slab import SlabAllocator
from .types import EATEntry, ExpertID, SHARD_SIZE_BYTES, ShardID, Tier


_Key = Tuple[ExpertID, ShardID]
LockingStrategy = Literal["single", "striped", "lockfree_read"]
_STRATEGIES: Tuple[str, ...] = ("single", "striped", "lockfree_read")


class ExpertAccessTable:
    """Thread-safe Expert Access Table — mappa (expert_id, shard_idx) -> EATEntry.

    Args:
        capacity:         Mantenuto per compatibilità di firma con le versioni
                          precedenti (era la capacità del Bloom filter, rimosso
                          2026-08-12 — vedi docstring di modulo) — non usato
                          internamente, nessuna struttura dimensionata su di esso.
        n_slots:          Numero di slot Slab Allocator.
        locking_strategy: "single" (default), "striped" o "lockfree_read" —
                          vedi docstring di modulo (issue #23).
        n_shards:         Numero di shard indipendenti. Usato solo con
                          locking_strategy="striped"; ignorato altrimenti
                          (1 shard implicito).
    """

    def __init__(self, capacity: int = 16_384, n_slots: int = 4,
                 locking_strategy: LockingStrategy = "single",
                 n_shards: int = 16) -> None:
        if locking_strategy not in _STRATEGIES:
            raise ValueError(f"locking_strategy sconosciuta: {locking_strategy!r}")
        self._slab = SlabAllocator(n_slots=n_slots)
        self._locking_strategy: LockingStrategy = locking_strategy
        self._n_shards = n_shards if locking_strategy == "striped" else 1
        self._shards: List[Dict[_Key, EATEntry]] = [dict() for _ in range(self._n_shards)]
        self._shard_locks: List[threading.Lock] = [threading.Lock() for _ in range(self._n_shards)]

    def _shard_idx(self, key: _Key) -> int:
        return hash(key) % self._n_shards if self._n_shards > 1 else 0

    def _collect(self, predicate) -> list[EATEntry]:
        """Scansiona tutti gli shard, uno alla volta sotto il proprio lock.

        Con locking_strategy="striped" e più shard questo è lo snapshot
        rilassato descritto in docstring di modulo. Con un solo shard
        ("single"/"lockfree_read") equivale esattamente al comportamento
        pre-issue #23: un solo lock, snapshot coerente.
        """
        result: list[EATEntry] = []
        for table, lock in zip(self._shards, self._shard_locks):
            with lock:
                result.extend(e for e in table.values() if predicate(e))
        return result

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
        key = (expert_id, shard_idx)
        idx = self._shard_idx(key)
        with self._shard_locks[idx]:
            table = self._shards[idx]
            if key in table:
                raise KeyError(f"shard già presente: {key}")
            entry = EATEntry(expert_id=expert_id, shard_idx=shard_idx, tier=tier)
            table[key] = entry
            return entry

    def lookup(self, expert_id: ExpertID, shard_idx: ShardID) -> Optional[EATEntry]:
        """Recupera un EATEntry — lookup diretto sul dict (2026-08-12: nessun
        fast-negative path Bloom davanti, vedi docstring di modulo).

        Con locking_strategy="lockfree_read" (Opzione C, issue #23) questo
        salta il lock: si appoggia all'atomicità di dict.get() sotto il
        GIL, scelta deliberata solo per questo metodo — vedi docstring di
        modulo.

        Returns:
            EATEntry se presente, None altrimenti.
        """
        key = (expert_id, shard_idx)
        idx = self._shard_idx(key)
        if self._locking_strategy == "lockfree_read":
            return self._shards[idx].get(key)
        with self._shard_locks[idx]:
            return self._shards[idx].get(key)

    def update_tier(self, expert_id: ExpertID, shard_idx: ShardID, new_tier: Tier) -> None:
        """Aggiorna il tier di uno shard (chiamato dal Tier Manager post-promozione/evizione).

        Thread-safe tramite lock (dello shard competente) + version bump.
        """
        key = (expert_id, shard_idx)
        idx = self._shard_idx(key)
        with self._shard_locks[idx]:
            entry = self._shards[idx].get(key)
            if entry is None:
                raise KeyError(f"shard non presente: {key}")
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
        key = (expert_id, shard_idx)
        idx = self._shard_idx(key)
        with self._shard_locks[idx]:
            return self._shards[idx].pop(key, None)

    def access(self, expert_id: ExpertID, shard_idx: ShardID) -> Optional[EATEntry]:
        """Registra un accesso (touch) e restituisce la entry aggiornata."""
        key = (expert_id, shard_idx)
        idx = self._shard_idx(key)
        with self._shard_locks[idx]:
            entry = self._shards[idx].get(key)
            if entry is None:
                return None
            entry.touch()
            return entry

    # ── bulk ops ───────────────────────────────────────────────────────────────

    def get_tier(self, tier: Tier) -> list[EATEntry]:
        """Restituisce tutte le entry in un dato tier (per Tier Manager)."""
        return self._collect(lambda e: e.tier == tier)

    def eviction_candidates(self, tier: Tier, n: int) -> list[EATEntry]:
        """Top-n candidati all'eviction nel tier dato (SEE score — vedi Tier Manager).

        Fallback LRU se SEE non disponibile.
        """
        candidates = self._collect(lambda e: e.tier == tier)
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
        candidates = self._collect(lambda e: e.tier == tier)
        candidates.sort(key=lambda e: (e.access_count, e.last_access_ts), reverse=True)
        return candidates[:n]

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Inizializza Slab Allocator e strutture interne."""
        self._slab.initialize()

    def shutdown(self) -> None:
        """Shutdown graceful — rilascia Slab Allocator."""
        self._slab.shutdown()
        for table, lock in zip(self._shards, self._shard_locks):
            with lock:
                table.clear()

    # ── stats / iteration ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        total = 0
        for table, lock in zip(self._shards, self._shard_locks):
            with lock:
                total += len(table)
        return total

    def __iter__(self) -> Iterator[EATEntry]:
        return iter(self._collect(lambda e: True))

    def stats(self) -> dict:
        """Metriche per Prometheus / Grafana."""
        entries = self._collect(lambda e: True)
        by_tier: Dict[str, int] = {}
        for entry in entries:
            by_tier[entry.tier.name] = by_tier.get(entry.tier.name, 0) + 1
        return {
            "total_entries": len(entries),
            "by_tier": by_tier,
        }
