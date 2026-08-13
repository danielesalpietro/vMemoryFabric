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
        """Shutdown graceful — rilascia Slab Allocator.

        NOTE: cancella incondizionatamente la tabella. Chi vuole persistere la
        hotness tra riavvii (EPM, issue #27) deve chiamare export_snapshot()
        *prima* di shutdown() — dopo, non c'è più nulla da esportare.
        """
        self._slab.shutdown()
        with self._lock:
            self._table.clear()

    # ── EPM — Expert Position Memory (issue #27) ─────────────────────────────────
    #
    # Checkpoint leggero della hotness, non una policy di scheduling: salva
    # (access_count, recency) a fine run e li ricarica come prior nel run
    # successivo. La posizione fisica nel tier NON sopravvive al riavvio (i
    # tensori VRAM sono stato CUDA del processo morto) — `tier` viaggia nello
    # snapshot solo a scopo informativo/debug, load_snapshot() non lo tocca
    # mai. Il chiamante deve trattare il prior come hint di priorità di
    # prefetch (consumato da GCSGWorker._seed_eat_entries(), vedi
    # scheduler.gcsg), non come ripristino di stato fisico.

    def export_snapshot(self) -> dict:
        """Serializza la hotness corrente in un dict JSON-safe.

        `last_access_ts` è su clock monotonic — non comparabile tra processi
        diversi — quindi viene esportato come `age_seconds` (delta rispetto
        a "ora") anziché come valore assoluto: il prossimo processo può così
        ricostruire l'ordine di recency senza assumere continuità del clock.

        Returns:
            dict con "version", "exported_at" (wall clock, solo informativo)
            e "entries": {"<expert_id>:<shard_idx>": {access_count,
            age_seconds, tier}}.
        """
        now = time.monotonic()
        with self._lock:
            entries = {
                f"{expert_id}:{shard_idx}": {
                    "access_count": entry.access_count,
                    "age_seconds": max(0.0, now - entry.last_access_ts),
                    "tier": entry.tier.name,
                }
                for (expert_id, shard_idx), entry in self._table.items()
            }
        return {"version": 1, "exported_at": time.time(), "entries": entries}

    def load_snapshot(self, snapshot: dict, decay: float = 0.5) -> int:
        """Applica un checkpoint di hotness alle entry già presenti in tabella.

        Non crea entry nuove — richiede che il seeding strutturale (insert())
        sia già avvenuto, tipicamente in un hook come `_seed_eat_entries()` —
        e non tocca mai `tier`: vedi nota EPM sopra sul perché la posizione
        fisica non è ripristinabile. Le chiavi dello snapshot non ancora
        presenti in tabella vengono ignorate silenziosamente.

        Args:
            snapshot: dict prodotto da export_snapshot() (di un run precedente).
            decay:    Fattore in [0, 1] applicato ad access_count, per evitare
                       che un checkpoint vecchio pesi quanto ore di traffico
                       fresco. Va scelto esplicitamente — 1.0 ("nessun
                       decadimento") non è il default, va richiesto a mano.

        Returns:
            Numero di entry effettivamente aggiornate.

        Raises:
            ValueError: decay fuori da [0, 1] o versione snapshot non supportata.
        """
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay {decay} fuori range [0, 1]")
        if snapshot.get("version") != 1:
            raise ValueError(f"snapshot version non supportata: {snapshot.get('version')!r}")

        now = time.monotonic()
        updated = 0
        with self._lock:
            for key, data in snapshot.get("entries", {}).items():
                expert_id_str, shard_idx_str = key.split(":", 1)
                entry = self._table.get((int(expert_id_str), int(shard_idx_str)))
                if entry is None:
                    continue
                entry.access_count = round(data["access_count"] * decay)
                entry.last_access_ts = now - data["age_seconds"]
                entry.version += 1
                updated += 1
        return updated

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
