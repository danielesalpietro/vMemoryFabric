"""M1 — Bloom Filter 2-livelli per EAT.

Livello 1: expert-level  (expert_id presente nell'EAT?)
Livello 2: shard-level   (shard (expert_id, shard_idx) presente nell'EAT?)

False positive rate target: 1% per livello.
Implementazione: Counting Bloom Filter custom (numpy/array), non più
pybloom_live — vedi issue #4 sotto per il perché.

Sostituito 2026-08-12 (Sprint 4, issue #4/#17, sotto-obiettivo 5):
pybloom_live.BloomFilter non supporta cancellazione per costruzione (un
Bloom filter classico, bit singoli, non può "dimenticare" un elemento
senza rischiare di dimenticarne anche altri che condividono un bit).
remove_expert() esisteva solo come stub (`raise NotImplementedError`) da
Sprint 1 — MAI chiamato da nessun altro punto del codice (verificato via
grep prima di questa modifica, non assunto), quindi gli shard evicted
restavano falsi positivi permanenti nel Bloom filter per l'intera vita
del processo, esattamente il bug descritto in issue #4. Fix: un Counting
Bloom Filter (contatori invece di bit singoli — add() incrementa,
remove() decrementa, "presente" se il contatore è > 0) supporta la
cancellazione reale con la stessa matematica di dimensionamento
(m, k) di un Bloom filter classico — il TODO già lasciato nel docstring
di modulo da Sprint 1 ("sostituire con implementazione custom numpy se
pybloom non raggiunge il target di latenza") anticipava esattamente
questo tipo di sostituzione, per un motivo diverso (latenza, issue #1)
ma con la stessa soluzione.

Lookup O(1) in ~80 ns target (DDR4 resident).
"""
from __future__ import annotations
import hashlib
import math
from array import array


class _CountingBloomFilter:
    """Counting Bloom Filter: contatori a 8 bit invece di bit singoli.

    Stessa matematica di dimensionamento di un Bloom filter classico
    (m = -n·ln(p)/ln(2)², k = (m/n)·ln(2) — vedi Wikipedia "Bloom filter"
    per la derivazione standard, non reinventata qui): il tasso di falsi
    positivi dipende da quanti slot sono "attivi" (counter > 0), non dal
    loro valore, quindi la stessa formula si applica identica.

    Hashing: doppio hash (tecnica di Kirsch-Mitzenmacher,
    h_i(x) = h1(x) + i·h2(x) mod m) da due digest blake2b indipendenti —
    evita una dipendenza esterna aggiuntiva (mmh3 o simili), hashlib è
    nella stdlib.

    Contatori saturati a 255, non wraparound: un add() oltre 255 sullo
    stesso slot non incrementa oltre; irraggiungibile per il workload di
    questo progetto (~256 expert × 64 shard) ma dichiarato per onestà,
    non assunto innocuo senza dirlo.
    """

    _COUNTER_MAX = 255

    def __init__(self, capacity: int, error_rate: float = 0.01) -> None:
        n = max(capacity, 1)
        m = max(1, int(-(n * math.log(error_rate)) / (math.log(2) ** 2)))
        k = max(1, round((m / n) * math.log(2)))
        self._m = m
        self._k = k
        self._counters = array("B", bytes(m))

    def _indices(self, item: str):
        h1 = int.from_bytes(hashlib.blake2b(item.encode(), digest_size=8).digest(), "big")
        h2 = int.from_bytes(
            hashlib.blake2b(item.encode(), digest_size=8, person=b"osx-bf-h2").digest(), "big",
        )
        for i in range(self._k):
            yield (h1 + i * h2) % self._m

    def add(self, item: str) -> None:
        for idx in self._indices(item):
            if self._counters[idx] < self._COUNTER_MAX:
                self._counters[idx] += 1

    def remove(self, item: str) -> None:
        """Decrementa i contatori di item — sicuro anche se non erano mai
        stati aggiunti (nessuna eccezione, i contatori non scendono sotto
        0): il chiamante (BloomFilter.remove_shard/remove_expert) è
        responsabile di non chiamarlo per elementi mai aggiunti, ma un
        errore lì degrada a no-op silenzioso, non corruzione dello stato
        di altri elementi."""
        for idx in self._indices(item):
            if self._counters[idx] > 0:
                self._counters[idx] -= 1

    def __contains__(self, item: str) -> bool:
        return all(self._counters[idx] > 0 for idx in self._indices(item))


class BloomFilter:
    """Bloom filter 2-livelli wrappato per EAT — vedi docstring di modulo
    per il passaggio a Counting Bloom Filter (2026-08-12, issue #4)."""

    def __init__(self, capacity: int = 16_384, error_rate: float = 0.01) -> None:
        """
        Args:
            capacity:   Numero massimo di elementi attesi (default: 256 expert × 64 shard).
            error_rate: False positive rate target per livello.
        """
        self._expert_bf = _CountingBloomFilter(capacity=capacity, error_rate=error_rate)
        self._shard_bf = _CountingBloomFilter(capacity=capacity, error_rate=error_rate)
        self._shard_count = 0   # per __len__ — un Counting BF non può derivarlo esattamente dai contatori

    # ── write ──────────────────────────────────────────────────────────────────

    def add(self, expert_id: int, shard_idx: int) -> None:
        """Registra (expert_id, shard_idx) in entrambi i livelli."""
        self._expert_bf.add(f"e:{expert_id}")
        self._shard_bf.add(f"s:{expert_id}:{shard_idx}")
        self._shard_count += 1

    def remove_shard(self, expert_id: int, shard_idx: int) -> None:
        """Rimuove SOLO la entry a livello shard per (expert_id, shard_idx)
        (2026-08-12, issue #4). Il livello expert non viene toccato qui
        di proposito: un expert può avere altri shard ancora presenti —
        vedi remove_expert() per quando invece va rimosso anche quello.
        Chiamato da EAT.evict().
        """
        self._shard_bf.remove(f"s:{expert_id}:{shard_idx}")
        self._shard_count = max(0, self._shard_count - 1)

    def remove_expert(self, expert_id: int) -> None:
        """Rimuove expert_id dal livello expert (2026-08-12, issue #4 —
        prima uno stub `raise NotImplementedError`, mai chiamato da
        nessun punto del codice, verificato via grep prima di questa
        modifica).

        Va chiamato SOLO quando il chiamante ha già verificato che non
        restano altri shard di quell'expert nell'EAT — altrimenti
        may_contain_expert() darebbe falsi negativi per shard ancora
        validi (il livello expert è condiviso da tutti gli shard dello
        stesso expert_id). EAT.evict() fa esattamente questo check prima
        di chiamare questo metodo.
        """
        self._expert_bf.remove(f"e:{expert_id}")

    # ── read ───────────────────────────────────────────────────────────────────

    def may_contain_expert(self, expert_id: int) -> bool:
        """True se expert_id è *probabilmente* presente (falsi positivi possibili)."""
        return f"e:{expert_id}" in self._expert_bf

    def may_contain_shard(self, expert_id: int, shard_idx: int) -> bool:
        """True se (expert_id, shard_idx) è *probabilmente* presente."""
        return f"s:{expert_id}:{shard_idx}" in self._shard_bf

    # ── stats ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Numero di elementi nel shard-level BF — contatore esplicito
        (add()/remove_shard() lo mantengono), non derivato dai contatori
        del Counting BF: con overlap tra gli slot di elementi diversi non
        c'è modo di ricavare un conteggio esatto dai soli contatori."""
        return self._shard_count
