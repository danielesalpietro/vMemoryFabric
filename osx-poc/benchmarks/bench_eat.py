"""Benchmark M1 — EAT: latenza/throughput, baseline, contention, scalabilita slab.

Usage:
    python benchmarks/bench_eat.py
    make bench-eat          (via Makefile)

Storia (issue #1, deciso 2026-08-12): questo benchmark è quello che ha
prodotto i numeri dietro la decisione di rimuovere il Bloom filter da
EAT — misurato consistentemente più lento (~5-14x, poi ri-misurato a
~6.8-8.1x con l'implementazione Counting BF di issue #4) di un lookup
diretto sul dict, per una struttura già O(1) in-memory. `EAT.lookup()`
ora fa lookup diretto (vedi src/eat/eat.py), quindi le due sezioni sotto
dovrebbero convergere sullo stesso ordine di grandezza — tenute
entrambe come regression check: se `eat` tornasse a divergere
sensibilmente da `baseline_plain_dict` senza una modifica intenzionale a
EAT, sarebbe un segnale da investigare, non atteso.

Sezioni:
    eat                  — EAT (dict + RLock, nessun layer intermedio),
                            lookup hit/miss separati
    baseline_plain_dict  — stesso workload, dict+RLock "nudo" (nessuna
                            EATEntry, nessun bookkeeping oltre il dict
                            stesso) — isola qualunque overhead residuo di
                            EAT rispetto a un dict grezzo
    eat_vs_baseline_delta_us — differenza p50 (EAT − baseline), hit e miss —
                            atteso vicino a zero ora, non più un vantaggio
                            "Bloom" da giustificare
    contention           — 4 reader concorrenti + 1 writer, per misurare il
                            costo del lock sotto traffico misto stile M2/M3
                            (issue #2 — vedi LOGBOOK.md 2026-08-12 per la nota
                            sul fatto che il traffico reale oggi è single-thread).
                            Usa locking_strategy="single" (default EAT) —
                            baseline di regressione invariata, issue #23.
    contention_by_strategy — stesso scenario di bench_contention(), ripetuto
                            per le tre locking_strategy introdotte in issue
                            #23 (single/striped/lockfree_read), a parità di
                            n_readers/n_prefill/n_writes — per decidere quale
                            opzione (A/B/C) attacca davvero la tail latency
                            misurata sopra, invece di scegliere a intuito.
    slab_scale           — alloc/free timing a 4 vs 32 slot (1 GB vs 8 GB),
                            per verificare empiricamente l'O(1) del free-list

Output: JSON con tutte le sezioni.
"""
import json
import random
import threading
import time

from eat import ExpertAccessTable, Tier
from eat.slab import SlabAllocator
from eat.types import SHARD_SIZE_BYTES

N_ENTRIES = 10_000
N_LOOKUPS = 50_000
HIT_RATIO = 0.8  # frazione di lookup su chiavi presenti


def _percentiles(latencies_us: list) -> dict:
    latencies_us = sorted(latencies_us)
    n = len(latencies_us)
    if n == 0:
        return {"p50_us": None, "p95_us": None, "p99_us": None}
    return {
        "p50_us": latencies_us[int(n * 0.50)],
        "p95_us": latencies_us[min(int(n * 0.95), n - 1)],
        "p99_us": latencies_us[min(int(n * 0.99), n - 1)],
    }


def _gen_workload(seed: int = 0) -> list:
    """(expert_id, shard_idx, is_hit) — stesso seed per EAT e baseline, confronto equo."""
    rng = random.Random(seed)
    n_hits = int(N_LOOKUPS * HIT_RATIO)
    keys = (
        [(0, rng.randrange(N_ENTRIES), True) for _ in range(n_hits)]
        + [(0, rng.randrange(N_ENTRIES, N_ENTRIES * 2), False) for _ in range(N_LOOKUPS - n_hits)]
    )
    rng.shuffle(keys)
    return keys


# ── EAT ────────────────────────────────────────────────────────────────────

def bench_eat() -> dict:
    eat = ExpertAccessTable(capacity=N_ENTRIES * 2, n_slots=4)

    start = time.perf_counter()
    for shard_idx in range(N_ENTRIES):
        eat.insert(expert_id=0, shard_idx=shard_idx, tier=Tier.NVME)
    insert_throughput = N_ENTRIES / (time.perf_counter() - start)

    hit_latencies, miss_latencies = [], []
    for expert_id, shard_idx, is_hit in _gen_workload(seed=0):
        t0 = time.perf_counter()
        eat.lookup(expert_id=expert_id, shard_idx=shard_idx)
        dt_us = (time.perf_counter() - t0) * 1e6
        (hit_latencies if is_hit else miss_latencies).append(dt_us)

    return {
        "insert_throughput_ops_sec": insert_throughput,
        "lookup_latency_hit": _percentiles(hit_latencies),
        "lookup_latency_miss": _percentiles(miss_latencies),
        "lookup_latency_overall": _percentiles(hit_latencies + miss_latencies),
    }


# ── Baseline: dict + RLock "nudo" ────────────────────────────────────────────

class _PlainDictBaseline:
    """Baseline di misura per il benchmark — stessa semantica insert/lookup
    di un dict.get() sotto lock, senza EATEntry né alcun bookkeeping
    aggiuntivo. Non fa parte dell'API pubblica dell'EAT: isola, per
    differenza, qualunque overhead residuo di EAT rispetto a un dict
    grezzo (atteso vicino a zero dal 2026-08-12 — vedi docstring di modulo).
    """

    def __init__(self) -> None:
        self._table: dict = {}
        self._lock = threading.RLock()

    def insert(self, expert_id: int, shard_idx: int) -> None:
        with self._lock:
            self._table[(expert_id, shard_idx)] = True

    def lookup(self, expert_id: int, shard_idx: int):
        with self._lock:
            return self._table.get((expert_id, shard_idx))


def bench_baseline() -> dict:
    baseline = _PlainDictBaseline()

    start = time.perf_counter()
    for shard_idx in range(N_ENTRIES):
        baseline.insert(expert_id=0, shard_idx=shard_idx)
    insert_throughput = N_ENTRIES / (time.perf_counter() - start)

    hit_latencies, miss_latencies = [], []
    for expert_id, shard_idx, is_hit in _gen_workload(seed=0):
        t0 = time.perf_counter()
        baseline.lookup(expert_id=expert_id, shard_idx=shard_idx)
        dt_us = (time.perf_counter() - t0) * 1e6
        (hit_latencies if is_hit else miss_latencies).append(dt_us)

    return {
        "insert_throughput_ops_sec": insert_throughput,
        "lookup_latency_hit": _percentiles(hit_latencies),
        "lookup_latency_miss": _percentiles(miss_latencies),
        "lookup_latency_overall": _percentiles(hit_latencies + miss_latencies),
    }


# ── Contention: 4 reader concorrenti + 1 writer ─────────────────────────────

def bench_contention(n_readers: int = 4, n_prefill: int = 5_000, n_writes: int = 20_000,
                      locking_strategy: str = "single") -> dict:
    eat = ExpertAccessTable(capacity=(n_prefill + n_writes) * 2, n_slots=4,
                             locking_strategy=locking_strategy)
    for shard_idx in range(n_prefill):
        eat.insert(expert_id=0, shard_idx=shard_idx, tier=Tier.NVME)

    stop = threading.Event()
    reader_latencies = [[] for _ in range(n_readers)]

    def writer() -> None:
        for shard_idx in range(n_prefill, n_prefill + n_writes):
            eat.insert(expert_id=0, shard_idx=shard_idx, tier=Tier.NVME)
        stop.set()

    def reader(idx: int) -> None:
        rng = random.Random(1000 + idx)
        latencies = reader_latencies[idx]
        while not stop.is_set():
            shard_idx = rng.randrange(n_prefill)
            t0 = time.perf_counter()
            eat.lookup(expert_id=0, shard_idx=shard_idx)
            latencies.append((time.perf_counter() - t0) * 1e6)

    threads = [threading.Thread(target=writer)] + [
        threading.Thread(target=reader, args=(i,)) for i in range(n_readers)
    ]
    writer_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    writer_elapsed = time.perf_counter() - writer_start

    all_reader_latencies = [lat for bucket in reader_latencies for lat in bucket]
    return {
        "locking_strategy": locking_strategy,
        "n_readers": n_readers,
        "n_writes": n_writes,
        "writer_throughput_ops_sec_under_contention": n_writes / writer_elapsed,
        "reader_lookups_completed": len(all_reader_latencies),
        "reader_lookup_latency_under_contention": _percentiles(all_reader_latencies),
    }


def bench_contention_by_strategy() -> dict:
    """Confronta le tre locking_strategy (issue #23: A=single, B=striped,
    C=lockfree_read) sullo stesso scenario di contesa 4 reader + 1 writer —
    unica variabile è la strategia, tutto il resto invariato rispetto a
    bench_contention()."""
    return {
        strategy: bench_contention(locking_strategy=strategy)
        for strategy in ("single", "striped", "lockfree_read")
    }


# ── Slab allocator — scalabilita del free-list ──────────────────────────────

def bench_slab_scale(slot_counts: tuple = (4, 32)) -> dict:
    results = {}
    for n_slots in slot_counts:
        slab = SlabAllocator(n_slots=n_slots)  # shard_size di default = SHARD_SIZE_BYTES (256 MB)
        slab.initialize()

        alloc_latencies = []
        slot_indices = []
        for i in range(n_slots):
            t0 = time.perf_counter()
            slot_idx = slab.alloc(expert_id=0, shard_idx=i, size_bytes=SHARD_SIZE_BYTES)
            alloc_latencies.append((time.perf_counter() - t0) * 1e6)
            slot_indices.append(slot_idx)

        free_latencies = []
        for slot_idx in slot_indices:
            t0 = time.perf_counter()
            slab.free(slot_idx)
            free_latencies.append((time.perf_counter() - t0) * 1e6)

        results[f"n_slots_{n_slots}"] = {
            "pool_size_gb": round(n_slots * SHARD_SIZE_BYTES / (1024 ** 3), 3),
            "alloc_latency_us": _percentiles(alloc_latencies),
            "free_latency_us": _percentiles(free_latencies),
        }
    return results


def main() -> None:
    eat_result = bench_eat()
    baseline_result = bench_baseline()

    delta_us = {
        "hit_p50": eat_result["lookup_latency_hit"]["p50_us"] - baseline_result["lookup_latency_hit"]["p50_us"],
        "miss_p50": eat_result["lookup_latency_miss"]["p50_us"] - baseline_result["lookup_latency_miss"]["p50_us"],
    }

    result = {
        "status": "done",
        "sprint": "1, revisited 2026-08-12 (issue #1 — Bloom filter removed)",
        "module": "EAT",
        "n_entries": N_ENTRIES,
        "n_lookups": N_LOOKUPS,
        "hit_ratio": HIT_RATIO,
        "eat": eat_result,
        "baseline_plain_dict": baseline_result,
        "eat_vs_baseline_delta_us": delta_us,
        "contention": bench_contention(),
        "contention_by_strategy": bench_contention_by_strategy(),
        "slab_scale": bench_slab_scale(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
