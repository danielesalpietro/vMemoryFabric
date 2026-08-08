"""Benchmark M1 — EAT latenza e throughput.

Usage:
    python benchmarks/bench_eat.py
    make bench-eat          (via Makefile)

Output: JSON con P50/P95/P99 latenza lookup e throughput insert.
"""
import json
import random
import time

from eat import ExpertAccessTable, Tier

N_ENTRIES = 10_000
N_LOOKUPS = 50_000
HIT_RATIO = 0.8  # frazione di lookup su chiavi presenti


def bench_insert(eat: ExpertAccessTable) -> float:
    """Ritorna throughput insert (ops/sec)."""
    start = time.perf_counter()
    for shard_idx in range(N_ENTRIES):
        eat.insert(expert_id=0, shard_idx=shard_idx, tier=Tier.NVME)
    elapsed = time.perf_counter() - start
    return N_ENTRIES / elapsed


def bench_lookup(eat: ExpertAccessTable) -> dict:
    """Ritorna P50/P95/P99 latenza lookup (µs), mix hit/miss."""
    rng = random.Random(0)
    n_hits = int(N_LOOKUPS * HIT_RATIO)
    keys = (
        [(0, rng.randrange(N_ENTRIES)) for _ in range(n_hits)]
        + [(0, rng.randrange(N_ENTRIES, N_ENTRIES * 2)) for _ in range(N_LOOKUPS - n_hits)]
    )
    rng.shuffle(keys)

    latencies_us = []
    for expert_id, shard_idx in keys:
        start = time.perf_counter()
        eat.lookup(expert_id=expert_id, shard_idx=shard_idx)
        latencies_us.append((time.perf_counter() - start) * 1e6)

    latencies_us.sort()
    n = len(latencies_us)
    return {
        "p50_us": latencies_us[int(n * 0.50)],
        "p95_us": latencies_us[int(n * 0.95)],
        "p99_us": latencies_us[int(n * 0.99)],
    }


def main() -> None:
    eat = ExpertAccessTable(capacity=N_ENTRIES * 2, n_slots=4)

    insert_throughput = bench_insert(eat)
    lookup_latencies = bench_lookup(eat)

    result = {
        "status": "done",
        "sprint": 1,
        "module": "EAT",
        "n_entries": N_ENTRIES,
        "n_lookups": N_LOOKUPS,
        "hit_ratio": HIT_RATIO,
        "insert_throughput_ops_sec": insert_throughput,
        "lookup_latency": lookup_latencies,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
