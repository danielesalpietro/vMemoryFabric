"""Benchmark M2 — EMH Tier Manager: latenza promozione per tier.

Sezioni:
    nvme_to_ddr4  — latenza promote(NVME→DDR4) per N shard, P50/P95/P99.
                    Gira ovunque torch sia importabile (nessuna CUDA reale
                    richiesta per questo hop).
    ddr4_to_vram  — latenza promote(DDR4→VRAM). Guardia esplicita su
                    torch.cuda.is_available(): se assente, la sezione
                    riporta {"status": "skipped", ...} invece di far
                    fallire l'intero benchmark o inventare numeri da un
                    path che non ha davvero toccato la GPU.

Deviazione benchmark-only, dichiarata esplicitamente: usa shard sintetici
più piccoli dei 256 MB (SHARD_SIZE_BYTES) di produzione, per tenere il
run in tempi ragionevoli. La latenza assoluta NVMe/PCIe non è quindi
comparabile 1:1 col target di produzione — l'ordine di grandezza
relativo tra i due hop sì.

Usage:
    python benchmarks/bench_tier.py
    make bench-tier          (via Makefile)
"""
import asyncio
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from eat import ExpertAccessTable, Tier
from tier import TierManager

N_SHARDS = 20
BENCH_SHARD_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB — vedi nota sopra


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


def _seed_nvme_shards(nvme_path: Path, eat: ExpertAccessTable) -> None:
    payload = np.full(BENCH_SHARD_SIZE_BYTES, 0xAB, dtype=np.uint8).tobytes()
    for shard_idx in range(N_SHARDS):
        eat.insert(expert_id=0, shard_idx=shard_idx, tier=Tier.NVME)
        path = nvme_path / "0" / f"{shard_idx}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


# ── NVMe → DDR4 ──────────────────────────────────────────────────────────────

async def bench_nvme_to_ddr4() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        nvme_path = Path(tmp)
        eat = ExpertAccessTable(capacity=N_SHARDS * 2, n_slots=N_SHARDS)
        eat.initialize()
        _seed_nvme_shards(nvme_path, eat)
        mgr = TierManager(eat=eat, nvme_path=str(nvme_path), gpu_device=0)

        latencies_us = []
        for shard_idx in range(N_SHARDS):
            t0 = time.perf_counter()
            await mgr.promote(expert_id=0, shard_idx=shard_idx, target_tier=Tier.DDR4)
            latencies_us.append((time.perf_counter() - t0) * 1e6)

        eat.shutdown()
        return {
            "n_shards": N_SHARDS,
            "shard_size_bytes": BENCH_SHARD_SIZE_BYTES,
            "latency_us": _percentiles(latencies_us),
        }


# ── DDR4 → VRAM ──────────────────────────────────────────────────────────────

async def bench_ddr4_to_vram() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA non disponibile su questo host"}

    with tempfile.TemporaryDirectory() as tmp:
        nvme_path = Path(tmp)
        eat = ExpertAccessTable(capacity=N_SHARDS * 2, n_slots=N_SHARDS)
        eat.initialize()
        _seed_nvme_shards(nvme_path, eat)
        mgr = TierManager(eat=eat, nvme_path=str(nvme_path), gpu_device=0)

        for shard_idx in range(N_SHARDS):
            await mgr.promote(expert_id=0, shard_idx=shard_idx, target_tier=Tier.DDR4)

        latencies_us = []
        for shard_idx in range(N_SHARDS):
            t0 = time.perf_counter()
            await mgr.promote(expert_id=0, shard_idx=shard_idx, target_tier=Tier.VRAM)
            latencies_us.append((time.perf_counter() - t0) * 1e6)

        eat.shutdown()
        return {
            "n_shards": N_SHARDS,
            "shard_size_bytes": BENCH_SHARD_SIZE_BYTES,
            "latency_us": _percentiles(latencies_us),
        }


async def _main_async() -> dict:
    return {
        "status": "done",
        "sprint": 2,
        "module": "TierManager",
        "note": (
            f"shard sintetici da {BENCH_SHARD_SIZE_BYTES} byte, non i "
            "256 MB di produzione — vedi docstring del modulo"
        ),
        "nvme_to_ddr4": await bench_nvme_to_ddr4(),
        "ddr4_to_vram": await bench_ddr4_to_vram(),
    }


def main() -> None:
    result = asyncio.run(_main_async())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
