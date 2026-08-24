"""Benchmark M2 — EMH-2 (PMEM): banda grezza + latenza promozione.

Issue #33/#45 follow-up — "passo 4" (osx-poc/LOGBOOK_NEW_Z8.md). Richiede
un mount DAX reale — `PMEMTransfer`/`TierManager(pmem_path=...)` restano
disponibili ovunque (nessun import condizionale), ma questo benchmark
degrada esplicitamente a "skipped" se `PMEM_PATH` non è una directory
scrivibile, invece di far fallire l'intero script — stesso pattern di
`bench_ddr4_to_vram()` in bench_tier.py per `torch.cuda.is_available()`.
Non gira su Docker-on-Windows/WSL2/RunPod (nessun mount DAX lì), solo su
un host con una region PMEM in modalità fsdax montata (vedi tier/pmem.py).

PMEM_PATH di default punta al bind-mount del container (/data/pmem via
docker-compose.override.yml, gitignored — copiare da
docker-compose.override.yml.example e adattare il path host).

Sezioni:
    raw_bandwidth  — scrittura/lettura sequenziale su un singolo slot
                     grande (default 512 MB, stessa dimensione di
                     perf_test_hardware.py bench_ram(), per
                     confrontabilità diretta con la banda DDR4-DDR4 già
                     misurata su questo host), GB/s.
    nvme_to_pmem   — latenza promote(NVME->PMEM) per N shard, P50/P95/P99.
    pmem_to_ddr4   — latenza promote(PMEM->DDR4) per N shard (riusa gli
                     shard promossi da nvme_to_pmem, non li riseeda).

Deviazione benchmark-only dichiarata esplicitamente, stessa di
bench_tier.py: nvme_to_pmem/pmem_to_ddr4 usano shard sintetici da
BENCH_SHARD_SIZE_BYTES (4 MB), non i 256 MB di produzione — l'ordine di
grandezza relativo conta qui, non il valore assoluto. raw_bandwidth
invece usa apposta un buffer grande per dare un numero di banda
direttamente interpretabile (non un proxy).

A differenza di bench_ddr4_to_vram() (issue #48: VRAM è capacity-bound a
24 GB, serviva evict() dopo ogni promote misurata), qui non c'è eviction
nel loop: PMEM (252 GB) e DDR4 (~236 GB su questo host) hanno entrambi
ampio margine per tenere tutti gli N_SHARDS residenti insieme
(N_SHARDS=100 x 256 MB di slot fisso = 25.6 GB su ciascun tier) — non
serve il pattern "libera dopo ogni misura" per restare entro capacità.

Nota: questo benchmark lascia un file pool persistente
(`<PMEM_PATH>/emh2_pool.bin`, ~25.6 GB) sul mount DAX tra una run e
l'altra — deliberato (evita di ri-fallocare blocchi ad ogni run se la
dimensione non cambia, vedi PMEMTransfer.initialize()), rimuovibile a
mano se serve liberare spazio.

Usage:
    python benchmarks/bench_pmem_tier.py
    make bench-pmem-tier   (via Makefile)
"""
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from eat import ExpertAccessTable, Tier
from tier import PMEMTransfer, TierManager

N_SHARDS = 100
BENCH_SHARD_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB — stessa convenzione di bench_tier.py

PMEM_PATH = os.environ.get("OSX_PMEM_PATH", "/data/pmem")
RAW_BANDWIDTH_TEST_SIZE_MB = 512  # stessa dimensione di perf_test_hardware.py bench_ram()


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


def _pmem_available() -> bool:
    return Path(PMEM_PATH).is_dir() and os.access(PMEM_PATH, os.W_OK)


def _skip_reason() -> dict:
    return {"status": "skipped", "reason": f"{PMEM_PATH} non montato/scrivibile su questo host"}


# ── raw bandwidth ──────────────────────────────────────────────────────────────

def bench_raw_bandwidth() -> dict:
    """Scrittura poi lettura sequenziale su un singolo slot da
    RAW_BANDWIDTH_TEST_SIZE_MB, stessa metodologia di perf_test_hardware.py
    bench_ram() (np.full per il payload, un solo giro cronometrato — non
    una media su ripetizioni, coerente con com'è già bench_ram())."""
    if not _pmem_available():
        return _skip_reason()

    size_bytes = RAW_BANDWIDTH_TEST_SIZE_MB * 1024 * 1024
    pt = PMEMTransfer(
        mount_path=PMEM_PATH, n_slots=1, shard_size=size_bytes,
        pool_filename="bench_raw_bandwidth.bin",
    )
    pt.initialize()
    try:
        slot = pt.alloc(expert_id=-1, shard_idx=-1, size_bytes=size_bytes)
        payload = np.full(size_bytes, 0xAB, dtype=np.uint8)

        t0 = time.perf_counter()
        pt.write(slot, payload)
        pt.flush()
        write_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = np.array(pt.read(slot))  # forza una copia reale, non solo la view mmap
        read_s = time.perf_counter() - t0
    finally:
        pt.shutdown()
        (Path(PMEM_PATH) / "bench_raw_bandwidth.bin").unlink(missing_ok=True)

    return {
        "status": "done",
        "test_size_mb": RAW_BANDWIDTH_TEST_SIZE_MB,
        "write_gbps": (size_bytes / write_s) / 1e9,
        "read_gbps": (size_bytes / read_s) / 1e9,
    }


# ── nvme_to_pmem / pmem_to_ddr4 ──────────────────────────────────────────────

def _seed_nvme_shards(nvme_path: Path, eat: ExpertAccessTable, shard_indices) -> None:
    payload = np.full(BENCH_SHARD_SIZE_BYTES, 0xAB, dtype=np.uint8).tobytes()
    for shard_idx in shard_indices:
        eat.insert(expert_id=0, shard_idx=shard_idx, tier=Tier.NVME)
        path = nvme_path / "0" / f"{shard_idx}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


async def bench_nvme_pmem_ddr4() -> dict:
    """nvme_to_pmem + pmem_to_ddr4 nello stesso EAT/TierManager —
    pmem_to_ddr4 riusa gli shard promossi da nvme_to_pmem invece di
    riseedarli, misurando il percorso reale a due hop EMH-3->EMH-2->EMH-1c
    (non incatenato in un solo promote() — vedi docstring di
    TierManager.promote() per il perché di due chiamate separate)."""
    if not _pmem_available():
        skip = _skip_reason()
        return {"nvme_to_pmem": skip, "pmem_to_ddr4": skip}

    with tempfile.TemporaryDirectory() as tmp:
        nvme_path = Path(tmp)
        eat = ExpertAccessTable(capacity=(N_SHARDS + 1) * 2, n_slots=N_SHARDS + 1)
        eat.initialize()
        _seed_nvme_shards(nvme_path, eat, range(N_SHARDS))
        mgr = TierManager(
            eat=eat, nvme_path=str(nvme_path), gpu_device=0,
            pmem_path=PMEM_PATH, pmem_n_slots=N_SHARDS + 1,
        )

        nvme_to_pmem_latencies = []
        for shard_idx in range(N_SHARDS):
            t0 = time.perf_counter()
            await mgr.promote(expert_id=0, shard_idx=shard_idx, target_tier=Tier.PMEM)
            nvme_to_pmem_latencies.append((time.perf_counter() - t0) * 1e6)

        pmem_to_ddr4_latencies = []
        for shard_idx in range(N_SHARDS):
            t0 = time.perf_counter()
            await mgr.promote(expert_id=0, shard_idx=shard_idx, target_tier=Tier.DDR4)
            pmem_to_ddr4_latencies.append((time.perf_counter() - t0) * 1e6)

        mgr.pmem.shutdown()
        eat.shutdown()

    return {
        "nvme_to_pmem": {
            "n_shards": N_SHARDS,
            "shard_size_bytes": BENCH_SHARD_SIZE_BYTES,
            "latency_us": _percentiles(nvme_to_pmem_latencies),
        },
        "pmem_to_ddr4": {
            "n_shards": N_SHARDS,
            "shard_size_bytes": BENCH_SHARD_SIZE_BYTES,
            "latency_us": _percentiles(pmem_to_ddr4_latencies),
        },
    }


async def _main_async() -> dict:
    result = {
        "status": "done",
        "issue": "#33/#45 — EMH-2 PMEM tier (passo 4)",
        "pmem_path": PMEM_PATH,
        "note": (
            "sezioni nvme_to_pmem/pmem_to_ddr4 usano shard sintetici da "
            f"{BENCH_SHARD_SIZE_BYTES} byte (stessa convenzione di "
            "bench_tier.py); raw_bandwidth usa un buffer da "
            f"{RAW_BANDWIDTH_TEST_SIZE_MB} MB per un numero di banda "
            "direttamente interpretabile, non un proxy"
        ),
        "raw_bandwidth": bench_raw_bandwidth(),
    }
    result.update(await bench_nvme_pmem_ddr4())
    return result


def main() -> None:
    result = asyncio.run(_main_async())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
