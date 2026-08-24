"""Benchmark M2 — EMH Tier Manager: latenza promozione per tier.

Sezioni:
    nvme_to_ddr4         — latenza promote(NVME→DDR4) per N shard,
                           P50/P95/P99. Gira ovunque torch sia importabile
                           (nessuna CUDA reale richiesta per questo hop).
    ddr4_to_vram         — latenza promote(DDR4→VRAM), file-based (il
                           path NVMe→DDR4→VRAM "classico" di M2). Guardia
                           esplicita su torch.cuda.is_available(): se
                           assente, la sezione riporta
                           {"status": "skipped", ...} invece di far
                           fallire l'intero benchmark o inventare numeri
                           da un path che non ha davvero toccato la GPU.
    promote_live_tensor  — latenza di TierManager.promote_live_tensor()
                           (2026-08-12, issue #17) — il path REALMENTE
                           usato da GCSGWorker per lo shadow pool, non un
                           file-shard da NVMe ma un tensore CPU già in
                           memoria. Misura separatamente pin=False
                           (comportamento storico, quello esercitato
                           sulla Z8/WSL2) e pin=True (quello verificato
                           sicuro sotto carico sostenuto solo su Linux
                           reale — vedi LOGBOOK.md, soak test e checklist
                           2026-08-12). Sprint 4 sotto-obiettivo 4 —
                           "shard promotion latency within 1.5x
                           theoretical bandwidth" (README, non misurabile
                           finché questo path non esisteva).

Deviazione benchmark-only, dichiarata esplicitamente: usa shard sintetici
più piccoli dei 256 MB (SHARD_SIZE_BYTES) di produzione, per tenere il
run in tempi ragionevoli. La latenza assoluta NVMe/PCIe non è quindi
comparabile 1:1 col target di produzione — l'ordine di grandezza
relativo tra i due hop sì. Stessa deviazione, stesso motivo, per
promote_live_tensor: un vero parametro AWQ dominante può arrivare a
decine di MB (mai misurato con precisione — vedi la checklist
gcsg_tier_manager per il perché), non i 4 MB sintetici qui.

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

N_SHARDS = 100
BENCH_SHARD_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB — vedi nota sopra

# Shard dedicato al warm-up CUDA in bench_ddr4_to_vram() (vedi GitHub issue #3):
# fuori dal range [0, N_SHARDS) misurato, cosi' non si sovrappone a nessuno
# degli shard temporizzati.
_WARMUP_SHARD_IDX = N_SHARDS


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


def _seed_nvme_shards(nvme_path: Path, eat: ExpertAccessTable, shard_indices) -> None:
    payload = np.full(BENCH_SHARD_SIZE_BYTES, 0xAB, dtype=np.uint8).tobytes()
    for shard_idx in shard_indices:
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
        _seed_nvme_shards(nvme_path, eat, range(N_SHARDS))
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
        # +1 slot/capacity per lo shard di warm-up dedicato, oltre a quelli misurati.
        eat = ExpertAccessTable(capacity=(N_SHARDS + 1) * 2, n_slots=N_SHARDS + 1)
        eat.initialize()
        _seed_nvme_shards(nvme_path, eat, [*range(N_SHARDS), _WARMUP_SHARD_IDX])
        mgr = TierManager(eat=eat, nvme_path=str(nvme_path), gpu_device=0)

        # Warm-up: paga il costo one-off di inizializzazione del contesto/
        # allocator CUDA (torch.from_numpy(...).to(device)) su uno shard MAI
        # toccato dal loop cronometrato sotto — issue #3. Senza questo, la
        # prima promote() del loop misurato assorbe quel costo e con pochi
        # campioni p95/p99 collassano sullo stesso outlier. Shard dedicato
        # (non shard 0 riciclato): promote() non supporta una transizione
        # VRAM→VRAM sullo stesso shard.
        await mgr.promote(expert_id=0, shard_idx=_WARMUP_SHARD_IDX, target_tier=Tier.DDR4)
        await mgr.promote(expert_id=0, shard_idx=_WARMUP_SHARD_IDX, target_tier=Tier.VRAM)
        await mgr.evict(expert_id=0, shard_idx=_WARMUP_SHARD_IDX)  # see issue #48

        for shard_idx in range(N_SHARDS):
            await mgr.promote(expert_id=0, shard_idx=shard_idx, target_tier=Tier.DDR4)

        # Each VRAM slot is the fixed SHARD_SIZE_BYTES (256 MB) slab slot, not
        # BENCH_SHARD_SIZE_BYTES (4 MB) — SlabAllocator.get_buffer() always
        # returns the whole slot. With N_SHARDS=100 and nothing evicting the
        # promoted tensor, this loop used to accumulate 100 x 256 MB = 25.6 GB
        # of permanently-resident VRAM, exceeding a single RTX 3090's 24 GB
        # capacity (guaranteed CUDA OOM, not host-specific — see issue #48).
        # evict() after each timed sample keeps steady-state VRAM usage at
        # ~1 shard, closer to how a real tier manager would behave, and runs
        # outside the timed t0/latencies_us window so it doesn't skew the
        # promotion latency being measured.
        latencies_us = []
        for shard_idx in range(N_SHARDS):
            t0 = time.perf_counter()
            await mgr.promote(expert_id=0, shard_idx=shard_idx, target_tier=Tier.VRAM)
            latencies_us.append((time.perf_counter() - t0) * 1e6)
            await mgr.evict(expert_id=0, shard_idx=shard_idx)

        eat.shutdown()
        return {
            "n_shards": N_SHARDS,
            "shard_size_bytes": BENCH_SHARD_SIZE_BYTES,
            "latency_us": _percentiles(latencies_us),
        }


# ── promote_live_tensor (2026-08-12, issue #17, Sprint 4 sotto-obiettivo 4) ──

# PCIe Gen3 ~8 GB/s unidirezionale — stesso numero già in tier/gpu.py
# docstring ("Teorico pinned: ~32 ms" per 256MB, cioè 256MiB/32ms ≈ 8GB/s).
# Base per il criterio di accettazione del README ("within 1.5x
# theoretical bandwidth"), non un nuovo numero inventato qui.
_PCIE_GEN3_BANDWIDTH_BYTES_PER_SEC = 8 * 1024 ** 3


def _theoretical_transfer_s(size_bytes: int) -> float:
    return size_bytes / _PCIE_GEN3_BANDWIDTH_BYTES_PER_SEC


async def bench_promote_live_tensor() -> dict:
    """Latenza di TierManager.promote_live_tensor() — il path REALE usato
    da GCSGWorker per lo shadow pool (non promote(), che è per shard-file
    da NVMe). Misura pin=False e pin=True separatamente: pin=False è
    quanto già gira sulla Z8/WSL2 (dove in_wsl() disattiva pin=True per
    design), pin=True è il ramo verificato sicuro solo su Linux reale
    (soak test + checklist, 2026-08-12) — questo benchmark quantifica per
    la prima volta QUANTO costa in latenza, non solo se è sicuro.

    Il criterio di accettazione del README ("shard promotion latency
    within 1.5x theoretical bandwidth") è calcolato qui contro la
    dimensione sintetica del benchmark (BENCH_SHARD_SIZE_BYTES), non i
    256 MB di produzione — stessa deviazione dichiarata nel docstring di
    modulo, l'ordine di grandezza relativo pin=True vs pin=False è il
    dato che conta di più qui, non il valore assoluto.
    """
    import torch
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA non disponibile su questo host"}

    theoretical_s = _theoretical_transfer_s(BENCH_SHARD_SIZE_BYTES)
    acceptance_threshold_s = theoretical_s * 1.5   # README: "within 1.5x theoretical bandwidth"

    results = {}
    for pin in (False, True):
        eat = ExpertAccessTable(capacity=N_SHARDS * 2, n_slots=N_SHARDS)
        eat.initialize()
        with tempfile.TemporaryDirectory() as tmp:
            # nvme_path non è mai letto/scritto da promote_live_tensor()
            # (bypassa AsyncNVMeIO per costruzione — vedi la sua
            # docstring in tier/manager.py) — una dir temporanea vuota
            # basta solo perché TierManager.__init__ la richiede comunque.
            mgr = TierManager(eat=eat, nvme_path=tmp, gpu_device=0)

            # Payload pre-allocati FUORI dal loop cronometrato — misura
            # solo il costo di promote_live_tensor() (pin + transfer),
            # non l'allocazione del tensore CPU sorgente. Stesso principio
            # di bench_ddr4_to_vram() sopra, che pre-popola DDR4 prima di
            # cronometrare solo l'hop DDR4->VRAM.
            payloads = [
                torch.full((BENCH_SHARD_SIZE_BYTES,), 0xAB, dtype=torch.uint8)
                for _ in range(N_SHARDS)
            ]

            latencies_us = []
            for shard_idx, data in enumerate(payloads):
                t0 = time.perf_counter()
                await mgr.promote_live_tensor(
                    expert_id=0, shard_idx=shard_idx, cpu_data=data, pin=pin,
                )
                latencies_us.append((time.perf_counter() - t0) * 1e6)

        eat.shutdown()
        pct = _percentiles(latencies_us)
        p50_s = (pct["p50_us"] or 0) / 1e6
        results[f"pin_{pin}"] = {
            "n_shards": N_SHARDS,
            "shard_size_bytes": BENCH_SHARD_SIZE_BYTES,
            "latency_us": pct,
            "within_1.5x_theoretical_bandwidth_at_p50": p50_s <= acceptance_threshold_s,
        }

    return {
        "theoretical_transfer_s_at_bench_size": theoretical_s,
        "acceptance_threshold_s_1.5x": acceptance_threshold_s,
        **results,
    }


async def _main_async() -> dict:
    return {
        "status": "done",
        "sprint": "2 (nvme_to_ddr4/ddr4_to_vram) + 4 (promote_live_tensor, issue #17)",
        "module": "TierManager",
        "note": (
            f"shard sintetici da {BENCH_SHARD_SIZE_BYTES} byte, non i "
            "256 MB di produzione — vedi docstring del modulo"
        ),
        "nvme_to_ddr4": await bench_nvme_to_ddr4(),
        "ddr4_to_vram": await bench_ddr4_to_vram(),
        "promote_live_tensor": await bench_promote_live_tensor(),
    }


def main() -> None:
    result = asyncio.run(_main_async())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
