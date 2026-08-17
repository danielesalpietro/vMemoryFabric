"""Issue #33 — performance-test: caratterizzazione sistematica del
"terreno di gioco" hardware di un host/pod (CPU, RAM, GPU/VRAM, PCIe),
riutilizzabile per qualunque futura decisione di tuning invece di misure
ad-hoc ripetute ogni volta (proposta del project owner, LOGBOOK_ISSUE33.MD
2026-08-17 "continued 17"/"continued 18").

Non è un benchmark del codice GCSG (per quello vedi bench_cpu_kernel.py,
bench_route_forward.py, bench_hybrid.py — tutti legati a un path di codice
specifico): questo script misura solo l'hardware sotto, con lo stesso
principio già usato per isolare la causa reale del rallentamento
cpu-offload ("continued 17": microbenchmark PyTorch puro fuori da
vLLM/GCSG, non la pipeline intera).

Sezioni:
    cpu   — core realmente disponibili al processo (cgroup-aware, non
            `os.cpu_count()` grezzo — vedi BOOTSTRAP_ANTI_ALZHEIMER.md §5.9,
            `nproc` mente su RunPod), supporto AVX2/AVX-512 (da
            /proc/cpuinfo, non assunto dal nome della CPU), throughput
            matmul isolato (GFLOPS) a batch multipli — stessa metodologia
            di bench_cpu_kernel.py ma su torch.matmul puro, non
            _ShadowExpertINT4, per isolare l'hardware dal path GCSG.
    ram   — GB disponibili cgroup-aware (riusa
            scheduler.gcsg._read_cgroup_available_gb(), stessa funzione
            già in produzione per _check_cpu_ram_budget()), banda memoria
            (copy GB/s) su un tensore grande.
    gpu   — spec statiche + una singola lettura NVML (nome, VRAM
            totale/libera, driver) — non telemetria continua, quella è
            già coperta da nvidia-smi dmon nel pattern di raccolta
            esistente (BOOTSTRAP §4).
    pcie  — banda H2D/D2H, pinned vs pageable, a varie dimensioni —
            generalizzato da scripts/bench_pcie_bandwidth_wsl2.py (creato
            per un dubbio specifico a WSL2, mai eseguito altrove prima di
            "continued 17"), qui inline per restare eseguibile in
            isolamento (nessun cross-import tra script, stessa convenzione
            di bench_awq_cpu_pipeline.py) e per la sezione "cpu"/"ram" che
            quel file non copriva.

Ogni sezione è {"status": "skipped", "reason": ...} se il prerequisito
manca (no CUDA per gpu/pcie, nessun cgroup leggibile per ram) — mai un
numero inventato, stesso pattern di bench_cpu_kernel.py/bench_hybrid.py.

Usage:
    PYTHONPATH=src python benchmarks/perf_test_hardware.py [--out results.json]
    make perf-test-hardware                       (via Makefile)
    OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336 \\
        PYTHONPATH=src python benchmarks/perf_test_hardware.py    (dimensioni Mixtral reali)

Il JSON prodotto è l'input di perf_tuning_report.py (stesso file, via
--out o piped su stdin).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch
from scheduler.gcsg import _read_cgroup_available_gb

_SEED = 0
_HIDDEN = int(os.environ.get("OSX_BENCH_HIDDEN", 512))
_INTERMEDIATE = int(os.environ.get("OSX_BENCH_INTERMEDIATE", 1536))
_BATCH_SIZES = (1, 8, 32)
_N_REPEATS = 30
_N_WARMUP = 5


def _percentiles(latencies_s: list[float]) -> dict:
    latencies_s = sorted(latencies_s)
    n = len(latencies_s)
    if n == 0:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "p50_ms": latencies_s[int(n * 0.50)] * 1e3,
        "p95_ms": latencies_s[min(int(n * 0.95), n - 1)] * 1e3,
        "p99_ms": latencies_s[min(int(n * 0.99), n - 1)] * 1e3,
    }


def _read_cgroup_cpu_count() -> float | None:
    """Core realmente allocati a QUESTO cgroup, non `os.cpu_count()`
    grezzo — gemello CPU di `_read_cgroup_available_gb()` (che copre la
    RAM). Stesso identico problema documentato in
    BOOTSTRAP_ANTI_ALZHEIMER.md §5.9: `nproc` dentro un pod RunPod ha
    riportato 256 (la topologia dell'HOST fisico condiviso), non la
    quota cgroup reale (~27 core misurati manualmente in "continued 17").
    Non ancora esistente altrove nel codebase — qui perché
    perf_test_hardware.py è il primo consumer che ne ha bisogno per
    calcolare i core realmente disponibili prima di lanciare il
    benchmark matmul.

    Prova cgroup v2 (`cpu.max`, "$MAX $PERIOD" o "max"), poi v1
    (`cpu.cfs_quota_us`/`cpu.cfs_period_us`, -1 = nessun limite), poi
    `os.cpu_count()` (ambienti senza cgroup — CI, macOS/dev locale).
    Ritorna float perché una quota frazionaria (es. 2.5 core) è legale
    e comune sotto Kubernetes/RunPod.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            raw = f.read().strip()
        if raw != "max":
            quota_us, period_us = raw.split()
            return int(quota_us) / int(period_us)
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota_us = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period_us = int(f.read().strip())
        if quota_us > 0:
            return quota_us / period_us
    except (OSError, ValueError):
        pass
    return float(os.cpu_count()) if os.cpu_count() else None


def _cpu_flags() -> list[str]:
    """Flag reali da /proc/cpuinfo, non dedotti dal nome del modello —
    "continued 17" ha scoperto un AMD EPYC 7C13 (Zen3) SENZA AVX-512
    solo leggendo qui, dopo aver assunto per errore che fosse presente
    (lo Xeon Ice Lake di Malmö, pod precedente, ce l'aveva). Ritorna
    lista vuota se /proc/cpuinfo non è leggibile (non-Linux)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return line.split(":", 1)[1].split()
    except OSError:
        pass
    return []


def _bench_matmul_gflops(device: str) -> dict:
    """SwiGLU a due matmul (gate_up, down), stessa forma/formula FLOPs di
    bench_cpu_kernel.py — qui su torch.matmul puro (non _ShadowExpertINT4)
    per isolare l'hardware dal path GCSG, e condivisa tra bench_cpu()
    (device="cpu") e bench_gpu() (device="cuda") cosi' il rapporto
    GFLOPS CPU/GPU nel JSON finale confronta la STESSA identica
    operazione, non due metodologie diverse."""
    generator = torch.Generator(device=device).manual_seed(_SEED)
    w13 = torch.randn(2 * _INTERMEDIATE, _HIDDEN, generator=generator, device=device)
    w2 = torch.randn(_HIDDEN, _INTERMEDIATE, generator=generator, device=device)

    results = {}
    for batch in _BATCH_SIZES:
        hidden_states = torch.randn(batch, _HIDDEN, generator=generator, device=device)

        def _forward():
            gate_up = hidden_states @ w13.T
            activated = gate_up[:, :_INTERMEDIATE]
            return activated @ w2.T

        for _ in range(_N_WARMUP):
            _forward()
        if device == "cuda":
            torch.cuda.synchronize()

        latencies_s = []
        for _ in range(_N_REPEATS):
            t0 = time.perf_counter()
            _forward()
            if device == "cuda":
                torch.cuda.synchronize()
            latencies_s.append(time.perf_counter() - t0)

        pct = _percentiles(latencies_s)
        flops = (
            2 * batch * _HIDDEN * (2 * _INTERMEDIATE)
            + 2 * batch * _INTERMEDIATE * _HIDDEN
        )
        p50_s = (pct["p50_ms"] or 0) / 1e3
        results[f"batch_{batch}"] = {
            "latency": pct,
            "gflops_at_p50": (flops / p50_s / 1e9) if p50_s > 0 else None,
        }

    return {
        "hidden": _HIDDEN,
        "intermediate": _INTERMEDIATE,
        "n_repeats": _N_REPEATS,
        **results,
    }


def bench_cpu() -> dict:
    n_cores = _read_cgroup_cpu_count()
    flags = _cpu_flags()
    avx_support = {
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
        "avx512_vnni": "avx512vnni" in flags,
    }

    return {
        "status": "done",
        "model_name": platform.processor() or platform.machine(),
        "cgroup_cores_available": n_cores,
        "os_cpu_count_raw": os.cpu_count(),
        "note": (
            "os_cpu_count_raw puo' riportare la topologia dell'HOST "
            "condiviso sotto Docker/RunPod, non i core allocati a "
            "questo cgroup (BOOTSTRAP_ANTI_ALZHEIMER.md SS5.9) - usare "
            "cgroup_cores_available per OMP_NUM_THREADS/MKL_NUM_THREADS"
        ),
        "avx_support": avx_support,
        "matmul_gflops": _bench_matmul_gflops("cpu"),
    }


def bench_ram() -> dict:
    available_gb = _read_cgroup_available_gb()

    n_floats = 128 * 1024 * 1024  # 512MB fp32, abbastanza grande da non
    # stare in cache L2/L3 e misurare banda RAM reale, non cache hit.
    src = torch.randn(n_floats, dtype=torch.float32)
    dst = torch.empty_like(src)

    for _ in range(3):
        dst.copy_(src)
    n_iters = 10
    t0 = time.perf_counter()
    for _ in range(n_iters):
        dst.copy_(src)
    elapsed = time.perf_counter() - t0
    size_bytes = n_floats * 4
    copy_gbps = (size_bytes * n_iters) / elapsed / 1e9

    return {
        "status": "done",
        "cgroup_available_gb": available_gb,
        "note": (
            "None se nessun cgroup leggibile (dev locale/CI) - vedi "
            "_read_cgroup_available_gb() in scheduler/gcsg.py, gia' in "
            "produzione per _check_cpu_ram_budget()"
        ),
        "copy_bandwidth_gbps": copy_gbps,
        "copy_test_size_mb": size_bytes / (1024 ** 2),
    }


def bench_gpu() -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA non disponibile su questo host"}

    import pynvml

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as exc:
        return {"status": "skipped", "reason": f"NVML error: {exc}"}

    major, minor = torch.cuda.get_device_capability(0)
    return {
        "status": "done",
        "name": name if isinstance(name, str) else name.decode(),
        "driver_version": driver if isinstance(driver, str) else driver.decode(),
        "compute_capability": f"{major}.{minor}",
        "vram_total_gb": mem.total / (1024 ** 3),
        "vram_free_gb": mem.free / (1024 ** 3),
        "note": (
            "singola lettura statica, non telemetria continua - per quella "
            "vedi `nvidia-smi dmon`/`--query-gpu` nel pattern di raccolta "
            "esistente (BOOTSTRAP_ANTI_ALZHEIMER.md SS4)"
        ),
        "matmul_gflops": _bench_matmul_gflops("cuda"),
    }


def _bench_h2d(size_bytes: int, pinned: bool, n_iters: int = 20) -> float:
    n_floats = size_bytes // 4
    cpu_tensor = torch.randn(n_floats, dtype=torch.float32)
    if pinned:
        cpu_tensor = cpu_tensor.pin_memory()
    _ = cpu_tensor.to("cuda", non_blocking=pinned)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = cpu_tensor.to("cuda", non_blocking=pinned)
    torch.cuda.synchronize()
    return (size_bytes * n_iters) / (time.perf_counter() - t0) / 1e9


def _bench_d2h(size_bytes: int, pinned: bool, n_iters: int = 20) -> float:
    n_floats = size_bytes // 4
    gpu_tensor = torch.randn(n_floats, dtype=torch.float32, device="cuda")
    dst = torch.empty(n_floats, dtype=torch.float32)
    if pinned:
        dst = dst.pin_memory()
    dst.copy_(gpu_tensor, non_blocking=pinned)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        dst.copy_(gpu_tensor, non_blocking=pinned)
    torch.cuda.synchronize()
    return (size_bytes * n_iters) / (time.perf_counter() - t0) / 1e9


def bench_pcie() -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA non disponibile su questo host"}

    sizes_mb = (1, 16, 64, 256)
    by_size = {}
    for mb in sizes_mb:
        size_bytes = mb * 1024 * 1024
        by_size[f"{mb}mb"] = {
            "h2d_pageable_gbps": _bench_h2d(size_bytes, pinned=False),
            "h2d_pinned_gbps": _bench_h2d(size_bytes, pinned=True),
            "d2h_pageable_gbps": _bench_d2h(size_bytes, pinned=False),
            "d2h_pinned_gbps": _bench_d2h(size_bytes, pinned=True),
        }
    return {
        "status": "done",
        "note": (
            "generalizzato da scripts/bench_pcie_bandwidth_wsl2.py "
            "(creato per un dubbio specifico a WSL2/GPU-PV, mai "
            "eseguito su RunPod prima di 'continued 17'). Se pinned "
            "resta sotto ~3GB/s anche sui tensori grandi: indizio di "
            "un tetto WSL2/GPU-PV, non del workload (BOOTSTRAP SS5.4)"
        ),
        "by_size": by_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                         help="Se dato, scrive anche il JSON su questo path "
                              "(oltre a stampare su stdout).")
    args = parser.parse_args()

    result = {
        "status": "done",
        "issue": "#33 — performance-test framework",
        "host": platform.node(),
        "note": (
            f"dimensioni matmul sintetiche hidden={_HIDDEN} "
            f"intermediate={_INTERMEDIATE} (default, frazione di Mixtral "
            "reale 4096x14336) - rilanciare con "
            "OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336 per un "
            "numero comparabile al target di produzione"
        ),
        "cpu": bench_cpu(),
        "ram": bench_ram(),
        "gpu": bench_gpu(),
        "pcie": bench_pcie(),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
