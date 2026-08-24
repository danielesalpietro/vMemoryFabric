"""Fase 1 (issue #33) — throughput del forward SwiGLU INT4 su CPU.

Domanda a cui questo benchmark risponde (LOGBOOK_ISSUE33.MD 2026-08-16):
il backend CPU di PyTorch (oneDNN/MKL, AVX-512/VNNI su Cascade Lake) è già
abbastanza veloce da giustificare NON scrivere un kernel AVX-512 custom?
`_ShadowExpertINT4` (`scheduler/gcsg.py:422-457`) è device-agnostic per
costruzione — vedi `tests/test_cpu_kernel.py` — quindi lo stesso identico
codice usato oggi in produzione su GPU è quello misurato qui su CPU, senza
bisogno di scrivere nulla di nuovo per questa fase.

Sezioni:
    forward_latency  — P50/P95/P99 di _ShadowExpertINT4.__call__() su CPU,
                        a batch size multiple, dimensioni sintetiche
                        configurabili via env (vedi sotto).
    gflops_estimate  — FLOPs teorici del forward (2 matmul SwiGLU) diviso
                        la latenza P50 misurata sopra — un limite inferiore
                        di throughput utile, non un picco hardware.
    gpu_reference    — stessa misura, stesso identico input, sul path GPU
                        già esistente (via GPUTransfer.to_vram + la stessa
                        _ShadowExpertINT4) — SOLO se torch.cuda.is_available(),
                        altrimenti {"status": "skipped", ...} come già fa
                        bench_tier.py, non un numero inventato.

Deviazione dichiarata: le dimensioni di default (HIDDEN/INTERMEDIATE env
var sotto) sono una frazione di Mixtral reale (4096×14336, vedi docstring
di modulo di gcsg.py) per restare veloce su qualunque host di sviluppo.
Per un numero comparabile al target di produzione, rilanciare con:

    OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336 \\
        python benchmarks/bench_cpu_kernel.py

sull'hardware reale (Z8 G4, Xeon Gold 6244 — self-hosted runner
Z8-G4-RTX3090) — questa run va ripetuta lì per un dato significativo,
esattamente come bench_tier.py dichiara per i suoi shard sintetici.

Usage:
    python benchmarks/bench_cpu_kernel.py
    make bench-cpu-kernel          (via Makefile)
"""
from __future__ import annotations

import json
import os
import time

import torch

from scheduler.gcsg import _quantize_int4, _ShadowExpertINT4

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


def _flops_per_forward(batch: int, hidden: int, intermediate: int) -> int:
    """2 matmul (m,k)x(k,n) -> 2*m*k*n FLOPs ciascuno:
    gate_up = hidden_states(batch,hidden) @ w13.T(hidden,2*intermediate)
    out     = activated(batch,intermediate) @ w2.T(intermediate,hidden)
    SiLU/split/mul non contati (trascurabili vs. i matmul a queste taglie)."""
    gate_up_flops = 2 * batch * hidden * (2 * intermediate)
    down_flops = 2 * batch * intermediate * hidden
    return gate_up_flops + down_flops


def _build_shadow(hidden: int, intermediate: int, generator: torch.Generator,
                   device: str) -> _ShadowExpertINT4:
    w13 = torch.randn(2 * intermediate, hidden, generator=generator, device=device)
    w2 = torch.randn(hidden, intermediate, generator=generator, device=device)
    return _ShadowExpertINT4([_quantize_int4(w13)], [_quantize_int4(w2)])


def _bench_forward(device: str) -> dict:
    # torch.randn(..., generator=g, device=device) richiede g sullo stesso
    # device del tensore da generare — un Generator("cpu") passato con
    # device="cuda" solleva "Expected a 'cuda' device type for generator
    # but found 'cpu'" (bug reale, trovato sul primo run su hardware reale
    # Z8/Xeon 6244+RTX 3090: la sezione cpu passava perché device="cpu"
    # matcha, quella gpu_reference no — vedi LOGBOOK_ISSUE33.MD).
    generator = torch.Generator(device=device).manual_seed(_SEED)
    shadow = _build_shadow(_HIDDEN, _INTERMEDIATE, generator, device)

    results = {}
    for batch in _BATCH_SIZES:
        hidden_states = torch.randn(batch, _HIDDEN, generator=generator, device=device)

        for _ in range(_N_WARMUP):
            shadow(hidden_states, layer_id=0)
        if device == "cuda":
            torch.cuda.synchronize()

        latencies_s = []
        for _ in range(_N_REPEATS):
            t0 = time.perf_counter()
            shadow(hidden_states, layer_id=0)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies_s.append(time.perf_counter() - t0)

        pct = _percentiles(latencies_s)
        flops = _flops_per_forward(batch, _HIDDEN, _INTERMEDIATE)
        p50_s = (pct["p50_ms"] or 0) / 1e3
        results[f"batch_{batch}"] = {
            "latency": pct,
            "gflops_at_p50": (flops / p50_s / 1e9) if p50_s > 0 else None,
        }
    return results


def bench_forward_latency_cpu() -> dict:
    return {
        "hidden": _HIDDEN,
        "intermediate": _INTERMEDIATE,
        "n_repeats": _N_REPEATS,
        **_bench_forward("cpu"),
    }


def bench_forward_latency_gpu_reference() -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA non disponibile su questo host"}
    return {
        "hidden": _HIDDEN,
        "intermediate": _INTERMEDIATE,
        "n_repeats": _N_REPEATS,
        **_bench_forward("cuda"),
    }


def main() -> None:
    result = {
        "status": "done",
        "issue": "#33 Fase 1",
        "module": "_ShadowExpertINT4 (scheduler/gcsg.py) — CPU path",
        "note": (
            f"dimensioni sintetiche hidden={_HIDDEN} intermediate={_INTERMEDIATE}, "
            "frazione di Mixtral reale (4096x14336) — vedi docstring di modulo "
            "per come rilanciare a scala reale su hardware reale"
        ),
        "cpu": bench_forward_latency_cpu(),
        "gpu_reference": bench_forward_latency_gpu_reference(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
