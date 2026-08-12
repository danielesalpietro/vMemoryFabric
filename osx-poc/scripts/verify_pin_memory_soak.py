#!/usr/bin/env python3
"""Soak test del pinning (Sprint 4 / Tekniska, §9, issue #17): la domanda
lasciata esplicitamente aperta dal report — una singola `pin_memory()` +
`is_pinned()` "sembra" a posto anche se il dato viene silenziosamente
corrotto o se l'allocatore degrada sotto uso ripetuto. Nessuno script
esistente copre questo: `isolate_*`, `verify_marlin_pinned_proxy.py`,
`smoke_test_fetta2_pinmemory.py` sono tutti isolamenti one-shot (una
chiamata forzata con `in_wsl` patchato), non un loop ripetuto — questa è
la prima verifica del genere in questa indagine, va eseguita fuori da
WSL2 (RunPod pod reale) per avere significato.

Ogni iterazione: alloca un buffer pageable con dati noti, lo pinna
(esercita l'allocatore pinned ripetutamente, non solo una volta),
transfer H2D non_blocking, sincronizza, transfer D2H di ritorno,
confronto byte-esatto con l'originale (torch.equal — nessuna computazione
nel mezzo, un roundtrip corretto deve combaciare esattamente, non solo
"entro tolleranza"). Cattura corruzione silenziosa, non solo crash.

Dimensione shard = SHARD_SIZE_MB da eat/types.py (256MB) — stessa unità
di trasferimento reale di TierManager/EAT, non un numero arbitrario.
N_ITERATIONS = 1000 — il numero proposto fin dalla prima analisi di
questa indagine, mai eseguito finora.

Usage:
    PYTHONPATH=src python scripts/verify_pin_memory_soak.py
    PYTHONPATH=src python scripts/verify_pin_memory_soak.py --iterations 100  # run rapido
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

try:
    from eat.types import SHARD_SIZE_MB
except ImportError:
    SHARD_SIZE_MB = 256  # fallback se eat/types.py non è importabile isolatamente

SHARD_NUMEL = SHARD_SIZE_MB * 1024 * 1024 // 4  # float32
N_ITERATIONS_DEFAULT = 1000


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        _fail("CUDA non disponibile — questo test ha senso solo su GPU reale")

    device = torch.cuda.get_device_name(0)
    print(f"Device: {device}")
    print(f"Shard size: {SHARD_SIZE_MB} MB ({SHARD_NUMEL:,} float32 elements)")
    print(f"Iterazioni: {args.iterations}")
    print()

    torch.manual_seed(args.seed)

    cycle_times_ms: list[float] = []
    pin_times_ms: list[float] = []
    transfer_times_ms: list[float] = []
    mismatches = 0

    t_start = time.perf_counter()
    for i in range(args.iterations):
        t_cycle0 = time.perf_counter()

        cpu_orig = torch.randn(SHARD_NUMEL, dtype=torch.float32)

        t_pin0 = time.perf_counter()
        cpu_pinned = cpu_orig.pin_memory()
        t_pin1 = time.perf_counter()
        if not cpu_pinned.is_pinned():
            _fail(f"iterazione {i}: is_pinned() è False dopo pin_memory()")

        t_xfer0 = time.perf_counter()
        gpu_buf = cpu_pinned.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        cpu_roundtrip = gpu_buf.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        t_xfer1 = time.perf_counter()

        if not torch.equal(cpu_orig, cpu_roundtrip):
            mismatches += 1
            n_diff = int((cpu_orig != cpu_roundtrip).sum())
            print(f"  MISMATCH iterazione {i}: {n_diff}/{SHARD_NUMEL} elementi diversi "
                  f"dopo roundtrip pinnato — corruzione silenziosa rilevata")

        t_cycle1 = time.perf_counter()
        cycle_times_ms.append((t_cycle1 - t_cycle0) * 1000)
        pin_times_ms.append((t_pin1 - t_pin0) * 1000)
        transfer_times_ms.append((t_xfer1 - t_xfer0) * 1000)

        del cpu_orig, cpu_pinned, gpu_buf, cpu_roundtrip

        if (i + 1) % max(1, args.iterations // 20) == 0 or i == args.iterations - 1:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i + 1}/{args.iterations}] elapsed={elapsed:.1f}s "
                  f"last_cycle={cycle_times_ms[-1]:.1f}ms mismatches={mismatches}")

    total_elapsed = time.perf_counter() - t_start

    print()
    print("=" * 60)
    print(f"Completate {args.iterations} iterazioni in {total_elapsed:.1f}s")
    print(f"Mismatch byte-esatti: {mismatches}/{args.iterations}")
    print()

    def _stats(name: str, values: list[float]) -> None:
        first_10pct = values[: max(1, len(values) // 10)]
        last_10pct = values[-max(1, len(values) // 10):]
        mean_first = statistics.mean(first_10pct)
        mean_last = statistics.mean(last_10pct)
        drift = (mean_last / mean_first - 1) * 100 if mean_first > 0 else 0.0
        print(f"{name}: min={min(values):.2f}ms mean={statistics.mean(values):.2f}ms "
              f"max={max(values):.2f}ms stddev={statistics.stdev(values):.2f}ms "
              f"| primi 10% media={mean_first:.2f}ms ultimi 10% media={mean_last:.2f}ms "
              f"(drift {drift:+.0f}%)")

    _stats("Cycle totale ", cycle_times_ms)
    _stats("Pin alloc    ", pin_times_ms)
    _stats("H2D+D2H xfer ", transfer_times_ms)

    print()
    if mismatches > 0:
        print(f"SIGNAL: {mismatches} corruzioni silenziose rilevate su pinned memory "
              f"reale (fuori WSL2) — la domanda del §9 NON è chiusa positivamente, "
              f"servirebbe indagine ulteriore prima di fidarsi del path integrato.")
        sys.exit(1)
    else:
        print(f"SIGNAL: {args.iterations}/{args.iterations} roundtrip byte-esatti, "
              f"nessuna corruzione silenziosa su pinned memory reale (fuori WSL2). "
              f"Controllare comunque il drift sopra — se significativo, indica "
              f"degradazione dell'allocatore sotto uso ripetuto anche senza corruzione.")


if __name__ == "__main__":
    main()
