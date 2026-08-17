"""Issue #33 — performance-tuning: legge il JSON prodotto da
perf_test_hardware.py e ne deriva raccomandazioni concrete, invece di
tentativi isolati come il test A/B thread-count di "continued 17"
(LOGBOOK_ISSUE33.MD 2026-08-17) — quel test ha risposto a UNA domanda
specifica per QUEL pod; questo script generalizza il ragionamento così
la prossima volta non serva rifarlo a mano.

Ogni raccomandazione è ancorata a un fatto già misurato/documentato per
issue #33, non a un'euristica generica:
    - il forward per-token (batch=1) è memory-bandwidth-bound, non
      thread-bound — falsificato con OMP_NUM_THREADS=27 vs 2, stesso
      ~0.10s/call (BOOTSTRAP_ANTI_ALZHEIMER.md §5.11). Questo script non
      raccomanda MAI più thread per il forward, solo per la build del
      pool (dequant, tensori grandi, quello sì parallelizzabile).
    - il budget RAM CPU-pool usa le stesse costanti già in produzione in
      `GCSGWorker._check_cpu_ram_budget()` (scheduler/gcsg.py):
      ~3GB/expert (path 1, INT4 simulato), ~21.5GB/expert (path AWQ,
      fp32 cache), margin_gb=24.0 di default.
    - il verdetto cpu-offload confronta GFLOPS CPU vs GPU sulla STESSA
      operazione (bench_cpu().matmul_gflops vs bench_gpu().matmul_gflops
      in perf_test_hardware.py, non due metodologie diverse), poi lo
      annota contro il range già osservato in produzione (24-26x nella
      pipeline reale, 44.6x nel microbenchmark isolato — vedi
      "continued 17") invece di trattarlo come un numero isolato.

Usage:
    PYTHONPATH=src python benchmarks/perf_test_hardware.py --out /tmp/hw.json
    PYTHONPATH=src python benchmarks/perf_tuning_report.py --in /tmp/hw.json
    # oppure in pipe:
    PYTHONPATH=src python benchmarks/perf_test_hardware.py | \\
        PYTHONPATH=src python benchmarks/perf_tuning_report.py
    make perf-tuning-report HW_JSON=/tmp/hw.json    (via Makefile)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Stesse costanti di GCSGWorker._check_cpu_ram_budget() (scheduler/gcsg.py)
# — duplicate qui deliberatamente invece di importate: questo script deve
# restare eseguibile anche solo con un JSON in mano, senza un ambiente
# vLLM/torch completo (a differenza di perf_test_hardware.py, che quello
# lo richiede per misurare). Se quelle costanti cambiano in produzione,
# aggiornarle anche qui — non è un valore derivato, è un fatto di dominio
# duplicato consapevolmente.
_PER_EXPERT_CPU_GB_PATH1 = 3.0
_PER_EXPERT_CPU_GB_AWQ = 21.5
_RAM_MARGIN_GB_DEFAULT = 24.0

_KNOWN_PIPELINE_SLOWDOWN_RANGE = (24.0, 26.0)   # "continued 17"/"continued 18"
_KNOWN_ISOLATED_SLOWDOWN = 44.6                 # microbenchmark puro, "continued 17"

_PCIE_PINNED_FLOOR_GBPS = 3.0   # sotto questo valore: indizio WSL2/GPU-PV, non il workload


def _cpu_offload_verdict(hw: dict) -> dict:
    cpu, gpu = hw.get("cpu", {}), hw.get("gpu", {})
    if cpu.get("status") != "done":
        return {"verdict": "n/a", "reason": "sezione cpu non disponibile nel JSON"}
    if gpu.get("status") != "done":
        return {
            "verdict": "n/a",
            "reason": "sezione gpu non disponibile (no CUDA su questo host) "
                      "- impossibile calcolare il rapporto CPU/GPU",
        }

    cpu_gflops = cpu["matmul_gflops"]["batch_1"]["gflops_at_p50"]
    gpu_gflops = gpu["matmul_gflops"]["batch_1"]["gflops_at_p50"]
    if not cpu_gflops or not gpu_gflops:
        return {"verdict": "n/a", "reason": "gflops_at_p50 mancante (run troppo rumoroso?)"}

    predicted_slowdown = gpu_gflops / cpu_gflops
    lo, hi = _KNOWN_PIPELINE_SLOWDOWN_RANGE
    if lo * 0.5 <= predicted_slowdown <= _KNOWN_ISOLATED_SLOWDOWN * 1.5:
        consistency = (
            f"coerente con il range già osservato in produzione "
            f"({lo:.0f}-{hi:.0f}x pipeline reale, {_KNOWN_ISOLATED_SLOWDOWN:.1f}x "
            "microbenchmark isolato, LOGBOOK_ISSUE33.MD 'continued 17') - "
            "questo host si comporta come quelli già misurati"
        )
    else:
        consistency = (
            f"FUORI dal range già osservato ({lo:.0f}-{hi:.0f}x / "
            f"{_KNOWN_ISOLATED_SLOWDOWN:.1f}x) - questo host ha caratteristiche "
            "CPU/GPU diverse da quelle già misurate, non assumere che le "
            "conclusioni precedenti si applichino senza verifica"
        )

    avx512 = cpu.get("avx_support", {}).get("avx512f")
    return {
        "verdict": (
            "cpu-offload utile solo per risparmio VRAM, non per velocità "
            f"- forward CPU stimato ~{predicted_slowdown:.1f}x più lento del GPU "
            "sulla stessa operazione (batch=1, il caso reale del forward per-token)"
        ),
        "predicted_slowdown_factor": predicted_slowdown,
        "consistency_with_known_data": consistency,
        "avx512f_present": avx512,
        "avx512_note": (
            None if avx512
            else "AVX-512 assente - coerente con un limite di throughput "
                 "per-core genuino (non un problema di configurazione), "
                 "come l'AMD EPYC 7C13/Zen3 già misurato in 'continued 17'"
        ),
    }


def _batching_signal(hw: dict) -> dict:
    """Segnale (non una raccomandazione di implementazione — l'opzione
    "batching GEMV→GEMM" resta una direzione separata, non scelta in
    questa sessione) su quanto i core CPU idle durante il forward
    single-token potrebbero rendere se il traffico fosse batchato:
    rapporto GFLOPS a batch=32 vs batch=1 sullo stesso host."""
    cpu = hw.get("cpu", {})
    if cpu.get("status") != "done":
        return {"signal": "n/a", "reason": "sezione cpu non disponibile"}
    g1 = cpu["matmul_gflops"]["batch_1"]["gflops_at_p50"]
    g32 = cpu["matmul_gflops"]["batch_32"]["gflops_at_p50"]
    if not g1 or not g32:
        return {"signal": "n/a", "reason": "gflops_at_p50 mancante"}
    ratio = g32 / g1
    return {
        "gflops_ratio_batch32_over_batch1": ratio,
        "signal": (
            f"batchare il forward potrebbe recuperare fino a ~{ratio:.1f}x "
            "throughput su questo host (core oggi idle durante un forward "
            "1x4096 GEMV) - non implementato, solo un segnale quantitativo "
            "per la direzione 'batching GEMV→GEMM' se/quando verrà scelta"
            if ratio > 1.5 else
            "nessun guadagno significativo osservato batchando su questo "
            "host - la direzione 'batching' varrebbe meno qui che altrove"
        ),
    }


def _thread_count_recommendation(hw: dict) -> dict:
    cpu = hw.get("cpu", {})
    if cpu.get("status") != "done":
        return {"recommendation": "n/a", "reason": "sezione cpu non disponibile"}
    n_cores = cpu.get("cgroup_cores_available")
    return {
        "omp_mkl_num_threads_for_pool_build": (
            int(n_cores) if n_cores else None
        ),
        "note_pool_build": (
            "usare TUTTI i core cgroup disponibili per la build del pool "
            "(dequant AWQ/quantizzazione INT4, tensori grandi, "
            "genuinamente parallelizzabile - misurato: 183s con 2 thread "
            "vs 88s con thread pieni, 'continued 17')"
        ),
        "note_per_token_forward": (
            "NON aumentare i thread per il forward per-token (batch=1): "
            "falsificato con dati che aiuti — 0.0967s/call a 27 thread vs "
            "0.100-0.103s/call a 2 thread, praticamente identico "
            "(memory-bandwidth-bound, non thread-bound). Non ripetere "
            "questo test (BOOTSTRAP_ANTI_ALZHEIMER.md §5.11)."
        ),
    }


def _ram_budget(hw: dict) -> dict:
    ram = hw.get("ram", {})
    if ram.get("status") != "done" or ram.get("cgroup_available_gb") is None:
        return {"recommendation": "n/a", "reason": "RAM cgroup non determinabile su questo host"}

    available_gb = ram["cgroup_available_gb"]
    budget_gb = available_gb - _RAM_MARGIN_GB_DEFAULT
    n_path1 = max(0, int(budget_gb // _PER_EXPERT_CPU_GB_PATH1))
    n_awq = max(0, int(budget_gb // _PER_EXPERT_CPU_GB_AWQ))
    return {
        "cgroup_available_gb": available_gb,
        "margin_reserved_gb": _RAM_MARGIN_GB_DEFAULT,
        "usable_budget_gb": budget_gb,
        "max_cpu_shadow_experts_path1_int4": n_path1,
        "max_cpu_shadow_experts_path_awq_fp32": n_awq,
        "note": (
            "stesse costanti di GCSGWorker._check_cpu_ram_budget() "
            "(scheduler/gcsg.py) - se questo host costruisce un pool più "
            "grande di questi numeri, aspettarsi lo stesso fallimento "
            "osservato una volta (RAM cgroup all'84%, zero prompt "
            "completati su 16 in 600s, 'continued 17')"
        ),
    }


def _pcie_flag(hw: dict) -> dict:
    pcie = hw.get("pcie", {})
    if pcie.get("status") != "done":
        return {"flag": "n/a", "reason": pcie.get("reason", "sezione pcie non disponibile")}

    largest = list(pcie["by_size"].values())[-1]
    pinned_min = min(largest["h2d_pinned_gbps"], largest["d2h_pinned_gbps"])
    if pinned_min < _PCIE_PINNED_FLOOR_GBPS:
        return {
            "flag": "SOSPETTO",
            "pinned_min_gbps": pinned_min,
            "reason": (
                f"banda pinned sotto {_PCIE_PINNED_FLOOR_GBPS} GB/s anche sul "
                "tensore più grande misurato - indizio di un tetto WSL2/GPU-PV "
                "(BOOTSTRAP_ANTI_ALZHEIMER.md §5.1/§5.4), non del workload. "
                "Se il sintomo che ha portato a girare questo perf-test era "
                "lentezza cpu-offload: NON è la causa dominante nota (quella è "
                "il forward per-token, §5.11) ma vale la pena escluderla."
            ),
        }
    return {
        "flag": "ok",
        "pinned_min_gbps": pinned_min,
        "reason": "nessun tetto PCIe evidente su questo host",
    }


def build_report(hw: dict) -> dict:
    return {
        "status": "done",
        "issue": "#33 — performance-tuning report",
        "source_host": hw.get("host"),
        "cpu_offload_verdict": _cpu_offload_verdict(hw),
        "thread_count": _thread_count_recommendation(hw),
        "ram_budget": _ram_budget(hw),
        "pcie": _pcie_flag(hw),
        "batching_signal": _batching_signal(hw),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=None,
                         help="Path al JSON di perf_test_hardware.py. "
                              "Se omesso, legge da stdin.")
    args = parser.parse_args()

    raw = args.in_path.read_text() if args.in_path is not None else sys.stdin.read()
    hw = json.loads(raw)

    print(json.dumps(build_report(hw), indent=2))


if __name__ == "__main__":
    main()
