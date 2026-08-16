"""Fase 5 (issue #33) — impatto end-to-end reale di enable_cpu_offload=True
(issue #33, spento di default — vedi GCSGWorker.configure_cpu_offload()).

Domanda a cui questo benchmark risponde, quella che Fase 4 non copriva:
Fase 4 ha misurato che l'overhead di *dispatch* (route_forward()) è
trascurabile (~0.08% del tempo di compute) — ma questo non dice nulla
sull'impatto AGGREGATO di spostare davvero traffico dal pool GPU (veloce,
~0.3ms/call su questi shard sintetici, Fase 1) al pool CPU (più lento,
~2.5-3ms/call, ~7-10x più lento su hardware reale — vedi LOGBOOK_ISSUE33.MD
2026-08-16). Con shadow_pool_size=2 (produzione, memory math in
scheduler/gcsg.py) e classify_hot_cold() che ne instrada uno "caldo" su
GPU e uno "freddo" su CPU (Fase 0/3), il costo aggregato dipende da QUANTO
traffico reale finisce sull'expert freddo — non misurabile isolando un
solo forward come in Fase 1/4.

Metodologia: due expert (shadow_pool_size=2, come in produzione), traffico
sintetico diviso tra i due secondo una quota "fredda" configurabile
(cold_share — quale frazione delle chiamate shadow va all'expert meno
caldo dei due). Per ogni cold_share, due scenari sullo STESSO traffico:
    baseline    — enable_cpu_offload=False (comportamento pre-issue-#33,
                  default reale): entrambi gli expert selezionati restano
                  su GPU, self._cpu_shadow_pool resta vuoto,
                  route_forward() instrada tutto a GPU per il suo
                  fallback esistente (nessun percorso di codice diverso
                  da qui a scenario "on" — stessa self._shadow_pool reale
                  in entrambi gli scenari, quello che cambia è solo se
                  self._cpu_shadow_pool è popolato o no).
    cpu_offload — enable_cpu_offload=True: l'expert classificato "freddo"
                  (classify_hot_cold(), Fase 0) instrada al pool CPU.

Riporta latenza aggregata totale e throughput (calls/sec) per entrambi
gli scenari, ad ogni cold_share — il crossover a cui l'aggregato inizia a
peggiorare visibilmente è il dato che manca per decidere quando
enable_cpu_offload=True conviene rispetto al risparmio di VRAM che
abilita (vedi memory math in scheduler/gcsg.py: ~3 GB/expert).

Deviazione dichiarata: stesse dimensioni sintetiche di bench_cpu_kernel.py
(frazione di Mixtral reale, hidden=512/intermediate=1536 di default) —
vedi quel modulo per come rilanciare a scala reale. GPU reale usata quando
disponibile (torch.cuda.is_available()), altrimenti {"status": "skipped"},
stesso pattern di bench_cpu_kernel.py/bench_tier.py — questo benchmark
esiste apposta per confrontare GPU vs CPU, non ha senso senza una GPU
reale.

Usage:
    python benchmarks/bench_hybrid.py
    make bench-hybrid          (via Makefile)
"""
from __future__ import annotations

import json
import os
import random
import time

import torch
from scheduler.gcsg import GCSGGuard, GCSGWorker, _quantize_int4, _ShadowExpertINT4

_SEED = 0
_HIDDEN = int(os.environ.get("OSX_BENCH_HIDDEN", 512))
_INTERMEDIATE = int(os.environ.get("OSX_BENCH_INTERMEDIATE", 1536))
_N_CALLS = 500          # chiamate simulate per scenario per cold_share
_COLD_SHARES = (0.05, 0.10, 0.25, 0.50)   # quota di traffico verso l'expert "freddo"
_HOT_EXPERT_ID = 0
_COLD_EXPERT_ID = 1


def _percentiles(latencies_s: list[float]) -> dict:
    latencies_s = sorted(latencies_s)
    n = len(latencies_s)
    return {
        "p50_ms": latencies_s[int(n * 0.50)] * 1e3,
        "p95_ms": latencies_s[min(int(n * 0.95), n - 1)] * 1e3,
        "p99_ms": latencies_s[min(int(n * 0.99), n - 1)] * 1e3,
    }


def _build_shadow(hidden: int, intermediate: int, generator: torch.Generator,
                   device: str) -> _ShadowExpertINT4:
    w13 = torch.randn(2 * intermediate, hidden, generator=generator, device=device)
    w2 = torch.randn(hidden, intermediate, generator=generator, device=device)
    return _ShadowExpertINT4([_quantize_int4(w13)], [_quantize_int4(w2)])


def _build_worker(seed: int, cpu_offload_enabled: bool) -> GCSGWorker:
    """Worker minimale, stesso pattern di tests/test_scheduler.py — nessun
    vLLM live richiesto (route_forward() non tocca self._base). Un solo
    GPU pool reale (entrambi gli expert, sempre — è quello che
    _load_shadow_pool() costruisce SEMPRE per il path 1, indipendentemente
    dal flag: il flag governa solo il pool CPU aggiuntivo, vedi
    scheduler/gcsg.py). Il pool CPU esiste solo se cpu_offload_enabled.

    Due generator separati, non uno condiviso tra device (bug reale,
    trovato scrivendo questo stesso benchmark prima di lanciarlo — lo
    stesso "Expected a 'X' device type for generator but found 'Y'" già
    isolato in Fase 1 per bench_cpu_kernel.py): torch.randn(...,
    generator=g, device=d) richiede g.device == d.
    """
    cuda_generator = torch.Generator(device="cuda").manual_seed(seed)
    cpu_generator = torch.Generator().manual_seed(seed)

    shadow_gpu_hot = _build_shadow(_HIDDEN, _INTERMEDIATE, cuda_generator, "cuda")
    shadow_gpu_cold = _build_shadow(_HIDDEN, _INTERMEDIATE, cuda_generator, "cuda")

    worker = GCSGWorker.__new__(GCSGWorker)
    worker.guard = GCSGGuard(shadow_pool_size=2, check_vram=False)
    worker._tier_manager = None
    worker._n_experts_cached = 2
    worker._shadow_pool = {_HOT_EXPERT_ID: shadow_gpu_hot, _COLD_EXPERT_ID: shadow_gpu_cold}
    worker._cpu_offload_enabled = cpu_offload_enabled

    if cpu_offload_enabled:
        shadow_cpu_cold = _build_shadow(_HIDDEN, _INTERMEDIATE, cpu_generator, "cpu")
        worker._cpu_shadow_pool = {_COLD_EXPERT_ID: shadow_cpu_cold}
        worker._hot_expert_ids = {_HOT_EXPERT_ID}   # solo l'expert 0 è "caldo"
    else:
        worker._cpu_shadow_pool = {}
        worker._hot_expert_ids = set()   # irrilevante: cpu pool vuoto, route_forward cade sempre su GPU

    return worker


def _simulate_traffic(worker: GCSGWorker, hidden_states_hot: torch.Tensor,
                       hidden_states_cold: torch.Tensor, call_order: list[int]) -> dict:
    """Esegue call_order (sequenza di _HOT_EXPERT_ID/_COLD_EXPERT_ID già
    mescolata) attraverso route_forward(), misurando latenza per-call e
    tempo totale — torch.cuda.synchronize() dopo ogni call GPU per una
    latenza reale, non solo il tempo di lancio del kernel asincrono."""
    latencies_s = []
    t_total0 = time.perf_counter()
    for expert_id in call_order:
        hs = hidden_states_hot if expert_id == _HOT_EXPERT_ID else hidden_states_cold
        t0 = time.perf_counter()
        worker.route_forward(expert_id, layer_id=0, hidden_states=hs)
        torch.cuda.synchronize()   # innocuo se l'ultima op era su CPU, necessario se su GPU
        latencies_s.append(time.perf_counter() - t0)
    total_s = time.perf_counter() - t_total0
    return {
        "total_s": total_s,
        "throughput_calls_per_s": len(call_order) / total_s if total_s > 0 else None,
        "latency": _percentiles(latencies_s),
    }


def bench_cold_share(cold_share: float, seed: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states_hot = torch.randn(1, _HIDDEN, generator=generator, device="cuda")
    hidden_states_cold = torch.randn(1, _HIDDEN, generator=generator, device="cuda")

    n_cold = round(_N_CALLS * cold_share)
    n_hot = _N_CALLS - n_cold
    call_order = [_HOT_EXPERT_ID] * n_hot + [_COLD_EXPERT_ID] * n_cold
    rng = random.Random(seed)   # non torch: solo per mescolare l'ordine delle chiamate, non i pesi
    rng.shuffle(call_order)

    baseline_worker = _build_worker(seed, cpu_offload_enabled=False)
    offload_worker = _build_worker(seed, cpu_offload_enabled=True)

    for _ in range(20):   # warm-up, fuori dal timing — CUDA cold-start (issue #3)
        baseline_worker.route_forward(_HOT_EXPERT_ID, 0, hidden_states_hot)
        baseline_worker.route_forward(_COLD_EXPERT_ID, 0, hidden_states_cold)
        offload_worker.route_forward(_HOT_EXPERT_ID, 0, hidden_states_hot)
        offload_worker.route_forward(_COLD_EXPERT_ID, 0, hidden_states_cold)
    torch.cuda.synchronize()

    baseline = _simulate_traffic(baseline_worker, hidden_states_hot, hidden_states_cold, call_order)
    cpu_offload = _simulate_traffic(offload_worker, hidden_states_hot, hidden_states_cold, call_order)

    slowdown = (
        cpu_offload["total_s"] / baseline["total_s"] if baseline["total_s"] > 0 else None
    )
    return {
        "cold_share": cold_share,
        "n_calls": _N_CALLS,
        "n_cold_calls": n_cold,
        "baseline_gpu_only": baseline,
        "cpu_offload_enabled": cpu_offload,
        "aggregate_slowdown_factor": slowdown,
    }


def main() -> None:
    if not torch.cuda.is_available():
        print(json.dumps({
            "status": "skipped",
            "reason": "CUDA non disponibile — questo benchmark confronta GPU vs CPU, non ha senso senza GPU reale",
        }, indent=2))
        return

    results = [bench_cold_share(share, seed=_SEED) for share in _COLD_SHARES]

    result = {
        "status": "done",
        "issue": "#33 Fase 5",
        "note": (
            f"dimensioni sintetiche hidden={_HIDDEN} intermediate={_INTERMEDIATE}, "
            "stessa frazione di Mixtral reale usata in bench_cpu_kernel.py (Fase 1); "
            f"{_N_CALLS} call simulate per scenario per cold_share"
        ),
        "by_cold_share": results,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
