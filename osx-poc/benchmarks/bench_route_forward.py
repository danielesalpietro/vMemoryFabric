"""Fase 4 (issue #33) — l'overhead di dispatch di route_forward() è
trascurabile rispetto al tempo di compute AVX-512, o serve davvero il
meccanismo separato-processo/shared-memory stile exllamav3 MoeCpuHost?

Il piano (LOGBOOK_ISSUE33.MD, 2026-08-16) pone Fase 4 come un gate, non
un'implementazione: "misurare l'overhead del round-trip prima di decidere
se serve un meccanismo di sync separato — skip Fase 4 se è trascurabile
rispetto al tempo di compute AVX-512". Fase 3 (2026-08-17) ha già scelto
la via in-process (route_forward()/_RoutedShadowPool, tutto nello stesso
processo Python, nessun job ring, nessuna shared memory) — quindi
"l'overhead del round-trip" qui è per costruzione l'overhead di dispatch
Python puro di quella scelta: lookup dict/set, chiamate a closure, la
logica di ranking di GCSGGuard.run_shadow(). Non c'è un vero round-trip
inter-processo da misurare perché quel meccanismo non esiste in questo
design — è esattamente l'opzione che Fase 4 deve escludere o confermare.

Sezioni:
    dispatch_overhead — P50/P95/P99 di tre livelli di indirection, con una
                         callable no-op (isola l'overhead Python puro dal
                         tempo di compute reale, misurato separatamente
                         sotto):
                           raw_dict_call   — pool[expert_id](hs, layer_id)
                                             diretto, il caso pre-Fase-3.
                           route_forward   — GCSGWorker.route_forward(),
                                             la nuova indirection di
                                             Fase 3.
                           run_shadow_full — l'intero path reale:
                                             GCSGGuard.run_shadow() con
                                             _RoutedShadowPool, incluso il
                                             ranking degli expert per
                                             gating score.
    compute_reference  — tempo di compute REALE di _ShadowExpertINT4 su
                          CPU, stessa metodologia di bench_cpu_kernel.py
                          (Fase 1), ricalcolato qui per un confronto
                          pulito nello stesso run invece di citare un
                          numero di una sessione precedente.
    verdict            — overhead di dispatch (run_shadow_full, il caso
                          reale) come frazione del tempo di compute p50 —
                          la risposta diretta alla domanda del gate.

Deviazione dichiarata: stesse dimensioni sintetiche di bench_cpu_kernel.py
(frazione di Mixtral reale) — vedi quel modulo per come rilanciare a scala
reale via OSX_BENCH_HIDDEN/OSX_BENCH_INTERMEDIATE.

Usage:
    python benchmarks/bench_route_forward.py
    make bench-route-forward          (via Makefile)
"""
from __future__ import annotations

import json
import os
import time

import torch
from eat import ExpertAccessTable, Tier
from scheduler.gcsg import (
    GatingContext,
    GCSGGuard,
    GCSGWorker,
    _quantize_int4,
    _RoutedShadowPool,
    _ShadowExpertINT4,
)
from tier import TierManager

_SEED = 0
_HIDDEN = int(os.environ.get("OSX_BENCH_HIDDEN", 512))
_INTERMEDIATE = int(os.environ.get("OSX_BENCH_INTERMEDIATE", 1536))
_N_SAMPLES = 30       # campioni per il calcolo dei percentili
_N_INNER = 2_000       # chiamate per campione, per ammortizzare l'overhead di time.perf_counter() stesso
_N_WARMUP_INNER = 200


def _percentiles(latencies_s: list[float]) -> dict:
    latencies_s = sorted(latencies_s)
    n = len(latencies_s)
    if n == 0:
        return {"p50_us": None, "p95_us": None, "p99_us": None}
    return {
        "p50_us": latencies_s[int(n * 0.50)] * 1e6,
        "p95_us": latencies_s[min(int(n * 0.95), n - 1)] * 1e6,
        "p99_us": latencies_s[min(int(n * 0.99), n - 1)] * 1e6,
    }


def _time_batched(fn, n_samples: int, n_inner: int, n_warmup_inner: int) -> dict:
    """Media n_inner chiamate per campione (non un timing per-call) — a
    queste scale (attese sub-microsecondo per una chiamata dict+closure),
    l'overhead di time.perf_counter() stesso (~50-100ns per chiamata)
    rischierebbe di dominare il segnale misurato, non solo di aggiungersi
    ad esso."""
    for _ in range(n_warmup_inner):
        fn()
    latencies_s = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        for _ in range(n_inner):
            fn()
        latencies_s.append((time.perf_counter() - t0) / n_inner)
    return _percentiles(latencies_s)


def _build_worker_with_both_pools() -> tuple[GCSGWorker, int]:
    """Worker minimale, stesso pattern di tests/test_scheduler.py
    (GCSGWorker.__new__() + attributi a mano — nessun vLLM live richiesto,
    route_forward()/_RoutedShadowPool non toccano self._base). Un
    TierManager/EAT reale, un solo expert_id (0), presente in entrambi i
    pool — il caso che route_forward() deve arbitrare, esattamente lo
    scenario di dispatch che questo benchmark isola.

    Le due pool usano callable NO-OP (`lambda hs, layer_id: hs`), non
    _ShadowExpertINT4 reali: questo benchmark misura l'overhead di
    dispatch (lookup dict/set, chiamate a closure, ranking di
    run_shadow()), non il tempo di compute — quello è
    bench_compute_reference(), separato deliberatamente. Un primo
    tentativo (bug, corretto prima di fidarsi di questo numero) usava
    _ShadowExpertINT4 reali qui: il risultato era identico al compute
    reference (~3.2ms per tutte e tre le sezioni) perché il matmul reale
    dominava completamente il segnale — la conclusione "overhead non
    trascurabile" era falsa, misurava compute due volte, non dispatch.
    """
    eat = ExpertAccessTable(capacity=100, n_slots=4)
    mgr = TierManager(eat=eat, nvme_path="/tmp/bench_route_forward", gpu_device=0)
    mgr.eat.insert(expert_id=0, shard_idx=0, tier=Tier.DDR4)

    worker = GCSGWorker.__new__(GCSGWorker)
    worker.guard = GCSGGuard(shadow_pool_size=1, check_vram=False)
    worker._tier_manager = mgr
    worker._n_experts_cached = 1
    worker._shadow_pool = {0: lambda hs, layer_id: hs}
    worker._cpu_shadow_pool = {0: lambda hs, layer_id: hs}
    worker._hot_expert_ids = set()   # expert 0 classificato "freddo" — il caso di interesse per Fase 4
    return worker, 0


def bench_dispatch_overhead() -> dict:
    generator = torch.Generator().manual_seed(_SEED)
    worker, expert_id = _build_worker_with_both_pools()
    hidden_states = torch.randn(1, _HIDDEN, generator=generator)
    layer_id = 0

    raw_pool = worker._cpu_shadow_pool   # stesso target fisico di route_forward() nel caso "freddo"

    def _raw_dict_call():
        raw_pool[expert_id](hidden_states, layer_id)

    def _route_forward_call():
        worker.route_forward(expert_id, layer_id, hidden_states)

    routed_pool = _RoutedShadowPool(worker)
    ctx = GatingContext(
        token_id=0, request_id="bench", gating_scores=[0.99], token_entropy=0.01,
    )

    def _run_shadow_full_call():
        worker.guard.run_shadow(ctx, routed_pool, hidden_states=hidden_states, layer_id=layer_id)

    return {
        "raw_dict_call": _time_batched(_raw_dict_call, _N_SAMPLES, _N_INNER, _N_WARMUP_INNER),
        "route_forward": _time_batched(_route_forward_call, _N_SAMPLES, _N_INNER, _N_WARMUP_INNER),
        "run_shadow_full": _time_batched(_run_shadow_full_call, _N_SAMPLES, _N_INNER, _N_WARMUP_INNER),
    }


def bench_compute_reference() -> dict:
    """Stessa metodologia di bench_cpu_kernel.py (Fase 1), ricalcolata qui
    per un confronto pulito nello stesso run — non una citazione di un
    numero misurato in una sessione precedente su un carico macchina
    potenzialmente diverso."""
    generator = torch.Generator().manual_seed(_SEED)
    w13 = torch.randn(2 * _INTERMEDIATE, _HIDDEN, generator=generator)
    w2 = torch.randn(_HIDDEN, _INTERMEDIATE, generator=generator)
    shadow = _ShadowExpertINT4([_quantize_int4(w13)], [_quantize_int4(w2)])
    hidden_states = torch.randn(1, _HIDDEN, generator=generator)

    def _forward_call():
        shadow(hidden_states, layer_id=0)

    return _time_batched(_forward_call, _N_SAMPLES, n_inner=50, n_warmup_inner=10)


def main() -> None:
    dispatch = bench_dispatch_overhead()
    compute = bench_compute_reference()

    compute_p50_us = compute["p50_us"]
    dispatch_p50_us = dispatch["run_shadow_full"]["p50_us"]
    overhead_fraction = (
        dispatch_p50_us / compute_p50_us if compute_p50_us else None
    )

    result = {
        "status": "done",
        "issue": "#33 Fase 4",
        "note": (
            f"dimensioni sintetiche hidden={_HIDDEN} intermediate={_INTERMEDIATE}, "
            "stessa frazione di Mixtral reale usata in bench_cpu_kernel.py (Fase 1)"
        ),
        "dispatch_overhead_us": dispatch,
        "compute_reference_us": compute,
        "verdict": {
            "run_shadow_full_p50_us": dispatch_p50_us,
            "compute_forward_p50_us": compute_p50_us,
            "dispatch_as_fraction_of_compute": overhead_fraction,
            "recommendation": (
                "skip Fase 4 (job ring stile MoeCpuHost non necessario — "
                "l'overhead di dispatch in-process e' trascurabile rispetto "
                "al tempo di compute)"
                if overhead_fraction is not None and overhead_fraction < 0.01
                else "rivalutare — l'overhead di dispatch non e' chiaramente trascurabile"
            ),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
