"""Benchmark — EPM: Expert Position Memory, checkpoint di hotness tra run (issue #27).

Dataset sintetico deliberato, non un vero MMLU/routing reale — nessun
modello, nessuna GPU necessaria. Isola la SOLA domanda che EPM deve
rispondere: "il prior caricato da un run precedente migliora la
selezione A FREDDO del run successivo, prima che un solo token reale sia
stato instradato in quel run?" — dal rumore di un vero benchmark MoE, che
mescolerebbe questa domanda con qualità del modello, quantizzazione, ecc.

Riusa il codice di produzione reale — ExpertAccessTable, TierManager,
GCSGWorker._select_shadow_expert_ids()/finalize_epm_run() — via lo stesso
pattern GCSGWorker.__new__() + attributi assegnati a mano già usato in
tests/test_scheduler.py::TestGCSGTierManagerWiring, non una
reimplementazione della logica di selezione. Non chiama
_epm_capture_initial_position() (legge self._shadow_pool, popolato da
_load_shadow_pool() che carica tensori GPU reali — fuori scope qui):
la selezione "a freddo" è letta direttamente da
_select_shadow_expert_ids(), che è la stessa funzione pura che
_epm_capture_initial_position() incapsula.

Popolarità expert simulata (Zipfian a due livelli, non uniforme —
coerente con quanto osservato per il routing MoE reale): un piccolo
sottoinsieme di expert "caldi" riceve un peso 10x rispetto agli altri.
Gli hot expert sono scelti DELIBERATAMENTE fuori da range(shadow_pool_size)
(cioè non {0, 1}) — altrimenti il baseline round-robin "vincerebbe per
coincidenza" invece che perché è davvero cieco alla hotness.

Sezioni:
    stationary   — hot expert fissi per tutti i run: confronta il match
                   rate della selezione A FREDDO (prima del traffico
                   reale di quel run) tra EPM on/off. Stesso seed RNG tra
                   le due condizioni -> stesso identico traffico simulato,
                   l'unica variabile è se il prior viene caricato.
    regime_shift — a metà dei run gli hot expert cambiano (non-
                   stazionarietà, cautela #2 dell'issue #27): confronta
                   quanto in fretta converge la selezione a freddo con
                   decay di default (0.5) contro decay=1.0 ("nessun
                   decadimento") dopo il cambio.

Usage:
    PYTHONPATH=src python benchmarks/bench_epm.py
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from eat import ExpertAccessTable, Tier
from tier import TierManager
from scheduler import epm
from scheduler.gcsg import GCSGWorker, GCSGGuard

N_EXPERTS = 8       # stesso layout Mixtral 8x7B usato in tutto il progetto
N_LAYERS = 4
POOL_SIZE = 2
HOT_WEIGHT = 10
COLD_WEIGHT = 1


def _seed_structural(eat: ExpertAccessTable) -> None:
    """Equivalente di GCSGWorker._seed_eat_entries() senza un vLLM Worker reale."""
    for expert_id in range(N_EXPERTS):
        for layer_id in range(N_LAYERS):
            eat.insert(expert_id, layer_id, tier=Tier.DDR4, size_bytes=0)


def _make_worker(tier_manager: TierManager) -> GCSGWorker:
    worker = GCSGWorker.__new__(GCSGWorker)
    worker.guard = GCSGGuard(shadow_pool_size=POOL_SIZE, check_vram=False)
    worker._tier_manager = tier_manager
    worker._n_experts_cached = N_EXPERTS
    worker._epm_run_id = None
    worker._epm_initial_selection = None
    worker._shadow_pool = {}
    return worker


def _match_rate(selection: list[int], true_hot: list[int]) -> float:
    return len(set(selection) & set(true_hot)) / len(true_hot)


def _simulate_traffic(eat: ExpertAccessTable, rng: random.Random,
                       weights: list[int], n_tokens: int) -> None:
    for _ in range(n_tokens):
        expert_id = rng.choices(range(N_EXPERTS), weights=weights, k=1)[0]
        layer_id = rng.randrange(N_LAYERS)
        eat.access(expert_id, layer_id)


def _run_one(nvme_dir: str, use_epm: bool, decay: float, prior_snapshot: dict | None,
             weights: list[int], n_tokens: int, rng: random.Random, run_id: str) -> dict:
    eat = ExpertAccessTable(capacity=1000, n_slots=4)
    tier_manager = TierManager(eat=eat, nvme_path=nvme_dir, gpu_device=0)
    _seed_structural(eat)

    if use_epm and prior_snapshot is not None:
        eat.load_snapshot(prior_snapshot, decay=decay)

    worker = _make_worker(tier_manager)
    worker._epm_run_id = run_id
    cold_selection = sorted(worker._select_shadow_expert_ids(N_EXPERTS))
    worker._epm_initial_selection = cold_selection

    _simulate_traffic(eat, rng, weights, n_tokens)

    with tempfile.TemporaryDirectory() as state_dir:
        record = worker.finalize_epm_run(
            snapshot_path=Path(state_dir) / "snap.json",
            history_path=Path(state_dir) / "hist.json",
        )

    return {
        "cold_selection": cold_selection,
        "final_selection": sorted(record["final_selection"]),
        "next_prior": eat.export_snapshot(),
    }


def bench_stationary(n_runs: int = 6, n_tokens: int = 200, decay: float = 0.5, seed: int = 0) -> dict:
    true_hot = [2, 5]
    weights = [HOT_WEIGHT if e in true_hot else COLD_WEIGHT for e in range(N_EXPERTS)]

    out = {}
    for condition, use_epm in (("no_epm", False), ("epm", True)):
        rng = random.Random(seed)   # stesso seed tra le condizioni -> stesso traffico
        prior = None
        runs = []
        with tempfile.TemporaryDirectory() as nvme_dir:
            for run_idx in range(n_runs):
                result = _run_one(
                    nvme_dir, use_epm, decay, prior, weights, n_tokens, rng,
                    run_id=f"{condition}-{run_idx}",
                )
                runs.append({
                    "run": run_idx,
                    "cold_selection": result["cold_selection"],
                    "cold_match_true_hot": _match_rate(result["cold_selection"], true_hot),
                    "final_selection": result["final_selection"],
                })
                if use_epm:
                    prior = result["next_prior"]
        out[condition] = runs

    # media sui run 1..n-1: il run 0 è identico per costruzione (nessun prior
    # ancora esistente in nessuna delle due condizioni) — includerlo
    # diluirebbe il confronto invece di chiarirlo.
    def _avg_from_run1(runs):
        tail = runs[1:]
        return sum(r["cold_match_true_hot"] for r in tail) / len(tail) if tail else None

    out["true_hot"] = true_hot
    out["avg_cold_match_from_run1"] = {
        "no_epm": _avg_from_run1(out["no_epm"]),
        "epm": _avg_from_run1(out["epm"]),
    }
    return out


def bench_regime_shift(n_runs: int = 8, n_tokens: int = 200, seed: int = 0) -> dict:
    shift_at = n_runs // 2
    hot_before = [2, 5]
    hot_after = [3, 6]

    out = {}
    for condition, decay in (("decay_0.5_default", 0.5), ("decay_1.0_no_decay", 1.0)):
        rng = random.Random(seed)
        prior = None
        runs = []
        with tempfile.TemporaryDirectory() as nvme_dir:
            for run_idx in range(n_runs):
                true_hot = hot_before if run_idx < shift_at else hot_after
                weights = [HOT_WEIGHT if e in true_hot else COLD_WEIGHT for e in range(N_EXPERTS)]
                result = _run_one(
                    nvme_dir, True, decay, prior, weights, n_tokens, rng,
                    run_id=f"{condition}-{run_idx}",
                )
                runs.append({
                    "run": run_idx,
                    "phase": "before_shift" if run_idx < shift_at else "after_shift",
                    "cold_selection": result["cold_selection"],
                    "cold_match_current_hot": _match_rate(result["cold_selection"], true_hot),
                })
                prior = result["next_prior"]
        out[condition] = runs

    def _runs_to_converge(runs):
        """Quanti run DOPO lo shift servono prima che la selezione a freddo
        torni a match=1.0 con gli hot expert correnti — None se non converge
        entro la finestra."""
        after = [r for r in runs if r["phase"] == "after_shift"]
        for i, r in enumerate(after):
            if r["cold_match_current_hot"] == 1.0:
                return i
        return None

    out["shift_at_run"] = shift_at
    out["hot_before"] = hot_before
    out["hot_after"] = hot_after
    out["runs_to_reconverge_after_shift"] = {
        condition: _runs_to_converge(runs)
        for condition, runs in out.items() if condition.startswith("decay_")
    }
    return out


def _print_stationary_summary(result: dict) -> None:
    print(f"\n=== stationary — hot expert reali: {result['true_hot']} "
          f"(round-robin di default sarebbe {list(range(POOL_SIZE))}) ===")
    print(f"{'run':>3}  {'no_epm cold':<14} match  {'epm cold':<14} match")
    for r0, r1 in zip(result["no_epm"], result["epm"]):
        print(f"{r0['run']:>3}  {str(r0['cold_selection']):<14} "
              f"{r0['cold_match_true_hot']:>4.0%}  "
              f"{str(r1['cold_selection']):<14} {r1['cold_match_true_hot']:>4.0%}")
    avg = result["avg_cold_match_from_run1"]
    print(f"\nMatch medio a freddo (run 1..N-1, esclude il primo run — identico "
          f"per costruzione): no_epm={avg['no_epm']:.0%}  epm={avg['epm']:.0%}")


def _print_regime_shift_summary(result: dict) -> None:
    print(f"\n=== regime_shift — hot {result['hot_before']} -> {result['hot_after']} "
          f"al run {result['shift_at_run']} ===")
    for condition in ("decay_0.5_default", "decay_1.0_no_decay"):
        print(f"\n  {condition}:")
        for r in result[condition]:
            marker = " <-- shift" if r["run"] == result["shift_at_run"] else ""
            print(f"    run {r['run']} [{r['phase']:<12}] cold={str(r['cold_selection']):<10} "
                  f"match_current_hot={r['cold_match_current_hot']:>4.0%}{marker}")
    print(f"\nRun dopo lo shift necessari per riconvergere a match=100%: "
          f"{result['runs_to_reconverge_after_shift']}")


def main() -> None:
    stationary = bench_stationary()
    regime_shift = bench_regime_shift()

    _print_stationary_summary(stationary)
    _print_regime_shift_summary(regime_shift)

    result = {
        "status": "done",
        "issue": "#27 — EPM",
        "n_experts": N_EXPERTS,
        "n_layers": N_LAYERS,
        "pool_size": POOL_SIZE,
        "hot_weight": HOT_WEIGHT,
        "cold_weight": COLD_WEIGHT,
        "stationary": stationary,
        "regime_shift": regime_shift,
    }
    print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
