#!/usr/bin/env python3
"""GCSGWorker <-> TierManager/EAT wiring — end-to-end verification checklist
(issue #17, sub-goal 1, implemented 2026-08-12).

Companion to scripts/smoke_test_gcsg_worker.py and
scripts/smoke_test_gcsg_mixtral8x7b.py, which validate GCSGWorker's own
mechanics/quality but never construct a TierManager at all. This script
is the FIRST thing to run against real hardware for the new
TierManager/EAT wiring added in src/scheduler/gcsg.py — everything it
checks was only unit-tested with fakes/CPU tensors (91 passed / 18
skipped, PYTHONPATH=src python -m pytest tests/, see LOGBOOK.md
2026-08-12) before this.

--quantization: both "awq" (path 3, AWQ ModuleList — default) and
"awq_marlin" (path 2, Marlin-packed FusedMoE) are wired through
TierManager as of 2026-08-12. AWQ was wired and verified first
(deliberately — Marlin is the more fragile mechanism in gcsg.py, see
_PinnedMarlinExperts' ATTENZIONE docstring on a real CUDA-allocator
fragmentation hang found 2026-08-10); Marlin was wired later the same
day, after AWQ passed this exact checklist twice on real hardware with
perfect determinism. Loading the SAME checkpoint
(casperhansen/mixtral-instruct-awq) with plain "awq" instead of
"awq_marlin" forces vLLM through vllm.model_executor.models.mixtral_quant
.MixtralMoE (ModuleList of MixtralMLP, one module per expert) instead of
FusedMoE+Marlin-packed tensors — "awq" is slower (no Marlin kernel),
that's expected, not a regression to chase. Run this script with BOTH
values — they exercise different code paths
(_promote_module_via_tier_manager vs. _build_marlin_tensor_promoter) and
neither result substitutes for the other.

Checklist mechanized here, in the same priority order as LOGBOOK.md's
"NOT run on real hardware" list for this feature:

    1. asyncio.run() inside _promote_module_via_tier_manager(), called
       from load_model() (sync, inside vLLM's real worker process) does
       not raise ("asyncio.run() cannot be called from a running event
       loop" or similar) — if load_model() below completes at all with
       tier_manager wired, this already passed.
    2. The real .to('cuda')/pin_memory() transfer via
       TierManager.promote_live_tensor() completes and the EAT entries
       for the promoted expert(s) actually show Tier.VRAM afterwards —
       not just "didn't crash".
    3. Whether a real per-layer AWQ expert's dominant parameter fits
       under SHARD_SIZE_BYTES (256MB) — surfaced by whether
       worker._shadow_pool actually contains the expected expert_ids;
       if EAT.insert() rejected a shard as too large,
       _pin_awq_expert_to_gpu() already degrades safely (excludes that
       expert_id, logs a warning) rather than crashing, so watch the
       log for "impossibile pinnare" as well as the assertion below.
    4. Real per-token EAT traffic (_evaluate_gcsg_for_rows' new
       EAT.access() call) actually accumulates during generate().
    5. refresh_shadow_pool_selection() is callable post-generate()
       without raising, and reflects whatever hotness accumulated.
    6. EPM (issue #27, 2026-08-13): on by default — a prior snapshot from
       a file scoped to MODEL_PATH (epm.snapshot_path_for_model(), e.g.
       state/epm_eat_snapshot__*.json — different model -> different
       file, no snapshot found -> clean cold start, no content-level
       validation needed) is loaded via GCSGWorker.configure_eat_snapshot()
       before LLM(...), and
       worker.finalize_epm_run() writes this run's snapshot + a history
       record (initial/final shadow pool position, continued_from_
       previous) after the checklist above. Two consecutive runs of this
       script (same --quantization) are the actual verification: the
       second run's "initial position" printed below should match the
       first run's "final position" — continued_from_previous=True is
       that check made explicit. --no-epm disables both load and save,
       forcing every run back to the round-robin cold start.

NOT covered here (needs a separate, larger run): a real MMLU comparison
against the existing 72.28%/72.3% baseline with tier_manager wired —
LOGBOOK.md's priority item 4. Do that only after this script is green.

Usage:
    PYTHONPATH=src python scripts/smoke_test_gcsg_tier_manager.py  # path 3, AWQ
    PYTHONPATH=src python scripts/smoke_test_gcsg_tier_manager.py --quantization awq_marlin  # path 2
    PYTHONPATH=src python scripts/smoke_test_gcsg_tier_manager.py --no-epm  # sempre cold start
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

from eat import ExpertAccessTable, Tier
from tier import TierManager
from vllm import LLM, SamplingParams

from scheduler import epm
from scheduler.gcsg import GCSGWorker

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"

START = time.monotonic()


def _elapsed() -> str:
    return f"T+{time.monotonic() - START:6.1f}s"


def _log(msg: str) -> None:
    print(f"[{_elapsed()}] {msg}", flush=True)


def _watchdog(timeout: float = 1200.0) -> None:
    # Higher than smoke_test_gcsg_mixtral8x7b.py's 900s: plain "awq" has
    # no Marlin kernel, and this is the first time this exact path has
    # ever run against the real checkpoint — no prior timing to anchor a
    # tighter bound on.
    time.sleep(timeout)
    _log(f"WATCHDOG: {timeout:.0f}s timeout reached — no completion. Sending SIGTERM.")
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(5)
    _log("WATCHDOG: SIGTERM didn't stop the process within 5s — sending SIGKILL.")
    try:
        os.kill(os.getpid(), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _heartbeat(interval: float = 30.0) -> None:
    while True:
        time.sleep(interval)
        _log("heartbeat — process still alive, waiting on the next log line")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", choices=["awq", "awq_marlin"], default="awq",
                         help="path 3 (AWQ ModuleList) or path 2 (Marlin) — see module docstring")
    parser.add_argument("--no-epm", action="store_true",
                         help="Disabilita EPM (issue #27): non carica lo snapshot di hotness "
                              "del run precedente (sempre cold start round-robin) e non salva "
                              "snapshot/storico di questo run. EPM è attivo di default.")
    args = parser.parse_args()

    threading.Thread(target=_watchdog, args=(1200.0,), daemon=True).start()
    threading.Thread(target=_heartbeat, args=(30.0,), daemon=True).start()

    _log("Building EAT + TierManager (real instances, no GPU touched yet)...")
    eat = ExpertAccessTable(capacity=1000, n_slots=4)
    tier_manager = TierManager(eat=eat, nvme_path="/data/nvme", gpu_device=0)
    GCSGWorker.configure_tier_manager(tier_manager)
    _log("TierManager wired via GCSGWorker.configure_tier_manager() — vLLM will pick it "
         "up when it constructs the worker itself (no direct kwarg path exists, see the "
         "docstring on configure_tier_manager()/_pending_tier_manager in gcsg.py).")

    # Path scoped su MODEL_PATH (issue #27): expert_id è solo un intero
    # posizionale, un prior di un modello diverso applicato alla cieca
    # sarebbe un bias silenzioso, non un cold start pulito — cambiare
    # modello punta automaticamente a un file diverso, vedi la nota di
    # modulo in scheduler/epm.py.
    epm_snapshot_path = epm.snapshot_path_for_model(MODEL_PATH)
    epm_history_path = epm.history_path_for_model(MODEL_PATH)

    if args.no_epm:
        GCSGWorker.configure_eat_snapshot(None)
        _log("EPM disabilitato (--no-epm): cold start round-robin, nessuno snapshot/storico salvato.")
    else:
        prior = epm.load_snapshot_file(epm_snapshot_path)
        GCSGWorker.configure_eat_snapshot(prior)
        _log(f"EPM: {'snapshot precedente caricato da ' + str(epm_snapshot_path) if prior else 'nessuno snapshot trovato in ' + str(epm_snapshot_path) + ' — cold start'}.")

    _log(f"Loading {MODEL_PATH} via EngineArgs(worker_cls=GCSGWorker), "
         f"quantization={args.quantization}, cpu_offload_gb=4 ...")
    llm = LLM(
        model=MODEL_PATH,
        worker_cls="scheduler.gcsg.GCSGWorker",
        quantization=args.quantization,
        cpu_offload_gb=4,
        gpu_memory_utilization=0.95,
        max_num_seqs=16,
        enforce_eager=True,
        max_model_len=3328,
        hf_overrides={"head_dim": 128},
    )
    _log("load_model() completed with tier_manager wired — checklist item 1 (asyncio.run() "
         "bridging inside load_model() didn't raise) PASSES by construction: we got here.")

    worker = llm.llm_engine.model_executor.driver_worker
    if not isinstance(worker, GCSGWorker):
        _fail(f"driver_worker is {type(worker)}, not GCSGWorker — worker_cls not honored")
    if worker._tier_manager is not tier_manager:
        _fail(
            "worker._tier_manager is not the instance passed to configure_tier_manager() — "
            "the pending-config handoff didn't work. Check GCSGWorker.__init__'s fallback "
            "logic and that configure_tier_manager() ran before LLM(...) above."
        )
    print("worker._tier_manager is the real TierManager instance. [checklist wiring OK]")

    # ── checklist 2/3: EAT seeded, shadow pool promoted to real VRAM ────────

    all_entries = eat.get_tier(Tier.DDR4) + eat.get_tier(Tier.VRAM)
    n_experts_in_pool = len(worker._shadow_pool)
    print(f"\nEAT entries after load_model(): {len(all_entries)} total "
          f"({len(eat.get_tier(Tier.DDR4))} DDR4, {len(eat.get_tier(Tier.VRAM))} VRAM).")
    if not all_entries:
        _fail("EAT has zero entries after load_model() — _seed_eat_entries() didn't run "
              "or found nothing; check for a 'seed EAT fallito' warning above.")
    print("EAT seeded with at least one entry. [checklist item 2 partial: seeding OK]")

    shadow_expert_ids = sorted(worker._shadow_pool.keys())
    print(f"Shadow pool populated: expert(s) {shadow_expert_ids} "
          f"(shadow_pool_size configured: {worker.guard.shadow_pool_size}).")
    if not shadow_expert_ids:
        _fail(
            "worker._shadow_pool is empty — either _pin_awq_expert_to_gpu() failed for "
            "every candidate expert (check for 'impossibile pinnare' warnings above — this "
            "is checklist item 3, SHARD_SIZE_BYTES may be too small for a real dominant "
            "parameter) or _select_shadow_expert_ids() returned nothing."
        )

    n_layers = len(worker.model_runner.model.model.layers)
    # AWQ path (path 3): promotion is tracked per real expert_id.
    # Marlin path (path 2): the proxy is shared by the whole pool per
    # layer, tracked under a sentinel expert_id=-1 (see
    # GCSGWorker._marlin_pool_shard_key() for why a real expert_id
    # can't be used there) — check both, whichever the loaded
    # quantization actually populated.
    vram_promoted = [
        (e, layer_id) for e in shadow_expert_ids
        for layer_id in range(n_layers)
        if (entry := eat.lookup(e, layer_id)) is not None and entry.tier == Tier.VRAM
    ]
    vram_promoted_marlin_sentinel = [
        entry for entry in eat.get_tier(Tier.VRAM) if entry.expert_id == -1
    ]
    if not vram_promoted and not vram_promoted_marlin_sentinel:
        _fail(
            "Neither real-expert-id nor sentinel(-1) entries reached Tier.VRAM in EAT — "
            "the real .to('cuda')/pin_memory() transfer via TierManager.promote_live_tensor() "
            "either didn't run or didn't update EAT's tier. This is checklist item 2's real "
            "check — 'load_model() didn't crash' alone (above) is not enough evidence."
        )
    if vram_promoted_marlin_sentinel:
        print(f"Marlin-path sentinel entries (expert_id=-1) reached Tier.VRAM: "
              f"{len(vram_promoted_marlin_sentinel)} confirmed.")
    print(f"At least one (expert_id, layer_id) pair reached Tier.VRAM via TierManager: "
          f"{len(vram_promoted)} confirmed promotions. [checklist item 2 OK — real transfer verified]")

    # ── checklist 4: real per-token EAT traffic during generate() ───────────

    prompts = [
        "[INST] What is 2+2? [/INST]",
        "[INST] Write a one-line Python function that reverses a string. [/INST]",
        "[INST] Explain quantum entanglement in one sentence. [/INST]",
    ]
    _log("Running generate()...")
    outputs = llm.generate(prompts, SamplingParams(max_tokens=32, temperature=0.0))
    for o in outputs:
        print(f"  prompt={o.prompt!r} -> {o.outputs[0].text!r}")
    non_empty = sum(1 for o in outputs if o.outputs[0].text.strip())
    print(f"generate() completed, {len(outputs)} outputs, {non_empty}/{len(outputs)} non-empty.")

    trafficked = [e for e in (eat.get_tier(Tier.DDR4) + eat.get_tier(Tier.VRAM)) if e.access_count > 0]
    print(f"\nEAT entries with real routing traffic (access_count > 0) after generate(): "
          f"{len(trafficked)} / {len(all_entries)}.")
    if not trafficked:
        _fail(
            "Zero EAT entries show access_count > 0 after generate() — "
            "_evaluate_gcsg_for_rows()'s new EAT.access() call on the real top-1 routed "
            "expert either didn't fire or isn't reaching the real EAT instance."
        )
    print("Real per-token EAT traffic confirmed. [checklist item 4 OK]")

    guard_stats = worker.guard.stats()
    print(f"\nGCSGGuard stats: {guard_stats}")

    # ── checklist 5: refresh_shadow_pool_selection() callable post-traffic ──

    pool_before = sorted(worker._shadow_pool.keys())
    try:
        worker.refresh_shadow_pool_selection()
    except Exception as e:
        _fail(f"refresh_shadow_pool_selection() raised: {e!r}")
    pool_after = sorted(worker._shadow_pool.keys())
    print(f"\nrefresh_shadow_pool_selection() ran without raising. "
          f"Pool before: {pool_before}, after: {pool_after} "
          f"({'changed' if pool_before != pool_after else 'unchanged'} — either is fine on "
          f"3 short prompts, this only confirms the call path itself works).")
    print("[checklist item 5 OK]")

    # ── checklist 6: EPM — snapshot + storico posizioni (issue #27) ─────────

    if args.no_epm:
        print("\nEPM disabilitato (--no-epm) — checklist item 6 skipped.")
    else:
        print(f"\nEPM: posizione iniziale di questo run (subito dopo load_model(), prima di "
              f"generate()): {worker._epm_initial_selection}.")
        record = worker.finalize_epm_run(snapshot_path=epm_snapshot_path, history_path=epm_history_path)
        if record is None:
            _fail("finalize_epm_run() ha restituito None con tier_manager wired e "
                  "_n_experts_cached impostato — non dovrebbe succedere qui.")
        print(f"EPM: run finalizzato — {record}")
        print(f"EPM: snapshot scritto in {epm_snapshot_path}, storico in "
              f"{epm_history_path} (ultimi {epm.MAX_HISTORY_RUNS} run).")
        if record["continued_from_previous"]:
            print("EPM: continued_from_previous=True — la posizione iniziale di QUESTO run "
                  "coincide con la posizione finale dell'ULTIMO run nello storico: la memoria "
                  "è stata usata per davvero, non solo caricata senza effetto.")
        else:
            print("EPM: continued_from_previous=False — o questo è il primo run "
                  "(niente storico precedente), oppure il prior caricato non ha determinato "
                  "la stessa selezione del run precedente (verificabile solo eseguendo "
                  "questo script due volte di fila con lo stesso --quantization).")
        print("[checklist item 6 OK]")

    print(f"\nSMOKE TEST: GREEN — TierManager/EAT wiring verified end-to-end on real "
          f"hardware for the '{args.quantization}' path (asyncio bridging, real GPU "
          f"transfer + EAT tier update, real routing traffic, refresh callable).")
    other_path = "AWQ (path 3)" if args.quantization == "awq_marlin" else "Marlin (path 2)"
    print(f"NOT verified here: MMLU quality on this path (LOGBOOK.md priority item 4 — run "
          f"separately, compare against the 72.28%/72.3% baseline), and the {other_path} "
          f"path (this run only exercises '{args.quantization}').")


if __name__ == "__main__":
    main()
