#!/usr/bin/env python3
"""GCSGWorker end-to-end smoke test — real Mixtral-8x7B-Instruct-AWQ.

Follow-up to scripts/smoke_test_gcsg_worker.py (hf-internal-testing/
Mixtral-tiny), which validated wiring mechanics only. This one runs the
actual model the whole GCSG memory math / expert_map / paper is about.

Checkpoint swapped 2026-08-09 (see LOGBOOK): `TheBloke/Mixtral-8x7B-
Instruct-v0.1-AWQ` is a known-bad quantization (vllm-project/vllm#2359,
filed 2024-01-05 — same NaN-on-generate symptom, never fixed for this
specific file) — confirmed directly on this exact stack, not just by
citation: identical config (vLLM 0.6.6.post1, awq_marlin,
cpu_offload_gb=4, hf_overrides head_dim=128) produces NaN on TheBloke's
file and clean text on `casperhansen/mixtral-instruct-awq`, nothing else
changed. This script now points at the working checkpoint.

VRAM: weights load at ~18.83GiB in this Marlin-repacked form, comfortably
under the 21.6GiB budget (gpu_memory_utilization=0.90 x 24GiB) — offload
isn't strictly needed for VRAM here, but cpu_offload_gb=4 is kept anyway:
removing it hits a separate, confirmed real bug (Marlin repacking hangs
without the scratch-space headroom offload provides, independent of
checkpoint — see LOGBOOK, needs its own issue) that has nothing to do
with GCSG and isn't worth re-triggering here.

quantization="awq_marlin" (not "awq"): the plain "awq" path loads Mixtral
through vllm.model_executor.models.mixtral_quant.MixtralMoE, which stores
each expert as a separate MixtralMLP module (w1/w2/w3 individually AWQ-
packed) in a ModuleList — a completely different structure from the
FusedMoE/w13_weight layout _load_shadow_pool() was written and verified
against (on the unquantized tiny model). "awq_marlin" restores the same
mixtral.MixtralMoE/FusedMoE structure (num_experts attribute present, same
navigation path) — confirmed by inspection before switching to it — but the
weight tensors themselves are Marlin-packed (w13_qweight/w13_scales/
w13_qzeros), not the plain w13_weight fp16 tensors _load_shadow_pool()
reads on the unquantized path. _load_shadow_pool() dispatches on
hasattr(experts, "num_experts") and delegates to the real quantized module
(_AWQShadowExpert) for this path instead of extracting weights by hand —
this is the first time that code runs against a checkpoint whose
generate() actually works, so it's the first real end-to-end check of it.

Watchdog: a hard 900s timeout + 30s heartbeat, same pattern used to debug
the NaN issue — this is GCSGWorker's first real run against a healthy
checkpoint, and shadow-pool loading / hook registration has never been
exercised end-to-end here before, so a silent hang shouldn't cost another
30 minutes of guessing where it stalled.

Usage:
    PYTHONPATH=src python scripts/smoke_test_gcsg_mixtral8x7b.py
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

from vllm import LLM, SamplingParams

from scheduler.gcsg import GCSGWorker

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"

START = time.monotonic()


def _elapsed() -> str:
    return f"T+{time.monotonic() - START:6.1f}s"


def _log(msg: str) -> None:
    print(f"[{_elapsed()}] {msg}", flush=True)


def _watchdog(timeout: float = 900.0) -> None:
    time.sleep(timeout)
    _log(f"WATCHDOG: timeout di {timeout:.0f}s raggiunto — nessun completamento. Invio SIGTERM.")
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(5)
    _log("WATCHDOG: SIGTERM non ha terminato il processo entro 5s — invio SIGKILL.")
    try:
        os.kill(os.getpid(), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _heartbeat(interval: float = 30.0) -> None:
    while True:
        time.sleep(interval)
        _log("heartbeat — processo ancora vivo, in attesa del prossimo log")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    threading.Thread(target=_watchdog, args=(900.0,), daemon=True).start()
    threading.Thread(target=_heartbeat, args=(30.0,), daemon=True).start()

    _log(f"Loading {MODEL_PATH} via EngineArgs(worker_cls=GCSGWorker), "
         f"quantization=awq_marlin, cpu_offload_gb=4 ...")
    llm = LLM(
        model=MODEL_PATH,
        worker_cls="scheduler.gcsg.GCSGWorker",
        quantization="awq_marlin",
        cpu_offload_gb=4,
        # gpu_memory_utilization=0.95, max_num_seqs=16, max_model_len=3328 —
        # SOLO per ambienti di test/validazione, non di produzione (issue
        # #10/#16). Con shadow pool pinnato in GPU (~1.02-1.05GiB sempre
        # residenti), gpu_memory_utilization=0.90 (il vecchio default, dai
        # tempi in cui lo shadow pool era in hook-only) non lascia budget
        # sufficiente per la KV-cache — verificato empiricamente
        # (scripts/probe_kv_blocks.py, 2026-08-10): stessi valori usati in
        # eval_mmlu_gcsg.py per coerenza, vedi lì per il dettaglio del calcolo.
        gpu_memory_utilization=0.95,
        max_num_seqs=16,
        enforce_eager=True,
        max_model_len=3328,
        hf_overrides={"head_dim": 128},
    )
    _log("load_model() completed — GCSGWorker attached, real Mixtral-8x7B loaded. [checklist 1-3 OK]")

    worker = llm.llm_engine.model_executor.driver_worker
    if not isinstance(worker, GCSGWorker):
        _fail(f"driver_worker is {type(worker)}, not GCSGWorker — worker_cls not honored")

    cache_config = llm.llm_engine.cache_config
    num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
    print(f"KV-cache blocks: num_gpu_blocks={num_gpu_blocks}")
    if not num_gpu_blocks or num_gpu_blocks <= 0:
        _fail(f"num_gpu_blocks={num_gpu_blocks} — no KV-cache space")
    print("KV-cache blocks > 0 confirmed. [checklist step 3 OK]")

    # Mixtral-8x7B-Instruct expects the Llama2-style [INST] ... [/INST]
    # chat wrapper — raw completion prompts on an Instruct-tuned model
    # commonly produce empty/degenerate output, which is what the first
    # run of this script saw on all 3 raw prompts.
    prompts = [
        "[INST] What is 2+2? [/INST]",
        "[INST] Write a one-line Python function that reverses a string. [/INST]",
        "[INST] Explain quantum entanglement in one sentence. [/INST]",
    ]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=32, temperature=0.0))
    for o in outputs:
        print(f"  prompt={o.prompt!r} -> {o.outputs[0].text!r}")
    non_empty = sum(1 for o in outputs if o.outputs[0].text.strip())
    print(f"generate() completed, {len(outputs)} outputs, {non_empty}/{len(outputs)} non-empty.")

    print(f"\n.gate hooks fired {len(worker.captured_router_logits)} times.")
    print(f"Real request_ids seen: {worker.seen_request_ids}")
    guard_stats = worker.guard.stats()
    print(f"GCSGGuard stats: {guard_stats}")
    print(f"Shadow pool loaded: expert(s) {sorted(worker._shadow_pool.keys())}.")

    print("\nCHECKLIST 1-3: GREEN — real Mixtral-8x7B loads and runs under GCSGWorker.")


if __name__ == "__main__":
    main()
