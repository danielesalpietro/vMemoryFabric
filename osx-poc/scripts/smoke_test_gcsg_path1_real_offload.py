#!/usr/bin/env python3
"""GCSGWorker path 1 (`_ShadowExpertINT4`) under REAL offload — Sprint 4
sotto-obiettivo 6, issue #17.

The one shadow path never exercised beyond the tiny, non-offloaded test
model (GCSG report §7 — "Path 1 (`_ShadowExpertINT4`) untested under
offload"). Paths 2 (Marlin) and 3 (AWQ ModuleList) both got real,
hardware-verified runs this sprint; path 1 hasn't, because it only
triggers on a checkpoint vLLM loads with RAW fp16 FusedMoE weights
(`w13_weight`) — i.e. NOT pre-quantized. The real Mixtral-8x7B checkpoint
this project otherwise uses is always AWQ-quantized; to hit path 1 for
real, this script loads the original, unquantized
`mistralai/Mixtral-8x7B-Instruct-v0.1` instead.

WHY THIS IS A BIGGER ASK THAN ANY PRIOR SMOKE TEST IN THIS PROJECT:

    - Checkpoint size: ~93GB (46.7B real parameters at fp16 — NOT
      8x7B=56B, Mixtral's non-expert layers are shared across experts).
      ~4x the size of the AWQ checkpoint (~23GB) every other script in
      this repo downloads. First download of this specific checkpoint in
      this project — budget real time for it, no established baseline
      timing exists yet (the AWQ checkpoint took 38s-88s depending on
      datacenter; this one has no precedent).
    - cpu_offload_gb: on a 24GB GPU, roughly ~75-80GB of those 93GB must
      be offloaded to host RAM — an order of magnitude more than the
      cpu_offload_gb=4 used everywhere else in this project. Host RAM is
      not the constraint (pod ~125GB, Z8 256GB DDR4, both comfortably
      above ~80GB) — GPU VRAM budget is. DO NOT guess this value and
      launch straight into a full run: use scripts/probe_kv_blocks.py
      first (extended today specifically for this — --model-path,
      --cpu-offload-gb, --quantization none) to find a cpu_offload_gb
      that leaves a positive, workable KV-cache budget, THEN pass that
      value here via --cpu-offload-gb. The default below (78) is an
      ESTIMATE from the arithmetic above, not a measured value — nobody
      has run this yet.
    - Expected slowness: this project's own Root Cause II finding
      (GCSG report §5) showed cpu_offload_gb 4->8 alone causing a 9x
      slowdown under WSL2, and pin_memory=True (available here, real
      Linux) mitigates but doesn't eliminate the cost of a CPU->GPU
      swap-in on every offloaded-layer forward pass. At ~78GB offloaded
      (~20x the cpu_offload_gb value that produced that 9x number), a
      MUCH larger slowdown is plausible and NOT itself a failure signal
      — same "slow != hung" discipline as every offload-related
      investigation in this project. The watchdog below is generous
      specifically because of this; heartbeat logging is there so a long
      wait is visibly still making progress, not silently stuck.

Checklist mechanized, in priority order:

    1. load_model() completes with the unquantized checkpoint + heavy
       offload — is_fused check in _load_shadow_pool() correctly
       dispatches to path 1 (hasattr(experts, "num_experts") True,
       hasattr(experts, "w13_qweight") False -> NOT Marlin -> raw
       w13_weight path).
    2. Shadow pool populated via _ShadowExpertINT4 — INT4-simulated
       quantize/dequantize (_quantize_int4) actually runs against REAL
       Mixtral-8x7B weight tensors (hidden_size=4096) for the first time
       ever — previously only verified at the tiny test model's
       hidden_size=1024.
    3. generate() completes (however slowly) and produces non-empty,
       non-degenerate output — NOT a quality/MMLU claim, just "the
       offloaded-plus-shadow forward pass produces real text."
    4. Gate hooks fire, shadow activations happen (or the mechanism is at
       least exercised — a legitimately low activation rate on a handful
       of short prompts is not a failure by itself, see
       smoke_test_gcsg_worker.py's own note on this).
    5. Direct numerical check of _ShadowExpertINT4's forward output at
       REAL dimensions (hidden_size=4096, not 1024) — finite, correct
       shape. This is the actual "does the INT4 quantize/dequantize/SwiGLU
       math hold up at real scale" proof, independent of whether
       should_activate_shadow() happened to fire during generate().

NOT covered here: MMLU quality, performance characterization beyond "did
it finish", any claim about whether ~78GB offload is remotely production-
viable (it almost certainly isn't — this is a correctness/mechanics
check, not a deployment recommendation).

Usage:
    # 1. Find a working cpu_offload_gb first (fast, no eval, no shadow pool):
    PYTHONPATH=src python scripts/probe_kv_blocks.py \\
        --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --quantization none --cpu-offload-gb 78 --max-num-seqs 1 --max-model-len 512

    # 2. Then run this script with whatever value from step 1 actually worked:
    PYTHONPATH=src python scripts/smoke_test_gcsg_path1_real_offload.py --cpu-offload-gb 78
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

START = time.monotonic()


def _elapsed() -> str:
    return f"T+{time.monotonic() - START:6.1f}s"


def _log(msg: str) -> None:
    print(f"[{_elapsed()}] {msg}", flush=True)


def _watchdog(timeout: float) -> None:
    time.sleep(timeout)
    _log(f"WATCHDOG: {timeout:.0f}s timeout reached — no completion. Sending SIGTERM.")
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(10)
    _log("WATCHDOG: SIGTERM didn't stop the process within 10s — sending SIGKILL.")
    try:
        os.kill(os.getpid(), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _heartbeat(interval: float = 30.0) -> None:
    while True:
        time.sleep(interval)
        _log("heartbeat — process still alive, waiting on the next log line "
             "(expected to be slow here — see module docstring)")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-offload-gb", type=float, default=78,
                         help="ESTIMATE, not measured — run probe_kv_blocks.py first, see module docstring")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=512,
                         help="deliberately small — this run's point is mechanics, not throughput")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--watchdog-s", type=float, default=3600.0,
                         help="generous default — see module docstring on why heavy offload is expected to be slow")
    args = parser.parse_args()

    threading.Thread(target=_watchdog, args=(args.watchdog_s,), daemon=True).start()
    threading.Thread(target=_heartbeat, args=(30.0,), daemon=True).start()

    _log("Import vllm...")
    from vllm import LLM, SamplingParams

    from scheduler.gcsg import GCSGWorker

    _log(f"Loading {MODEL} — quantization=None (raw fp16, path 1), "
         f"cpu_offload_gb={args.cpu_offload_gb}, "
         f"gpu_memory_utilization={args.gpu_memory_utilization} ...")
    _log("This is a ~93GB checkpoint — first download in this project will take real time.")
    llm = LLM(
        model=MODEL,
        worker_cls="scheduler.gcsg.GCSGWorker",
        quantization=None,
        cpu_offload_gb=args.cpu_offload_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        hf_overrides={"head_dim": 128},
    )
    _log("load_model() completed. [checklist item 1 OK]")

    worker = llm.llm_engine.model_executor.driver_worker
    if not isinstance(worker, GCSGWorker):
        _fail(f"driver_worker is {type(worker)}, not GCSGWorker — worker_cls not honored")

    if not worker._shadow_pool:
        _fail(
            "worker._shadow_pool is empty — _load_shadow_pool() didn't populate path 1. "
            "Check for a 'shadow pool non caricato' warning above (VRAM preflight in "
            "GCSGGuard._check_vram_budget can legitimately reduce/refuse the pool if "
            "cpu_offload_gb left too little headroom — see the memory math in gcsg.py's "
            "module docstring)."
        )
    print(f"Shadow pool loaded via path 1: expert(s) {sorted(worker._shadow_pool.keys())}. "
          f"[checklist item 2 OK]")

    prompts = ["[INST] What is 2+2? [/INST]"]
    _log(f"Running generate() on {len(prompts)} short prompt(s), max_tokens=8 — expect this "
         f"to be slow (heavy offload), that alone is not a failure.")
    outputs = llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0))
    for o in outputs:
        print(f"  prompt={o.prompt!r} -> {o.outputs[0].text!r}")
    non_empty = sum(1 for o in outputs if o.outputs[0].text.strip())
    if non_empty == 0:
        _fail("generate() completed but produced only empty output on every prompt — "
              "degenerate output (same symptom class as the TheBloke checkpoint's NaN bug, "
              "2026-08-09 — not assumed to be the same cause, but worth treating with the "
              "same suspicion, not waved off).")
    print(f"generate() completed, {non_empty}/{len(outputs)} non-empty. [checklist item 3 OK]")

    n_captured = len(worker.captured_router_logits)
    guard_stats = worker.guard.stats()
    print(f"\n.gate hooks fired {n_captured} times.")
    print(f"GCSGGuard stats: {guard_stats}")
    if n_captured == 0:
        _fail("captured_router_logits is empty — .gate hooks never fired at all "
              "(different from a low/zero activation RATE, which alone would be fine).")
    print(f"Gate hooks fired for real. Shadow activations this run: "
          f"{guard_stats['shadow_activations']} (activation_rate={guard_stats['activation_rate']:.1%} "
          f"— low or zero on a single short prompt is not a pass/fail signal by itself, "
          f"same caveat as smoke_test_gcsg_worker.py). [checklist item 4 OK]")

    # ── checklist item 5: direct numerical proof at REAL dimensions ─────────
    import torch

    any_expert_id = next(iter(worker._shadow_pool))
    first_layer_experts = worker.model_runner.model.model.layers[0].block_sparse_moe.experts
    if not hasattr(first_layer_experts, "w13_weight"):
        _fail(
            f"layer 0's experts module has no w13_weight ({type(first_layer_experts)}) — "
            "the shadow pool populated via SOME path, but not the raw-fp16 FusedMoE path "
            "(path 1) this script exists to check. Did the checkpoint actually load "
            "unquantized? quantization=None was passed explicitly above."
        )
    hidden_size = first_layer_experts.w13_weight.shape[-1]
    dummy_hidden = torch.randn(hidden_size, dtype=torch.float16, device="cuda")
    shadow_output = worker._shadow_pool[any_expert_id](dummy_hidden, layer_id=0)
    if shadow_output.shape != dummy_hidden.shape:
        _fail(f"shadow expert output shape {tuple(shadow_output.shape)} != "
              f"input shape {tuple(dummy_hidden.shape)}")
    if not torch.isfinite(shadow_output).all():
        _fail("shadow expert output contains NaN/Inf — INT4 quantize/dequantize or SwiGLU "
              "math breaks at real Mixtral-8x7B dimensions (hidden_size="
              f"{hidden_size}), even though it was verified at the tiny model's "
              "hidden_size=1024 (2026-08-09). Do not assume the tiny-model verification "
              "generalizes — this is exactly the gap this script exists to check.")
    print(f"\nDirect shadow-expert forward at REAL scale (expert {any_expert_id}, "
          f"hidden_size={hidden_size}): output shape {tuple(shadow_output.shape)}, finite, "
          f"sample values {shadow_output[:3].tolist()}. [checklist item 5 OK]")

    print("\nSMOKE TEST: GREEN — path 1 (_ShadowExpertINT4) verified end-to-end under real "
          f"offload (cpu_offload_gb={args.cpu_offload_gb}) on real Mixtral-8x7B dimensions, "
          "not just the tiny test model.")
    print("NOT verified here: MMLU quality on this path, whether this offload configuration "
          "is remotely production-viable, performance beyond 'did it finish'.")


if __name__ == "__main__":
    main()
