#!/usr/bin/env python3
"""GCSGWorker end-to-end smoke test — hf-internal-testing/Mixtral-tiny.

NOT Mixtral-8x7B. This validates wiring MECHANICS only:
    1. GCSGWorker attaches to a real vLLM engine and load_model() completes
    2. The .gate forward hooks fire and capture router_logits, one per layer
       per forward pass, with the expected (num_tokens, num_local_experts) shape
    3. Real request_ids are accessible inside execute_model() (via
       ExecuteModelRequest.seq_group_metadata_list — see note below)
    4. GCSGGuard's per-request contamination bookkeeping works end-to-end when
       driven by real request_ids from vLLM

It does NOT measure quality (MMLU), performance, or real shadow-pool VRAM
behavior — see GCSGWorker's docstring (src/scheduler/gcsg.py) and LOGBOOK
2026-08-09 for why: the real Mixtral-8x7B AWQ checkpoint (~23 GiB) is too
tight against the 3090's 24 GiB to be a fair mechanics test, so this uses a
same-architecture tiny model instead. Those results stay pending on the real
model.

Two environment quirks specific to this tiny model, both worked around here
rather than in GCSGWorker itself (real Mixtral-8x7B doesn't need either):
  - FlashAttention (vLLM's default backend) crashes on this model's shapes
    with "cu_seqlens_q must have dtype int32". Forcing XFormers here sidesteps
    it — this is about the tiny model's unusual dimensions, not GCSGWorker.
  - `hf_overrides={"head_dim": 32}` works around a real transformers/vLLM
    pin-combo gap (see the LLM(...) call below) that DOES also affect real
    Mixtral-8x7B — not tiny-model-specific, flagged here so it isn't lost.

Usage:
    PYTHONPATH=src python scripts/smoke_test_gcsg_worker.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

from vllm import LLM, SamplingParams

from scheduler.gcsg import GCSGWorker

MODEL = "hf-internal-testing/Mixtral-tiny"
EXPECTED_LAYERS = 2
EXPECTED_EXPERTS = 8


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    print(f"Loading {MODEL} via EngineArgs(worker_cls=GCSGWorker) ...")
    llm = LLM(
        model=MODEL,
        # hf-internal-testing/Mixtral-tiny ships no tokenizer files at all, and
        # vLLM's auto-resolution for its `mixtral` model_type falls back to a
        # MistralCommonTokenizer that errors on kwargs this transformers/vLLM
        # pairing passes. Vocab size matches (32000) — swap in a real,
        # standalone tokenizer repo. Irrelevant to what this smoke test checks
        # (MoE routing mechanics), so any compatible tokenizer is fine here.
        tokenizer="hf-internal-testing/llama-tokenizer",
        # worker_cls does NOT accept a class object in this vLLM version —
        # vllm.worker.worker_base.init_worker() does
        # resolve_obj_by_qualname(qualname).rsplit(".", 1), i.e. it expects a
        # "module.ClassName" string it imports and resolves itself. Passing
        # the GCSGWorker class directly crashes with
        # AttributeError: type object 'GCSGWorker' has no attribute 'rsplit'.
        worker_cls="scheduler.gcsg.GCSGWorker",
        gpu_memory_utilization=0.5,
        enforce_eager=True,   # skip CUDA graph capture — irrelevant to what we're checking, saves time
        max_model_len=512,
        # transformers==4.57.6's MixtralConfig exposes `head_dim` as an
        # attribute that EXISTS but is None when config.json doesn't set it
        # explicitly (neither this tiny model's config.json nor real
        # Mixtral-8x7B's does). vLLM 0.6.6.post1's get_head_size() does
        # `hasattr(config, "head_dim")` — True even when the value is None —
        # so it returns None instead of falling through to
        # hidden_size // num_attention_heads. This is a real pin-combo
        # incompatibility, not tiny-model-specific: confirmed the same gap
        # exists in the real Mixtral-8x7B config.json. hf_overrides is
        # vLLM's documented mechanism for exactly this. 1024 // 32 = 32.
        hf_overrides={"head_dim": 32},
    )
    print("load_model() completed — GCSGWorker attached without crash. [1/4 OK]")

    worker = llm.llm_engine.model_executor.driver_worker
    if not isinstance(worker, GCSGWorker):
        _fail(f"driver_worker is {type(worker)}, not GCSGWorker — worker_cls not honored")

    prompts = ["The capital of France is", "def fibonacci(n):"]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0))
    print(f"generate() completed, {len(outputs)} outputs.")

    # ── 2: hooks fired, correct shape ────────────────────────────────────────
    n_captured = len(worker.captured_router_logits)
    if n_captured == 0:
        _fail("captured_router_logits is empty — .gate hooks never fired")
    print(f".gate hooks fired {n_captured} times.")

    bad_shapes = [
        tuple(t.shape) for t in worker.captured_router_logits
        if t.shape[-1] != EXPECTED_EXPERTS
    ]
    if bad_shapes:
        _fail(f"router_logits last dim != {EXPECTED_EXPERTS} experts: {bad_shapes}")
    print(f"router_logits shapes all end in {EXPECTED_EXPERTS} experts (last dim) — "
          f"sample shape: {tuple(worker.captured_router_logits[0].shape)}. [2/4 OK]")

    # ── 3: real request_ids accessible in execute_model() ───────────────────
    if not worker.seen_request_ids:
        _fail("seen_request_ids is empty — seq_group_metadata_list was never "
              "accessible/non-empty in execute_model()")
    print(f"Real request_ids seen via ExecuteModelRequest.seq_group_metadata_list "
          f"for {len(worker.seen_request_ids)} request(s): {worker.seen_request_ids}. [3/4 OK]")

    # ── 4: per-request contamination bookkeeping ────────────────────────────
    guard_stats = worker.guard.stats()
    if guard_stats["total_tokens_evaluated"] == 0:
        _fail("GCSGGuard.stats()['total_tokens_evaluated'] == 0 — should_activate_shadow "
              "was never called from execute_model()")
    per_request_rates = {
        rid: worker.guard.contamination_rate(rid) for rid in worker.seen_request_ids
    }
    print(f"GCSGGuard stats: {guard_stats}")
    print(f"Per-request contamination rates (synthetic gating values, "
          f"real request_ids): {per_request_rates}. [4/4 OK]")

    print("\nSMOKE TEST: GREEN — mechanics verified on hf-internal-testing/Mixtral-tiny.")
    print("NOT verified: quality (MMLU), performance, real shadow-pool VRAM behavior — "
          "pending on Mixtral-8x7B (see LOGBOOK 2026-08-09).")


if __name__ == "__main__":
    main()
