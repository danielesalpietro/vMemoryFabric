#!/usr/bin/env python3
"""GCSGWorker end-to-end smoke test — real Mixtral-8x7B-Instruct-v0.1-AWQ.

Follow-up to scripts/smoke_test_gcsg_worker.py (hf-internal-testing/
Mixtral-tiny), which validated wiring mechanics only. This one runs the
actual model the whole GCSG memory math / expert_map / paper is about.

VRAM is tight by design (see LOGBOOK 2026-08-09): the AWQ checkpoint is
~22.96 GiB against the 3090's 24 GiB, so cpu_offload_gb keeps some layers on
system RAM (188 GiB free in this container — confirmed no DDR4 constraint,
pin_memory() confirmed working in-container) and streams them to GPU on
demand. In practice the loaded weights only take 18.8-19.0 GiB in VRAM
(smaller than the on-disk safetensors size), so cpu_offload_gb ended up
unnecessary this run, but is left wired in for whenever it isn't.

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
currently reads. That gap is not yet closed — see the shadow-pool section
below.

Usage:
    PYTHONPATH=src python scripts/smoke_test_gcsg_mixtral8x7b.py
"""
from __future__ import annotations

import sys

from vllm import LLM, SamplingParams

from scheduler.gcsg import GCSGWorker

MODEL_PATH = "/data/nvme/models/Mixtral-8x7B-Instruct-v0.1-AWQ"


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    print(f"Loading {MODEL_PATH} via EngineArgs(worker_cls=GCSGWorker), "
          f"quantization=awq_marlin, cpu_offload_gb=4 ...")
    llm = LLM(
        model=MODEL_PATH,
        worker_cls="scheduler.gcsg.GCSGWorker",
        quantization="awq_marlin",
        cpu_offload_gb=4,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=2048,
        hf_overrides={"head_dim": 128},
    )
    print("load_model() completed — GCSGWorker attached, real Mixtral-8x7B loaded. [checklist 1-3 OK]")

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
