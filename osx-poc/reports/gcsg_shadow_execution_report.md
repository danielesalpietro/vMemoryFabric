# GCSG: Quality-Safe Shadow Execution Under Aggressive Quantization — Preliminary Report

**Status:** Preliminary / baseline. First report of its kind for this component — the
numbers below establish the reference point future GCSG runs are compared
against, not a final result.

**Date:** 2026-08-11
**Project:** OSX — Operating System for Experts (repo: `vMemoryFabric`), Sprint 3
("Oskarshamn")
**Scope:** M3 — Gating Confidence Shadow Guard (`GCSGWorker`/`GCSGGuard`,
`src/scheduler/gcsg.py`) only. Does **not** validate M1 (Expert Access Table)
or M2 (EMH Tier Manager) end-to-end — see [Limitations](#7-limitations).

---

## Abstract

We evaluate whether shadow execution through precision-degraded experts
(AWQ/Marlin INT4) can be inserted into a real Mixture-of-Experts inference
path without a measurable quality cost. On a full MMLU-5shot sweep (570
questions, 57 subjects) against Mixtral-8x7B-Instruct (AWQ), real shadow
execution scores 72.11% versus a 72.3% hook-only baseline — a −0.19
percentage-point degradation, inside the project's <2% non-functional
target by a wide margin. Reaching this number required root-causing two
separate failure modes that blocked it: a CUDA `illegal memory access`
crash shared by two quantization code paths (traced to vLLM's CPU-offload
wrapper not covering individually-called expert modules, combined with
pageable — not pinned — host memory under WSL2), and a severe,
variable-latency slowdown initially mistaken for a deadlock (traced to the
same pageable-memory mechanism operating correctly but slowly). Both
findings are cross-checked against vLLM's own upstream issue tracker,
which confirms the underlying platform limitation is real, structural, and
currently unaddressed upstream.

---

## 1. Motivation

Mixture-of-Experts inference activates a small subset of experts per
token. GCSG's premise is that when the router's top expert choice is
high-confidence (`gating_score > θ_gate`) and the token is low-entropy
(`token_entropy < θ_entropy`), a cheaper, precision-degraded version of
that expert can execute "in the shadow" of the real one — validating, at
low cost, whether serving through degraded experts more broadly would
preserve output quality. The risk this experiment targets is direct: if
shadow execution measurably degrades output quality, the premise doesn't
hold and the rest of the OSX system built around it (EAT/Tier Manager
promotion decisions, quantized shard caching) inherits a false assumption.
MMLU-5shot is used as the quality proxy because it is standard, cheap
relative to a full downstream eval, and already had a hook-only baseline
recorded in an earlier session (72.3%, LOGBOOK 2026-08-09) to compare
against.

---

## 2. Experimental Setup

Recorded in detail because the two failure modes in §4–5 turned out to be
almost entirely properties of this specific environment, not of the model
or the GCSG design — reproducing this result elsewhere requires knowing
exactly what's being reproduced.

### 2.1 Hardware

| Component | Spec |
|---|---|
| Workstation | HP Z8 G4 (`Z8-G4-RTX3090`) — doubles as dev box and the project's self-hosted CI GPU runner |
| GPU | NVIDIA RTX 3090, 24 GB VRAM, single GPU (no dual-GPU / AER path exercised) |
| Host OS | Windows, Docker Desktop with the **WSL2** backend |
| NVIDIA driver | 610.74 (host-side; verified against the CUDA 12.1.1 base image at Sprint 3 kickoff, 2026-08-08) |

### 2.2 Software stack

| Layer | Version |
|---|---|
| Container base image | `nvidia/cuda:12.1.1-devel-ubuntu22.04` |
| Python | 3.12 (via deadsnakes PPA — Ubuntu 22.04 ships 3.10 by default) |
| PyTorch | `2.5.1+cu124` (own bundled CUDA 12.4 runtime, independent of the base image's 12.1.1 devel toolchain) |
| vLLM | `0.6.6.post1` |
| transformers | `4.57.6` |
| Container orchestration | Docker Compose, `runtime: nvidia` via the NVIDIA Container Toolkit |

Pinned CUDA host memory (`cudaMallocHost`/`pin_memory=True`) is **not**
available in this configuration — vLLM detects WSL2 at startup and
disables it by default (`vllm.platforms.interface.in_wsl()` →
`Platform.is_pin_memory_available() == False`), citing NVIDIA's own
CUDA-on-WSL guidance that pinned system memory availability is limited
under WSL2. This single fact is the common ancestor of both failure modes
investigated in §4 and §5.

### 2.3 Model and quantization

- **Checkpoint:** `casperhansen/mixtral-instruct-awq` (Mixtral-8x7B-Instruct-v0.1,
  AWQ 4-bit), loaded from local NVMe (`/data/nvme/models/mixtral-instruct-awq`).
- **Quantization backend:** `quantization="awq_marlin"` — vLLM repacks the
  AWQ tensors into Marlin's packed kernel format at load time
  (`ops.awq_marlin_moe_repack()`), which is why the Marlin-specific crash
  in §4 exists as a distinct path from the plain AWQ `ModuleList` path
  (issue #16) despite both originating from the same checkpoint family.
- **`hf_overrides`:** `{"head_dim": 128}`.

### 2.4 vLLM engine configuration (final MMLU run)

```
cpu_offload_gb=4
gpu_memory_utilization=0.95
max_num_seqs=16
max_model_len=3328
enforce_eager=True
```

`cpu_offload_gb=4` keeps 6 of 32 decoder layers CPU-resident at load time
(verified via `scripts/map_offload_state.py`); the remaining 26 stay
GPU-resident for the whole run. `gpu_memory_utilization=0.95` and
`max_model_len=3328` (vs. an earlier, arbitrary 4096) are both measured,
not assumed — see `scripts/probe_kv_blocks.py` and
`scripts/measure_mmlu_prompt_lengths.py` (real 5-shot prompt lengths:
min=282, p50=547, p90=1197, p99=2961, max=3306 tokens). `enforce_eager=True`
disables CUDA Graph capture/replay for the whole run — relevant because it
rules out a CUDA-Graph-related explanation for the slowdown investigated
in §5.

### 2.5 GCSG configuration

```
theta_gate          = 0.85
theta_entropy       = 0.70
theta_contamination = 0.05
shadow_pool_size     = 2   (top-2 experts, round-robin selection —
                             not yet guided by real hotness; see §7)
```

Shadow pool experts are pinned GPU-resident explicitly at worker init
(`_pin_awq_expert_to_gpu()` / `_build_marlin_shadow_pool()` +
`_PinnedMarlinExperts`, commit `e59a16d`) — this is the fix that came out
of §4 and is why the shadow path itself is excluded as a cause in §5.

### 2.6 Orchestration

The full 570-question run does not complete in a single `generate()` call
reliably within this environment (see §5) — it is split into 18 slices of
32 prompts, each run in its own fresh Docker container/`GCSGWorker`
process (`scripts/run_mmlu_in_slices.sh`), with a per-slice watchdog of
2700s and an external `timeout` of 3000s (raised from an original
250s/300s that was killing legitimate slow-but-completing slices before
§5's root cause was understood). Results are persisted per-slice to
`mmlu_results_overnight_20260811.jsonl` so a single slice's timeout or
failure does not lose the rest of the run.

---

## 3. GCSG Design

`GCSGGuard.should_activate_shadow()` gates activation per token on three
conditions (all must hold): `bf16_available == False`, top gating score
`> theta_gate`, token entropy `< theta_entropy`, and per-request
contamination rate `< theta_contamination`. `GCSGWorker` hooks
`layer.block_sparse_moe.gate`'s forward (`_register_gate_hooks()`) to
capture real router logits and hidden states per layer, per forward pass,
then evaluates the gate condition and — if it passes — calls the
corresponding shadow expert via `run_shadow()`.

Three shadow-execution paths exist, selected automatically per checkpoint
based on how vLLM instantiated the experts:

1. **`_ShadowExpertINT4`** — raw fp16 `FusedMoE` weights, manually
   quantized to a simulated INT4 and dequantized on the fly. Used for
   non-pre-quantized checkpoints (e.g. the tiny test model). Not exercised
   under real offload in this report — see §7.
2. **`_AWQShadowExpert`** — checkpoint pre-quantized as a `ModuleList` of
   `MixtralMLP` (one module per expert, AWQ-packed). Delegates to the real
   module's own forward; no manual dequant.
3. **`_MarlinFusedShadowExpert`** — checkpoint quantized with
   `awq_marlin`, where `experts` is a single `FusedMoE` holding all
   experts' Marlin-packed tensors together. Isolates one expert via a
   one-hot `router_logits` bias rather than reverse-engineering the packed
   tensor layout.

This report's MMLU run exercises path 3 (Marlin), since that is the
checkpoint's actual quantization backend (§2.3).

---

## 4. Root Cause I: the CUDA Crash (Issues #10, #16)

Two shadow paths — Marlin-packed (#10) and AWQ `ModuleList` (#16) —
independently crashed with `CUDA error: illegal memory access` when
exercised for the first time against the real checkpoint under real
offload. Both traced to the same mechanism, confirmed by direct
manipulation rather than inferred from correlation alone:

- `vllm.model_executor.models.utils.maybe_offload_to_cpu()` wraps only the
  **decoder layer's** `forward()` — never the individual `experts`
  submodule's `forward()` (verified via `forward.__qualname__` inspection,
  `scripts/map_offload_state.py`).
- GCSG's shadow paths call the expert module directly, bypassing the
  decoder layer's forward entirely — so for any layer CPU-offloaded under
  `cpu_offload_gb`, the shadow call reaches a CUDA kernel with
  CPU-resident tensors.
- Isolated three-way (offloaded+unpinned → crash; offloaded+pinned → OK;
  non-offloaded → OK) against both the AWQ and Marlin paths independently,
  same signature, same variable flips it both times
  (`scripts/isolate_awq_offload_variables.py`,
  `scripts/isolate_marlin_offload_variables.py`).

**Fix (commit `e59a16d`):** pin every shadow-pool expert GPU-resident
*before* registering it in the pool — `.to('cuda')` for AWQ `ModuleList`
modules, a GPU-resident sliced proxy (`_PinnedMarlinExperts`) built only
for the layers actually offloaded (26/32 already-resident layers reuse
the original module directly, zero extra allocation — an earlier attempt
that pinned all 32 layers indiscriminately caused a separate, confirmed
CUDA-allocator fragmentation hang under `determine_num_available_blocks()`,
fixed by scoping the proxy construction to offloaded layers only). Both
proxies verified numerically against the original tensors before
integration (`scripts/verify_marlin_pinned_proxy.py`), then end-to-end
through the real worker (`scripts/verify_shadow_pool_pinning_e2e.py`).

---

## 5. Root Cause II: the Non-Deadlock Stall

With the crash fixed, full-coverage MMLU runs still failed to complete
within the original watchdog window at certain batch compositions —
initially read as a hang. Direct GPU-level instrumentation overturned that
reading:

- `nvidia-smi dmon` sampled through a "stalled" window: **SM utilization
  pinned at 100% continuously**, no idle gap — inconsistent with a
  host-side deadlock, which would show 0% SM and a blocked (`D`-state)
  process.
- The in-process watchdog's `SIGKILL` did not reliably stop the process;
  every repro eventually completed on its own. Five independent repeats of
  the same slice all completed (never truly hung) with wall time varying
  238s–590s for byte-identical output.
- Single-variable isolation on a previously-clean slice: only
  `cpu_offload_gb` (4 → 8) reproduces a **9× slowdown** with accuracy
  unchanged — content and batch composition, the leading hypotheses from
  the prior session, do not move the needle once this variable is
  controlled.
- Disabling shadow execution entirely (`shadow_pool_size=0`, hook-only)
  does **not** fix the slowdown — first-item latency is slightly worse
  (88.20s vs. 70.23s with shadow active), excluding the already-hardened
  shadow path (§4) as a contributor.
- Direct confirmation, not just correlation: forcing `pin_memory=True` via
  the same `in_wsl` monkeypatch used in §4, on the real (not isolated)
  evaluation path, cuts first-item latency **~5.4×** (88.20s → 16.28s),
  moving in lockstep with the one variable flipped.

**Conclusion:** the real model's own offloaded-layer forward pass — which
correctly goes through the wrapped `layer.forward()`, unlike the bypass in
§4 — performs its CPU→GPU swap-in over pageable host memory because vLLM
disables pinned memory under WSL2. This is slow and highly variable, but
not incorrect and not a hang. Cross-checked against vLLM's own upstream
issue tracker before accepting a "must be a WSL2 problem" conclusion:
[vllm-project/vllm#1084](https://github.com/vllm-project/vllm/issues/1084)
has a vLLM maintainer stating this is "unavoidable" on WSL, citing
NVIDIA's own CUDA-on-WSL documentation; more strikingly,
[vllm-project/vllm#37883](https://github.com/vllm-project/vllm/issues/37883),
filed against vLLM 0.17.1/0.18.0 — far newer than this project's pinned
0.6.6.post1 — describes the same `non_blocking=True`-without-pinned-memory
mechanism causing an outright crash in a different offload code path, and
is closed `not_planned`. The wrapper-granularity detail from §4 (decoder
layer wrapped, individual expert module not) was not on record in either
upstream issue; it was added as a comment to #37883 during this
investigation.

**This is not a fix.** It is a structural limitation of vLLM's CPU-offload
feature under WSL2, confirmed unaddressed upstream on a much newer
release. `pin_memory=True` remains diagnostic-only in this project — never
validated under sustained real load, not proposed for production use.

---

## 6. MMLU-5shot Evaluation

Full run: 18 slices × 32 prompts (last slice 18 prompts), 570/570
questions, 57/57 subjects, zero slice failures, unattended overnight
(raised watchdog/timeout values per §5's understanding that slow ≠ hung).

| Metric | Value |
|---|---|
| Prompts evaluated | 570 / 570 (100%) |
| Subjects covered | 57 / 57 (100%) |
| Correct | 411 |
| Unresolved | 0 |
| **Overall accuracy** | **72.11%** |
| Hook-only baseline (earlier session) | 72.3% |
| **Degradation** | **−0.19 pp** (target: < 2 pp) |
| Shadow activations (sum across all 18 slices) | 562,338 |
| Total `generate()` time (summed) | 3,690s (~1h 03m) |
| Average slice time | 205.0s |
| Slowest slice | `[32:64)` — 1,784.7s (~29.7 min) — the previously-flagged slow region, now understood per §5 |

Full per-subject breakdown: `mmlu_final_report.md`. Lowest-scoring
subjects are STEM-heavy (`abstract_algebra`, `college_physics`,
`electrical_engineering`, `formal_logic`, `high_school_mathematics`, all
40.0%) — expected for a 4-bit quantized 8x7B model, not attributed to
shadow execution specifically since the hook-only baseline was not
re-broken down per-subject for direct comparison (see §7).

**Note on the shadow-activations figure:** the value above (562,338) is
the corrected sum across all 18 independent slice processes, each with
its own `GCSGGuard` state. An earlier draft of `mmlu_final_report.md`
reported 13,756 — the last slice's own counter, mistaken for a run-wide
total. No per-slice `total_tokens_evaluated` is logged in the results
file, so an aggregate activation *rate* cannot be derived from this run;
a rerun with that field added would be needed.

---

## 7. Limitations

Stated explicitly, per this project's own established convention (see the
M1 technical report's limitations section) rather than left implicit:

- **Scope.** This validates GCSG (M3) shadow-execution quality on a
  single GPU, with offload managed by vLLM's own `cpu_offload_gb` — not
  the project's EAT (M1) or Tier Manager (M2). The shadow pool's expert
  selection is a round-robin placeholder (`range(shadow_pool_size)`), not
  guided by real hotness; that integration is EAT/Tier Manager work not
  yet done. M1's own extended benchmark found its Bloom filter ~5–14×
  slower than a plain dict on this workload (issue #1) — an open question
  independent of this report. Dual-GPU/AER (issue #8) was not exercised.
  "vMemoryFabric is alive" as a claim about the full system is **not**
  what this report supports; what it supports is narrower and stated in
  the abstract.
- **Single run, no statistical repetition.** The 72.11% figure is one
  overnight run. Earlier benchmark work on this same hardware (M1
  `bench_eat.py`) observed ~32% run-to-run variance on unrelated timing
  metrics with no code change — accuracy is a different, more discrete
  metric, but repeated full runs have not been done to establish a
  confidence interval here.
- **`pin_memory=True` is diagnostic only.** Used to confirm the mechanism
  in §5, never validated under sustained production-scale load; not
  proposed as a shippable configuration change.
- **Path 1 (`_ShadowExpertINT4`) untested under offload.** Only paths 2
  (Marlin) and 3 (AWQ) were exercised against the real checkpoint under
  real offload; path 1 is only verified against a tiny, non-offloaded test
  model.
- **Per-subject shadow-vs-baseline comparison not available.** The 72.3%
  hook-only baseline exists as an aggregate figure from an earlier
  session; a subject-by-subject baseline breakdown to compare against
  §6's table was not captured in that session and would need a rerun.

---

## 8. Related Work (upstream references)

- [vllm-project/vllm#1084](https://github.com/vllm-project/vllm/issues/1084)
  — WSL2 pinned-memory unavailability, confirmed by a vLLM maintainer as
  an unavoidable platform constraint (2023, still open in practice).
- [vllm-project/vllm#37883](https://github.com/vllm-project/vllm/issues/37883)
  — the same `pin_memory=False` + `non_blocking=True` mechanism causing an
  outright crash in a different (UVA/NVFP4) offload path, on a much newer
  vLLM release, closed `not_planned`. This project's wrapper-granularity
  finding (§4) was added there as a comment during this investigation.
  **Update (2026-08-12):** an independent reporter confirmed the identical
  failure mode still reproduces on a stack with essentially nothing in
  common with this project's own — vLLM 0.27.1, PyTorch 2.13.0+cu130, an
  RTX 5090 (Blackwell, SM120, 24GB), and a different model family entirely
  (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, hybrid NemotronH,
  not Mixtral). Same trigger (`--cpu-offload-gb 4`, Marlin *and* the newer
  Humming MoE backend both affected), same fix (`--cpu-offload-gb 0`,
  everything GPU-resident). This is now confirmed across ~20 vLLM releases,
  three GPU generations (Ampere → Ada → Blackwell), and two unrelated model
  architectures — strengthens §5's conclusion that this is a structural
  WSL2 platform limitation, not specific to this project's stack, and that
  waiting for an upstream fix (rather than the EAT/Tier-Manager-mediated
  approach in §9) is not a viable path.
- MoE quantization / precision-degraded expert literature — not yet
  surveyed; to be added before any external submission.

---

## 9. Conclusions and Future Work

GCSG's core premise — that shadow execution through degraded experts can
be validated at low quality cost — holds on this first full run, with
margin (−0.19pp against a 2pp budget). The path to that number required
resolving two failure modes now understood to be properties of the
Docker-on-Windows/WSL2 development environment, not of the GCSG design or
the model, and cross-confirmed against vLLM's own upstream issue tracker.

Before this can support a claim about the full OSX/vMemoryFabric system
rather than GCSG in isolation:

1. Route the shadow pool's promotion/eviction through EAT/Tier Manager
   (M1/M2) instead of vLLM's `cpu_offload_gb`, replacing round-robin
   expert selection with real hotness-driven selection. **Correction
   (post-review):** this does not automatically sidestep the WSL2
   pinned-memory limitation — §5's platform constraint applies to any
   CUDA process under WSL2, not specifically to vLLM's offload path. M1
   (EAT) itself never touches this question at all; it is pure
   in-process bookkeeping with no transfer code. What controlling the
   transfer *does* buy: avoiding `non_blocking=True` on an unpinned
   buffer (the exact crash-class mechanism in §4) is fully within the
   project's control regardless of pinning. Whether a manually pinned
   buffer (`torch.Tensor.pin_memory()`, called directly rather than
   through vLLM's `is_pin_memory_available()` gate) is actually fast and
   stable under sustained load on this platform is a separate, open,
   testable question this project has not answered rigorously — an
   early, informal check from before this investigation began
   (`torch.zeros(1024).pin_memory()` reporting `is_pinned() == True`)
   exists, but was never soak-tested. This needs its own direct
   verification before assuming Tier-Manager-mediated transfer would be
   materially faster than what vLLM does today.
2. Repeat this same MMLU evaluation on that path once it exists, as the
   next data point against this report's baseline.
3. Exercise path 1 (`_ShadowExpertINT4`) under real offload for parity
   with paths 2/3.
4. Establish a confidence interval via repeated full runs rather than a
   single overnight pass.

---

## Appendix: Scripts and Artifacts

- Crash isolation: `scripts/isolate_awq_shadow_call_crash.py`,
  `scripts/isolate_awq_offload_variables.py`,
  `scripts/isolate_marlin_offload_variables.py`
- Pinning verification: `scripts/verify_marlin_pinned_proxy.py`,
  `scripts/verify_shadow_pool_pinning_e2e.py`,
  `scripts/verify_awq_manual_shadow_expert.py`,
  `scripts/verify_marlin_shadow_expert.py`
- Offload-state mapping: `scripts/map_offload_state.py`
- MMLU harness: `scripts/eval_mmlu_gcsg.py`,
  `scripts/run_mmlu_in_slices.sh`,
  `scripts/measure_mmlu_prompt_lengths.py`, `scripts/probe_kv_blocks.py`
- Results: `mmlu_final_report.md`,
  `mmlu_results_overnight_20260811.jsonl`
- Full investigation trail: `LOGBOOK.md`, 2026-08-09 through 2026-08-11
  entries
- Issues: [#10](https://github.com/danielesalpietro/vMemoryFabric/issues/10),
  [#16](https://github.com/danielesalpietro/vMemoryFabric/issues/16)
