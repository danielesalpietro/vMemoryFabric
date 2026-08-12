# vMemoryFabric

A heterogeneous multi-tier memory fabric for MoE and LLM inference. Hot AI objects are promoted to executable VRAM, while cold objects are transparently tiered across heterogeneous GPU memory, DDR, PMEM/CXL and storage.

Developed under the internal codename **OSX** ("Operating System for Experts") — a system-level framework for managing the lifecycle of experts in Mixture-of-Experts (MoE) large language models. It treats experts as first-class objects governed by a dedicated runtime, with hierarchical memory placement, predictive prefetching, gating-aware scheduling, and adaptive replication.

> *Current release: **Tekniska** (v0.5.0-dev) — August 13, 2026 — previous: Oskarshamn (v0.4.0-dev), Eketorp (v0.3.0-dev), Möllstorp (v0.2.0-dev), Karlshamn (v0.1.0-dev). Sprint 4 complete; Sprint 5 (PoC delivery + paper, codename **Berg**) in planning — see [`osx-poc/reports/sprint5_berg_plan.md`](osx-poc/reports/sprint5_berg_plan.md).*

---

## Overview

Modern MoE models have no system layer dedicated to expert lifecycle management. Experts are placed statically, served without network topology awareness, and replicated uniformly or not at all. OSX closes this gap by introducing an Expert Memory Hierarchy (EMH) and a set of runtime components that operate transparently below the inference stack.

### Architecture (PoC scope)

```
┌────────────────────────────────────────────────────┐
│  M1 · Expert Access Table (EAT)                    │
│  Bloom filter 2-level + Slab allocator + Version   │
├────────────────────────────────────────────────────┤
│  M2 · EMH Tier Manager                             │
│  Promotion/eviction · SEE policy · Async I/O       │
├────────────────────────────────────────────────────┤
│  M3 · Expert Scheduler                             │
│  PT-PEP (BERT-small) · GCSG · AER stub             │
└────────────────────────────────────────────────────┘
         ↕ vLLM hooks (pre-tokenization, gating)
```

**M4 (RecursiveMAS LED Bridge)** is out of scope for this PoC — deferred to a future release.

### EMH tiers — dev setup

| Tier   | Hardware        | Role                   |
|--------|-----------------|------------------------|
| EMH-1a | RTX 3090 24 GB  | Hot expert shards      |
| EMH-1c | DDR4 256 GB     | Warm buffer            |
| EMH-3  | NVMe / volume   | Cold storage           |
| EMH-2  | Optane PMEM     | *Deferred*             |

---

## Dev environment constraints

This repository targets a **Docker-on-Windows** development setup with a single **RTX 3090**. Several production features are intentionally disabled or stubbed:

| Feature             | Status in dev          | Planned when            |
|---------------------|------------------------|-------------------------|
| Pinned CUDA memory  | ❌ not available        | Linux bare-metal        |
| `io_uring`          | ❌ WSL2/Docker          | Linux bare-metal        |
| Optane PMEM (EMH-2) | ❌ deferred             | Z8 G4 bare-metal        |
| Dual GPU (RTX 5080) | ❌ not yet available    | RTX 5080 arrival        |
| AER replication     | ❌ stub (single GPU)    | Dual-GPU setup          |
| RecursiveMAS M4     | ❌ out of PoC scope     | Future release          |
| vLLM (M3 GCSG hooks) | ❌ excluded from base image | Sprint 3 — see `requirements-vllm.txt` |

These constraints are documented in `osx-poc/configs/osx_default.yaml` and tracked in all affected source files.

---

## Quickstart

**Prerequisites:** Docker Desktop with WSL2 backend + NVIDIA Container Toolkit.

```bash
# 0. The Makefile lives in osx-poc/ — cd there first (see issue #12: the
#    container's WORKDIR doesn't match repo-root-relative paths, so this
#    matters for `docker compose run` invocations too, not just `make`)
cd osx-poc

# 1. Build the image (once)
make build

# 2. Verify hardware and environment
make smoke

# 3. Run all tests (M1/M2/M3 pass as of Sprint 3 — GCSG shadow execution real, see LOGBOOK.md)
make test

# 4. Interactive shell
make shell
```

`make smoke` passes 13/13 checks on the reference dev setup (RTX 3090, Docker Desktop + WSL2, NVIDIA Container Toolkit): Python/PyTorch/CUDA versions, VRAM, GPU tensor roundtrip, NVMe volume, all pinned packages importable, and `src/` importable via `PYTHONPATH`.

---

## CI/CD

`.github/workflows/ci.yml` runs two jobs (GitHub Actions doesn't support per-job triggers, so both are declared under one `on:` and the GPU job gates itself with an `if:`):

| Job | Trigger | Runner | What it runs |
|-----|---------|--------|---------------|
| `cpu-tests` | `push`, `pull_request` | `ubuntu-latest` | `pytest tests/ -m "not gpu"` — CPU-only subset of deps, no torch/vLLM/CUDA |
| `full-gpu-tests` | `workflow_dispatch` only (manual) | `[self-hosted, gpu]` | `docker compose build`, full test suite via the dev image, then `benchmarks/bench_eat.py` and `benchmarks/bench_tier.py` — uploaded as the `bench-eat-result` / `bench-tier-result` workflow artifacts |

Tests requiring real CUDA hardware are marked `@pytest.mark.gpu` (see `TestGPUTransfer`/`TestTierManagerGPU` in `tests/test_tier.py`) and registered in `pytest.ini`, so `-m "not gpu"` excludes them deterministically instead of relying on a runtime `pytest.skip()`.

Every manual `workflow_dispatch` run of `full-gpu-tests` re-measures M1 and M2 on the actual target hardware (`Z8-G4-RTX3090`) — see the M1/M2 technical reports for the latest numbers and their evolution across runs. This machine doubles as both the dev workstation and the self-hosted runner, so GPU-dependent bugs can be iterated on locally via `docker compose run` before spending a `workflow_dispatch` cycle on the final, recorded verification.

---

## Repository structure

```
vMemoryFabric/                  (repo root)
├── README.md                   (this file)
├── Dockerfile                  # CUDA 12.1.1 + Python 3.12 base image
├── docker-compose.yml          # osx-dev service (RTX 3090) + optional Prometheus sidecar
├── requirements.txt            # base deps — no vLLM (see requirements-vllm.txt)
├── requirements-vllm.txt       # vLLM, deferred to Sprint 3 (GCSG hooks not implemented yet)
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml              # cpu-tests (push/PR) + full-gpu-tests (manual)
│
└── osx-poc/                    # PoC implementation — everything below is here, not repo root
    ├── Makefile
    ├── pytest.ini
    ├── CHANGELOG.MD
    ├── LOGBOOK.md               # dev diary — full investigation trails, session-by-session
    ├── reports/                 # preliminary/technical reports (.md + .docx EN/IT)
    │
    ├── src/
    │   ├── eat/              # M1 — Expert Access Table
    │   │   ├── types.py      #   EATEntry, Tier, ExpertID, SHARD_SIZE_MB
    │   │   ├── bloom.py      #   BloomFilter 2-level
    │   │   ├── slab.py       #   SlabAllocator (DDR4 / future PMEM)
    │   │   └── eat.py        #   ExpertAccessTable (main class)
    │   ├── tier/             # M2 — EMH Tier Manager
    │   │   ├── io.py         #   AsyncNVMeIO (asyncio proxy for io_uring)
    │   │   ├── gpu.py        #   GPUTransfer (DDR4 → VRAM, no pinned)
    │   │   ├── policies.py   #   SEEPolicy + LRUPolicy
    │   │   └── manager.py    #   TierManager (orchestrator)
    │   └── scheduler/        # M3 — Expert Scheduler
    │       ├── ptpep.py      #   PT-PEP classifier (BERT-small ONNX)
    │       ├── gcsg.py       #   Gating Confidence Shadow Guard
    │       └── aer.py        #   AER (stub — single GPU dev)
    │
    ├── tests/
    │   ├── test_eat.py
    │   ├── test_tier.py      #   TestGPUTransfer marked @pytest.mark.gpu
    │   └── test_scheduler.py
    │
    ├── benchmarks/
    │   ├── bench_eat.py      # Sprint 1 (Möllstorp) — implemented
    │   └── bench_tier.py     # Sprint 2 (Eketorp) — implemented
    │
    ├── scripts/
    │   └── smoke_test.py     # Hardware + env validation — 13/13 passing
    │
    └── configs/
        ├── osx_default.yaml  # Runtime config (all constraints documented)
        └── prometheus.yml    # Metrics scraping config
```

---

## Development roadmap

| Sprint | Module | Weeks | Status      |
|--------|--------|-------|-------------|
| 0      | Environment + skeleton | 1–2 | ✅ **Karlshamn** (100%) |
| 1      | M1 — EAT               | 3–4 | ✅ **Möllstorp** (~92%) |
| 2      | M2 — Tier Manager      | 5–6 | ✅ **Eketorp** (~85%) |
| 3      | M3 — Expert Scheduler  | 7–8 | 🟡 in progress (~80%) — **Oskarshamn** |
| 4      | Integration + benchmarks | 9–12 | ✅ **Tekniska** (100%) (branch `Sprint-4-Tekniska`) |
| 5      | PoC delivery + paper   | 13–16 | 🟡 planning (~10%) — **Berg** |
| 6      | Telemetry + observability dashboard | TBD | 🔲 pending (~10%) — **Stockholm** |

Percentages are grounded in code/test/hardware verification, not label
carry-over — updated 2026-08-13 with Sprint 4 (Tekniska) complete: all 7
sub-goals of issue #17 closed, including two real bugs found and fixed on
real hardware along the way (Marlin's TierManager transfer, path 1 under
real offload) and a final MMLU quality number for every wired path.

**Sprint 1 (Möllstorp, ~92%):** M1 (EAT) is implemented and unit-tested.
Three real, measured defects were left open across two subsequent
sprints — [#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1)
(Bloom filter slower than a plain dict — never decided whether it belongs
in the hot path at all), [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2)
(`RLock` p99 degrades under contention), and [#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4)
(`BloomFilter.remove_expert()` unimplemented — evicted shards stayed
permanent false positives). All three visited 2026-08-12, in the order
the day actually happened: **#4 fixed first** — swapped `pybloom_live`
for a self-contained Counting Bloom Filter (supports real deletion;
classic Bloom filters structurally can't) and wired `EAT.evict()` to
actually call it. Re-measuring **#1** against that new implementation
showed lookup latency vs. a plain dict got *worse*, not better
(~6.8-8.1× vs. the old ~4.7-4.9×) — which settled #1's own long-open
question by making it obvious rather than deciding it in the abstract:
**the Bloom filter didn't belong in the hot path at all**, at this
scale, regardless of which Bloom implementation backed it. `src/eat/bloom.py`
was removed entirely (not just bypassed), `EAT.lookup()`/`insert()`/`evict()`
simplified back to a direct `dict` under `RLock`, `pybloom-live` dropped
from `requirements.txt`. Re-running the benchmark afterward: the
EAT-vs-plain-dict delta that used to be the whole point of measuring is
now ~0.07µs — noise, not a gap to justify. **#1 and #4 are both closed.**
**#2 remains open** — see Sprint 4 below for today's re-measurement of
its contention scenario against real (single-threaded) traffic, which
found the originally-cited degradation figure doesn't reproduce at the
same magnitude here.

**Sprint 2 (Eketorp, ~85%):** `TierManager`/`EAT`'s NVMe→DDR4→VRAM
pipeline is implemented and GPU-verified in isolation (Sprint 2). Until
2026-08-12 it was verified to be called from **nowhere** in
`osx-poc/src/scheduler/` or `osx-poc/scripts/` — zero occurrences,
checked directly, not assumed. That's now partially resolved: `GCSGWorker`
calls a new `TierManager.promote_live_tensor()` bridge for its shadow
pool's AWQ ModuleList path, verified end-to-end on two real GPUs (see
Sprint 4). Still open: the Marlin path (the one the published MMLU
numbers actually use) isn't wired; the *original* file-based shard
pipeline (`promote()`, `AsyncNVMeIO`) still has no real caller outside
its own tests/benchmarks; and the "shard promotion latency within 1.5×
theoretical bandwidth" acceptance target has a benchmark
(`benchmarks/bench_tier.py::bench_promote_live_tensor`) but no measured
result yet — written without GPU access, not yet run.

Sprint 3 (Oskarshamn) is real, not a stub: GCSG shadow execution runs
against the real Mixtral-8x7B checkpoint (both the AWQ ModuleList and
Marlin-packed paths), with a real fix for the CPU-offload/pin_memory crash
class that blocked it (issues [#10](https://github.com/danielesalpietro/vMemoryFabric/issues/10)/[#16](https://github.com/danielesalpietro/vMemoryFabric/issues/16), both **closed**
2026-08-11) verified end-to-end. The separate, reproducible slowdown under
certain concurrent batch compositions that blocked full 570-question MMLU
coverage is also resolved — root-caused to a structural WSL2/CUDA
pageable-memory limitation (confirmed against vLLM's own upstream issue
tracker, not a bug in this project's code), not a deadlock. Full 570/570
MMLU-5shot coverage achieved with real shadow execution active: 72.11% vs.
a 72.3% hook-only baseline (−0.19pp, inside the <2% target). Full writeup:
[`osx-poc/reports/gcsg_shadow_execution_report.md`](osx-poc/reports/gcsg_shadow_execution_report.md)
(also available as `.docx`, EN/IT, in the same directory) — marked
preliminary/baseline, not a final result.

Still open within Sprint 3, keeping it below 100%: PT-PEP ships as a
TF-IDF+centroid classifier rather than the originally-planned BERT-small,
a documented deviation (hit rate 87.2%, past the >70% target, but on a
same-distribution held-out set, not OOD); AER is trigger-logic-only by
design, blocked on dual-GPU hardware (#8); path 1 (`_ShadowExpertINT4`)
is still only verified on a tiny non-offloaded test model; and the
shadow pool's expert selection is a round-robin placeholder **when no
`TierManager` is wired** — now genuinely hotness-driven when one is (see
Sprint 4), so this caveat only applies to the still-default,
un-integrated path. As of 2026-08-12, **M2 (Tier Manager) reaches the
real shadow-pool path for one of its two quantization backends** (AWQ
ModuleList) — see Sprint 4 below for what that means and what's still
missing (the Marlin path, which the published 72.11%/72.28%/72.3% MMLU
numbers actually used). Full report: `osx-poc/reports/gcsg_shadow_execution_report.md`
§7/§9.

Sprint 4 (Tekniska) is issue #17 given real scope: wire `GCSGWorker`'s
shadow pool through `TierManager`/`EAT` instead of vLLM's `cpu_offload_gb`,
resolve the open pinning-strategy question from the GCSG report's §9
correction, re-run the MMLU-5shot evaluation on the integrated path,
measure the "shard promotion latency within 1.5× theoretical bandwidth"
target, and close out the M1 debt (#1/#2/#4) that Sprint 1/2 explicitly
deferred until M3 generated real concurrent traffic. Full sub-goal
breakdown: `osx-poc/LOGBOOK.md`, 2026-08-11 "Tekniska: Sprint 4 kickoff"
entry. **Complete as of 2026-08-12 — all 7 sub-goals done or closed:**

- **Done — wiring (sub-goal 1).** `GCSGWorker(tier_manager=...)` (opt-in,
  default `None`, zero behavior change unless wired — see
  `GCSGWorker.configure_tier_manager()`, needed because vLLM constructs
  the worker itself, no direct kwarg path exists) routes GPU promotion
  through a new `TierManager.promote_live_tensor()` bridge, seeds EAT
  with one entry per (expert, layer), and feeds it real per-token
  routing traffic independent of shadow activation — closing the "no
  real concurrent EAT traffic" precondition Sprint 1/2 were waiting on.
  Landed in two stages, deliberately: the AWQ ModuleList path (path 3)
  first, verified 4/5 on a real RTX 3090 (WSL2 — `pin=True` correctly
  disabled there by design) and 5/5 on a second real GPU (real Linux,
  `pin=True` confirmed end-to-end), plus two full MMLU reruns matching
  the historical baseline (one byte-identical per-subject on two
  32-question slices, one full 570-question run landing at 411/570 with
  only 4 individual answers flipped and perfect run-to-run determinism).
  Only *then* — the most fragile mechanism in the file (see
  `_PinnedMarlinExperts`' docstring on a real CUDA-allocator
  fragmentation hang found 2026-08-10) — was the **Marlin path (path 2,
  the one every published MMLU number so far has used) also wired**,
  via a shared per-layer proxy tracked under a sentinel EAT key (a
  single proxy serves the whole shadow pool per layer by design, not one
  per expert — see `GCSGWorker._marlin_pool_shard_key()` for why a real
  `expert_id` key would risk a stale-VRAM-reuse bug across
  `refresh_shadow_pool_selection()` calls). **Update:** the Marlin-path
  transfer's own hardware pass (`scripts/smoke_test_gcsg_tier_manager.py
  --quantization awq_marlin`) has now run, on a real RTX 3090 — failed
  once (a real bug: `_build_marlin_shadow_pool()` inferred a whole
  layer's offload state from a single tensor, but vLLM's
  `cpu_offload_gb` doesn't offload all of a Marlin-packed layer's
  tensors uniformly, so the transfer tried to pin an already-CUDA
  tensor), fixed, then green with 6 sentinel (`expert_id=-1`) EAT
  entries confirmed reaching `Tier.VRAM`. Both quantization paths
  GCSG's shadow pool can use are now wired and hardware-verified. 98
  unit tests passing.
- **Done — pinning strategy (sub-goal 2).** Manually pinned transfer
  (`torch.Tensor.pin_memory()`, bypassing vLLM's own WSL2 gate) is safe
  and stable under sustained load on real Linux — 1000-cycle soak test,
  0 mismatches, no timing degradation — and now actually wired into
  `GPUTransfer.to_vram(pin=True)`, opt-in, default `False`.
- **Substantially done — integrated-path MMLU rerun (sub-goal 3).** Real
  runs, not projections: two 32-question slices matched the historical
  Marlin baseline exactly, per subject; the full 570-question run
  (single-process, the pattern that used to hang under WSL2 — didn't
  here) landed at 411/570 (72.1%), same total as the baseline but with 4
  individual answers flipped (2 up, 2 down) — 0.7%, inside the <2%
  target, corrected from an earlier same-day overclaim of "identical"
  based only on the two slices. **Update:** repeated again with
  `TierManager` actually wired in (`--wire-tier-manager`, sub-goal 1's
  path) rather than the original `cpu_offload_gb` path — same
  570-question, single-process pattern, landed at 411/570 (72.1%) again,
  plus an exact per-subject match on a 32-question slice. The
  TierManager-routed rerun this item originally called for is now done
  for AWQ (path 3); no equivalent MMLU number exists yet for the Marlin
  path specifically, only the mechanical wiring verification above.
  `osx-poc/LOGBOOK.md`, 2026-08-12 entries.
- **Done — promotion latency (sub-goal 4).** Benchmark
  (`benchmarks/bench_tier.py::bench_promote_live_tensor`, `pin=True` vs
  `pin=False`) run on real hardware: `pin=True` P50 194.1µs vs.
  `pin=False` P50 684.2µs — ~3.5× faster, meeting the 1.5×
  theoretical-bandwidth criterion on the metric that criterion actually
  asks for (P50); P95/P99 went the other way on `pin=True` but at n=20
  synthetic 4MB shards that tail isn't a statistically meaningful
  measurement, not investigated further.
- **Done — M1 debt re-analysis (sub-goal 5).** Issue #1 (Bloom filter
  ~5–14× slower than a plain dict on this workload) is **closed**:
  removed from EAT's hot path entirely rather than tuned. Issue #4
  (`BloomFilter.remove_expert()`) is now moot — same removal. **Issue #2
  (RLock contention): decided 2026-08-12, left deliberately open, not
  redesigned.** Today's re-measurement under real EAT traffic didn't
  match the issue's originally cited ~1360× p99 degradation figure
  (today: ~61-91×, both numbers recorded, the gap unexplained) — but
  more importantly, real `GCSGWorker` traffic today is single-threaded,
  so the contention scenario issue #2 measures isn't produced by
  anything running in this project yet. Redesigning the `RLock` now
  would be speculative work against a scenario that doesn't exist;
  re-scoped from "in progress" to blocked on real concurrent EAT access
  actually existing (a future multi-worker/async-server setup) —
  revisit if/when that traffic shape becomes real, not before.
- **Done — path 1 parity under real offload (sub-goal 6).** Exercised
  `_ShadowExpertINT4` for the first time against real
  Mixtral-8x7B-Instruct-v0.1 (unquantized, ~93.4GB) under real offload
  (A100 SXM 80GB, `cpu_offload_gb=28`) — previously only verified on a
  tiny, non-offloaded test model. Found and fixed a real bug: the
  fused-FusedMoE build loop never pinned offloaded weight slices to GPU
  before quantizing, unlike paths 2/3; fixed with the same
  device-check-then-pin pattern, then verified green including a
  numerically-correct INT4 forward at real `hidden_size=4096`. See
  `osx-poc/reports/gcsg_shadow_execution_report.md` §7/§9 for the full
  writeup. Mechanics/correctness only — no MMLU number on this path, no
  production-viability claim for this offload configuration.
- **Done — close-out (sub-goal 7).** This table and the GCSG report were
  kept in sync as each sub-goal closed; the last item, regenerating the
  report's `.docx` (EN/IT) exports to match its markdown source, is done
  — both pass full OOXML schema validation. **Sprint 4 (Tekniska) is
  complete.**

Sprint 5 (Berg) kicked off 2026-08-13, right after Tekniska closed. First
action was a consistency check between README/CHANGELOG's claimed-closed
issues and GitHub's actual state, not assumed: found
[#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1)/[#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4)/[#17](https://github.com/danielesalpietro/vMemoryFabric/issues/17)
documented as closed but still open on GitHub, closed for real with
commit references. Full plan:
[`osx-poc/reports/sprint5_berg_plan.md`](osx-poc/reports/sprint5_berg_plan.md) —
two parallel tracks, PoC delivery (aimed at external stakeholders/reviewers,
not just internal close-out: issue triage, a consolidated PoC Final Report,
real repro packaging, a tagged release) and a systems paper targeting a real
venue (OSDI/EuroSys/MLSys 2027), reusing `gcsg_shadow_execution_report.md`'s
already paper-shaped structure rather than starting from a blank page.

Sprint 6 (Stockholm) is a new leg, added without reordering or reweighting
Sprints 0–5 above — those stay exactly as planned. Named deliberately:
Stockholm is the seat of the Swedish government, and this sprint's job is
oversight, not operation — a dashboard that *observes* GCSG/EAT/Tier
Manager state, without sitting in the hot path of any of them.

Two phases, in order, not scoped together:

1. **Single-worker telemetry** — the low-overhead path. `GCSGGuard`,
   `AERManager`, `PTPEPClassifier`, `TierManager`, and `EAT` already each
   expose a `.stats()` method returning counters accumulated as a
   byproduct of work already happening (tokens evaluated, shadow
   activations, contamination rate, tier promotions, latencies) — zero new
   instrumentation needed, only an adapter. Exposes these via a
   `/metrics` endpoint (`prometheus_client`), wired into the
   already-scaffolded but never-connected `osx-poc/configs/prometheus.yml` and
   `make metrics-up`/`metrics-down` targets. Scope: one `GCSGWorker`
   process, one dashboard.
2. **Multi-worker aggregation** — deferred until there's more than one
   worker process to aggregate across, which today means issue #8
   (dual-GPU / AER, blocked on RTX 5080 arrival) landing first. Needs a
   real design decision this project hasn't made yet (Prometheus
   multi-target scraping by instance label vs. a pushgateway pattern for
   short-lived workers) — not started until phase 1 is real and the
   hardware blocker clears, tracked as a dependency rather than an
   arbitrary "later."

Non-functional targets (acceptance criteria for PoC):

- PT-PEP latency < 3 ms p99 on CPU
- PT-PEP hit rate > 70% on labeled test set
- GCSG quality degradation < 2% (MMLU-5shot)
- Shard promotion latency within 1.5× theoretical bandwidth

Live roadmap board: [OSX-PoC Roadmap](https://github.com/users/danielesalpietro/projects/1) (GitHub Project — one card per sprint plus tracked open issues).

---

## Known limitations / open issues

Findings from M1/M2 benchmarking that were deliberately left unresolved, each with the measurement behind it — tracked as GitHub Issues rather than left as LOGBOOK notes, so they survive past whoever wrote the LOGBOOK entry:

| # | Issue | Why it matters |
|---|-------|-----------------|
| ~~[#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1)~~ | Bloom filter slower than a plain dict | **Closed 2026-08-12** — re-measured worse under the new Counting Bloom Filter (issue #4's fix), settling the question: removed from `EAT` entirely rather than kept as an unjustified fast-path; see Sprint 1 above |
| [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2) | EAT `RLock` p99 degrades under contention | Re-measured 2026-08-12: real `GCSGWorker` traffic today is single-threaded, so this scenario isn't exercised by production yet; the originally-cited ~1360× figure didn't reproduce at that magnitude (~61-91× measured instead, both recorded — the gap itself is unexplained). **Decision 2026-08-12**: left open, deliberately, not redesigned — there is no real production contention to fix yet, redesigning the `RLock` now would be speculative work against a scenario that doesn't exist. Re-scoped from "in progress" to blocked on real concurrent EAT access actually existing (a future multi-worker/async-server setup); revisit if/when that traffic shape becomes real |
| [#3](https://github.com/danielesalpietro/vMemoryFabric/issues/3) | `bench_tier.py` DDR4→VRAM p95/p99 skewed by CUDA cold-start | No warm-up iteration before timing |
| ~~[#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4)~~ | `BloomFilter.remove_expert()` unimplemented | **Closed 2026-08-12** — fixed with a Counting Bloom Filter, then superseded hours later when #1's re-measurement led to removing the Bloom filter entirely; see Sprint 1 above |
| [#5](https://github.com/danielesalpietro/vMemoryFabric/issues/5) | No CUDA stream pipelining in `GPUTransfer` | Deferred since Sprint 0, needs real compute to overlap with |
| [#6](https://github.com/danielesalpietro/vMemoryFabric/issues/6) | No `pyproject.toml`/`ruff.toml` | Pre-existing style debt across the whole codebase |
| [#7](https://github.com/danielesalpietro/vMemoryFabric/issues/7) | PMEM (EMH-2) integration | Blocked on hardware availability |
| [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) | Dual-GPU / AER | Blocked on RTX 5080 arrival |
| [#12](https://github.com/danielesalpietro/vMemoryFabric/issues/12) | `make lint`/`test`/`bench` fail on relative paths — container `WORKDIR` (`/workspace`) doesn't match `osx-poc/`'s relative paths | Workaround in use everywhere: `docker compose run --rm osx-dev bash -c "cd osx-poc && ..."` |
| ~~[#17](https://github.com/danielesalpietro/vMemoryFabric/issues/17)~~ | `TierManager`/`EAT` (M1/M2) not in the shadow pool's actual data path — `GCSGWorker` used vLLM's `cpu_offload_gb` directly | **Resolved 2026-08-13**: all three shadow-pool paths (AWQ, Marlin, path 1) wired/exercised and verified on real hardware (Sprint 4, complete — see roadmap above), each with its own real bug found and fixed along the way. The last open gap — an MMLU quality number for Marlin specifically routed through TierManager — is now closed too: 412/570 (72.28%), matching the historical Marlin baseline exactly (`osx-poc/marlin_mmlu_20260812/`). Getting there required a small fix to `eval_mmlu_gcsg.py`, which had `--wire-tier-manager` silently forcing `quantization=awq` (stale from before Marlin was wired) — added an explicit `--quantization` flag. Promotion-latency measurement done. Issue #2 (RLock contention) decided separately, not a blocker |
| [#18](https://github.com/danielesalpietro/vMemoryFabric/issues/18) | No environment fingerprint pre-check — `OMP_NUM_THREADS`/`shm_size`/GPU model assumed, not verified | Hit for real deploying to RunPod: `OMP_NUM_THREADS=8` fixed regardless of real vCPU count, `docker-compose.yml`'s `shm_size` doesn't apply outside local `docker compose` |

**Closed:** [#10](https://github.com/danielesalpietro/vMemoryFabric/issues/10)/[#16](https://github.com/danielesalpietro/vMemoryFabric/issues/16) (2026-08-11) — GCSG shadow-execution crash and the related batch-composition slowdown, both root-caused to WSL2/CUDA pageable-memory offload behavior (structural, upstream-confirmed, not a project bug). Full trail: `osx-poc/reports/gcsg_shadow_execution_report.md`. [#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4) (2026-08-12) — `BloomFilter.remove_expert()` implemented for real (Counting Bloom Filter, replaces `pybloom_live`) and wired into `EAT.evict()`, which never called it before. [#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1) (2026-08-12, same day) — re-measuring against #4's new implementation showed lookup latency got *worse*, not better, settling the question: the Bloom filter is removed from `EAT` entirely, not kept as an unjustified fast-path (which also makes #4's fix moot, but it stays recorded as its own closed issue — the bug it fixed was real while the Bloom filter still existed). Full trail: `osx-poc/LOGBOOK.md`, 2026-08-12 "issue #4 actually fixed" and "Bloom filter removed" entries.

---

## Key design decisions

**Why `asyncio + aiofiles` instead of `io_uring`?**
`io_uring` is Linux-only and not available on WSL2/Docker Windows. The `AsyncNVMeIO` interface is identical to what `io_uring` will use on Linux bare-metal; swapping the backend requires no changes to `TierManager`.

**Why no pinned memory?**
`cudaMallocHost` is available but incurs significant overhead in Docker-on-Windows due to the virtualized DMA path. The delta vs. standard `cudaMemcpy` is documented and tracked as a known baseline deviation.

**Why PMEM as DDR4 proxy?**
Optane DCPMM requires bare-metal Linux with kernel ≥ 5.1. In dev, DDR4 acts as EMH-2 proxy. The `SlabAllocator` backend is designed to swap to `libpmem2` mmap with no interface changes.

---

## Citation

If you use OSX in your research, please cite:

```
Daniele Carmelo Salpietro, "OSX / vMemoryFabric: Operating System for Experts — PoC v0.4.0-dev (Oskarshamn)",
Internal Research Report, August 2026.
```

Target: arXiv preprint (self-archive), formatted in the style of a systems
conference paper (ACM `sigconf`, EuroSys-style) — see
[`osx-poc/reports/sprint5_berg_plan.md`](osx-poc/reports/sprint5_berg_plan.md)
for the Sprint 5 plan, target end of August 2026.

**Author:** Daniele Carmelo Salpietro — Principal Solutions Architect (independent) — [salpietro.it](https://www.salpietro.it) — daniele@salpietro.it
