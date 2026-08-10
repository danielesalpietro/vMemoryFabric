# OSX — Operating System for Experts

**OSX** is a system-level framework for managing the lifecycle of experts in Mixture-of-Experts (MoE) large language models. It treats experts as first-class objects governed by a dedicated runtime — with hierarchical memory placement, predictive prefetching, gating-aware scheduling, and adaptive replication.

> *Current release: **Eketorp** (v0.3.0-dev) — August 8, 2026 — previous: Möllstorp (v0.2.0-dev), Karlshamn (v0.1.0-dev)*

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

These constraints are documented in `configs/osx_default.yaml` and tracked in all affected source files.

---

## Quickstart

**Prerequisites:** Docker Desktop with WSL2 backend + NVIDIA Container Toolkit.

```bash
# 1. Build the image (once)
make build

# 2. Verify hardware and environment
make smoke

# 3. Run all tests (M1/M2 pass as of Sprint 2; M3 still NotImplementedError — expected)
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
├── Dockerfile                  # CUDA 12.1.1 + Python 3.12 base image
├── docker-compose.yml          # osx-dev service (RTX 3090) + optional Prometheus sidecar
├── requirements.txt            # base deps — no vLLM (see requirements-vllm.txt)
├── requirements-vllm.txt       # vLLM, deferred to Sprint 3 (GCSG hooks not implemented yet)
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml              # cpu-tests (push/PR) + full-gpu-tests (manual)
│
└── osx-poc/
    ├── Makefile
    ├── pytest.ini
    ├── README.md                (this file)
    ├── CHANGELOG.MD
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
| 0      | Environment + skeleton | 1–2 | ✅ **Karlshamn** |
| 1      | M1 — EAT               | 3–4 | ✅ **Möllstorp** |
| 2      | M2 — Tier Manager      | 5–6 | ✅ **Eketorp**  |
| 3      | M3 — Expert Scheduler  | 7–8 | 🔲 pending  |
| 4      | Integration + benchmarks | 9–12 | 🔲 pending |
| 5      | PoC delivery + paper   | 13–16 | 🔲 pending |
| 6      | Telemetry + observability dashboard | TBD | 🔲 pending — **Stockholm** |

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
   already-scaffolded but never-connected `configs/prometheus.yml` and
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
| [#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1) | Bloom filter ~5-14× slower than a plain dict | Undecided whether it belongs in the EAT hot path at all |
| [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2) | EAT `RLock` p99 degrades ~1360× under contention | M3 adds real concurrent traffic on top of this |
| [#3](https://github.com/danielesalpietro/vMemoryFabric/issues/3) | `bench_tier.py` DDR4→VRAM p95/p99 skewed by CUDA cold-start | No warm-up iteration before timing |
| [#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4) | `BloomFilter.remove_expert()` unimplemented | Evicted shards remain permanent false positives |
| [#5](https://github.com/danielesalpietro/vMemoryFabric/issues/5) | No CUDA stream pipelining in `GPUTransfer` | Deferred since Sprint 0, needs real compute to overlap with |
| [#6](https://github.com/danielesalpietro/vMemoryFabric/issues/6) | No `pyproject.toml`/`ruff.toml` | Pre-existing style debt across the whole codebase |
| [#7](https://github.com/danielesalpietro/vMemoryFabric/issues/7) | PMEM (EMH-2) integration | Blocked on hardware availability |
| [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) | Dual-GPU / AER | Blocked on RTX 5080 arrival |

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
OSX Research Team, "OSX: Operating System for Experts — PoC v0.1.0 (Karlshamn)",
Internal Research Report, August 2026.
```

Target venues: OSDI 2027 / EuroSys 2027 / MLSys 2027.
