\# Changelog



All notable changes to OSX-PoC are documented here.

Format follows \[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning follows \[Semantic Versioning](https://semver.org/).



Releases are named after cities where significant project milestones were reached.



\---



\## \[Möllstorp] — v0.2.0-dev — 2026-08-08



\*Named after Möllstorp, Öland, Sweden, where M1 — Expert Access Table was implemented.\*



\### Added



\*\*M1 — Expert Access Table (real implementation)\*\*



\- `src/eat/bloom.py` — `BloomFilter.add/may\_contain\_expert/may\_contain\_shard/\_\_len\_\_` implemented on top of `pybloom\_live`, string-encoded keys (`e:{expert\_id}`, `s:{expert\_id}:{shard\_idx}`) for deterministic hashing. `remove\_expert` still raises `NotImplementedError` — standard Bloom filters can't delete; deferred to a future Counting Bloom Filter evaluation, unused by `EAT.evict()` by design (see Fixed/Dev constraints below)

\- `src/eat/slab.py` — `SlabAllocator` pool lifecycle (`initialize`/`shutdown`), `alloc`/`free`/`get\_buffer`, `free\_slots`/`used\_slots`. Pool backed by `np.empty((n\_slots, shard\_size), dtype=np.uint8)` — `empty` not `zeros`, content is a placeholder for the future PMEM/DDR4 payload so there's no reason to zero-fill pages

\- `src/eat/eat.py` — `ExpertAccessTable` full CRUD: `insert`, `lookup` (Bloom fast-negative before touching the dict), `update\_tier` (bumps `version`), `evict`, `access` (`touch()`), `get\_tier`, `eviction\_candidates` (LRU by `last\_access\_ts`, SEE score deferred to M2 per the existing docstring), `initialize`/`shutdown`, `stats()`



\*\*Benchmarks\*\*



\- `benchmarks/bench\_eat.py` — real P50/P95/P99 lookup latency + insert throughput, JSON output (replaces the `"status": "pending"` placeholder)



\### Design decisions



\- \*\*SlabAllocator stays standalone for M1\*\* — `EAT.insert()`/`evict()` operate only on the dict + Bloom filter; they do not call `slab.alloc()`/`free()`. `EATEntry`'s documented 28-byte layout has no `slot\_idx` field, and physical DDR4 promotion/eviction is the Tier Manager's responsibility (M2, Sprint 2). `EAT.initialize()`/`shutdown()` still drive the `SlabAllocator`'s lifecycle in lockstep so the two stay wired together for M2 to build on

\- Bloom filter keys are explicit strings rather than raw tuples/ints, to not depend on `pybloom\_live`'s internal key-hashing behavior for compound keys



\### Tests



\- `tests/test\_eat.py` — every stub (`pytest.raises(NotImplementedError)` / `pytest.skip("TODO Sprint 1")`) replaced with real assertions: Bloom add/contains/false-positive-rate, Slab alloc/free/exhaustion/variable-tail, EAT CRUD + LRU eviction ordering, 8-thread concurrent insert/read-write/version-counter races, and a real lookup-latency smoke timing. Added `test\_initialize\_shutdown` (previously uncovered) and a couple of extra edge-case tests (`test\_insert\_duplicate\_raises`, `test\_update\_tier\_missing\_raises`, `test\_evict\_missing\_returns\_none`, `test\_access\_missing\_returns\_none`) not present in the Sprint 0 skeleton

\- `make test-eat` — 24/24 passed (verified via a lightweight `python:3.12-slim` container mirroring CI's `cpu-tests` dependency set, not the full CUDA dev image — nothing in M1 touches torch/CUDA)



\### CI/CD



\- `.github/workflows/ci.yml` — `full-gpu-tests` now also runs `benchmarks/bench\_eat.py` on the self-hosted `Z8-G4-RTX3090` runner after the test suite, uploading the result as the `bench-eat-result` workflow artifact. Every manual `workflow\_dispatch` re-measures M1 on real target hardware instead of the numbers staying container-host-only



\### Evaluation — extended benchmark + technical report



\- `benchmarks/bench\_eat.py` — extended past the initial P50/P95/P99 + insert-throughput numbers with three additional measurements, requested during report review to close the gaps an OSDI-style reviewer would flag first:

  \- \*\*Baseline comparison\*\* — a plain-dict, no-Bloom-filter baseline (`\_PlainDictBaseline`, benchmark-only, not part of the EAT's public API) run against the identical workload. Result: the Bloom filter measured \~5× slower on lookups and \~14.6× slower on inserts than the plain dict — it costs latency rather than saving it, because the structure it guards is already an O(1) in-memory dict. Flagged in the report as an open question for M2/M3 rather than a justified design choice

  \- \*\*Contention profile\*\* — 4 concurrent reader threads + 1 writer thread against a shared EAT. Reader p99 lookup latency rose \~1,360× under contention (single-digit µs → several ms), exposing a real tail-latency cost from the `RLock`-based concurrency model in `eat.py` (see the "CAS counter" design note in Möllstorp's Design decisions above)

  \- \*\*Slab allocator at scale\*\* — alloc/free timing at 4 vs. 32 slots (1 GB vs. 8 GB). No measurable latency growth, consistent with the free-list's claimed O(1) behavior across that 8× increase (still \~2 orders of magnitude below the \~1,000-slot production target)

\- \*OSX Technical Report — M1: Expert Access Table\* (internal document, not committed to this repository) — formal design/implementation/evaluation writeup including the baseline, contention, and slab-scale results above, and an explicit limitations section. Re-run against real target hardware twice via CI (`workflow\_dispatch` runs `31259547453` and `31260457679`), after an initial container-host-only draft — the two same-hardware runs differed by \~32% in measured insert throughput, noted in the report as unquantified run-to-run variance rather than smoothed over



\### Status



\*\*Released.\*\* M1 (Expert Access Table) is fully implemented, tested, and benchmarked: 24/24 tests passing in `tests/test\_eat.py`; real hardware numbers from `benchmarks/bench\_eat.py` on `Z8-G4-RTX3090` (p50 ≈ 2.6 µs / p95 ≈ 3.8 µs / p99 ≈ 4.6 µs lookup latency, ≈ 177k inserts/sec); and, per the extended evaluation above, two open design questions with measured costs attached — the Bloom filter's net benefit in the current single-process design, and the RLock's tail-latency behavior under concurrent read/write load — both recorded as Sprint 2/M2 candidates rather than fixed here, per the decision to leave `EATEntry`'s layout and M1's core design untouched until end-to-end numbers exist (Sprint 4). M2/M3 remain Sprint 0 skeletons (`NotImplementedError`), as planned.



\---



\*Next release: Sprint 2 — “Eketorp” — M2 (Tier Manager) implementation complete, `make test-tier` passing green.\*



\---



\## \[Karlshamn] — v0.1.0-dev — 2026-08-07



\*Named after Karlshamn, Sweden, where the PoC development environment was established.\*



\### Added



\*\*Environment\*\*

\- `Dockerfile` — CUDA 12.1.1 + Python 3.12 + full M1/M2/M3 stack

\- `requirements.txt` — pinned dependencies (PyTorch 2.3.0+cu121, transformers 4.41.2, onnxruntime 1.18.0; vLLM deferred — see `requirements-vllm.txt`)

\- `docker-compose.yml` — single RTX 3090, NVMe volume, optional Prometheus sidecar

\- `Makefile` — targets: `build`, `smoke`, `test`, `test-eat`, `test-tier`, `test-scheduler`, `bench`, `lint`, `typecheck`, `shell`, `metrics-up`, `clean`

\- `pytest.ini` — asyncio auto mode, short traceback



\*\*M1 — Expert Access Table (skeleton)\*\*

\- `src/eat/types.py` — `EATEntry`, `Tier` (VRAM/DDR4/NVME), `ExpertID`, `ShardID`

\- `src/eat/bloom.py` — `BloomFilter` 2-level interface (expert-level + shard-level)

\- `src/eat/slab.py` — `SlabAllocator` interface (DDR4 backend, PMEM-ready)

\- `src/eat/eat.py` — `ExpertAccessTable` interface: insert, lookup, update\_tier, evict, access, get\_tier, eviction\_candidates



\*\*M2 — EMH Tier Manager (skeleton)\*\*

\- `src/tier/io.py` — `AsyncNVMeIO` interface (asyncio+aiofiles proxy for io\_uring)

\- `src/tier/gpu.py` — `GPUTransfer` interface (standard cudaMemcpy, no pinned memory)

\- `src/tier/policies.py` — `SEEPolicy` (α=0.3, β=0.3, γ=0.4) + `LRUPolicy` interfaces

\- `src/tier/manager.py` — `TierManager` interface: promote, evict, evict\_to\_free\_vram, prefetch



\*\*M3 — Expert Scheduler (skeleton)\*\*

\- `src/scheduler/ptpep.py` — `PTPEPClassifier` interface: 8 domains, ONNX runtime, confidence threshold

\- `src/scheduler/gcsg.py` — `GCSGGuard` interface: θ\_gate=0.85, θ\_entropy=0.70, θ\_contamination=0.05

\- `src/scheduler/aer.py` — `AERManager` stub: replication\_factor=1 (single GPU dev), sync\_lora\_delta no-op



\*\*Tests\*\*

\- `tests/test\_eat.py` — BloomFilter, SlabAllocator, EAT CRUD, concurrency, latency smoke

\- `tests/test\_tier.py` — AsyncNVMeIO, GPUTransfer, SEE/LRU policies, TierManager, EAT↔Tier integration

\- `tests/test\_scheduler.py` — PT-PEP, GCSG, AER, Scheduler↔Tier integration



\*\*Benchmarks (placeholder)\*\*

\- `benchmarks/bench\_eat.py` — Sprint 1 target

\- `benchmarks/bench\_tier.py` — Sprint 2 target



\*\*Config\*\*

\- `configs/osx\_default.yaml` — full runtime config with dev constraints documented inline

\- `configs/prometheus.yml` — Prometheus scrape config (port 9090)



\*\*Scripts\*\*

\- `scripts/smoke\_test.py` — 13 independent checks: Python version, PyTorch, CUDA, VRAM ≥22 GB, tensor roundtrip, NVMe volume, vLLM, transformers, onnxruntime, aiofiles, prometheus\_client, src/ importable, pinned memory info



\### Added — CI/CD (extra scope)



Not part of the original Sprint 0 plan; added opportunistically while closing out the environment.



\- `.github/workflows/ci.yml` — two jobs. `cpu-tests` runs on every push/PR (`ubuntu-latest`, `pytest -m "not gpu"`, CPU-only dep subset — no torch/vLLM/CUDA). `full-gpu-tests` runs only on manual `workflow\_dispatch` (`runs-on: [self-hosted, gpu]`), since GitHub Actions has no per-job trigger — the workflow listens for push/PR/workflow\_dispatch and the GPU job gates itself with `if: github.event\_name == 'workflow\_dispatch'`

\- `.gitignore` (repo root) — excludes `\_\_pycache\_\_/`, `.pytest\_cache/`, `.mypy\_cache/`, `.ruff\_cache/`, `.coverage`

\- `osx-poc/SELF-HOSTED.MD` (new) — setup guide for the self-hosted GPU runner: hardware/software prerequisites, pre-registration GPU-in-Docker check, registration steps with the required `gpu` label, security note on why the job stays gated to `workflow\_dispatch`

\- Self-hosted runner \*\*Z8-G4-RTX3090\*\* registered against the repo, `gpu` custom label attached, running as a dedicated `GitRunner` local service account (member of `docker-users`, granted "Log on as a service")



\### Fixed



\*\*Docker build\*\*

\- `Dockerfile` — Python 3.12 requires the deadsnakes PPA on Ubuntu 22.04 (jammy ships 3.10 only); bootstrap pip via `ensurepip` since apt's `python3-pip` targets the default interpreter, not deadsnakes'

\- `requirements.txt` — pinned `torch==2.3.0+cu121` / `torchvision==0.18.0+cu121` (was `2.3.1`, an incompatible pair once vLLM entered dependency resolution)

\- `requirements.txt` — removed `vllm==0.4.3` from the base install: it pins `vllm-flash-attn==2.5.8.post2`, no longer published on PyPI, and the GCSG hooks that need it are still `NotImplementedError` stubs (Sprint 3)

\- `requirements-vllm.txt` (new) — parks the vLLM pin separately; install manually once the GCSG hook code exists and the right vLLM version can be verified against it



\*\*Smoke test\*\*

\- `scripts/smoke\_test.py` — `check\_vllm()` no longer fails the suite when vLLM is absent; not installed is the expected state until Sprint 3

\- `scripts/smoke\_test.py` — `check\_aiofiles()` / `check\_prometheus\_client()` read the version via `importlib.metadata` instead of the module's `\_\_version\_\_` attribute, which neither package exposes



\*\*M1 — Expert Access Table\*\*

\- `src/eat/types.py` — added the `SHARD\_SIZE\_MB` / `SHARD\_SIZE\_BYTES` constants, which `src/tier/io.py` was already importing from here but that were only ever defined in `src/eat/slab.py`; `slab.py` now imports them from `.types` instead of redefining them



\*\*CI (found on the first real run, not by inspection)\*\*

\- `.github/workflows/ci.yml` — `cpu-tests` now installs `torch==2.3.0+cpu` (\~190 MB, no bundled CUDA libs) instead of skipping torch entirely: `TierManager.\_\_init\_\_` unconditionally constructs a `GPUTransfer`, which needs `torch` importable even for tests that never touch real CUDA — `TestTierManager` was failing with 4 errors until this was added



\### Dev constraints (documented, not bugs)



\- Pinned CUDA memory (`cudaMallocHost`) not used — Docker/Windows overhead; standard `cudaMemcpy` path instead

\- `io\_uring` replaced by `asyncio + aiofiles` — not available on WSL2/Docker Windows

\- PMEM (EMH-2) absent — DDR4 acts as proxy; `SlabAllocator` is PMEM-ready via `libpmem2` mmap

\- Single GPU (RTX 3090) — AER disabled, dual-GPU benchmarks deferred to RTX 5080 arrival

\- M4 (RecursiveMAS LED Bridge) — out of PoC scope, deferred



\### Status



\*\*Released.\*\* All module interfaces are defined; all methods raise `NotImplementedError` — the intended baseline for Sprint 1, `make test` shows all tests failing deterministically as expected. The dev environment itself is verified end-to-end: `make build` produces a working image and `make smoke` passes 13/13 checks (Python 3.12, PyTorch 2.3.0+cu121, CUDA, RTX 3090 24 GB VRAM, GPU tensor roundtrip, NVMe volume, all pinned packages, `src/` importable).

Both CI jobs verified green through an actual run, not just by reading the YAML: `cpu-tests` on a real push, `full-gpu-tests` via manual `workflow\_dispatch` against the newly-registered `Z8-G4-RTX3090` runner (54 collected, 27 passed / 27 skipped, 0 failed).



\---



\*Next release: Sprint 1 — M1 (EAT) implementation complete, `make test-eat` passing green.\*

