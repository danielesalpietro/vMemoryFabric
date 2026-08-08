# Logbook

Dev diary for OSX-PoC — the "how we actually got here" story behind the
`CHANGELOG.md` entries. One section per working session.

---

## 2026-08-08 — Möllstorp, continued: technical report, real-hardware benchmarking, honest negative results

**Release:** [Möllstorp] v0.2.0-dev — same release as below, later the same day.

### What we set out to do

Write up M1 formally — a PhD-style technical report, deliberately not
oversold as a peer-reviewed contribution — and then actually stress-test the
claims in it instead of letting a first draft stand. Two rounds of that
stress-testing happened this session.

### Round 1: get off the wrong hardware

The report's first draft measured everything on a throwaway
`python:3.12-slim` container, because that's what was available locally —
flagged honestly in the draft as a "wrong hardware" limitation. Since the
project's self-hosted GPU runner (`Z8-G4-RTX3090`) turned out to be online,
closed that gap for real instead of leaving it as a caveat: added a step to
`full-gpu-tests` that runs `benchmarks/bench_eat.py` on the actual target
workstation and uploads the JSON as a workflow artifact, then triggered it
via `workflow_dispatch`. Target hardware came back faster (as expected) but
not qualitatively different — lookup latency was still microseconds, not the
nanoseconds the original module docstring implies, which is itself a useful
result: it rules out "the container was the problem" as an explanation for
that gap.

### Round 2: the questions a reviewer would actually ask

Went back to the report with a reviewer's eye (prompted by a very specific,
useful piece of feedback) and it was right: §5 demonstrated feasibility, not
benefit. Fixed that by extending `benchmarks/bench_eat.py` with three more
measurements, all without touching `src/eat/*` — deliberately, since the
28-byte `EATEntry` layout gap is real but not worth optimizing before M2
exists to tell us which component is actually the bottleneck:

1. **A plain-dict baseline**, no Bloom filter, same workload, same seed.
   Result was not the one we expected going in: the baseline is **~5× faster
   on lookups and ~14.6× faster on inserts**. The Bloom filter's fast-negative
   path doesn't save time here — it costs 1.4–2.1 µs extra per lookup —
   because the thing it's protecting against (a dict miss) is already O(1)
   in-memory. A Bloom filter earns its keep against something slow (disk,
   network); it doesn't earn much against another hash table. Worth stating
   plainly rather than quietly keeping the Bloom filter because it was
   already built.
2. **A contention profile** — 4 readers + 1 writer against a shared EAT.
   Reader p50 latency degraded a predictable ~10×, but p95 degraded 482× and
   p99 degraded **1,360×** (single-digit µs to several milliseconds). Python's
   `RLock` has no fairness guarantee, so under sustained writer activity a
   minority of readers queue behind bad luck rather than bad average-case
   behavior. This turns the "CAS counter" documentation gap flagged in the
   report's design section from a wording nitpick into something with a
   measured cost — updated that section to say so.
3. **Slab allocator at 32 slots (8 GB) vs. the 4-slot (1 GB) default.**
   No latency growth — consistent with the free-list's O(1) design, at least
   across that 8× increase. Still two orders of magnitude short of the
   ~1,000-slot production target, so this narrows that limitation rather than
   closing it.

Also caught, and reported rather than smoothed over: running the identical
benchmark on the identical RTX 3090 twice gave insert throughput numbers
133,406 and 176,720 ops/s — a ~32% spread with no code change between runs.
Added "single-run measurements, no statistical repetition" as its own
limitation rather than quietly picking the better number.

### Why this matters going into Sprint 2

Both new findings are now concrete M2-adjacent design questions instead of
open-ended ones: whether the Bloom filter belongs in the hot path at all
(§8, future work), and whether the RLock needs to become a real
reader-writer lock or lock-free structure before M2's Tier Manager and M3's
Scheduler start generating the concurrent traffic the contention benchmark
approximated. Recorded both as Sprint 2 candidates rather than fixing them
now, per the same reasoning as the SlabAllocator-standalone decision below:
don't optimize a component in isolation before the system that will actually
stress it exists.

### End of day state

- Technical report: title, abstract, 9 sections + appendices, revised twice
  (container-only draft → real-hardware confirmation → baseline/contention/
  scale extension). Not committed to the repository (delivered directly).
- `benchmarks/bench_eat.py` — now four sections: EAT-with-Bloom, plain-dict
  baseline, contention, slab-scale. Committed.
- `.github/workflows/ci.yml` — `full-gpu-tests` runs the benchmark and
  uploads `bench-eat-result` on every `workflow_dispatch`. Committed.
- Two full-gpu-tests runs completed green on real hardware this session
  (`31259547453`, `31260457679`), both pytest and the benchmark.

Next session: Sprint 2 — "Eketorp" — M2 (EMH Tier Manager): promotion/
eviction, SEE policy, async NVMe I/O, `make test-tier` green. First items on
the list, straight from this session's findings: decide the Bloom filter's
fate, and design the concurrency model the contention benchmark says the
current RLock won't survive unchanged.

---

## 2026-08-08 — Möllstorp: M1 (EAT) implemented, `make test-eat` green

**Release:** [Möllstorp] v0.2.0-dev — closed out today.

### What we set out to do

Pick up exactly where Karlshamn left off: implement the three M1 pieces — the
2-level Bloom filter, the DDR4 Slab Allocator, and the `ExpertAccessTable`
CRUD/lifecycle methods — and turn `tests/test_eat.py` from a suite that
deliberately expects `NotImplementedError` into one that exercises real
behavior.

### The two real design calls

1. **Should `EAT.insert()` reach into the SlabAllocator when `tier=DDR4`?**
   `EATEntry`'s documented 28-byte layout has no `slot_idx` field, and doing
   this properly would mean adding a reverse `(expert_id, shard_idx) → slot_idx`
   index inside `SlabAllocator` that nothing currently asks for. Decided
   against it: for Sprint 1, `SlabAllocator` stays a standalone, independently
   tested component. `EAT.initialize()`/`shutdown()` still drive its lifecycle,
   but physical DDR4 promotion/eviction is the Tier Manager's job — that's
   M2, Sprint 2, and it's the natural place to decide how slot ownership
   should actually be tracked.
2. **Bloom filter keys.** Used explicit strings (`e:{expert_id}`,
   `s:{expert_id}:{shard_idx}`) instead of passing raw ints/tuples into
   `pybloom_live`, so correctness doesn't depend on how that library hashes
   compound keys internally.

### Implementation

Straightforward once those two calls were made — `bloom.py`, `slab.py`,
`eat.py` filled in against the docstrings and type hints already left by the
Sprint 0 skeleton, which turned out to specify the contract precisely enough
that there was very little guessing involved.

### Tests

Rewrote every test in `tests/test_eat.py`: replaced each
`pytest.raises(NotImplementedError)` / `pytest.skip("TODO Sprint 1")` with a
real assertion, and added a handful that didn't exist before —
`test_initialize_shutdown` (the lifecycle methods had zero coverage),
duplicate-insert / missing-key edge cases, and an LRU-ordering check for
`eviction_candidates`. The concurrency tests actually spin up 8 threads: one
inserts 10k disjoint keys and checks for lost writes, another mixes
concurrent readers and writers, and a third fires many concurrent
`update_tier` calls at a single entry and asserts the final `version` equals
the exact call count — proving the `RLock` serializes the CAS bump instead of
losing increments.

### Verification — real run, not a read-through

No local Python outside Docker in this environment, and the full CUDA dev
image wasn't built here yet (would've meant a slow multi-minute rebuild just
to run logic that never touches torch/CUDA). Instead mirrored what the CI
`cpu-tests` job actually does — install the CPU-only dep subset directly,
skip the CUDA image entirely — via a throwaway `python:3.12-slim` container:

```
docker run --rm -v "$(pwd)/osx-poc:/work" -w /work python:3.12-slim bash -c \
  "pip install -q pytest==8.2.1 numpy==1.26.4 pybloom-live==4.0.0 && \
   PYTHONPATH=src pytest tests/test_eat.py -v --tb=short"
```

**24 passed, 0 failed**, first try. `benchmarks/bench_eat.py` (rewritten from
the `"status": "pending"` placeholder to a real P50/P95/P99 + throughput
benchmark) came back with p50 ≈ 3.3 µs / p95 ≈ 3.5 µs / p99 ≈ 5.5 µs lookup
latency and ≈ 99k inserts/sec on the verification container — well inside the
documented <10 µs target, though that number is from a shared container host,
not the RTX 3090 dev box, so it's a sanity check rather than a real
benchmark run.

One minor detour: the first `docker run` invocation failed with
`the working directory 'C:/Program Files/Git/work' is invalid` — Git Bash's
MSYS layer was rewriting `/work` as a Windows path before Docker ever saw it.
Fixed with `MSYS_NO_PATHCONV=1`.

### End of day state

- `tests/test_eat.py` → 24/24 passing (verified via lightweight container, not `make test-eat`'s full CUDA image)
- `benchmarks/bench_eat.py` → real numbers, no longer a placeholder
- M1 (Bloom filter, Slab allocator, EAT CRUD) fully implemented
- M2/M3 untouched — still Sprint 0 skeletons, as planned
- CHANGELOG/README updated for the Möllstorp release

Next session: Sprint 2 — M2 (Tier Manager): promotion/eviction, SEE policy,
async NVMe I/O, `make test-tier` green. This is also where the
SlabAllocator-standalone decision above gets revisited — the Tier Manager is
the component that will actually call `slab.alloc()`/`free()` on promotion
and eviction.

---

## 2026-08-07 — Karlshamn: environment stood up, CI wired, GPU runner live

**Release:** [Karlshamn] v0.1.0-dev — closed out today.

### What we set out to do

Get `docker compose build` working, then everything that naturally follows
from having a real working environment: CI, and a way to actually run the
GPU-dependent tests somewhere other than a laptop.

### The build fight

`docker compose build` didn't work on the first try — or the second, or the
fifth. In order:

1. **Python 3.12 not found.** `nvidia/cuda:12.1.1-devel-ubuntu22.04` is
   Ubuntu 22.04 (jammy), which only ships Python 3.10 in its default repos.
   Fixed with the deadsnakes PPA.
2. **`pip` missing after switching to 3.12.** `apt`'s `python3-pip` installs
   pip for the *default* interpreter (3.10), not deadsnakes' 3.12. Fixed by
   bootstrapping with `python3.12 -m ensurepip`.
3. **torch/torchvision/vllm dependency conflict.** `torchvision==0.18.1`
   wanted `torch==2.3.1`; `vllm==0.4.3` wanted `torch==2.3.0` exactly. Pinned
   both to the `2.3.0` pairing.
4. **`xformers` (a vllm dependency) failed to build** — its `setup.py`
   imports `torch` outside pip's build-isolation sandbox, so it needs torch
   pre-installed and `--no-build-isolation`. Tried scoping that flag to just
   `vllm`; applying it too broadly then broke `gpustat`'s `setuptools_scm`
   build step, which relies on normal isolation to fetch its own build deps.
5. **The real wall: `vllm==0.4.3` pins `vllm-flash-attn==2.5.8.post2`, which
   is gone from PyPI** (only `2.6.1`/`2.6.2` remain). At that point we
   stepped back instead of hacking around it.

**Decision:** the vLLM pin was premature. The GCSG hooks that would use it
are still `NotImplementedError` stubs — there was no actual code depending
on vLLM 0.4.3's internal API yet, so nothing to protect by forcing the
version. Pulled `vllm` out of `requirements.txt` entirely into
`requirements-vllm.txt`, to be revisited at Sprint 3 once the hook code
exists and we can verify which version it actually needs against real code
instead of a comment.

Build went green right after. `make smoke` — 13/13 on the first real run
(RTX 3090, 24 GB VRAM, CUDA tensor roundtrip, NVMe volume, all packages).

Along the way, actually running the smoke test (as opposed to just getting
the build to pass) surfaced three unrelated pre-existing bugs: `SHARD_SIZE_MB`
imported from `eat.types` but only ever defined in `eat/slab.py`, and two
smoke-test checks reading `.__version__` on packages that don't expose it
(`aiofiles`, `prometheus_client`). Fixed all three same-day.

### CI — extra scope

Added `.github/workflows/ci.yml`: `cpu-tests` on every push/PR, `full-gpu-tests`
gated to manual `workflow_dispatch` (GitHub Actions has no per-job trigger,
so the whole workflow listens for push/PR/dispatch and the GPU job
self-gates with an `if:`).

**Tested it for real, not just by reading the YAML** — and that's exactly
what caught the next bug: the first push-triggered run of `cpu-tests` failed.
`TestTierManager`'s fixture constructs a real `TierManager`, whose
`__init__` unconditionally builds a `GPUTransfer` — which needs `torch`
*importable* (not real CUDA) to not raise. The CPU job didn't install torch
at all. Fixed by installing `torch==2.3.0+cpu` (~190 MB, no bundled CUDA
libs) instead of widening the `@pytest.mark.gpu` net over tests that don't
actually need a GPU.

### Standing up the self-hosted GPU runner

Registered `Z8-G4-RTX3090` against the repo, added the `gpu` custom label
via the API (`POST .../runners/{id}/labels` — faster than reconfiguring the
runner). First `workflow_dispatch` test run failed twice before working:

1. **PowerShell execution policy.** Windows runners execute each `run:` step
   as a `.ps1` script; the box had `Restricted` policy, blocking all of them.
   Fixed with `Set-ExecutionPolicy RemoteSigned -Scope LocalMachine`
   (GitHub's own recommended policy for Windows runners).
2. **`permission denied` on the Docker named pipe.** The runner service
   account wasn't in the local `docker-users` group. Decided to stop running
   the service under whatever default account it had and give it a proper
   dedicated identity: created a local `GitRunner` account, added it to
   `docker-users`, pointed the service at it via `sc.exe config obj=`.
   Hit a second snag here — `sc.exe config` doesn't grant "Log on as a
   service" the way the Services GUI does automatically, so the service
   still wouldn't start until that right was granted through
   `services.msc`.
3. **Self-inflicted one:** the first generated password contained `$ZVQ`,
   which PowerShell double-quoted strings interpolate as a variable
   reference — silently truncating the password differently depending on
   which command it was pasted into (`ConvertTo-SecureString "..."` vs. the
   literal GUI text field). Two different effective passwords, one account —
   guaranteed logon failure. Regenerated a password with no `$` in it and
   used single-quoted strings throughout to make the mistake impossible to
   repeat.

Once the service ran as `GitRunner`, `full-gpu-tests` went green:
**27 passed, 27 skipped, 0 failed** — the real suite, with real CUDA, not a
skip-because-no-GPU stub.

### Housekeeping

Set up `.gitignore` (none existed — caught `__pycache__/` before it got
committed), created a GitHub Project (`OSX-PoC Roadmap`) with one card per
sprint plus the runner setup, wrote `SELF-HOSTED.MD` so the runner setup
above doesn't have to be rediscovered from scratch next time.

### End of day state

- `docker compose build` → green
- `make smoke` → 13/13
- CI `cpu-tests` → green (push/PR)
- CI `full-gpu-tests` → green (manual, real GPU, 27/27 non-skipped passed)
- Project board created, Sprint 0 + runner setup marked Done
- Everything pushed to `develop`

Next session: Sprint 1 — M1 (EAT) implementation, `make test-eat` green.
