# Logbook

Dev diary for OSX-PoC — the "how we actually got here" story behind the
`CHANGELOG.md` entries. One section per working session.

---

## 2026-08-08 — Eketorp, wrap-up: issues filed, roadmap board updated, docs closed out

**Release:** [Eketorp] v0.3.0-dev — no code change, pure housekeeping to close the sprint out properly before starting M3.

### What we set out to do

Turn everything left as a LOGBOOK/CHANGELOG note across Möllstorp and
Eketorp into something that survives independently of those files:
GitHub Issues, so a future session (or a different person) picking up
this repo doesn't have to re-read three LOGBOOK sessions to know what's
still open.

### Filed 8 issues, evaluated before creating

Went through Karlshamn → Möllstorp → Eketorp looking for items
explicitly marked deferred/carried-over, not vague "could be nicer"
observations. Landed on 8: Bloom filter fate (#1) and RLock contention
(#2) — both measured, both deferred twice now; the `bench_tier.py`
warm-up gap (#3) found earlier today; `BloomFilter.remove_expert()`
leaving evicted shards as permanent false positives (#4), never
actually tracked anywhere before; missing CUDA stream pipelining (#5),
deferred since Karlshamn; no `ruff`/`pyproject.toml` config (#6),
noticed this session while verifying M2; and the two hardware-blocked
ones, PMEM (#7) and dual-GPU/AER (#8), which have been sitting as dev
constraints since day one with nowhere to track "what to do once the
hardware shows up."

Asked before creating rather than just doing it — issue creation is
visible/shared-state, and the user picked all 8.

### Project board

`OSX-PoC Roadmap` (GitHub Project) predates this session — created in
Karlshamn with draft-issue cards, one per sprint. Updated it to match
reality instead of drifting further from the CHANGELOG:

- Sprint 2 card → Done, body replaced with the real outcome (GPU
  verification, the bug found+fixed, real bench numbers) instead of
  the stale "Target: make test-tier passing green" placeholder
- Sprint 3 card → left at Todo (not started), but body now references
  issues #1/#2/#5 since they're directly relevant to how M3 should be
  approached
- All 8 new issues added to the board as real linked items, not just
  floating in the Issues tab

### Docs

`README.md` (both root and `osx-poc/`) updated to Eketorp — release
banner, CI section (`bench_tier.py` step, `TestTierManagerGPU`), repo
structure comments, roadmap table (Sprint 2 now ✅), and a new "Known
limitations / open issues" table linking all 8 issues with a one-line
reason each. `CHANGELOG.md`'s Eketorp entry got a `### Tracking`
section pointing at the same 8 issues and the board update, so the
release entry and the issue tracker cross-reference each other instead
of one silently going stale.

### End of day state

- 8 GitHub Issues filed and linked from README/CHANGELOG/project board
- Project board reflects actual sprint status, not Sprint-0-era
  placeholders
- README (root + osx-poc) current as of Eketorp

Next session: Sprint 3 — M3 (Expert Scheduler). Planning conversation
starts fresh, informed by issues #1/#2/#5 rather than re-deriving them
from LOGBOOK archaeology.

---

## 2026-08-08 — Eketorp, continued: real GPU verification, a real bug, honest benchmark numbers

**Release:** [Eketorp] v0.3.0-dev — same release as below, closed out for real this time.

### What we set out to do

Finish what the first Eketorp session left explicitly open: actually run
the `@pytest.mark.gpu` tests and `benchmarks/bench_tier.py` on real
hardware, instead of shipping M2 "done" on CPU-only coverage alone.

### The machine is the runner

Learned partway through this session: the box this Claude Code session
runs on **is** the self-hosted `Z8-G4-RTX3090` runner — confirmed via
`nvidia-smi` (RTX 3090 present) and the `GitHub Actions Runner` Windows
service running locally. That changed the iteration loop for the rest
of the session: instead of push → `workflow_dispatch` → wait → read
logs for every attempt, later iterations ran `docker compose run`
directly against the live-mounted repo. Saved at least two full CI
round-trips once the first attempt turned up a real bug.

### First `workflow_dispatch` run: a real bug, not flakiness

Pushed the M2 work (3 commits: implementation, CI step, docs) and
triggered `workflow_dispatch`. `CPU Tests` passed; `Full GPU Tests`
failed at the test-suite step —
`TestTierManagerGPU::test_evict_to_free_vram_evicts_candidate`, with
`evict_to_free_vram()` raising `MemoryError` ("tier VRAM vuoto") right
after successfully evicting the only VRAM entry. 58/59 GPU-marked
assertions passed; this was the one real gap.

Root cause: `vram_free_bytes()` wraps `torch.cuda.mem_get_info()`,
which reports **driver-level** free memory. PyTorch's caching
allocator doesn't return freed blocks to the driver on `del tensor` —
it holds them for reuse. So after `evict()` dropped its VRAM tensor,
`vram_free_bytes()` didn't move, the `evict_to_free_vram()` loop ran
again, found no more VRAM candidates, and correctly raised. First fix:
added `GPUTransfer.empty_cache()` (`torch.cuda.empty_cache()`), called
from `TierManager.evict()`'s VRAM branch. Pushed, re-triggered.

### Second attempt: same failure, more precisely

Still failed — `free_after_evict > free_with_shard` asserted
`24167579648 > 24167579648`, exactly equal down to the byte.
`empty_cache()` only returns blocks with **zero live references**, and
the local `tensor` variable inside `evict()` still referenced the
tensor after `del self._vram[key]` — that only dropped the *dict's*
reference. Fixed for real: `self._vram.pop(key)` followed by an
explicit `del tensor` before `empty_cache()` runs, so the last
reference is gone before asking the allocator to release the block.

### Verifying the real fix — locally first, then officially

Given the "this machine is the runner" finding, ran
`docker compose run --rm osx-dev bash -c "cd osx-poc && PYTHONPATH=src pytest tests/test_tier.py -v"`
directly against the local RTX 3090 before pushing again — **28/28
passed**, including a new regression test
(`test_evict_frees_vram_visible_to_mem_get_info`) asserting
`vram_free_bytes()` actually increases after an eviction, not just
that eviction doesn't raise. Ran the full suite too: 60/72 (12 skips
are unrelated M3 stubs). Then pushed the real fix and triggered
`workflow_dispatch` one more time for the official record — green in
both jobs (`CPU Tests` 35s, `Full GPU Tests` 45s), confirming the local
run wasn't specific to some container-state difference.

### Comparing M1 vs M2 on real numbers

Ran both `benchmarks/bench_eat.py` and `benchmarks/bench_tier.py`
locally and via the official CI artifacts (`bench-eat-result`,
`bench-tier-result`) — consistent between the two runs. EAT lookups
(pure in-memory): p50≈2.6µs. `TierManager.promote()` NVMe→DDR4 (real
file I/O via aiofiles, 4MB synthetic shard): p50≈5.2ms — about 2000×
slower, which is exactly the expected shape (disk I/O vs. a dict).
DDR4→VRAM: p50≈77ms but p95/p99 spike to 1.3-1.4 **seconds**,
reproduced near-identically in both runs. That's not noise — it's the
one-time CUDA context/allocator warm-up cost on the first real GPU
call in a fresh process, and `bench_tier.py` has no warm-up iteration
before timing, so with only 20 samples that one cold call dominates
the tail. Recorded as an honest benchmark-methodology limitation in
the CHANGELOG rather than a `TierManager` performance problem — not
fixed this session, flagged for whoever picks up the benchmark next.

### End of day state

- `src/tier/gpu.py` / `src/tier/manager.py` — real fix committed (pop +
  explicit `del` before `empty_cache()`), not just the first attempt
- `tests/test_tier.py` — 28/28 passing on real CUDA, new regression
  test for the VRAM-release bug
- CI run `31263503614` — both jobs green, `bench-tier-result` artifact
  uploaded and compared against `bench-eat-result`
- CHANGELOG updated from "partially verified" to actually verified,
  with the bug and the real numbers both recorded
- Saved a memory note: this machine == the GPU runner, useful for every
  future GPU-dependent debugging session on this repo

Next session: Sprint 3 — M3 (Expert Scheduler): PT-PEP classifier, GCSG
guard, AER stub.

---

## 2026-08-08 — Eketorp: M2 (Tier Manager) implemented, CPU-runnable subset green

**Release:** [Eketorp] v0.3.0-dev — closed out today (partially — see Status below).

### What we set out to do

Pick up where Möllstorp's LOGBOOK left off: implement M2 — `AsyncNVMeIO`,
`GPUTransfer`, `LRUPolicy`/`SEEPolicy`, and `TierManager` — turning
`src/tier/*` from Sprint-0 `NotImplementedError` skeletons into working
NVMe→DDR4→VRAM promotion/eviction, and decide what to actually do about
the two open findings Möllstorp flagged as "Sprint 2 candidates": the
Bloom filter's net-negative cost, and the RLock's tail-latency behavior
under contention.

### Planning first, before any code

Given how much M1 left genuinely open, this session started with a
planning pass rather than diving straight into the skeletons. Three
scoping questions got settled explicitly before writing anything:

1. **Bloom filter** — leave it alone. Nothing in M2 depends on its
   behavior, and revisiting it now would reopen "closed" M1 files for a
   finding that's better addressed once there are end-to-end numbers
   (Sprint 4, per Möllstorp's own CHANGELOG note).
2. **EAT's RLock** — also leave it alone, but with a concrete rule
   instead of ignoring the finding: `TierManager` only touches EAT for
   fast synchronous ops (`lookup`, `update_tier`), never while an NVMe
   read or GPU transfer is in flight. No new contention on top of what
   M1 already measured, no RLock redesign needed for M2 to function.
3. **Slot ownership** — `TierManager` keeps its own
   `{key: slot_idx}`/`{key: tensor}` bookkeeping rather than adding a
   `slot_idx` field to `EATEntry`. Keeps M1's documented 28-byte layout
   untouched, per the LOGBOOK's own note that M2 is "the natural place
   to decide how slot ownership should actually be tracked" — decided
   here as *don't put it in EATEntry*, not *don't decide at all*.

One thing that *did* require touching an M1 file, flagged explicitly
rather than folded in quietly: `ExpertAccessTable` needed a read-only
`slab` property. `EAT.initialize()`/`shutdown()` already drive the
`SlabAllocator`'s lifecycle — `TierManager` needs to alloc/free DDR4
slots on that *same* instance, not spin up a second, disconnected
allocator. Pure accessor, no change to locking or `EATEntry`.

### The bug planning actually caught

Draft one of the concurrency model said "TierManager never holds the
EAT lock across an `await`, and asyncio's cooperative scheduling
handles the rest." That's true for the *slab mutation itself* (no
`await` in the middle of `alloc`/`free`), but it missed a real race:
two concurrent `promote()` calls for the same `(expert_id, shard_idx)`
— the normal shape of a prefetch batch with overlapping shards — both
see the shard at NVME before either one's `await io.read_shard()`
returns, both allocate their own slab slot, both call
`eat.update_tier(DDR4)`. One slot ends up leaked. This is a real bug
under asyncio's single-threaded event loop, no OS threads required —
caught during the plan-review pass, not after writing the code.

Fix: `TierManager` keeps a lazily-created `dict[(expert_id, shard_idx), asyncio.Lock]`
and acquires the key's lock for the *entire* transition — from the
initial `eat.lookup()` through the final `eat.update_tier()`/slab
free — not just around the slow I/O. A regression test
(`test_concurrent_double_promote_no_slab_leak`) runs two `promote()`
calls concurrently via `asyncio.gather(..., return_exceptions=True)`
and asserts exactly one slab slot ends up allocated — the second call
legitimately raises `ValueError` (same-tier, since the lock serializes
it behind the first) rather than silently losing a slot.

Two smaller review points, both folded into the implementation:
`AsyncNVMeIO.read_shard` needed an *explicit* `dtype=np.uint8` in
`np.frombuffer` (not the implicit default) — deliberately matching
`SlabAllocator`'s pool dtype, since shards are opaque byte payloads
through this whole pipeline. And `evict()` needed its single-hop-only
behavior stated explicitly in the docstring and covered by a test,
specifically *because* `promote()` does chain NVME→VRAM in one call —
the asymmetry needs to be obvious to a reader, not inferred.

### Implementation

Once the design was settled, filling in `policies.py` → `io.py` →
`gpu.py` → `manager.py` against the docstrings/type hints left by the
Sprint 0 skeleton was straightforward — same experience as M1.
`SEEPolicy.score` normalizes access-count and recency to `(0, 1]`
before weighting them (no natural common scale between a counter and a
duration in seconds); σ stays a `0.0` stub in both branches, but
*doesn't* redistribute weight when a caller passes `context_vec` — a
deliberate difference from the `None` branch, so it doesn't silently
imply PT-PEP integration that doesn't exist until M3.

### Tests

Rewrote every test in `tests/test_tier.py`, same pattern as M1's
`test_eat.py`: every `pytest.raises(NotImplementedError)` /
`pytest.skip(...)` replaced with a real assertion. Split by what
actually needs CUDA, mirroring the existing `cpu-tests`/`full-gpu-tests`
CI job split — `TestTierManagerGPU`, the GPU half of `TestGPUTransfer`,
and the VRAM-side integration test are all `@pytest.mark.gpu`.

### Verification — CPU-runnable subset only, said plainly

No local GPU/CUDA image in this environment, same wall M1 hit. Verified
the non-GPU path the same way M1 did: a throwaway `python:3.12-slim`
container with the CI `cpu-tests` dependency set.

```
docker run --rm -v "$(pwd)/osx-poc:/work" -w /work python:3.12-slim bash -c \
  "pip install -q torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
   pip install -q pytest==8.2.1 pytest-asyncio==0.23.7 numpy==1.26.4 aiofiles==23.2.1 pybloom-live==4.0.0 && \
   PYTHONPATH=src pytest tests/ -m 'not gpu' -v"
```

**50 passed, 12 skipped (GPU-only), 9 GPU-marked deselected** — first
try, across `test_eat.py`, `test_scheduler.py`, and the rewritten
`test_tier.py` (18/18 on its own). `ran ruff` out of curiosity on the
touched files; it flagged pre-existing typing-style debt
(`Dict`/`List`/`Optional` vs. `dict`/`list`/`X | None`) that predates
this session across the whole codebase — no `pyproject.toml`/`ruff.toml`
exists, so `ruff check` runs on bare defaults against files this
session didn't touch too. Left alone as out of scope for M2.

**Not verified this session, and saying so plainly**: every
`@pytest.mark.gpu` test, the `ddr4_to_vram` section of the new
`benchmarks/bench_tier.py`, and the `full-gpu-tests` CI step added for
it. All of that needs a real `workflow_dispatch` run against the
self-hosted `Z8-G4-RTX3090` runner, same as M1's benchmark — not
triggered in this session. `make test-tier` (the full CUDA Docker
image) also wasn't run for the same reason; the lightweight container
above approximates the CPU-only subset of it, not the target itself.

### End of day state

- `src/tier/{policies,io,gpu,manager}.py` — fully implemented
- `src/eat/eat.py` — one new `slab` read-only property, otherwise
  untouched
- `tests/test_tier.py` — rewritten; 18/18 CPU-runnable tests pass, GPU
  half written but unexecuted
- `benchmarks/bench_tier.py` — real NVMe→DDR4 numbers obtainable
  anywhere; DDR4→VRAM section implemented but unexecuted (needs CUDA)
- `.github/workflows/ci.yml` — `bench_tier.py` step added to
  `full-gpu-tests`, unexecuted this session
- CHANGELOG/LOGBOOK updated, explicitly marked "partially verified"
  rather than claiming GPU coverage that wasn't run

Next session: trigger `workflow_dispatch` on `Z8-G4-RTX3090` to actually
run the `@pytest.mark.gpu` tests and the full `bench_tier.py` (including
`ddr4_to_vram`) on target hardware — the same close-the-loop step M1
took in its second Möllstorp session — before calling M2 done. After
that: Sprint 3 — M3 (Expert Scheduler).

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
