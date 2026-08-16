# Logbook — Issue #33

Dev diary scoped to GitHub issue #33 ("Tier DDR4 attivo: pattern
CPU-compute-offload ispirato a exllamav3 moe_cpu_host.py"), kept separate
from the main `LOGBOOK.md` because this piece of work is expected to span
many sessions and shouldn't dilute — or get lost inside — the general
project diary. Same convention as the main log: one section per working
session, prose, record corrections rather than erasing them.

---

## 2026-08-16 — Issue #33 opened, claim audited and confirmed, hybrid VRAM+CPU/AVX-512 plan agreed

**Release:** none — investigation and planning only, no code shipped yet.

### What we set out to do

Issue #33 claims that `GPUTransfer.to_vram()` (`osx-poc/src/tier/gpu.py:60-105`)
is today the *only* way DDR4 interacts with the rest of the system — a
blocking/passive transfer — and that no computation ever happens in the DDR4
tier, despite the "Tier Manager" name and the EMH-1a/EMH-1c tiers in the
README suggesting a more active role. Before acting on the issue's proposal
(a CPU-compute-offload pattern inspired by `exllamav3/model/moe_cpu_host.py`,
`MoeCpuHost`), the claim itself needed verifying against the actual code, not
taken on faith from the issue body.

### What we did

Read `tier/gpu.py` (`GPUTransfer.to_vram`/`to_ddr4`), `tier/manager.py`
(`TierManager._ddr4_to_vram`), and `scheduler/gcsg.py` end to end for any
compute path that might touch CPU-resident (DDR4) data.

Found:
- `GPUTransfer.to_vram()`/`to_ddr4()`: pure data movement — `pin_memory()` +
  `.to(device)` / `.cpu().numpy()`. No arithmetic.
- `TierManager._ddr4_to_vram()` (`manager.py:141-163`): pulls the buffer from
  the slab allocator and calls `self._gpu.to_vram(buffer)` — zero compute,
  only slot/EAT bookkeeping around the transfer.
- Stronger, indirect confirmation in `gcsg.py:1074-1092`: a comment
  documents a real bug found 2026-08-12 — when a layer is offloaded, the
  `w13`/`w2` slices stay CPU-resident, and `_quantize_int4()` was quantizing
  them in place *before* `.to('cuda')`, crashing with a device-mismatch error
  the moment the matmul ran (`Expected all tensors to be on the same device,
  cuda:0 and cpu`). Fix: force `.to("cuda")` before any compute. Notably,
  even in this buggy path no computation ever actually *completed* on
  DDR4-resident data — a CUDA kernel requires matching devices, so it just
  errored out immediately instead of silently running on CPU.
- Grepped for `device.type != "cuda"` guards across `gcsg.py`: the same
  pattern — check device, force `.to("cuda")` — appears before every real
  compute path (`_load_shadow_pool`, `_promote_module_via_tier_manager`,
  `_pin_awq_expert_to_gpu`, `_build_marlin_shadow_pool`). The codebase
  enforces GPU-residency before compute everywhere, not just in the one
  documented bug fix.

One open caveat surfaced during the audit: `_promote_module_via_tier_manager`
(`gcsg.py:1276-1327`) is itself flagged in its own docstring as *not verified
on real hardware* ("nessuna GPU in questo ambiente... primo item da
controllare sul pod") — the GPU-residency guarantee is confirmed by static
reading, not by an actual run on real hardware for that specific path.

Posted the findings as a comment on issue #33
(https://github.com/danielesalpietro/vMemoryFabric/issues/33#issuecomment-5309368795)
confirming the issue's claim in full: today DDR4 is pure passive
storage/staging (NVMe → DDR4 → VRAM and back on eviction), never a compute
site.

### Result

Claim confirmed. Discussion moved to whether to act on it now: project
owner has two Xeon Gold 6244 (8c/16t per socket, AVX-512, Cascade Lake) on
the same Z8 G4 box that already hosts the RTX 3090 self-hosted CI runner —
real, idle compute capacity that could run the "cold" side of a hybrid
MoE forward instead of just staging bytes through it.

Agreed not to jump straight to the `MoeCpuHost` sync architecture (separate
process, shared-memory job ring) — that's exactly the mistake issue #33
itself warns against: building the sync mechanism before deciding what
computation belongs on it. Worked through a phased plan instead:

- **Fase 0** — Define hot/cold routing criterion by reusing `SEEPolicy`
  scoring (`tier/policies.py:64-100`) already in EAT, instead of inventing a
  new one. Pure unit test, no hardware.
- **Fase 1** — Isolated CPU kernel. Check whether PyTorch's CPU backend
  (oneDNN/MKL, AVX-512/VNNI-capable on Cascade Lake) is fast enough before
  writing custom AVX-512 intrinsics. New `@pytest.mark.avx512` marker
  (mirrors `@pytest.mark.gpu`), numeric parity test against the existing
  `_ShadowExpertINT4.__call__()` GPU path, plus a standalone throughput
  bench.
- **Fase 2** — CPU-resident wrapper mirroring `_ShadowExpertINT4`'s
  `__call__(hidden_states, layer_id)` API, weights pinned in DDR4, never
  `.to("cuda")`. Same parametric test suite as the GPU path, device=cpu.
- **Fase 3** — Routing in `TierManager`/`GCSGWorker` (`route_forward()`),
  without touching the `Tier` enum — a "cold" expert stays logically
  `Tier.DDR4` in the EAT while being used directly for compute. Test
  asserts `GPUTransfer.to_vram()` is never called for a routed-cold expert —
  the deliberate, tested version of what was previously only an
  accidentally-enforced invariant.
- **Fase 4** — Only if Fase 1-3 justify it: evaluate whether the
  `MoeCpuHost`-style separate-process/shared-memory job ring is actually
  needed, or whether an in-process round-trip is good enough. Gate: measure
  round-trip overhead first: skip Fase 4 if it's negligible next to AVX-512
  compute time.
- **Fase 5** — End-to-end benchmark on real hardware (Z8 G4, RTX 3090 +
  both Xeon 6244), new `benchmarks/bench_hybrid.py` mirroring `bench_tier.py`,
  run on the `Z8-G4-RTX3090` self-hosted runner alongside `full-gpu-tests`.

Flagged one prerequisite quick win, independent of the phased plan: the
untested `_promote_module_via_tier_manager` pinning path should probably be
verified on real hardware before the hybrid routing (Fase 3) comes to depend
on it for the "hot" half of the split.

### Why this matters

This is the first real candidate workload for the DDR4 tier since issue #33
was opened — MoE cold-expert CPU compute is the "what to move to DDR4"
question the issue explicitly left unanswered as a precondition for any
sync-mechanism work. Nothing implemented yet; this log starts here so the
next session (or a different agent picking this up cold) has the full
reasoning chain — audit → confirmation → plan — without re-deriving it.

### Current state

- Issue #33: open, comment posted with the code audit, no PR yet.
- No code changes in this session — planning and verification only.
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: Fase 1 — bench PyTorch CPU backend (oneDNN/MKL AVX-512) vs.
`_ShadowExpertINT4`'s existing GPU path, decide whether a custom AVX-512
kernel is actually needed before writing one.

---

## 2026-08-16 (continued) — Fase 1: no custom kernel needed, `_ShadowExpertINT4` is already device-agnostic

**Release:** none — new tests + benchmark added, no production code changed.

### What we set out to do

Fase 1 of the plan above: check whether PyTorch's CPU backend (oneDNN/MKL,
AVX-512/VNNI-capable) is fast enough for the SwiGLU INT4 forward before
committing to writing custom AVX-512 intrinsics.

### What we did

Read `_ShadowExpertINT4.__call__()` (`gcsg.py:445-457`) again with this
specific question in mind, and found the answer was already sitting there:
the method is `hidden_states @ w13.T` → split → `F.silu` → `* up` →
`@ w2.T` — plain PyTorch ops with zero CUDA-specific code. It runs on
whatever device the input tensors are on. The "GPU path" in production is
GPU-only because `_load_shadow_pool()` forces `.to("cuda")` on the weights
*before* handing them to this class (the same guard documented in the
previous session's audit) — not because the class itself needs CUDA. So
Fase 1's actual deliverable wasn't a new kernel, it was proving this.

Added `tests/test_cpu_kernel.py` (4 tests, no `@pytest.mark.gpu`, runs in
the standard `cpu-tests` CI job):
- `test_runs_on_cpu_without_cuda` — builds `_ShadowExpertINT4` from
  CPU-quantized weights, calls it with a CPU `hidden_states`, asserts the
  output stays CPU (`not output.is_cuda`) — direct proof of the
  device-agnostic claim.
- `test_quantize_int4_round_trip_within_scale_tolerance` — `_quantize_int4`
  had no correctness test anywhere in the repo despite a comment in
  `test_scheduler.py:362` claiming it was "verified elsewhere" (grepped,
  found nothing — stale or aspirational comment). Verifies the per-element
  round-trip error stays within `scale/2`, as guaranteed by `torch.round()`.
- `test_shadow_expert_forward_matches_fp32_reference_within_quant_tolerance`
  — parity test: CPU INT4 output vs. an independently-computed fp32
  reference forward (same formula, unquantized weights). First attempt used
  a hand-derived per-element error bound (`scale/2` propagated through the
  two matmuls) — wrong by over an order of magnitude, because it ignored
  error accumulation over the summed `hidden`/`intermediate` dimensions.
  Replaced with an empirically-calibrated threshold: relative L2 error
  measured at 0.24–0.32 across 5 seeds (ad-hoc script, not committed),
  asserted `< 0.5`. Recorded as an explicit deviation, same convention
  `bench_tier.py` already uses for its own declared approximations.
- `TestThroughputSmoke` (`@pytest.mark.avx512`, new marker registered in
  `pytest.ini`) — loose wall-clock sanity check (not a real benchmark),
  skipped via a `/proc/cpuinfo` flag check if `avx512f` isn't present.

Added `benchmarks/bench_cpu_kernel.py` (mirrors `bench_tier.py`'s
structure/conventions): P50/P95/P99 latency and a GFLOPS-at-p50 estimate
for `_ShadowExpertINT4`'s forward on CPU, at 3 batch sizes, with a
`gpu_reference` section that runs the identical forward on CUDA for a
real side-by-side — `{"status": "skipped", ...}` if no GPU, not a fake
number. Default synthetic dims (512×1536) are a declared fraction of
Mixtral's real 4096×14336 (see module docstring for the env-var override
to run at real scale); this run still needs repeating on the actual Xeon
6244 hardware for a meaningful number, same caveat `bench_tier.py` already
carries for its own synthetic shard size.

Added `make test-cpu-kernel` / `make bench-cpu-kernel` targets. Did not
wire the new benchmark into `ci.yml`'s `full-gpu-tests` job yet — Fase 1 is
explicitly exploratory ("decide if a custom kernel is needed"), wiring a
benchmark into CI with an acceptance threshold is Fase 5 territory once the
kernel approach itself is settled.

Verified locally (no real GPU/AVX-512 hardware confirmed beyond a
container CPU that happens to expose `avx512f`): `pytest tests/ -m "not
gpu"` — 114 passed, 3 skipped (pre-existing), 0 failed, no regressions.
`bench_cpu_kernel.py` runs clean, `gpu_reference` skips cleanly with no
CUDA present, matching `bench_tier.py`'s established pattern.

### Result

No new CPU kernel was written, and per the plan's own Fase 1 gate, none is
needed yet: PyTorch's existing CPU backend already executes the real
production forward class unmodified. Fase 2 (a CPU-resident wrapper
"mirroring" `_ShadowExpertINT4`) is now simpler than planned — it doesn't
need a new class, `_ShadowExpertINT4` already qualifies as long as its
weights are constructed/kept CPU-resident, which is exactly what
`test_runs_on_cpu_without_cuda` demonstrates.

Numbers from this sandbox (not the real Xeon 6244, no AVX-512-vs-scalar
comparison possible without a second run on hardware without the flag):
batch=32 forward at 512×1536 → ~3.8ms p50, ~40 GFLOPS at p50. Not
production-representative — logged as a sandbox sanity number, not a
Fase 5 result.

### Why this matters

This closes the specific open question Fase 1 was scoped to answer, and
changes the shape of Fase 2: instead of building something new, Fase 2 is
now about wiring `_ShadowExpertINT4` into a CPU-resident code path inside
`TierManager`/`GCSGWorker` and testing that it's *never* promoted to CUDA
— reusing, not reinventing, issue #33's own "conferma indiretta" evidence
(the 2026-08-12 bug fix that forces `.to('cuda')` before compute) as the
thing Fase 2's test needs to assert does NOT happen for cold experts.

### Current state

- `tests/test_cpu_kernel.py`, `benchmarks/bench_cpu_kernel.py` added.
- `pytest.ini`: new `avx512` marker registered.
- `Makefile`: `test-cpu-kernel`, `bench-cpu-kernel` targets added.
- Full CPU suite green (114 passed / 3 skipped / 0 failed), no regressions.
- Not yet run on real Xeon 6244 hardware — sandbox numbers only.
- Branch: `claude/ddr4-ram-processing-unzseh`.

---

## 2026-08-16 (continued 2) — Fase 1 confirmed on real hardware (Z8, Xeon Gold 6244 + RTX 3090), plus one real bug found only by running there

**Release:** none — one bug fix (`bench_cpu_kernel.py`), no production code changed.

### What we set out to do

Get `test_cpu_kernel.py` and `bench_cpu_kernel.py` actually running on the real
Z8 (dual Xeon Gold 6244, RTX 3090) instead of the sandbox-only numbers logged
above — the project owner asked for this explicitly, wanting real numbers
before deciding anything about Fase 2/3.

### What we did

No direct shell access to the Z8 from this session (it's a separate
Windows 11 Pro machine — this session runs in an isolated cloud container
regardless of which physical device is driving the chat, confirmed via
`list_environments` showing only the cloud environment). Wired
`bench_cpu_kernel.py` into `full-gpu-tests` (`.github/workflows/ci.yml`),
the self-hosted-runner job already registered on that same Z8
(`Z8-G4-RTX3090`, `full-gpu-tests`, gated to manual `workflow_dispatch`),
then asked the project owner to trigger it from the GitHub Actions UI (this
session's GitHub App integration gets a `403 Resource not accessible by
integration` on `workflow_dispatch` itself — read-only for Actions from
here, confirmed twice).

**First run (`31971631020`) surfaced a real bug the sandbox couldn't**:
`_bench_forward()` built `torch.Generator(device="cpu")` unconditionally,
then passed it to `torch.randn(..., device="cuda")` in the `gpu_reference`
section — PyTorch requires the generator's device to match the tensor's,
so it raised `RuntimeError: Expected a 'cuda' device type for generator but
found 'cpu'`. The `cpu` section had already run correctly (matched by
construction), but the whole script crashed before printing anything, so
even that result was lost. This class of bug is invisible in a CPU-only
sandbox by definition — first real payoff of insisting on a hardware run
instead of accepting the sandbox numbers as final. Fixed by building the
generator on the same device as the tensors inside `_bench_forward()`
instead of hardcoding `"cpu"`.

**Second run (`31973334463`), on the fix commit, went green end to end**:
full suite (129 passed / 3 skipped, incl. `test_cpu_kernel.py`), M1/M2
benchmarks, and `bench_cpu_kernel.py` all succeeded. Real numbers
(hidden=512, intermediate=1536 — still the declared fraction of Mixtral's
real 4096×14336, not yet re-run at full scale):

| batch | CPU (Xeon 6244) p50 | CPU GFLOPS | GPU (RTX 3090) p50 | GPU GFLOPS | GPU/CPU |
|---|---|---|---|---|---|
| 1  | 2.59 ms | 1.82  | 0.32 ms | 14.59  | ~8.0×  |
| 8  | 2.51 ms | 15.03 | 0.36 ms | 106.30 | ~7.1×  |
| 32 | 2.96 ms | 50.96 | 0.29 ms | 526.53 | ~10.3× |

Unplanned finding, worth more than the raw numbers: CPU p50 latency is
almost flat across batch=1→32 (2.59ms → 2.96ms, batch 32× larger) — the
cost is dominated by weight load/dequant, not by per-token compute at this
scale. GPU shows the same flatness (kernel-launch-overhead bound at these
tiny synthetic dims, not compute bound). Practical read for Fase 3: on this
hardware, batching multiple calls to the *same* cold expert before running
them is close to free in latency and multiplies effective throughput —
a concrete argument for a batching/coalescing step in the eventual
hot/cold routing design, not just a hot/cold split.

**Also observed, not yet solved**: both `full-gpu-tests` runs rebuilt the
Docker image from scratch (`Build dev image`: ~26 min first run, ~17.5 min
second run) despite the Dockerfile ordering `pip install` of
torch/vLLM/CUDA deps (lines 53-60) *before* `COPY` of source code (lines
82-85) — meaning normal source-only commits should hit Docker's layer
cache and skip the expensive re-download entirely. Checked the Dockerfile
layer order specifically to rule out a cache-defeating COPY-before-install
anti-pattern — it's fine. So the cache loss is happening at the Docker
daemon/host level on the Z8 (Windows 11 Pro, Docker Desktop + WSL2), not in
anything this branch's commits touched. Leading hypothesis, not yet
confirmed: WSL2 disk space pressure (large model checkpoints + growing
build cache on the same disk) triggering an automatic or manual prune
between runs — flagged to the project owner with `docker system df` /
`docker images` / `Get-PSDrive` as the next diagnostic step, not yet run.
Separate concern from Fase 1 itself; logged here so it isn't lost before
the next `full-gpu-tests` dispatch.

**Also discovered mid-session**: the project owner opened a second,
independent Claude Code CLI session directly on the Z8 (local, not a
Claude Code Remote/CCR session — confirmed absent from both `ListAgents`
and `list_sessions`), with `gh` authenticated with `workflow` scope. That
session *can* trigger `workflow_dispatch` directly (unlike this one). Not
yet linked to this session via Remote Control — would remove the
project owner as a manual relay for every future run. Deferred, at the
project owner's request, to a planned `site-test-preparation.MD` covering
both the Docker/cache environment setup and this connectivity question
together, rather than solving it piecemeal mid-debugging.

### Result

Fase 1 is now closed on real hardware, not just sandbox-verified: the core
claim (`_ShadowExpertINT4` runs correctly, unmodified, CPU-resident, on the
real Xeon 6244) is confirmed by the full test suite passing there, and the
CPU-vs-GPU throughput gap is now a real number (~7-10× at this synthetic
scale) instead of a guess.

### Why this matters

Real hardware found a real bug (device-mismatched generator) that no amount
of sandbox testing could have caught, and produced one design-relevant
finding (batching cold-expert calls is nearly free in latency) that wasn't
anticipated in the original phased plan. Both are exactly the payoff the
project owner was asking for when insisting on a real run instead of
accepting sandbox numbers as sufficient.

### Current state

- `benchmarks/bench_cpu_kernel.py`: generator-device bug fixed (commit
  `2379920`), verified green on real hardware (run `31973334463`).
- Fase 1 real-hardware numbers logged above — synthetic scale (512×1536),
  not yet re-run at Mixtral's real 4096×14336.
- Docker build-cache-loss on the Z8 runner: open, diagnostic commands
  handed to the project owner, not yet run.
- Remote Control link between this session and the project owner's local
  Z8 Claude Code CLI session: deferred to `site-test-preparation.MD`
  (not yet written).
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: either (a) re-run `bench_cpu_kernel.py` at real Mixtral scale
(`OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336`) for a
production-representative number, or (b) start Fase 2 (wire
`_ShadowExpertINT4` into a CPU-resident path in `TierManager`/`GCSGWorker`,
gated by the batching insight from this session) — project owner's call.

Next session: Fase 2 — wire a CPU-resident `_ShadowExpertINT4` path into
`GCSGWorker`'s shadow pool construction (parallel to the existing GPU pool,
not replacing it), reusing the class as-is per this session's finding.

---

## 2026-08-16 (continued) — First real run on the Z8 (Windows 11 Pro, Xeon 6244 + RTX 3090): full suite green, bench script had a real bug

**Release:** none — one-line fix to a benchmark script, no production code changed.

### What we set out to do

Get Fase 1's tests and benchmark actually running on the real hardware
instead of the sandbox, per the project owner's request. Also cleared up a
wrong assumption made mid-session: issue #28 ("Bare-metal re-install Z8
G4") proposes migrating the Z8 to native Linux, but it's still open/never
executed — the project owner confirmed live that the machine is Windows 11
Pro, runner at `D:\actions-runner`, same as it's always been since Sprint
0. No environment was wiped or reset; earlier speculation in this
conversation about a bare-metal reinstall explaining a cold Docker cache
was wrong and got corrected against the actual issue text.

Mechanism used: no direct shell access to the Z8 exists from a Claude Code
Remote session (`list_environments` only shows the cloud "Default"
environment; `ListAgents` shows no reachable local agent on that machine).
The only path is the pre-existing `full-gpu-tests` GitHub Actions job
(`workflow_dispatch`-gated, runs on the `Z8-G4-RTX3090` self-hosted
runner). Added a step for `bench_cpu_kernel.py` to that job in `ci.yml`
(mirrors the existing `bench_eat.py`/`bench_tier.py` steps), pushed, then
hit a real capability gap: the GitHub App backing this session's tool
access returned `403 Resource not accessible by integration` on
`workflow_dispatch` — this session cannot trigger the workflow itself, only
read run/job/log state after the fact. Project owner triggered it manually
from the GitHub UI.

### What we did

Watched the run (`31971631020`) via `actions_get`/`actions_list` while it
was in progress — `docker compose build` took ~26 minutes (20:49→21:15
UTC), which read as "starting from zero" and prompted a legitimate
question about environment persistence across sessions (see below). Once
build finished:

- `CPU Tests` job (GitHub-hosted `ubuntu-latest`): green, as always.
- `Full GPU Tests` job, real hardware: full suite (`incl. GPU-marked
  tests`) **129 passed, 3 skipped**, 14s — this includes
  `test_cpu_kernel.py`'s 4 tests actually running on the real Xeon 6244 +
  RTX 3090 for the first time, not just the sandbox. M1 (`bench_eat.py`)
  and M2 (`bench_tier.py`) benchmarks both green, artifacts uploaded as
  usual.
- The new `bench_cpu_kernel.py` step: **failed**.
  `RuntimeError: Expected a 'cuda' device type for generator but found
  'cpu'` — `_bench_forward()` built `torch.Generator(device="cpu")`
  unconditionally, then passed it into `torch.randn(..., device=device)`
  for the `device="cuda"` case in `bench_forward_latency_gpu_reference()`.
  PyTorch requires the generator's device to match the tensor being
  generated. Real bug in the benchmark script (never exercised against
  real CUDA before this run — the sandbox has no GPU, so
  `gpu_reference` always hit the early `torch.cuda.is_available()` guard
  and skipped before reaching this code path). The CPU section of the same
  function never tripped it, since `device="cpu"` matched the
  hardcoded-CPU generator by coincidence, not by correctness — this is
  exactly the class of bug a real-hardware run and only a real-hardware
  run surfaces.

  Because `main()` builds one dict literal (`{"cpu": ..., "gpu_reference":
  ...}`), the exception during `gpu_reference` aborted the script before
  printing anything — the JSON output (including the already-computed
  `cpu` section) was lost, and the upload step correctly no-opped
  (`skipped`, no file to upload).

Fixed in `benchmarks/bench_cpu_kernel.py`: `generator = torch.Generator
(device=device).manual_seed(_SEED)` — one generator per device, built
inside `_bench_forward()` instead of hardcoded to `"cpu"`. Verified locally
(sandbox, CPU-only): runs clean, `gpu_reference` skips cleanly as expected
with no CUDA present here. Not yet re-verified against real CUDA — that
needs another run on the Z8.

### On "starting from zero" / environment persistence

Project owner's concern mid-run: Sprint 1-3 already had a working
environment with models on Docker volumes — why does `docker compose
build` look like it's rebuilding from scratch? Two things got untangled:
- The model checkpoints (Mixtral AWQ etc., wherever they live in a Docker
  volume from earlier sprints) are untouched by this — `full-gpu-tests`
  never loads them; `test_scheduler.py`'s fixtures use tiny
  synthetic/mocked layers, not the real checkpoint.
- What genuinely re-downloaded is the dev image itself (base CUDA + torch/
  vLLM via pip) in the "Build dev image" step. Neither this session's
  commits nor issue #28 (still unexecuted) explain a cold build cache on a
  machine that's been Windows/Docker Desktop all along. Left open,
  explicitly, rather than guessed at further: worth checking `docker
  images`/`docker system df` on the Z8 directly next time this job runs,
  to see whether the cache stays warm now that a build has actually
  happened, or whether something (Docker Desktop auto-cleanup, disk
  pressure) is silently evicting it between infrequent `workflow_dispatch`
  runs. Not resolved this session — parked, not forgotten.

Separately, the project owner raised a real standing need: a way for a
Claude Code session to move freely across a Sprint 1-3-like test
environment the way earlier sprints apparently could, instead of every
session starting cold. Explicitly deferred to a later session, not
designed here — noted so it isn't lost.

### Result

First real-hardware confirmation that Fase 1's core claim holds:
`_ShadowExpertINT4` runs correctly on the real Xeon 6244 (part of the
129-passed full suite). The benchmark script's own bug meant we still
don't have real CPU-vs-GPU throughput numbers from the Z8 — that requires
one more `workflow_dispatch` run with the fix in place.

### Current state

- `benchmarks/bench_cpu_kernel.py`: generator-device bug fixed, verified
  locally (CPU-only sandbox), not yet re-verified against real CUDA.
- Real-hardware confirmation (Z8, Windows 11 Pro, Xeon 6244 + RTX 3090):
  full suite 129 passed / 3 skipped, including `test_cpu_kernel.py`'s 4
  tests — Fase 1's correctness claim now holds on real hardware, not just
  sandbox.
- Real-hardware throughput numbers: still pending, blocked on a re-run.
- This session cannot trigger `workflow_dispatch` itself (403, integration
  lacks `actions: write`) — every run so far was triggered manually by the
  project owner from the GitHub UI.
- Open, parked question: why `docker compose build` needed ~26 minutes
  (cold-looking cache) on a machine with prior Sprint 1-3 history — not
  explained by issue #28 (unexecuted) or by anything changed this session.
- Open, parked request: a persistent/reusable test environment across
  sessions, closer to what Sprint 1-3 apparently had — deferred, not
  designed yet.
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: re-run `full-gpu-tests` (project owner triggers) to get real
`bench_cpu_kernel.py` numbers off the Z8, then proceed to Fase 2 as
planned above.
