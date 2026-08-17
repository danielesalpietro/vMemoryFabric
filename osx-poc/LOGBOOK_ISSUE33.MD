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

---

## 2026-08-17 — Fase 2: pool CPU-resident (DDR4) wired into GCSGWorker, parallelo al pool GPU

**Release:** none — solo codice/test, nessuna integrazione col routing reale
(Fase 3, non ancora iniziata).

### What we set out to do

Fase 2 del piano: "CPU-resident wrapper mirroring `_ShadowExpertINT4`'s
`__call__(hidden_states, layer_id)` API, weights pinned in DDR4, never
`.to('cuda')`" — wiring nel costruttore dello shadow pool di GCSGWorker,
parallelo al pool GPU esistente, non un suo sostituto. Sessione girata
direttamente su questa macchina (Z8-G4-RTX3090, accesso diretto a Docker/gh
locale confermato) invece che tramite CI, per iterare più rapidamente.

### What we did

Mappato con un agente Explore il resto di `gcsg.py` (righe 1007-1700, non
lette nella sessione Fase 1), `tier/manager.py`, `tier/policies.py`, `eat/`
prima di scrivere codice — confermato che `route_forward()` non esiste
ancora da nessuna parte (è nome pianificato per Fase 3, non un refuso), che
`SEEPolicy` non è mai stata riusata per hot/cold routing (Fase 0 resta solo
piano, mai implementata), e che `TierManager.promote_live_tensor()` non ha
un ramo "resta in DDR4" (coerente: quella logica è Fase 3, Fase 2 non deve
inventarla).

Implementato in `scheduler/gcsg.py`:
- `GCSGWorker.__init__`: nuovo `self._cpu_shadow_pool: dict[int, object] = {}`,
  parallelo a `self._shadow_pool` (GPU).
- `GCSGWorker._build_cpu_shadow_pool(layers, expert_ids)`: stesso loop di
  estrazione/quantizzazione del path 1 (FusedMoE fp16 grezzo) in
  `_load_shadow_pool()`, ma senza il `.to("cuda")` forzato del fix
  2026-08-12 — i pesi restano dove sono. Nessuna classe nuova:
  `_ShadowExpertINT4` è già device-agnostic (Fase 1, confermato su hardware
  reale), riusata così com'è. Non forza mai `.cpu()` nemmeno se un peso
  risultasse già CUDA-resident (caso raro, modello non offloaded) — un D2H
  esplicito sarebbe una decisione di eviction, fuori scope qui.
- `_load_shadow_pool()`: dopo aver costruito il pool GPU per il path 1,
  se `self._tier_manager` è wired (opt-in, stesso principio del resto del
  wiring issue #17), costruisce ANCHE il pool CPU per gli stessi
  `expert_ids`. Solo path 1 — i path 2/3 (Marlin, AWQ ModuleList) delegano
  a kernel CUDA reali mai validati su CPU, fuori scope.
- `refresh_shadow_pool_selection()`: ora svuota anche `_cpu_shadow_pool`
  prima di ricaricare (altrimenti expert_id rimossi dalla nuova selezione
  resterebbero fantasma nel pool CPU).

Test aggiunti in `tests/test_scheduler.py` (`TestGCSGCpuShadowPool`, 5
test): pool CPU costruito in parallelo al pool GPU quando tier_manager è
wired (con conferma diretta via device tracking — GPU pool promosso a
`cuda`, CPU pool mai toccato, resta `cpu`); pool CPU vuoto senza
tier_manager (comportamento di default invariato); `_build_cpu_shadow_pool`
usato end-to-end con tensori reali produce un forward CPU corretto e
finito; `refresh_shadow_pool_selection()` svuota entrambi i pool.

**Due bug reali trovati scrivendo questi test (non nella review, nel primo
run)**:
1. `refresh_shadow_pool_selection()`'s nuovo `self._cpu_shadow_pool.clear()`
   rompeva un test PRE-ESISTENTE (`test_refresh_reloads_when_selection_changes`)
   con `AttributeError` — il worker di quel test è costruito da un
   `_make_worker()` scritto prima di questa integrazione, senza
   `_cpu_shadow_pool` assegnato. Fix: `getattr(self, "_cpu_shadow_pool", None)`
   difensivo, stesso pattern già usato in questo file per `_tier_manager`/
   `_n_experts_cached` su worker costruiti via `__new__()` nei test.
2. Assunzione sbagliata nel mio stesso test: contavo 4 chiamate a
   `_quantize_int4` per pool (2 expert × 2 layer), dimenticando che w13 e
   w2 sono due chiamate separate — il conteggio reale è 8. Corretto
   l'assert, non il codice (il codice era giusto, il test no).

Full suite verificata su hardware reale (questa macchina, RTX 3090 + CUDA
12.1.1 nel container): 133 passed, 3 skipped, 0 failed — nessuna
regressione. `ruff check` pulito sui file toccati (un solo warning
pre-esistente e non correlato in `test_scheduler.py`, import block
un-sorted, non causato da questa sessione).

**Trovato en passant, non risolto qui**: i target del Makefile
(`test`, `test-scheduler`, `lint`, `typecheck`, ecc.) sono rotti come
scritti — girano con `docker compose run` il cui WORKDIR nel container è
`/workspace` (repo root, bind mount), ma il codice vive sotto
`/workspace/osx-poc/`. `make test-scheduler` fallisce con
"file or directory not found: tests/test_scheduler.py". CI non ha mai
esercitato questo bug perché `ci.yml` non usa affatto il Makefile — chiama
`docker compose run` direttamente con un wrapper `bash -c "cd osx-poc && ..."`
esplicito. Aggirato in questa sessione passando il path completo
(`osx-poc/tests/...`) invece di usare `make`. Non in scope per Fase 2, non
fixato qui. Correzione: avevo aperto la issue #40 per questo senza prima
controllare le issue esistenti — è un duplicato di #12 (aperta
2026-08-09, stessa causa, analisi anche più completa: propone
`working_dir: /workspace/osx-poc` in `docker-compose.yml` come fix,
alternativa migliore al semplice `cd osx-poc &&` per-target che avevo
suggerito). #40 chiusa come duplicato, riferimento corretto:
https://github.com/danielesalpietro/vMemoryFabric/issues/12

**Trovato anche in questa sessione (prima di iniziare Fase 2), non
correlato al lavoro DDR4**: il disco `D:` della macchina (dove vive
`docker_data.vhdx`, il disco WSL2 di Docker Desktop) è a 395 GB usati su
477 GB totali, ~81 GB liberi (17%). `docker system df`: 183.1 GB immagini
(68.98 GB reclaimable), 148 GB volumi locali (49.31 GB reclaimable), 31.3 GB
build cache (20.99 GB reclaimable), 689 MB container fermi (100%
reclaimable). Il grosso non appartiene a vMemoryFabric — immagini/volumi di
altri progetti sulla stessa macchina condivisa (`moe-infinity`,
`vllm-openai`, `vllm-granite`, `ollama_data`, `tabby_data`, container fermi
di `mike-for-dummies`/`grastorp`/`private-ai-coding`/`anythingllm`/
`open-webui`); il volume proprio di questo progetto (`vmemoryfabric_nvme-data`)
pesa 49.31 GB. Possibile collegamento (non dimostrato) con la domanda
parcheggiata nella sessione 2026-08-16 sul perché `docker compose build`
impiegasse ~26 minuti su `full-gpu-tests` — il build cache reclaimable
risale a "4 settimane fa" (non evictato), quindi più probabile un
degrado da pressione disco che un'eviction reale, ma non verificato a
fondo. Nessuna pulizia eseguita (distruttivo, tocca dati di altri
progetti sulla stessa macchina condivisa) — issue aperta:
https://github.com/danielesalpietro/vMemoryFabric/issues/41

### Result

Fase 2 chiusa: il meccanismo di costruzione del pool CPU-resident esiste,
è opt-in (zero rischio per il path di default), riusa `_ShadowExpertINT4`
senza modifiche, e non tocca in nessun modo la decisione di routing (che
resta Fase 3). Verificato su hardware reale, non solo in sandbox.

### Why this matters

Questo è il primo punto in cui DDR4 diventa effettivamente un sito di
compute utilizzabile all'interno di GCSGWorker, non solo una dimostrazione
isolata (Fase 1). Il pool CPU esiste ora accanto a quello GPU pronto per
Fase 3 a decidere quando instradarci — senza che Fase 2 abbia dovuto
anticipare quella decisione (hot/cold routing, `route_forward()`) o il
criterio di selezione (Fase 0, SEEPolicy reuse), entrambi ancora non
implementati.

### Current state

- `scheduler/gcsg.py`: `_cpu_shadow_pool`, `_build_cpu_shadow_pool()`,
  wiring in `_load_shadow_pool()`/`refresh_shadow_pool_selection()`.
- `tests/test_scheduler.py`: `TestGCSGCpuShadowPool` (5 test), tutti verdi
  su hardware reale insieme al resto della suite (133 passed / 3 skipped).
- Makefile: target rotti (path relativi assumono WORKDIR sbagliato),
  non fixato qui — issue preesistente #12 (avevo aperto #40 per errore
  senza controllare prima, chiusa come duplicato).
- Disco D: della macchina al 17% libero (81/477 GB), non correlato al
  lavoro DDR4, nessuna pulizia eseguita — issue #41.
- Fase 0 (criterio hot/cold via SEEPolicy) e Fase 3 (routing reale,
  `route_forward()`, assert su `GPUTransfer.to_vram()` mai chiamato per
  expert instradati a freddo) restano da fare, in quest'ordine o quello
  che il project owner preferisce.
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: Fase 0 (definire il criterio hot/cold riusando `SEEPolicy`,
pure unit test) o Fase 3 (routing in TierManager/GCSGWorker) — decisione
del project owner. Fase 3 dipende concettualmente da Fase 0 (serve un
criterio prima di poter instradare), quindi Fase 0 è la scelta più naturale
se non diversamente indicato.

---

## 2026-08-17 (continued) — Fase 0: criterio hot/cold, informato dall'esperienza di issue #32 e successive

**Release:** none — solo codice/test, nessuna integrazione col routing reale
(Fase 3, ancora da fare).

### What we set out to do

Fase 0 del piano: definire il criterio hot/cold riusando `SEEPolicy`
scoring invece di inventarne uno nuovo, pure unit test, nessun hardware.
Richiesta esplicita del project owner questa volta: non progettarlo a
vuoto, ma tesorizzare l'esperienza già raccolta in issue #32 ("Component
reuse da Petals/ExLlamaV2/V3/tabbyAPI") e nelle sue issue figlie/collegate.

### What we did

**Trovato prima di tutto un problema reale**: i due report che issue #32 e
tutte le sue sub-issue (incluse #33-37) citano come fondamento
(`osx-poc/reports/component_reuse_analysis.md`,
`osx-poc/reports/related_work_petals_exllama.md`) non esistono su
`develop` né su questo branch — vivono solo sul branch mai mergiato
`claude/petals-exllama-heterogeneous-v0frr4`. Recuperati via `git show`
per leggerli (non checkout, per non toccare lo stato del branch corrente).

**La scoperta rilevante per Fase 0**, da `component_reuse_analysis.md`
§2.1 (verificato sul sorgente reale di `exllamav3/model/moe_cpu_host.py`,
non per analogia): il criterio hot/cold REALE di quel sistema è "quanti
token sono stati assegnati a un expert" — gli esperti con più
assegnazioni vengono streammati verso la GPU, la "coda fredda" gira su
CPU con AVX-512. Questo è esattamente `access_count` di `EATEntry`, il
segnale che `SEEPolicy.score()` già pesa nel suo `freq_component`. Non
un'idea nuova da inventare: il piano "riusa SEEPolicy" del 2026-08-16 era
già la scelta giusta, ora validata contro un'implementazione reale
verificata sul codice, non solo per intuizione architetturale.

Letta anche issue #21 ("Collegare σ in SEEPolicy a PT-PEP") prima di
scrivere codice, per capire cosa NON fare: propone di alimentare il
termine γ·σ (oggi stub, γ=0.4 sprecato su un contributo nullo) col
segnale semantico di PT-PEP, ma con due riserve esplicite nella issue
stessa — (1) PT-PEP opera a livello di prompt intero, σ è pensato per lo
shard corrente, possibile mismatch di scope; (2) raccomandazione di
misurare se aiuta davvero prima di collegare, non assumerlo. Decisione
per Fase 0: NON toccare σ/PT-PEP — resterebbe un rischio non ancora
misurato ereditato in un criterio che deve restare puro/testabile senza
hardware. Riusato solo il path `context_vec=None` di `score()` (LRU
ponderata alpha/beta), già validato.

Letta anche issue #24 (misura diretta transfer-bound vs compute-bound,
non ancora eseguita, gated su budget POD) — non blocca Fase 0 (che è
struttura, non calibrazione) ma ne informa il design: `hot_fraction` è un
parametro esplicito, non una soglia hardcoded, perché il punto di taglio
reale tra "vale la VRAM" e "resta CPU" dipende da un dato che oggi non
esiste ancora.

Letta anche issue #27 (EPM, persistenza hotness EAT) per il principio
del cold start onesto — coerente con quanto già dichiarato in
`_select_shadow_expert_ids()` (Fase 2 precedente): senza traffico reale,
degrado deterministico via tie-break stabile, non un fallback nascosto.

**Implementato**: `SEEPolicy.classify_hot_cold(entries, hot_fraction=0.5,
now=None) -> tuple[list[int], list[int]]` in `tier/policies.py`. Aggrega
`score()` per `expert_id` sommando su tutti gli `shard_idx`/layer (stesso
principio di aggregazione già in `_select_shadow_expert_ids()`, qui via
`score()` invece di `access_count` grezzo), ordina decrescente, spacca a
`round(n * hot_fraction)` con minimo 1 hot se ci sono entry. Nessun
wiring in `GCSGWorker`/`TierManager` — quello resta Fase 3, per lo stesso
motivo per cui Fase 2 non ha anticipato il routing: la funzione è pronta,
la decisione di quando/come chiamarla no.

7 test in `TestHotColdClassification` (`tests/test_tier.py`): split per
score, aggregazione cross-layer (stesso schema del test analogo per
`_select_shadow_expert_ids`), tie-break stabile a cold start (access_count
uniforme, ordine di input preservato — non last_access_ts), arrotondamento
di `hot_fraction` con minimo 1, entries vuote, `hot_fraction` fuori range
solleva `AssertionError`. Tutti verdi al primo run — nessun bug trovato
questa volta (a differenza delle due sessioni precedenti).

Full suite verificata su hardware reale: 139 passed, 3 skipped, 0 failed.
`ruff check`/`mypy` puliti sui file toccati (stesso singolo warning
pre-esistente non correlato già visto nella sessione Fase 2, questa volta
anche su `test_tier.py` — stesso debito di stile diffuso, issue #6
chiusa ma non evidentemente su ogni file).

**Correzione da questa sessione**: la issue #40 aperta nella sessione
precedente (Makefile rotto) era un duplicato di #12, preesistente
dal 2026-08-09 — non controllato prima di aprirla. Chiusa come duplicato,
riferimento corretto a #12 nel log della sessione precedente sopra.

### Result

Fase 0 chiusa: criterio hot/cold esiste, puro, testato, riusa `SEEPolicy`
senza inventare nulla di nuovo — e la scelta di design (hotness via
access_count, non semantica via PT-PEP/σ) è ora giustificata da due fonti
indipendenti: il piano originale (Fase 0 del 2026-08-16) e il codice reale
di exllamav3 (component_reuse_analysis.md §2.1), non solo da intuizione.

### Why this matters

Fase 3 (routing reale) può ora consumare `classify_hot_cold()` senza dover
decidere da capo cosa significa "caldo" — la domanda che il piano
originale aveva lasciato esplicitamente aperta. L'esperienza già raccolta
in issue #32/#21/#24/#27 (fin qui dispersa in issue e report non
tutti raggiungibili da `develop`) è ora concentrata in un'unica
implementazione con la motivazione di ogni scelta scritta accanto al
codice, non solo nelle issue di origine.

### Current state

- `tier/policies.py`: `SEEPolicy.classify_hot_cold()`.
- `tests/test_tier.py`: `TestHotColdClassification` (7 test), suite
  completa 139 passed / 3 skipped su hardware reale.
- `route_forward()`/wiring in `GCSGWorker`/`TierManager` resta Fase 3, non
  iniziata.
- I due report fondanti di issue #32 esistono solo sul branch non
  mergiato `claude/petals-exllama-heterogeneous-v0frr4` — non segnalato
  come issue separata questa sessione (nessuna azione richiesta, solo
  annotato qui: chi riprende Fase 3 o le altre sub-issue di #32 dovrebbe
  saperlo).
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: Fase 3 — wiring di `classify_hot_cold()` in
`TierManager`/`GCSGWorker` (`route_forward()`), test che asserisce
`GPUTransfer.to_vram()` mai chiamato per un expert instradato a freddo
(il pool CPU di Fase 2 è già pronto a riceverlo).

---

## 2026-08-17 (continued) — merge report issue #32 su develop, poi Fase 3: routing hot/cold reale

**Release:** merge di 4 commit doc-only su `develop` (fast-forward), più
codice/test per Fase 3 su questo branch — nessuna integrazione vLLM live.

### Merge dei report issue #32 su develop

Prima di Fase 3, il project owner ha chiesto di valutare il merge del
branch `claude/petals-exllama-heterogeneous-v0frr4` (dove vivono i due
report citati sopra) — "se non crea problemi, procedi". Verificato:
`develop` è ancestor diretto del branch (nessuna divergenza, fast-forward
puro), diff limitato a `README.md` (+30 righe, registra il sub-progetto
"Marstrand" nella roadmap) e ai due nuovi file di report — zero modifiche
a codice esistente, zero rischio di conflitto. Pushato come fast-forward
diretto (`git push origin origin/claude/petals-exllama-heterogeneous-v0frr4:develop`),
senza checkout locale. `develop` ora a `1bee282`. Branch sorgente non
cancellato (non l'ho creato io — stesso principio già seguito per gli
altri branch di sessione).

### What we set out to do

Fase 3: wiring reale in `GCSGWorker` di `classify_hot_cold()` (Fase 0) sui
due pool costruiti in Fase 2 — `route_forward()`, senza toccare il `Tier`
enum, con test che verifica che un expert instradato a freddo non
attraversi mai la promozione GPU.

### What we did

Implementato in `scheduler/gcsg.py`:
- `_RoutedShadowPool`: adapter dict-like (`__contains__`/`__getitem__`)
  passato a `GCSGGuard.run_shadow()` al posto di `self._shadow_pool`
  diretto — `run_shadow()` resta byte-per-byte invariato, non sa nulla di
  GPU/CPU. Costruito inline al call site in `_evaluate_gcsg_for_rows()`
  (oggetto stateless, zero costo reale), non cacheato in `__init__` —
  scelta deliberata per non richiedere che ogni worker di test lo
  assegni a mano.
- `GCSGWorker.route_forward(expert_id, layer_id, hidden_states)`: decide
  GPU vs CPU per un expert presente in entrambi i pool usando
  `self._hot_expert_ids`, con fallback a "qualunque pool lo contenga" se
  l'expert manca da uno dei due (oggi non raggiunto dal path 1, che
  costruisce sempre gli stessi expert_id in entrambi — ma testato
  comunque come contratto esplicito del metodo).
- `GCSGWorker._refresh_hot_cold_classification()`: ricalcola
  `self._hot_expert_ids` da `SEEPolicy.classify_hot_cold()` su
  `tier_manager.eat.get_tier(Tier.DDR4)` — chiamata solo al load/refresh
  del pool (non ad ogni `route_forward()`, potenzialmente una volta per
  token: `EAT.get_tier()` scansiona l'intera tabella, stesso motivo per
  cui `_select_shadow_expert_ids()` non si ricalcola ad ogni forward).
  Fallback a cold start: TUTTI gli expert_id restano "caldi" (GPU-only,
  comportamento pre-Fase-3) invece di instradare tutto a freddo su un
  segnale che non esiste ancora — stesso principio del round-robin di
  `_select_shadow_expert_ids()`.
- Wired in `_load_shadow_pool()` (subito dopo aver costruito il pool CPU)
  e in `_evaluate_gcsg_for_rows()` (dispatch reale via
  `_RoutedShadowPool`).

**Bug reale trovato scrivendo i test (terza sessione di fila con almeno
un bug al primo run)**: `_refresh_hot_cold_classification()` accedeva a
`self._hot_cold_policy` senza `getattr` difensivo — `AttributeError` su
un worker di test costruito via `__new__()` con EAT non vuota (il ramo
cold-start/EAT-vuota lo aggirava per caso, senza mai raggiungere quella
riga, motivo per cui solo UN test su undici l'ha esposto). Fix:
`getattr(self, "_hot_cold_policy", None) or SEEPolicy()`, stesso pattern
difensivo già usato ovunque in questo file per worker costruiti senza
passare da `__init__()`.

11 test nuovi in `TestGCSGRouteForward` (`tests/test_scheduler.py`):
dispatch caldo/freddo puro su `route_forward()`, fallback quando un
expert manca da uno dei due pool, classificazione default cold-start
(tutti caldi) con e senza `tier_manager`, classificazione da traffico EAT
reale, `_RoutedShadowPool.__contains__` sull'unione dei due pool,
integrazione `_load_shadow_pool()` -> classificazione popolata, e i due
test end-to-end richiesti dal piano: un expert freddo instradato via
`_evaluate_gcsg_for_rows()` (hook `.gate` reale) tocca SOLO la callable
del pool CPU, mai quella GPU (la garanzia "mai `GPUTransfer.to_vram()`"
dimostrata per dispatch — quella funzione vive solo dietro le callable
del pool GPU, mai istanziata/chiamata nel path freddo) — e il caso
speculare per un expert caldo.

Full suite su hardware reale: 150 passed, 3 skipped, 0 failed (153
raccolti, +11 da questa sessione). `ruff check` pulito sui file toccati
dopo un fix minore (quote superflue su un type hint stringato, `UP037` —
`from __future__ import annotations` le rende inutili); stesso singolo
warning pre-esistente non correlato su `test_scheduler.py`.

### Result

Fase 3 chiusa: il routing hot/cold è reale, non solo la struttura (Fase
0) o il meccanismo di costruzione (Fase 2) — un expert classificato
freddo gira davvero su CPU/DDR4 senza mai toccare la GPU, verificato
end-to-end, non solo per costruzione. Le quattro fasi centrali del piano
originale (0, 1, 2, 3) sono ora tutte chiuse.

### Why this matters

Questo chiude il ciclo che issue #33 aveva aperto: DDR4 non è più solo
"passive storage/staging" (l'audit del 2026-08-16 che aveva confermato la
claim originale dell'issue) — è un sito di compute REALMENTE raggiungibile
dal routing reale del sistema, con un criterio di selezione tracciabile
fino al codice sorgente di un sistema di produzione (exllamav3) invece che
inventato, e senza aver dovuto toccare l'invariante che l'intero file
protegge gelosamente (Tier enum, `_promote_module_via_tier_manager`,
zero rischio per il path GPU già validato).

### Current state

- `scheduler/gcsg.py`: `_RoutedShadowPool`, `GCSGWorker.route_forward()`,
  `_refresh_hot_cold_classification()`, wiring in `_load_shadow_pool()`/
  `_evaluate_gcsg_for_rows()`.
- `tests/test_scheduler.py`: `TestGCSGRouteForward` (11 test).
- `develop`: aggiornato con i report issue #32 (fast-forward, 4 commit).
- Suite completa 150 passed / 3 skipped su hardware reale, lint pulito.
- Non ancora verificato su un vero engine vLLM/checkpoint reale (stesso
  limite dichiarato per ogni altra parte "logica pura" di questo file —
  `_promote_module_via_tier_manager`/`_build_marlin_tensor_promoter`
  restano gli unici pezzi CUDA-touching mai eseguiti su hardware reale
  in QUESTA integrazione, non toccati da Fase 0-3).
- `hot_fraction` (default 0.5 in `classify_hot_cold()`) non calibrato —
  issue #24 (misura transfer-vs-compute reale) resta aperta e non gated
  da questo lavoro.
- Fase 4 (valutare se serve il job ring stile `MoeCpuHost`, o se
  l'in-process round-trip di Fase 2/3 basta) e Fase 5 (benchmark
  end-to-end reale) restano da fare.
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: Fase 4 (misurare l'overhead del round-trip in-process prima
di decidere se serve un meccanismo di sync separato) o Fase 5 (benchmark
end-to-end su hardware reale, `bench_hybrid.py`) — decisione del project
owner. Il piano originale gate-a Fase 4 su "skip se l'overhead è
trascurabile rispetto al tempo di compute AVX-512" — quel numero non
esiste ancora.

---

## 2026-08-17 (continued) — Fase 4: overhead di dispatch trascurabile, job ring scartato (doppia motivazione)

**Release:** nuovo `benchmarks/bench_route_forward.py` + target Makefile
`bench-route-forward`. Nessuna modifica a codice di produzione — Fase 4 è
per definizione un gate di misura, non un'implementazione.

### What we set out to do

Il gate esplicito del piano: misurare l'overhead del round-trip
in-process (route_forward()/_RoutedShadowPool, Fase 3) prima di decidere
se serve davvero un meccanismo di sincronizzazione separato stile
`exllamav3/model/moe_cpu_host.py` (`MoeCpuHost`) — skip se l'overhead è
trascurabile rispetto al tempo di compute AVX-512 già misurato in Fase 1.

### What we did

**Chiarito prima di misurare, su richiesta del project owner, cosa
consiste davvero il job ring di MoeCpuHost** (rileggendo
`component_reuse_analysis.md` §2.1 e `related_work_petals_exllama.md`
§2, recuperati la sessione precedente): un processo separato possiede i
pesi in un'arena di memoria host pinned/huge-page; comunica col processo
GPU tramite un job ring in `multiprocessing.shared_memory` con
`cuda_host_register` per il DMA; gli esperti "caldi" vengono streammati
verso la GPU, i "freddi" calcolati in loco su CPU con AVX-512. Il pezzo
strutturalmente rilevante qui è `cuda_host_register`/pinned memory — è
esattamente il meccanismo che l'architettura a due processi userebbe per
evitare un vero round-trip.

**Secondo argomento, sollevato dal project owner prima di lanciare
qualunque benchmark**: siamo su WSL2, e questo stesso codebase ha già una
storia concreta di instabilità con pin_memory lì — non un'ipotesi
astratta. `_should_pin_transfers()` (`gcsg.py:1469-1483`) ritorna `False`
esplicitamente sotto WSL2 (`in_wsl()`), e la ragione è documentata: il
fix issue #10/#16 (2026-08-10) aveva isolato un crash CUDA "illegal
memory access" che non era nel kernel Marlin ma in
`vllm.model_executor.models.utils.maybe_offload_to_cpu()` — una copia
H2D asincrona non sicura sotto WSL2 senza pin_memory, mai validata sotto
carico sostenuto lì. Il job ring di MoeCpuHost dipende dallo stesso
meccanismo (`cuda_host_register`) che questo progetto ha già dovuto
disattivare per instabilità reale, non per cautela. Questo è un motivo
qualitativo per scartare il job ring INDIPENDENTE dal numero di overhead
— vale anche se il dispatch in-process risultasse costoso.

**Misura quantitativa**: `benchmarks/bench_route_forward.py`, tre
sezioni — `raw_dict_call` (baseline pre-Fase-3), `route_forward()`
(nuova indirection Fase 3), `run_shadow_full` (l'intero path reale via
`GCSGGuard.run_shadow()` + `_RoutedShadowPool`, incluso il ranking degli
expert) — confrontate contro `compute_reference` (tempo reale di
`_ShadowExpertINT4` su CPU, stessa metodologia di `bench_cpu_kernel.py`
Fase 1, ricalcolato nello stesso run per un confronto pulito).

**Bug reale trovato PRIMA di fidarsi del primo risultato** (non nella
review, il primo run del benchmark stesso): le pool di dispatch usavano
`_ShadowExpertINT4` REALI (pesi random, matmul veri), non callable no-op
— il benchmark misurava compute due volte, non dispatch. Sintomo
osservato prima di capire la causa: il container è rimasto a 686% CPU
per ~9 minuti (verificato con `docker exec ... ps aux` +
`nvidia-smi`, non un'assunzione — il processo era vivo e attivamente
occupato, non bloccato) — coerente con 180.000+ forward reali da
~2.6ms l'uno (2 pool GPU/CPU + run_shadow, tutte e tre le sezioni
chiamavano la stessa `_ShadowExpertINT4` vera). Primo risultato (scartato):
tutte e tre le sezioni ~3.2ms, identiche al compute reference — la
"conclusione" che ne seguiva ("overhead non trascurabile") era
matematicamente vera ma basata su un confronto rotto (compute contro
compute, non dispatch contro compute). Fix: callable no-op
(`lambda hs, layer_id: hs`) nelle due pool per la sola sezione
dispatch_overhead, `compute_reference` tenuto separato e reale.

**Risultato corretto**, hardware reale (RTX 3090, CUDA 12.1.1, dimensioni
sintetiche 512×1536 come Fase 1):

| Sezione | p50 |
|---|---|
| `raw_dict_call` | 0.082 μs |
| `route_forward()` | 0.33 μs |
| `run_shadow_full` (path reale) | 2.5 μs |
| `compute_reference` | 3000 μs (3.0 ms) |

Overhead di dispatch come frazione del tempo di compute: **~0.08%** — tre
ordini di grandezza sotto, non un margine ambiguo.

### Result

Fase 4 chiusa: **non serve il job ring stile MoeCpuHost**, per doppia
motivazione indipendente — quantitativa (overhead di dispatch
trascurabile, ~0.08% del tempo di compute) e qualitativa (il meccanismo
di pinning su cui si reggerebbe è già noto instabile su questo hardware
sotto WSL2, issue #10/#16). L'architettura in-process di Fase 2/3 resta
quella giusta, non un compromesso temporaneo.

### Why this matters

Chiude esplicitamente il gate che il piano originale aveva posto prima di
qualunque lavoro su un meccanismo di sincronizzazione separato — evita di
costruire (o anche solo prototipare su RunPod/Linux reale) un pezzo di
infrastruttura complesso per un problema di overhead che non esiste,
sovrapposto a un rischio di stabilità reale e già documentato su questa
piattaforma. Il bug trovato nel benchmark stesso è un promemoria diretto
del principio "misura prima di ottimizzare" applicato ricorsivamente:
anche lo strumento di misura va verificato prima di fidarsi del numero
che produce.

### Current state

- `benchmarks/bench_route_forward.py` aggiunto, `make bench-route-forward`.
- Nessuna modifica a codice di produzione — Fase 4 era un gate, ora
  chiuso con esito "skip".
- Suite completa invariata: 150 passed / 3 skipped, lint pulito.
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: Fase 5 — benchmark end-to-end su hardware reale
(`bench_hybrid.py`, mirror di `bench_tier.py`), sul runner
`Z8-G4-RTX3090` via `full-gpu-tests`. Con Fase 4 chiusa a "skip", Fase 5
misura l'intero pipeline così com'è (Fase 0-3), senza dover prima

---

## 2026-08-17 (continued) — flag esplicito enable_cpu_offload, spento di default

**Release:** codice + test, nessuna modifica di comportamento per chi già
usa `configure_tier_manager()`/`tier_manager=` senza il nuovo flag — resta
tutto invariato finché non viene richiesto esplicitamente.

### What we set out to do

Correzione al project owner ("sei troppo frettoloso"): chiudere Fase 4 a
"skip il job ring" non doveva implicare che il routing CPU/DDR4-resident
(Fase 2/3) diventasse automaticamente live ovunque un `TierManager` è
wired. Va tenuto, testato bene, ma spento di default — attivabile solo
esplicitamente — così l'impatto reale si può misurare e documentare più
avanti senza che nel frattempo sia già la via di default in produzione.

### What we did

Aggiunto `GCSGWorker.configure_cpu_offload(enabled: bool)` (classmethod,
stesso meccanismo di `configure_tier_manager()` — `_pending_enable_cpu_offload`
a livello di classe, perché vLLM costruisce il worker da solo e non c'è
un punto per iniettare un kwarg extra) e il parametro `enable_cpu_offload`
sul costruttore, per chi costruisce `GCSGWorker` direttamente (test).
Default **False** in entrambi i casi — deliberatamente diverso da
`tier_manager` (che risolve a `None`): qui la scelta è booleana e il
default deve essere "non attivare", non "eredita quello che c'è".

Il gate in `_load_shadow_pool()` ora richiede ENTRAMBE le condizioni —
`self._tier_manager is not None` E `self._cpu_offload_enabled` — per
costruire il pool CPU. Prima di questa sessione bastava la prima da
sola: chiunque avesse wired un TierManager per qualunque altro motivo
(es. solo hotness tracking EAT lato GPU, issue #17, che non ha nulla a
che fare col compute-offload DDR4) si sarebbe ritrovato anche il routing
CPU attivo come effetto collaterale non richiesto. `route_forward()` non
ha bisogno di nessuna modifica: con `_cpu_shadow_pool` vuoto (flag
spento), il suo fallback già esistente ("usa qualunque pool contenga
l'expert") instrada tutto al pool GPU esattamente come prima di Fase 2 —
lo stesso principio "zero rischio per il path già validato" di ogni
altra estensione in questo file, questa volta verificato per costruzione
invece che per un test dedicato aggiuntivo.

4 test nuovi: `configure_cpu_offload()` classmethod (default False,
imposta/pulisce lo stato pending — stesso schema dei tre test gemelli
per `configure_tier_manager()`), e un test diretto che nega
esplicitamente il caso precedente (`tier_manager` wired MA
`cpu_offload_enabled=False` esplicito, il default reale di produzione) —
pool CPU resta vuoto, pool GPU invariato. `TestGCSGCpuShadowPool._make_worker()`
aggiornato con un default locale `cpu_offload_enabled=True` (quella
classe testa specificamente il comportamento attivo — il default reale a
livello di produzione resta False, coperto a parte).

Full suite: 154 passed, 3 skipped, 0 failed (157 raccolti, +4). `ruff
check` pulito (stesso singolo warning pre-esistente non correlato).

### Result

Il routing CPU/DDR4-resident (Fase 2/3) è ora un opt-in esplicito e
separato dal resto del wiring TierManager, non un effetto collaterale
implicito. Nessuna regressione per l'uso esistente di `tier_manager=`.

### Why this matters

Disaccoppia due decisioni che erano rimaste implicitamente fuse: "voglio
un TierManager per l'hotness tracking/promozione GPU" (issue #17,
maturo, validato su hardware reale) e "voglio che gli expert freddi
girino su CPU invece che essere promossi" (issue #33, nuovo, Fase 5 non
ancora fatta — impatto reale non ancora misurato). Tenerle separate
significa che l'adozione della seconda può essere graduale e misurata,
invece che automatica per chiunque usi già la prima.

### Current state

- `scheduler/gcsg.py`: `GCSGWorker.configure_cpu_offload()`,
  `_pending_enable_cpu_offload` (default False), parametro
  `enable_cpu_offload` sul costruttore, gate aggiornato in
  `_load_shadow_pool()`.
- `tests/test_scheduler.py`: 4 test nuovi, `_make_worker()` di
  `TestGCSGCpuShadowPool` aggiornato.
- Suite completa 154 passed / 3 skipped, lint pulito.
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: Fase 5 — benchmark end-to-end su hardware reale
(`bench_hybrid.py`, mirror di `bench_tier.py`), sul runner
`Z8-G4-RTX3090` via `full-gpu-tests`. Con l'aggiunta che ogni run dovrà
esplicitamente attivare il flag (`configure_cpu_offload(True)` o
`enable_cpu_offload=True`) per esercitare davvero il path CPU — senza,
misurerebbe solo il comportamento di sempre (GPU-only), non quello che
Fase 5 deve validare.

---

## 2026-08-17 (continued) — Fase 5: impatto aggregato reale, e un bug di device mai esercitato prima

**Release:** nuovo `benchmarks/bench_hybrid.py` + target Makefile
`bench-hybrid`. Un fix reale a codice di produzione (`route_forward()`),
non solo benchmark — vedi sotto.

### What set out to do

Fase 5: misurare l'impatto end-to-end reale di `enable_cpu_offload=True`
(non solo il dispatch isolato di Fase 4, che non dice nulla sul costo
AGGREGATO di spostare traffico reale dal pool GPU veloce al pool CPU più
lento) — su hardware reale, con un mix di traffico realistico tra un
expert "caldo" e uno "freddo" a parità di shadow_pool_size=2
(produzione).

### What we did

Scritto `benchmarks/bench_hybrid.py`: due expert reali (shadow_pool_size
di produzione), traffico sintetico diviso per una quota "fredda"
configurabile (5%/10%/25%/50%), stesso traffico confrontato in due
scenari — `enable_cpu_offload=False` (baseline, entrambi su GPU) vs
`=True` (l'expert freddo su CPU) — via `route_forward()` reale, non
mockato.

**Bug reale trovato AL PRIMO LANCIO** (non in un unit test isolato — lo
stesso principio "hardware reale trova bug che il sandbox non trova" già
visto in Fase 1 sul generator): `route_forward()` passava
`hidden_states` intatto al pool CPU senza mai controllarne il device.
Nei test di Fase 2/3, `hidden_states` era sempre già CPU-resident per
costruzione (stesso device dei pesi CPU del test) — mai esercitato il
caso reale, dove `hidden_states` arriva CUDA-resident dalla forward pass
del modello (l'hook `.gate` gira dentro un modello che vive su GPU).
Crash: `RuntimeError: Expected all tensors to be on the same device,
cuda:0 and cpu` — lo stesso identico sintomo del bug 2026-08-12 sui PESI
(mai forzato `.to('cuda')` nel path 1 prima di quel fix), stavolta
sull'INPUT, mai sui pesi. Fix in `route_forward()`: `hidden_states =
hidden_states.cpu()` immediatamente prima della sola chiamata al pool
CPU, se non già CPU-resident — un solo D2H di un batch a una riga, non
dell'intero modello.

2 test nuovi (`TestGCSGRouteForward`): normalizzazione CUDA→CPU prima
del dispatch a freddo (fake minimale, nessuna CUDA reale richiesta —
stesso principio del fake usato per il bug fix 2026-08-12 sui pesi), e
il complementare (nessuna copia inutile se già CPU-resident). Due test
preesistenti aggiornati (usavano una stringa `"hs"` come hidden_states,
che non ha `.device` — sostituita con `torch.Tensor` reali dove il
codice ora vi accede).

**Risultato del benchmark** (dopo il fix, RTX 3090 + CUDA 12.1.1, 500
call simulate per scenario, dimensioni sintetiche 512×1536 — stessa
frazione di Mixtral reale di Fase 1):

| cold_share | rallentamento aggregato |
|---|---|
| 5% | 1.27× |
| 10% | 1.34× |
| 25% | 1.85× |
| 50% | 1.43× |

A differenza di Fase 4 (overhead di dispatch trascurabile, ~0.08%),
l'impatto AGGREGATO qui è reale e non banale: anche solo il 5% di
traffico instradato a freddo costa ~27% di rallentamento — il gap
GPU/CPU per-chiamata (~7-10×, Fase 1) domina la media pesata anche con
poca coda fredda. La quota 50% ha un rallentamento minore della quota
25% (1.43× contro 1.85×) — non un errore di misura confermato ma
nemmeno indagato a fondo: possibile rumore di sistema tra run (il
baseline stesso a 50% è più lento del baseline a 25%, 2492 vs 3405
calls/s), da rivedere con più campioni se questo numero diventa
decisionale.

**Deviazione dichiarata, non ancora risolta**: dimensioni sintetiche —
a queste taglie la GPU è probabilmente bound sul lancio del kernel, non
sul compute reale (nota già in Fase 1: "GPU shows the same flatness,
kernel-launch-overhead bound... not compute bound"), quindi il gap
misurato qui potrebbe restringersi a scala reale
(`OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336`) se il compute
reale della GPU cresce più del previsto. Non rilanciato a scala reale in
questa sessione.

Full suite: 159 passed, 3 skipped, 0 failed (+2 test nuovi, +2
aggiornati). `ruff check` pulito.

### Result

Fase 5 chiusa (prima iterazione, dimensioni sintetiche): l'impatto
aggregato di `enable_cpu_offload=True` è misurabile e reale, non
trascurabile come l'overhead di dispatch di Fase 4 — conferma diretta
che la decisione di tenerlo spento di default (sessione precedente) era
corretta, non solo prudente. Trovato e corretto en passant un bug di
produzione reale (device mismatch sull'input), non solo raccolto un
numero.

### Why this matters

Fase 4 e Fase 5 rispondono a due domande diverse che potevano sembrare
la stessa: "il meccanismo di dispatch costa qualcosa?" (no, trascurabile)
contro "il routing REALE del traffico verso CPU costa qualcosa?" (sì,
misurabilmente). Senza Fase 5, il flag `enable_cpu_offload` sarebbe
sembrato "sicuro da accendere sempre" sulla sola base di Fase 4 — invece
resta, giustamente, una scelta esplicita con un trade-off reale da
valutare caso per caso (VRAM risparmiata contro latenza aggregata persa),
non una funzionalità gratuita.

### Current state

- `scheduler/gcsg.py`: `route_forward()` normalizza hidden_states su CPU
  prima del dispatch a freddo (bug fix).
- `benchmarks/bench_hybrid.py` aggiunto, `make bench-hybrid`.
- `tests/test_scheduler.py`: 2 test nuovi, 2 aggiornati.
- Suite completa 159 passed / 3 skipped, lint pulito.
- Numeri a dimensioni sintetiche, non ancora a scala reale Mixtral.
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: rilancio di `bench_hybrid.py` a scala reale
(`OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336`) per confermare o
smentire se il gap GPU/CPU si restringe — decisivo per capire se
`enable_cpu_offload=True` sia mai la scelta giusta in produzione, o solo
uno strumento di misura/ricerca. In parallelo, discussione aperta con il
project owner su una configurazione RunPod (Linux reale, niente limiti
WSL2) per validazione indipendente — GPU/CPU/RAM ancora da decidere.

---

## 2026-08-17 (continued) — specifiche pod Sprint 4 recuperate, un dato reale su AVX-512 su RunPod

**Release:** nessuna — solo ricerca/documentazione, in preparazione a un
eventuale pod RunPod per issue #33.

### What we set out to do

Il project owner ha chiesto le specifiche del pod RunPod di Sprint 4
(GPU/CPU/RAM), in vista di provisionare un pod per issue #33 — e di
annotare qui che, prima di distruggere QUALUNQUE pod futuro, va
recuperata anche la configurazione hardware, non solo i risultati dei
test.

### What we did

Trovato che questa disciplina esisteva già ed era già stata applicata
correttamente in Sprint 4: `logs/sprint4_tekniska/pod_config/` (nel
repo, non perso) contiene uno snapshot completo — `lscpu.txt`,
`free.txt`, `gpu_summary.csv`, `nvidia_smi_full.txt`, `os_release.txt`,
`uname.txt`, `df.txt`, `pip_freeze.txt`, `git_state.txt` — catturato
2026-08-12 19:20 UTC (commit `dfa9a9d`) prima che quel pod venisse
spento (confermato: `LOGBOOK.md` riga ~646, "Before retiring this pod...
Confirmed clean before shutdown").

Sprint 4 ha usato **4 pod diversi** (swap multipli per problemi di
compatibilità/capacità, non pianificati):

| # | GPU | CPU | RAM | Esito |
|---|---|---|---|---|
| 1 | RTX A5000 24GB (CC 8.6) | 12 vCPU, AMD EPYC 7B13 | 25 GB | EU-RO-1, primo pod (2026-08-11) |
| 2 | RTX PRO 6000 Blackwell 96GB (sm_120) | — | — | Scartato subito: `torch==2.5.1+cu124` non supporta sm_120 (`no kernel image available`) |
| 3 | A100 SXM 80GB (sm_80) | — | — | Sub-goal 6 (device-mismatch bug fix sui pesi, 2026-08-12/13) |
| 4 | RTX 3090 24GB (CC 8.6) | 256 vCPU, AMD EPYC 7763 (2×64c/128t, Zen3/Milan) | 1.0 TiB | EU-CZ-1 — snapshot completo (`pod_config/`), run MMLU/TierManager principale (72.28%/72.3%) |

Kernel Linux 6.8.0-134-generic, Ubuntu 22.04.3 LTS, filesystem condiviso
MooseFS (`mfs#eu-cz-1.runpod.net`, 851T, non una quota per-pod — stessa
lezione già annotata più volte in `LOGBOOK.md` per questo dettaglio).

**Trovato controllando `lscpu.txt` riga per riga, non per sentito dire**:
il pod #4 (AMD EPYC 7763) **non ha `avx512f` nei CPU flags** — solo
`avx2`/`avx`/`fma`. EPYC 7763 è Zen3 (Milan, serie 7003): AMD ha
aggiunto AVX-512 solo da Zen4/Genoa (serie 9004) in poi. Questo conferma
concretamente (non ipoteticamente) l'avviso dato al project owner
discutendo la configurazione RunPod per issue #33: le CPU allocate ai
pod GPU RunPod spesso sono AMD EPYC, e il supporto AVX-512 varia molto
per generazione — un pod EPYC 7003-class per issue #33 renderebbe i
numeri di Fase 1/4/5 (calibrati su Xeon 6244 Cascade Lake, AVX-512+VNNI
reale) non rappresentativi, probabilmente con un gap GPU/CPU peggiore,
non uguale.

### Result

Nessun pod Sprint 4 è ancora attivo — tutti e 4 sono già stati spenti,
con la config HW già recuperata correttamente per il pod #4 (quello
rilevante, con lo snapshot completo). Nulla da recuperare urgentemente
ora. La disciplina va solo riapplicata al prossimo pod, non reinventata.

### Why this matters

Prima di provisionare un pod per issue #33, sappiamo già due cose
concrete invece di scoprirle a caldo: (1) come si fa a catturare la
config HW prima dello shutdown — template pronto in
`logs/sprint4_tekniska/pod_config/`, riusabile as-is (`lscpu`, `free -h`,
`nvidia-smi --query-gpu=... --format=csv`, `df -h`, `cat /etc/os-release`,
`uname -a`, `pip freeze`, `git log -1`); (2) che RunPod può assegnare CPU
senza AVX-512 anche su pod pensati per GPU potenti — va verificato
`/proc/cpuinfo | grep avx512f` SUBITO dopo il provisioning, prima di
fidarsi di qualunque numero di `bench_cpu_kernel.py`/`bench_hybrid.py`
raccolto lì.

### Current state

- Nessuna modifica a codice — solo ricerca.
- Checklist per il prossimo pod RunPod (issue #33): dopo il
  provisioning, replicare `logs/sprint4_tekniska/pod_config/` (stessi
  comandi) PRIMA di girare qualunque benchmark, e di nuovo PRIMA di
  distruggere il pod se qualcosa è cambiato nel frattempo (es. resize).
  Verificare `avx512f` esplicitamente come primo passo, non assumerlo
  dalla scheda prodotto.
- GPU/CPU/RAM del prossimo pod: ancora da decidere col project owner.
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: decisione su GPU/CPU/RAM per il pod issue #33, poi
provisioning + verifica AVX-512 + eventuale rilancio di `bench_hybrid.py`
a scala reale lì (visto che WSL2 comunque non blocca questo benchmark
specifico — nessun pin_memory coinvolto — un pod RunPod serve qui solo
per una CPU AVX-512 diversa/più rappresentativa, non per aggirare un
limite di questa macchina).

---

## 2026-08-17 (continued) — pod RunPod per issue #33 provisionato: Malmö

**Release:** nessuna — solo setup, in preparazione all'esecuzione reale del
piano di test concordato (Punto 2: regressione baseline 16/32/64/570 ×2;
Punto 3: conferma esplicita del no-op CPU-offload su path 2/3 a 16/32).

### Pod, dal project owner (non ancora verificato via SSH — vedi sotto)

```
Pod name:        vMemoryFabric - Malmö - CPU Offload DDR
GPU:              RTX A6000 x1 (GA102, CC 8.6 — stessa famiglia di
                  RTX 3090/A5000, 48GB VRAM invece di 24, solo margine
                  in più rispetto alla memory math esistente)
vCPU:             9 (Intel Xeon Gold 6342 @ 2.80GHz — Ice Lake-SP, 3ª
                  gen Xeon Scalable — a differenza dell'AMD EPYC 7763
                  di Sprint 4, gli Ice Lake Xeon Scalable hanno AVX-512
                  nativo; DA VERIFICARE via /proc/cpuinfo appena su, non
                  assunto dalla scheda prodotto, stessa disciplina di
                  sempre)
Memory:           50 GB
Container disk:   50 GB
Pod volume:       128 GB, mount path /data/nvme diretto (a differenza di
                  Sprint 4, dove il Network Volume si montava forzato su
                  /workspace e serviva un `ln -s` manuale — qui non
                  serve, parte già allineato a MODEL_PATH/TierManager)
Pricing:          $0.53/hr compute + $0.007/hr container storage +
                  $0.018/hr volume storage = $0.55/hr totale
Base image:       ghcr.io/danielesalpietro/vmemoryfabric:sprint-4-tekniska
                  (precedente a issue #33 — serve git checkout dei
                  sorgenti freschi, vedi sotto, non un rebuild immagine:
                  i layer pesanti CUDA/torch/vLLM non sono cambiati)
```

**Nome**: continua la convenzione di codename svedesi già in uso nel
progetto (Sprint: Karlshamn...Stockholm, Berg, Marstrand) — Malmö per il
lavoro CPU-offload DDR4 di issue #33 specificamente, non un nuovo Sprint
numerato.

**Checkpoint**: quasi certamente il volume è vuoto (stesso caso di
Sprint 4, primo pod) — non un problema pratico questa volta, confermato
dal project owner: il download da HuggingFace su RunPod è nell'ordine
dei secondi, non richiede pianificazione anticipata come temuto
inizialmente.

### Non ancora risolto: accesso SSH

Stesso problema ricorrente già documentato in `LOGBOOK.md` 2026-08-11
("Access: SSH only, no direct network path from this Claude session" —
l'egress di rete di questa sessione è allowlisted a domini specifici,
`runpod.io`/`api.runpod.io` danno `403` dal proxy dell'ambiente, SSH
diretto sulla porta 22 va in timeout). Il project owner aggiorna la
chiave SSH da console web RunPod per darmi accesso diretto — non ancora
completato al momento di questo log.

### Current state

- Pod in fase di provisioning/setup, non ancora verificato via SSH.
- Nessuna delle verifiche pianificate (AVX-512 reale, spazio disco,
  sincronizzazione sorgenti) ancora eseguita — solo le specifiche
  dichiarate dal project owner, non confermate da comandi reali su
  questa macchina.
- Piano di test concordato (Punto 2 + Punto 3, comandi già preparati
  nella sessione precedente) pronto a partire appena l'accesso è
  disponibile.
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: appena l'SSH è disponibile — verificare AVX-512
(`/proc/cpuinfo`), spazio disco (`df -h`), sincronizzare i sorgenti
(`git fetch && git checkout claude/ddr4-ram-processing-unzseh`), poi
eseguire il piano di test già concordato (Punto 2 baseline ×2, Punto 3
conferma no-op a 16/32).

---

## 2026-08-17 (continued) — Malmö in produzione: validazione reale, bug ambientale trovato, conferma empirica del no-op

**Release:** un fix (`OSX_MMLU_MODEL_PATH` override, commit `b719111`) —
diagnostico, non cambia il comportamento di default. Nessun'altra
modifica a codice di produzione: sessione di validazione, non sviluppo.

### What we set out to do

Eseguire il piano concordato sul pod Malmö: Punto 2 (regressione
baseline, `--wire-tier-manager` senza `--enable-cpu-offload`, fette
16→32→64→570 ×2) e Punto 3 (conferma esplicita del no-op di
`--enable-cpu-offload` sul path 2/3 reale, fette 16/32) — vedi sessione
precedente per il contesto completo (specifiche pod, motivazione).

### What we did

**Accesso SSH**: risolto — chiave locale (`id_ed25519_runpod_sprint4`,
la stessa di Sprint 4) aggiunta a `~/.ssh/authorized_keys` sul pod
direttamente dalla shell già aperta dal project owner, non serviva la
console web. Sorgenti: `/workspace` dell'immagine base risultava vuoto
(un `VOLUME` Docker anonimo probabilmente sovrascrive quello che il
Dockerfile aveva copiato — non indagato a fondo, non necessario) —
risolto con un `git clone` fresco diretto invece di investigare oltre.
Checkpoint: `casperhansen/mixtral-instruct-awq` scaricato da HuggingFace
in **28.9 secondi** (confermata la velocità RunPod↔HF preannunciata dal
project owner).

**Verifica AVX-512**: confermata via `lscpu` E via `/proc/cpuinfo` da
Python — feature set completo (`avx512f`, `avx512bw`, `avx512vl`,
`avx512_vnni`, più estensioni Ice Lake assenti sul Cascade Lake locale:
`avx512_vbmi2`, `avx512_bitalg`, `avx512_vpopcntdq`, `gfni`, `vaes`).

**Quota CPU reale vs. visibile**: `nproc` riporta 96 (topologia
dell'host, visibile al container), ma `/sys/fs/cgroup/cpu.max` =
"765000 100000" → quota reale **7.65 core** — stesso problema già
noto nel progetto (issue #18, "OMP_NUM_THREADS vs real vCPU count").
Impostato `OMP_NUM_THREADS=7` esplicitamente su ogni run per evitare
oversubscription (96 thread in contesa su 7.65 core di quota avrebbe
prodotto numeri fuorvianti, non rappresentativi).

**Punto 2, round 1 — tutto pulito**: 16 (44.1s), 32 (51.6s), 64 (133.9s),
570 completo (**600.8s, 72.1% — 411/570**, contro il baseline storico
72.3%/412/570 del 2026-08-09 — un solo risultato di differenza).
`shadow_activations`/`activation_rate` coerenti su tutte le fette
(~4.5-5.0%, come atteso dalla soglia θ_contamination=0.05). Nessun hang
nella storica zona di rischio (richieste 27-31, causa originaria issue
#10/#16) — confermato ancora una volta che era un artefatto WSL2, non
riprodotto su Linux reale. Un rallentamento transitorio (richieste
300-320, ~3.6s/it di picco) auto-risolto, stesso pattern "micro-slowdown
zone" già documentato in `LOGBOOK.md` per il burn-test Marlin.

**Bug ambientale trovato — non del nostro codice, verificato con un test
di controllo**: durante il Punto 3, `--enable-cpu-offload` a 16 domande
si è bloccato indefinitamente al caricamento del checkpoint (safetensors
shard 0%, stato processo `D` — disk sleep — `wchan:
folio_wait_bit_common`, zero byte letti in 15s consecutivi). Ipotesi del
project owner ("forse crea oggetti in RAM/filesystem col nuovo
parametro?") verificata e ESCLUSA con un test decisivo: lo stesso
identico comando SENZA `--enable-cpu-offload` (il Punto 2 baseline, già
girato 4 volte senza problemi) si è bloccato ALLO STESSO MODO subito
dopo — provando che il problema è ambientale/temporale (finestra di
congestione sul volume di rete condiviso), non legato al flag. `dd`
sullo stesso file leggeva a 4-5 GB/s nello stesso momento — il problema
è specifico all'mmap (usato da safetensors/vLLM per il caricamento), non
al filesystem in generale: classe di problema nota per FUSE (page fault
attraverso FUSE può bloccarsi in modi che una read() sequenziale non
incontra).

**Workaround, non un fix del filesystem** (fuori dal nostro controllo):
checkpoint scaricato una seconda volta direttamente su disco locale del
container (`/root/models/mixtral-instruct-awq`, 31.9s, bypassa
`/data/nvme` interamente). Aggiunto `OSX_MMLU_MODEL_PATH` come override
opzionale del `MODEL_PATH` hardcoded in `eval_mmlu_gcsg.py` (default
invariato) — commit `b719111`. Con questo, tutti i run successivi puliti
e veloci (nessun'altra ricorrenza).

**Punto 2, round 2 — determinismo perfetto**: 16, 32, 64 tutti
IDENTICI byte-per-byte al round 1 rispettivo (stessa accuratezza, stesso
`shadow_activations`, stessa `activation_rate`) — non solo "vicini",
esattamente uguali. Il run completo a 570 del round 2 non è stato
rilanciato: con tre fette su tre già concordanti al 100%, l'esito era
scontato (decisione del project owner, non un'omissione).

**Punto 3 — no-op confermato empiricamente, non solo per lettura del
codice**: a 16 e 32 domande, `--enable-cpu-offload` produce risultati
IDENTICI byte-per-byte al corrispondente run senza flag (stessa
accuratezza, stesso `shadow_activations`). Conferma diretta che il
checkpoint reale (path 3, AWQ ModuleList — non path 1, l'unico coperto
da `_build_cpu_shadow_pool()`, Fase 2) non viene mai toccato dal
routing CPU, esattamente come previsto dall'analisi statica della
sessione precedente.

**Telemetria GPU**: raccolta continua per l'intera sessione (547
campioni, ogni 5s, 00:32-01:18 UTC) — picco utilizzo 99%, picco VRAM
43729/46068 MiB (combacia col budget 0.95 di `gpu_memory_utilization`:
18.95GB pesi + 21GB KV-cache), potenza media 124.9W (sessione intera,
inclusi lunghi periodi di inattività tra un run e l'altro — sotto carico
attivo tocca ~290W).

**Recupero completo prima dello shutdown** (richiesto esplicitamente dal
project owner, due volte, durante la sessione): tutti i 14 file di log,
4 file di risultati JSONL, la telemetria GPU completa, e lo snapshot
HW/env del pod copiati in `logs/malmo_runpod/` — stesso principio già
applicato in Sprint 4 (`logs/sprint4_tekniska/`), verificato file per
file prima di considerare la sessione chiusa.

### Result

Punto 2 e Punto 3 entrambi chiusi con esito pulito. Nessuna regressione
da Fase 0-5 confermata su hardware reale con checkpoint reale (non più
solo sintetico/unit-test). Il no-op di `--enable-cpu-offload` sul
checkpoint reale è ora un fatto misurato, non solo dedotto dal codice.
Un bug ambientale reale (mmap-su-FUSE) diagnosticato e aggirato, non
lasciato come mistero.

### Why this matters

Chiude il cerchio aperto dalla domanda diretta del project owner
("cosa abbiamo ottenuto, a parte tanto codice?"): la risposta concreta è
che oggi sappiamo, con dati reali e non assunzioni, che (1) il lavoro di
Fase 0-5 non ha rotto nulla di già validato, (2) il flag è sicuro anche
restando inerte, (3) esattamente cosa manca per renderlo utile (vedi
Fase 6 sotto) — tre fatti verificati, non tre speranze.

### Fase 6 (nuova) — cosa serve per attivare cpu_offload sul checkpoint reale

Non ancora iniziata — scoping proposto al project owner, decisione
aperta se procedere:

- **Fase 6a — path 3 (AWQ ModuleList), il più fattibile**: `_pin_awq_expert_to_gpu()`
  oggi sposta i pesi AWQ-packed (`qweight`/`qzeros`/`scales`, formato
  GEMM group_size=128 — verificato in `quant_config.json` del checkpoint
  reale) su GPU; il forward (`_AWQShadowExpert`) delega a
  `MixtralMLP.forward()`, che usa un kernel CUDA AWQ fuso
  (dequant+matmul), nessun equivalente CPU in vLLM/AutoAWQ. Serve un
  dequantizzatore AWQ lato CPU (formato documentato, non
  reverse-engineering) che produca pesi fp16/fp32, poi riusa il forward
  "plain PyTorch" già validato da `_ShadowExpertINT4` (Fase 1,
  device-agnostic). Passo 0 prima di scrivere codice: verificare se
  AutoAWQ espone già un dequantizzatore Python/CPU riusabile invece di
  reimplementare il bit-unpacking a mano — non ancora controllato.
  Verifica numerica: parità fp32/CPU vs. kernel CUDA reale, stessa
  metodologia di Fase 1.
- **Fase 6b — path 2 (Marlin), più difficile, priorità più bassa**: il
  formato Marlin-packed (`w13_qweight` ecc.) è bit-interleaved
  specificamente per l'accesso a memoria del kernel Marlin su GPU
  (coalescenza dei warp) — non un semplice 4-bit impacchettato come AWQ.
  Invertirlo per CPU è stato già scartato come approccio in questo
  progetto in un contesto simile (issue #10, motivo diverso ma stessa
  difficoltà di fondo: formato opaco senza reimplementare il kernel).
  Alternativa da esplorare, non verificata: accedere ai pesi AWQ
  pre-repack Marlin, se ancora recuperabili altrove nella pipeline di
  caricamento.
- **Raccomandazione**: Fase 6a da sola basterebbe a rendere reale il
  test di oggi (path 3 è il default quando si usa `--wire-tier-manager`
  senza `--quantization` esplicito). Fase 6b rimandabile finché non
  esiste un motivo di produzione concreto.

### Current state

- `scripts/eval_mmlu_gcsg.py`: `OSX_MMLU_MODEL_PATH` override (workaround
  mmap-su-FUSE), `--enable-cpu-offload` verificato end-to-end su
  checkpoint reale.
- `logs/malmo_runpod/`: log completi, risultati JSONL, telemetria GPU,
  snapshot HW/env — tutto recuperato prima di considerare lo shutdown
  del pod.
- Fase 0-5 confermate senza regressioni su hardware reale + checkpoint
  reale (non solo sintetico).
- Fase 6 (path 2/3 CPU support) proposta, non iniziata — decisione
  aperta col project owner.
- Pod Malmö: pronto per lo shutdown non appena confermato il recupero
  completo (in corso).
- Branch: `claude/ddr4-ram-processing-unzseh`.

Next session: decisione su Fase 6 (procedere con 6a? Verificare prima se
AutoAWQ espone un dequant CPU riusabile?) — o chiudere issue #33 qui,
con l'infrastruttura pronta ma non ancora utile in produzione, come
esito onesto e documentato.

---

## 2026-08-17 (continued 3) — Ricerca prior art: MoE-Infinity non risolve Fase 6, il gap è dell'ecosistema intero

**Release:** nessuna — solo ricerca, nessun codice toccato.

### What we set out to do

Il project owner ricordava un progetto GitHub, MoE-Infinity, che "generava un
file di offload del modello" e chiedeva di verificare se avesse già
risolto il problema aperto di Fase 6 (dequantizzare AWQ/Marlin lato CPU per
rendere `--enable-cpu-offload` utile sul checkpoint reale, non solo
sicuro-ma-inerte).

### What we did

Ricerca web (`EfficientMoE/MoE-Infinity` su GitHub, più il paper arXiv
2401.14361) sul meccanismo di offload reale: cosa genera, come gestisce
pesi quantizzati, se calcola mai su CPU. Poi una seconda ricerca mirata,
allargata oltre il singolo progetto, su chi altro nell'ecosistema supporta
AWQ-per-MoE su CPU.

**Cosa fa davvero MoE-Infinity**: un sistema di cache/prefetch
activation-aware — Expert Activation Matrix (EAM) che traccia quali expert
si riattivano più spesso, per decidere quali tenere "caldi" e prefetchare
verso GPU appena servono. Il "file di offload" ricordato dal project owner
è questa cache/trace di attivazione, non un formato di pesi ricalcolabile:
il progetto **calcola sempre e solo su GPU** — stessa categoria del nostro
`GPUTransfer.to_vram()` pre-issue-#33 (staging passivo), non della Fase 2+
(compute CPU reale). Nessuna menzione di AVX-512, kernel CPU, o
dequantizzazione lato host in tutta la documentazione.

**Sul punto specifico che blocca Fase 6a**: confermato che non è un gap
solo nostro. Quando vLLM gestisce un checkpoint AWQ su un modello MoE lo
converte automaticamente in formato `awq_marlin` — e il kernel Marlin non
è supportato su CPU, per design (formato bit-interleaved pensato per la
coalescenza dei warp GPU, stesso identico ostacolo già descritto per Fase
6b). Trovata una feature request ancora aperta su `sglang-project/sglang`
(issue #5324, "[Feature] Support awq quantization for MoE model on CPU")
che chiede esplicitamente questa capacità — non implementata da nessun
progetto pubblico noto al momento di questa ricerca.

### Result

MoE-Infinity **non** risolve Fase 6 — non fa nulla di comparabile, è
un'ottimizzazione del "quando prefetchare" (rilevante come riferimento di
design per la Fase 0/SEE, non per Fase 6). Il dequantizzatore AWQ lato CPU
resta un problema genuinamente aperto anche fuori da questo progetto: se
lo implementassimo saremmo avanti rispetto allo stato dell'arte pubblico
conosciuto, non recuperando terreno perso.

### Why this matters

Chiude una domanda concreta prima di decidere se investire in Fase 6a: non
c'è una libreria/formato già pronto da riusare per il dequant AWQ-su-CPU,
quindi il costo stimato in Fase 6a (session precedente: "formato
documentato, non reverse-engineering... passo 0: verificare se AutoAWQ
espone già un dequantizzatore riusabile") resta la stima più realistica
disponibile — nessuna scorciatoia trovata. Anche il "non fare nulla" (Fase
6b, Marlin) è ora confermato coerente con quello che fa il resto
dell'ecosistema, non solo una scelta di comodo interna al progetto.

### Current state

- Nessuna modifica a codice — solo ricerca, registrata qui per evitare che
  una sessione futura rifaccia la stessa indagine prima di decidere su
  Fase 6.
- Decisione su Fase 6 (procedere con 6a, o chiudere issue #33 con
  l'infrastruttura pronta-ma-non-ancora-utile come esito documentato)
  ancora aperta col project owner.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: come sopra — decisione su Fase 6a (procedere? verificare
prima se AutoAWQ espone un dequant CPU riusabile, visto che nessun
progetto terzo lo fa già per noi) — o chiudere issue #33 qui.

---

## 2026-08-17 (continued 4) — Passo 0 di Fase 6a risolto: AutoAWQ espone già il dequant CPU che serve

**Release:** nessuna — solo ricerca, nessun codice toccato.

### What we set out to do

Chiudere concretamente il "Passo 0" lasciato aperto in Fase 6a: verificare
se `AutoAWQ`/`autoawq-kernels` espone già una routine di dequantizzazione
CPU riusabile per il formato AWQ GEMM (`qweight`/`qzeros`/`scales`,
group_size=128) del checkpoint reale — invece di scrivere un
unpacker/dequantizzatore da zero.

### What we did

Letto il sorgente reale di `casper-hansen/AutoAWQ` (licenza MIT):
`awq/modules/linear/gemm.py` (`WQLinearMMFunction.forward`, che sceglie tra
`awq_ext.dequantize_weights_cuda` — CUDA — e un fallback quando l'estensione
non è disponibile) e `awq/utils/packing_utils.py`, dove vive il fallback
stesso:

```python
def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * scales
    return iweight
```

Solo operazioni PyTorch standard (`bitwise_and`/`bitwise_right_shift`/
`bitwise_left_shift`, `view`, `repeat_interleave`, aritmetica) —
device-agnostic per costruzione, nessuna estensione CUDA richiesta. Non è
codice sperimentale: è il path "naive (slow) implementation" che AutoAWQ
stesso usa in produzione quando `awq_ext` non è disponibile — già
esercitato da terzi, non da scrivere né validare da zero. Input/output
combaciano esattamente col checkpoint reale (`qweight`/`qzeros`/`scales`,
`group_size=128`, `w_bit=4` — stessi tensori già letti in
`_pin_awq_expert_to_gpu()`/Fase 6 recon della sessione precedente).

### Result

**Passo 0: risposta sì, concreta.** La pipeline proposta (checkpoint AWQ →
`qweight`/`qzeros`/`scales` → `dequantize_gemm()` CPU → peso FP32 →
`_ShadowExpertINT4`, già device-agnostic da Fase 1 → oneDNN GEMM) è
realizzabile senza scrivere un solo rigo di unpack/dequant a mano —
riusando (con attribuzione, licenza MIT) o reimplementando le ~10 righe di
`dequantize_gemm`/`unpack_awq`/`reverse_awq_order`, che sono comunque
piccole abbastanza da poter essere vendorizzate nel progetto se si vuole
evitare una dipendenza runtime da `autoawq`.

### Why this matters

Cambia lo scope di Fase 6a da "progettare un dequantizzatore" a "misurare
una pipeline che già esiste altrove" — un lavoro molto più piccolo di
quanto stimato nella sessione Malmö. Riallinea anche la sequenza corretta
già discussa col project owner: (1) proof-of-path con `dequantize_gemm()`
+ `_ShadowExpertINT4` esistente, (2) benchmark che separa esplicitamente
costo di unpack/dequant, costo del GEMM fp32 (Fase 1 ha già escluso che sia
questo il collo di bottiglia), memoria temporanea, e costo ammortizzato se
i pesi dequantizzati vengono cachati — (3) solo se il dequant risulta
troppo lento, allora ha senso un kernel fuso AWQ custom (unpack+dequant+dot
in un solo passaggio, tuned per l'ISA della CPU target) invece del
dequant-poi-GEMM separato di questo proof-of-path.

### Current state

- Nessuna modifica a codice — solo ricerca, chiude formalmente il "Passo
  0" di Fase 6a lasciato aperto nella sessione Malmö.
- Prossimo passo concreto identificato, non ancora iniziato: script di
  proof-of-path (`dequantize_gemm()` + `_ShadowExpertINT4`) + benchmark a
  4 componenti (unpack/dequant, GEMM fp32, memoria temp, costo
  ammortizzato con caching).
- Decisione su Fase 6a (procedere con l'implementazione ora che il rischio
  tecnico è chiarito, o chiudere issue #33 qui) ancora col project owner.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: se si procede — proof-of-path minimo (`dequantize_gemm()` su
un singolo expert del checkpoint reale, poi attraverso `_ShadowExpertINT4`)
più il benchmark a 4 componenti sopra descritto.

---

## 2026-08-17 (continued 5) — Proof-of-path Fase 6a: funziona, e due numeri reali cambiano il design

**Release:** nuovo `scripts/verify_awq_cpu_dequant.py` — script di ricerca
autonomo, nessuna modifica a `scheduler/gcsg.py` o ad altro codice di
produzione. "Siamo ricercatori, procediamo ed andiamo a vedere" (project
owner) — proof-of-path eseguito su questa macchina (Z8 locale, checkpoint
reale già presente nel volume Docker), non sul pod Malmö (già spento).

### What we set out to do

Chiudere il proof-of-path minimo lasciato aperto nella sessione
precedente: `dequantize_gemm()` (vendorizzata da AutoAWQ, verificata) su
un expert reale del checkpoint locale, poi attraverso `_ShadowExpertINT4`,
misurando i 4 componenti di costo identificati (unpack/dequant, GEMM,
memoria, ammortizzato).

### What we did

**Verificato il sorgente di AutoAWQ direttamente** (non solo il riassunto
della sessione precedente) via `gh api repos/casper-hansen/AutoAWQ/contents/...`
— confermati `AWQ_ORDER`/`AWQ_REVERSE_ORDER` esatti (`[0,2,4,6,1,3,5,7]`/
`[0,4,1,5,2,6,3,7]`), esattamente il rischio di ordine di packing
interleaved segnalato nella sessione precedente, risolto correttamente
dal codice reale di AutoAWQ.

**Trovati i nomi reali dei tensori** nel checkpoint locale
(`model.safetensors.index.json`): `model.layers.{L}.block_sparse_moe.
experts.{E}.{w1,w2,w3}.{qweight,qzeros,scales}` — naming Mixtral standard
(w1=gate_proj, w3=up_proj, w2=down_proj).

**Scritto `scripts/verify_awq_cpu_dequant.py`**: carica i tensori AWQ
grezzi di un expert reale via `safetensors` diretto (nessun motore vLLM,
nessuna GPU — solo lettura file), dequantizza su CPU con
`dequantize_gemm()` vendorizzata, verifica shape/finitezza, esegue un
forward SwiGLU reale attraverso `_ShadowExpertINT4` (scale=1.0, i pesi
sono già dequantizzati — riuso la classe intatta, zero modifiche).

**Trovato subito, non assunto**: la convenzione di layout AWQ GEMM è
`(in_features, out_features)` — es. w1 dequantizzato ha shape (4096,
14336) = (hidden, intermediate), l'OPPOSTO della convenzione nn.Linear/
`_ShadowExpertINT4` `(out_features, in_features)`. Verificato
empiricamente contro `config.json` reale (hidden_size=4096,
intermediate_size=14336), non assunto dalla documentazione AWQ generica.
Serve una trasposizione esplicita prima di passare i pesi a
`_ShadowExpertINT4` — fatto, non stato un problema una volta identificato.

**Trovato un problema di performance reale, non anticipato**: il primo
forward a scala reale (hidden=4096, intermediate=14336, batch=4) in fp16
ha impiegato **3.3-3.4 secondi** — molto più dei ~70-85ms che la sola
scala (28× le dimensioni sintetiche di Fase 1) avrebbe fatto prevedere.
Ipotesi verificata con un confronto diretto (non assunta): il CPU backend
di PyTorch ha supporto fp16 molto meno ottimizzato di fp32. Confermato —
lo stesso identico forward in fp32 impiega **156-204ms**, un
**16.5× più veloce**. Regola pratica concreta per Fase 6a: i pesi
dequantizzati vanno SEMPRE castati a fp32 prima del forward CPU, mai
lasciati fp16 (anche se le `scales` del checkpoint sono fp16 — quello è
il formato di storage, non quello che deve girare nel kernel CPU).

**Trovato un vincolo di memoria reale**: cachare i pesi dequantizzati in
fp32 per un intero expert "freddo" (tutti e 32 i layer di Mixtral) costa
**21.0 GB di DDR4** — molto più dei ~3 GB/expert INT4 lato GPU che sono
il punto di paragone di sempre in questo progetto (memory math in
`gcsg.py`). Soluzione identificata, non implementata: dopo
`dequantize_gemm()` (AWQ→fp32, one-time), ri-quantizzare subito con
`_quantize_int4()` — lo STESSO schema già usato e validato per il path 1
(Fase 1) — e cachare quello (int8+scale, ~4× più compatto,
~5.25GB/expert stimati). Il forward resta `_ShadowExpertINT4` così com'è
(già fa `w_q.to(fp32)*scale` internamente) — nessun pezzo nuovo da
inventare, solo la sequenza AWQ-dequant→INT4-requant da collegare, che
riusa ESCLUSIVAMENTE codice già esistente e validato (AutoAWQ vendorizzato
+ `_quantize_int4` Fase 1 + `_ShadowExpertINT4` Fase 1).

### Result

**Proof-of-path Passo 1 (CPU-only) riuscito**: dequant produce pesi
finiti e numericamente ragionevoli, layout verificato (non assunto),
forward SwiGLU reale eseguito su pesi di un expert reale del checkpoint
di produzione. Due numeri concreti (16.5× fp16→fp32, 21GB→~5.25GB con
re-quantizzazione INT4) cambiano il design da "genericamente fattibile"
a "un piano preciso con numeri reali dietro". Passo 2 (parità numerica
contro il kernel CUDA reale) resta da fare, richiede GPU + un modello
vLLM caricato (o quantomeno il kernel AWQ invocato direttamente) — non
fatto in questa sessione.

### Why this matters

Questo è esattamente il valore di "andare a vedere" invece di stimare:
senza eseguire il proof-of-path, il design naturale (dequant → cache
fp32 → forward) avrebbe portato a un'implementazione 16.5× più lenta del
necessario E con un'impronta di memoria 4× più grande del necessario —
entrambi i problemi invisibili sulla carta, entrambi trovati eseguendo
codice reale su un checkpoint reale in meno di un'ora.

### Current state

- `scripts/verify_awq_cpu_dequant.py` aggiunto — script di ricerca
  autonomo (safetensors diretto, nessuna dipendenza da vLLM/GPU), lint
  pulito.
- Nessuna modifica a `scheduler/gcsg.py` — il proof-of-path è
  deliberatamente separato dal codice di produzione finché il Passo 2
  (parità GPU) non è fatto.
- Design concreto per Fase 6a ora chiaro: AWQ dequant (vendorizzato,
  one-time per expert-layer) → `_quantize_int4()` (riuso Fase 1) →
  cache INT8 → `_ShadowExpertINT4` forward fp32 (riuso Fase 1) — nessun
  pezzo nuovo, solo collegamento.
- Numeri reali raccolti: dequant ~850ms/expert-layer (naive, non
  ottimizzato), forward fp32 ~150-200ms/expert-layer/call a scala reale,
  memoria ~21GB/expert se cachato fp32 (~5.25GB stimati con
  re-quantizzazione INT4, non ancora misurato per davvero).
- Passo 2 (parità numerica contro kernel CUDA AWQ reale) non fatto —
  richiede GPU + vLLM, prossimo passo naturale prima di qualunque
  integrazione in `_build_cpu_shadow_pool()`.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: Passo 2 — parità numerica CPU-dequant vs. kernel CUDA AWQ
reale (richiede GPU), poi misurare per davvero il costo/memoria della
pipeline AWQ-dequant→INT4-requant (non solo stimato), poi — solo se i
numeri restano favorevoli — integrazione in `_build_cpu_shadow_pool()`
per path 3. Decisione finale su procedere ancora aperta col project
owner, ma ora con dati concreti invece di stime.

---

## 2026-08-17 (continued 6) — OpenVINO valutato come alternativa, spostato in issue dedicata #43

**Release:** nessuna — solo ricerca + housekeeping (issue tracker).

### What we set out to do

Il project owner ha chiesto, in parallelo al lavoro sul proof-of-path
sopra (eseguito da un'altra sessione in contemporanea): OpenVINO potrebbe
aiutare con Fase 6a?

### What we did

Ricerca web su `openvinotoolkit/openvino`, `vllm-project/vllm-openvino`,
e (trovato per caso durante la stessa ricerca) l'RFC nativo di vLLM per
l'offload MoE (`vllm-project/vllm#38256`). Confermato:

- OpenVINO 2026.0 ha kernel INT4 nativi CPU (AVX-512/AMX) + import diretto
  di checkpoint AWQ, con compressione data-aware specifica per i MatMul
  3D dei modelli MoE — strutturalmente meglio del dequant-poi-GEMM-fp32
  di Fase 6a per il solo aspetto "velocità di calcolo".
- Ma: supporto MoE ancora "preview" (validato solo su Qwen3-30B-A3B, non
  Mixtral+AWQ), e non componibile a grana di singolo expert dentro
  `GCSGWorker.route_forward()` — richiederebbe un motore di serving
  separato (OVMS/`vllm-openvino`) o la sostituzione dell'intero backend
  vLLM, un cambio architetturale molto più grande del piano Fase 6a
  attuale.
- Dato collaterale utile: l'RFC MoE offload nativo di vLLM (#38256, in
  corso) esclude esplicitamente il calcolo CPU ("hand-rolled CPU MoE...
  is a correctness trap") — conferma che nessun progetto a monte regala
  l'infrastruttura di calcolo CPU che issue #33 sta costruendo da zero.

Deciso, su richiesta esplicita del project owner, di non bloccare il
lavoro in corso (proof-of-path Fase 6a, sessione sopra) per approfondire
OpenVINO ora — aperta invece **issue #43** (sub-issue di #33) con
contesto, criteri di rivalutazione (non procedere finché il benchmark di
Fase 6a non mostra un collo di bottiglia reale, O il supporto MoE di
OpenVINO esce da preview su un checkpoint comparabile, O emerge un motivo
di produzione per servire l'intero modello CPU-only) e un primo passo a
basso costo se/quando si riapre (conversione IR di un singolo expert via
`optimum-intel`/NNCF, confronto diretto contro `dequantize_gemm()+
_ShadowExpertINT4`).

### Result

OpenVINO non è né adottato né scartato — parcheggiato con criteri
espliciti di rivalutazione invece di una decisione impulsiva o di essere
dimenticato. Il proof-of-path Fase 6a (sessione parallela sopra) resta il
piano primario e non è stato rallentato da questa indagine.

### Why this matters

Con due sessioni attive in parallelo sullo stesso branch (questa e quella
che ha scritto `verify_awq_cpu_dequant.py`), il rischio concreto era che
una pista di ricerca laterale (OpenVINO) finisse per competere per
attenzione con il lavoro di implementazione già in corso, o venisse persa
senza traccia. Un'issue dedicata con criteri di rientro espliciti risolve
entrambi i problemi con lo stesso costo di un paragrafo di LOGBOOK.

### Current state

- Issue #43 aperta, sub-issue di #33, nessun lavoro di codice associato.
- Nessuna modifica a codice in questa sessione — solo ricerca +
  organizzazione tracker.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: seguire il thread principale (Passo 2 del proof-of-path,
sessione sopra) — issue #43 resta in attesa dei suoi criteri di
rivalutazione, non un prossimo passo attivo.

---

## 2026-08-17 (continued 7) — vLLM ha già un CPU backend nativo: copre AWQ e copre MoE, ma non insieme

**Release:** nessuna — solo ricerca.

### What we set out to do

Il project owner ricordava che vLLM avesse già una "modalità CPU-only" e
chiedeva conferma — verificare se chiude (parzialmente o del tutto) il gap
di Fase 6a prima ancora di scomodare AutoAWQ/OpenVINO.

### What we did

Ricerca sulla documentazione ufficiale vLLM (`VLLM_TARGET_DEVICE=cpu`,
matrice hardware/quantizzazione supportata) e sul PR storico che ha
aggiunto MoE fuso AWQ (`vllm-project/vllm#6422`, `AWQMoEMethod`).

Confermato: **sì, esiste** un CPU backend nativo x86 in vLLM, con:
- AWQ supportato — ma solo su modelli **densi** (non-MoE).
- MoE supportato — ma solo **fp16/bf16 non quantizzato**, via
  `ipex.llm.modules.GatedMLPMOE` (Intel Extension for PyTorch),
  controllato da `VLLM_CPU_MOE_PREPACK`.
- **AWQ + MoE insieme, su CPU: non supportato.** Conferma testuale diretta
  dalla documentazione stessa di vLLM: *"vLLM converts AWQ to AWQ Marlin
  if the model is a MoE model, but Marlin kernel is not supported on
  CPU"* — stesso identico muro già trovato analizzando il checkpoint
  reale nella sessione Malmö (Fase 6, path 2/Marlin), ora confermato
  anche dal lato "cosa fa/non fa il CPU backend di vLLM stesso", non solo
  dal lato "cosa richiede il nostro checkpoint".

**Trovato un riferimento utile per il Passo 2** (parità numerica CPU vs.
kernel CUDA reale, già identificato come prossimo passo nel thread del
proof-of-path): `fused_moe_kernel_gptq_awq` in
`vllm/model_executor/layers/fused_moe/fused_moe.py` è l'implementazione
CUDA reale di `AWQMoEMethod` — la stessa classe che `_AWQShadowExpert`
(`gcsg.py`) già chiama oggi via `quant_method.apply()`. È il riferimento
numerico "verità" corretto per quel confronto, non un vicolo cieco.

### Result

Il CPU backend nativo di vLLM **non** chiude il gap di Fase 6a — copre
AWQ e MoE separatamente ma non la loro intersezione, la stessa che ci
serve. Non cambia il piano (AutoAWQ dequant vendorizzato +
`_ShadowExpertINT4`, proof-of-path già in corso nel thread parallelo).
Trovato però un puntatore concreto e utile per il Passo 2 di quel thread.

### Why this matters

Terza conferma indipendente (dopo l'analisi del checkpoint reale in
Malmö e la ricerca su MoE-Infinity/sglang) dello stesso fatto: AWQ-MoE-su-
CPU è un gap reale dell'intero ecosistema, non un dettaglio che
sarebbe bastato leggere meglio la documentazione di uno strumento
esistente. Aumenta la fiducia che investire in Fase 6a sia un lavoro
genuinamente utile, non un problema già risolto altrove e ignorato.

### Current state

- Nessuna modifica a codice — solo ricerca.
- Puntatore a `fused_moe_kernel_gptq_awq` disponibile per il Passo 2 del
  thread principale (proof-of-path Fase 6a).
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: come sopra — Passo 2 del proof-of-path (parità numerica),
usando `fused_moe_kernel_gptq_awq` come riferimento.

---

## 2026-08-17 (continued 8) — Cambio di prospettiva discusso: CPU-first invece di GPU-first, aperta issue #44

**Release:** nessuna — solo discussione + housekeeping (issue tracker).

### What we set out to do

Il project owner ha posto una domanda di fondo, non incrementale: e se
invertissimo la prospettiva? Oggi issue #33 è GPU-first (modello
VRAM-resident, Fase 6 aggiunge CPU offload opt-in per i soli expert
freddi). E se fosse **CPU-first** — modello DDR4-resident per
costruzione, GPU come acceleratore/cache solo per il sottoinsieme caldo?

### What we did

Discussione (nessuna ricerca nuova — riusa i finding già raccolti su
MoE-Infinity nella sessione OpenVINO di oggi). Valutazione:

**A favore**: coerente con la gerarchia EMH che il progetto dichiara
(NVMe→DDR4→VRAM, promozione verso l'alto — non "VRAM con overflow").
Sostenuto dai numeri reali già raccolti per issue #33: Z8 centinaia di GB
DDR4 vs 24GB VRAM, pod Malmö 503GiB RAM vs 46GB VRAM (`logs/malmo_runpod/
SUMMARY.md`) — un modello CPU-first non ha mai pressione di memoria GPU
per costruzione, elimina l'aritmetica fragile di "quanti shadow expert
entrano in VRAM" che oggi esiste solo perché il modello parte
GPU-resident (`GCSGGuard.__init__`, memory math in `gcsg.py`).

**Il punto decisivo**: MoE-Infinity (già valutato per Fase 6/OpenVINO,
sessione precedente) non è solo "rilevante come riferimento di design" —
è **esattamente** l'architettura CPU-first con cache GPU
activation-aware che questa domanda descrive, già maturo e benchmarked.
Decisione di principio: se si procede, si eredita il design (e dove
possibile il codice, licenza MIT da riverificare puntualmente prima di
qualunque porting) di MoE-Infinity, non lo si reinventa da zero dentro
`TierManager`/`GCSGWorker` — stesso principio già seguito per `MoeCpuHost`
di exllamav3 nell'apertura originale di issue #33.

**Perché è issue separata, non estensione di Fase 6**: Fase 6 è
incrementale dentro l'architettura GPU-first esistente (path canonico
invariato). Questa domanda tocca *quale tier è quello canonico* — stessa
classe di decisione di #43 (sostituire un motore), non "aggiungere un
percorso opt-in". Trattarla come parte di Fase 6 avrebbe rischiato di
far deragliare il proof-of-path già in corso (sessione parallela,
Passo 2 pendente) dentro una domanda architetturale molto più grande.

Aperta **issue #44** (sub-issue di #33): contesto, motivazione coi numeri
reali, principio "eredita MoE-Infinity, non reinventare", proposta di
interfaccia non-distruttiva (`GCSGWorker.configure_serving_mode(mode:
"gpu_first"|"cpu_first")`, default invariato), criteri di non-avvio
(aspettare l'esito di Fase 6; oppure un caso reale dove GPU-first con
overflow selettivo smette di bastare per capacità, non solo velocità;
oppure conferma puntuale della licenza MoE-Infinity prima di qualunque
porting di codice).

### Result

Nessuna decisione di procedere — la domanda è registrata con un percorso
di valutazione esplicito invece di restare una divagazione a voce.
Entrambe le issue #43 e #44 condividono lo stesso principio: non
bloccare/deragliare Fase 6a (proof-of-path in corso nella sessione
parallela) per esplorare alternative più grandi, ma non perdere il filo
del ragionamento nel frattempo.

### Why this matters

Con due sessioni attive sullo stesso branch, il rischio concreto di una
domanda architetturale di questa portata era di competere per attenzione
col lavoro di implementazione già a metà (Passo 2 del proof-of-path) o di
essere dimenticata a fine sessione. Un'issue con criteri di rientro
espliciti — stesso pattern già usato per #43 — risolve entrambi i
problemi.

### Current state

- Issue #44 aperta, sub-issue di #33 (ora 2 sub-issue: #43 OpenVINO, #44
  CPU-first/MoE-Infinity).
- Nessuna modifica a codice in questa sessione.
- Verificato via `git fetch` prima di scrivere e prima di pushare: nessun
  commit nuovo dalla sessione parallela nel frattempo — nessun conflitto.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: seguire il thread principale (Passo 2 del proof-of-path,
sessione parallela) — #43 e #44 restano in attesa dei rispettivi criteri
di rivalutazione, non prossimi passi attivi.
