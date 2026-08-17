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

---

## 2026-08-17 (continued 9) — Passo 2: parità numerica confermata contro il kernel CUDA AWQ reale

**Release:** nuovo `scripts/verify_awq_cpu_dequant_parity.py`. Nessuna
modifica a `scheduler/gcsg.py` — stessa disciplina della sessione
precedente, proof-of-path tenuto separato dal codice di produzione finché
la parità non è verificata (ora lo è).

### What we set out to do

Il Passo 2 lasciato aperto: confrontare l'output del nostro dequant CPU
(Passo 1) contro il kernel CUDA AWQ reale, sullo stesso expert dello
stesso checkpoint, stesso input — non un output "plausibile", una
verifica numerica diretta.

### What we did

Scritto `verify_awq_cpu_dequant_parity.py`: carica il modello reale via
`vllm.LLM()` (plain, non `GCSGWorker` — non servono gli hook GCSG per
questo confronto), estrae il vero modulo `MixtralMLP` di un expert,
esegue il suo forward reale (kernel CUDA AWQ vero) su un input sintetico,
poi lo stesso input attraverso dequant CPU + `_ShadowExpertINT4` sugli
STESSI tensori `qweight`/`qzeros`/`scales` (letti direttamente dal
modulo caricato, non da un file separato come nel Passo 1 — garantisce
che siano esattamente gli stessi, non solo dello stesso file).

Nota per chi legge entrambi i thread paralleli di questa giornata: la
sessione parallela ("continued 7" sopra) aveva segnalato
`fused_moe_kernel_gptq_awq` come riferimento numerico "verità" per questo
confronto — approccio valido, non usato qui per semplicità: chiamare
direttamente il modulo `MixtralMLP` reale (`module(hidden_states)`)
ottiene lo stesso ground truth senza dover invocare il kernel CUDA a
basso livello a mano, stesso principio già usato per `_AWQShadowExpert`
in produzione (delega al modulo reale, zero reimplementazione).

**Due blocchi reali incontrati e risolti, entrambi ambientali — non del
nostro dequant**:
1. Il backend flash-attention di default crasha nel profiling interno di
   vLLM (`cu_seqlens_q must have dtype int32`) con `LLM()` semplice
   (senza `worker_cls=GCSGWorker`) — mai incontrato prima in questa
   sessione perché ogni run precedente con checkpoint reale girava sul
   pod Malmö (Linux reale), non su questa macchina locale (Z8, WSL2).
   Aggirato con `VLLM_ATTENTION_BACKEND=XFORMERS`, non indagato a fondo
   (probabile drift di versione flash-attn/vLLM locale, fuori scope per
   Fase 6a).
2. `hf_overrides={"head_dim": 128}` mancante — senza, `num_heads` risolve
   a `None` e il calcolo della dimensione del blocco KV-cache crasha.
   Stesso override già presente in `eval_mmlu_gcsg.py` (dove passa
   inosservato, mai isolato prima come singolo requisito). Aggiunto.

**Risultato, due combinazioni layer/expert indipendenti**:

| Layer/Expert | Stato modulo | Errore relativo L2 | Differenza assoluta max |
|---|---|---|---|
| 0/0 | offloaded (pinning automatico riuscito) | 0.000483 | 0.0076 |
| 15/5 | già GPU-resident | 0.000555 | 0.0021 |

Entrambi ben sotto la soglia stretta (1e-2, non lo 0.5 di Fase 1 — qui
non c'è rumore di quantizzazione atteso, solo precisione fp16/fp32).
Il primo caso (layer 0, offloaded di default con `cpu_offload_gb=4`) ha
anche esercitato il pinning esplicito GPU (`module.to("cuda")` prima del
forward diretto) — la stessa identica situazione che nel 2026-08-12
(issue #10/#16) crashava senza quel fix, qui gestita correttamente senza
sorprese.

### Result

**Parità confermata, non solo plausibilità.** Il dequant CPU (AutoAWQ
vendorizzato) + `_ShadowExpertINT4` (Fase 1, invariata) produce lo
STESSO output del kernel CUDA AWQ reale su un expert reale del
checkpoint di produzione, entro la sola precisione numerica. La pipeline
proposta in Fase 6a non è più "probabilmente corretta per lettura del
codice" — è verificata contro il ground truth reale.

### Why this matters

Chiude il rischio tecnico più grande di Fase 6a (correttezza del dequant)
con un dato diretto, non un'inferenza. Le tre verifiche fatte finora
(Passo 1: shape/layout/plausibilità; costo fp16 vs fp32; Passo 2: parità
contro kernel reale) coprono correttezza E performance — quello che
manca ora è solo la misura reale della pipeline
AWQ-dequant→INT4-requant→cache (stimata, non misurata) prima di decidere
sull'integrazione in produzione.

### Current state

- `scripts/verify_awq_cpu_dequant_parity.py` aggiunto, lint pulito,
  verificato due volte (layer/expert diversi, un caso offloaded un caso
  GPU-resident).
- Nessuna modifica a `scheduler/gcsg.py`.
- Suite pytest invariata (159 passed / 3 skipped), nessuna regressione.
- Rischio tecnico principale di Fase 6a (correttezza) chiuso. Resta:
  misurare per davvero (non stimare) il costo della pipeline completa
  AWQ-dequant→`_quantize_int4()`-requant→cache, poi decidere
  sull'integrazione in `_build_cpu_shadow_pool()`.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: misurare la pipeline completa (dequant AWQ → requant INT4
via `_quantize_int4()` → cache → forward `_ShadowExpertINT4`) sul
checkpoint reale, non solo stimarla — poi decisione finale su
integrazione in `_build_cpu_shadow_pool()` per path 3, con dati completi
invece di stime parziali.

---

## 2026-08-17 (continued 10) — Misura completa della pipeline: la cache INT4 è inutilizzabile, la cache fp32 no

**Release:** nuovo `scripts/bench_awq_cpu_pipeline.py`. Nessuna modifica a
`scheduler/gcsg.py`. "Portiamo a casa le misurazioni di tutto questo
lavoro" (project owner) — chiude i pezzi lasciati stimati dopo Passo 1/2.

### What we set out to do

Misurare per davvero (non stimare) i 4 componenti mancanti della
pipeline Fase 6a: costo di ri-quantizzazione INT4, memoria reale
cachata, costo del forward a regime — e, aggiunto perché non testato
prima d'ora, se la doppia quantizzazione (AWQ reale →
`_quantize_int4()` simulato) degrada l'accuratezza oltre il kernel CUDA
reale.

### What we did

Scritto `bench_awq_cpu_pipeline.py`: stesso ground truth di Passo 2
(modulo `MixtralMLP` reale, kernel CUDA AWQ reale), confronta due
strategie di caching sullo stesso expert — `fp32` (cache diretta dei
pesi dequantizzati AWQ, Passo 1/2) contro `int4` (ri-quantizzazione via
`_quantize_int4()`, Fase 1, subito dopo il dequant AWQ).

**Risultato, layer 0 / expert 0**:

| | dequant AWQ | requant INT4 | forward/call | memoria/expert (32 layer) | errore vs. kernel reale |
|---|---|---|---|---|---|
| `fp32_cache` | 1210.8ms | — | 199.9ms | 21.00 GB | **0.000530** |
| `int4_cache` | 1210.8ms | 975.3ms | 312.4ms | 5.25 GB | **0.950429** |

### Result

**La cache INT4 è inutilizzabile così com'è — non un compromesso
accettabile, un fallimento.** Errore relativo 0.95 non è "un po' peggio
del previsto": è quasi totale distruzione del segnale, ben oltre anche
la soglia già generosa di Fase 1 (0.5, per un caso di quantizzazione
diretta su pesi random gaussiani sintetici). La causa è strutturale, non
un bug da correggere in `_quantize_int4()`: quella funzione fa
quantizzazione **per-tensore simmetrica** (uno scale unico per l'intero
tensore) — adeguata per Fase 1 (pesi random gaussiani sintetici,
distribuzione uniforme, nessun outlier). AWQ invece usa quantizzazione
**per-gruppo** (group_size=128, verificato in `quant_config.json` del
checkpoint reale) con zero-point asimmetrico — esiste apposta per
gestire la varianza reale dei pesi di un modello allenato (outlier
per-canale che con uno scale unico per-tensore schiacciano la precisione
ovunque altrove). Ri-quantizzare pesi REALI con lo schema SBAGLIATO
(grana sbagliata) non è una piccola perdita aggiuntiva di precisione —
è la stessa classe di errore che AWQ è stato specificamente progettato
per evitare, reintrodotta ignorando il motivo per cui esiste.

**La cache fp32 invece è pulita e utilizzabile**: errore 0.00053 (stesso
ordine di grandezza del solo Passo 2, nessun degrado aggiuntivo — logico,
non c'è un secondo round di quantizzazione), 199.9ms/call a regime,
21GB/expert su tutti i 32 layer.

### Why this matters

Chiude la domanda "quanto costa la pipeline completa" con un dato che
CAMBIA la conclusione, non solo la conferma: l'idea di "ri-quantizzare
per risparmiare memoria" (5.25GB contro 21GB, motivazione originale del
Passo 1) non è realizzabile riusando `_quantize_int4()` così com'è — 
servirebbe una quantizzazione per-gruppo (matching group_size=128 di
AWQ), lavoro nuovo non ancora scoping, non "solo collegare pezzi già
esistenti" come sembrava dopo il Passo 0. La cache fp32 (21GB/expert)
resta l'UNICA strategia misurata che funziona — più pesante in memoria
di quanto sperato, ma corretta. Con centinaia di GB di DDR4 disponibili
sia sul Z8 locale sia sul pod Malmö (503GiB), 21GB/expert non è
proibitivo in assoluto — solo più caro del previsto, e limita quanti
expert "freddi" si possono cachare contemporaneamente rispetto
all'ipotesi INT4 (4x meno expert cachabili a parità di RAM).

### Current state

- `scripts/bench_awq_cpu_pipeline.py` aggiunto, lint pulito, verificato
  su hardware reale (layer 0/expert 0).
- Nessuna modifica a `scheduler/gcsg.py`.
- **Decisione tecnica chiara ora disponibile**: se si procede
  all'integrazione in `_build_cpu_shadow_pool()`, la cache deve essere
  fp32 (21GB/expert), non INT4 — l'alternativa INT4 richiederebbe
  quantizzazione per-gruppo non ancora scritta, un pezzo di lavoro nuovo
  e non banale (fuori dallo scope "solo collegare" del Passo 0).
- Le quattro verifiche complete di Fase 6a: Passo 0 (dequant esiste già,
  AutoAWQ), Passo 1 (shape/layout/fp32-vs-fp16), Passo 2 (parità kernel
  reale), misura pipeline completa (fp32 sì, INT4-semplice no) — tutte
  fatte, tutte con dati reali.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: decisione finale col project owner — integrare in
`_build_cpu_shadow_pool()` con cache fp32 (21GB/expert, funziona, misura
completa disponibile), o scoping di una quantizzazione per-gruppo prima
di procedere (lavoro nuovo, non ancora dimensionato), o chiudere Fase 6a
qui con i dati raccolti come esito documentato.

---

## 2026-08-17 (continued 11) — Consigli esterni vagliati: contiguità confermata (-33%), test di guardia per il risultato negativo

**Release:** `bench_awq_cpu_pipeline.py` aggiornato (fix contiguità),
nuovo test in `tests/test_cpu_kernel.py`. Nessuna modifica a
`scheduler/gcsg.py`.

### What we set out to do

Il project owner ha girato 4 consigli tecnici esterni per "blindare" una
futura integrazione di Fase 6a (cache fp32): gestione memoria/GC, tuning
thread PyTorch, contiguità dei tensori, un test per il risultato negativo
dell'INT4. Vagliati uno per uno prima di implementare alla cieca — non
tutti erano applicabili o corretti nello specifico.

### What we did

**Scartato/rimandato**: gestione memoria aggressiva (`del`+`gc.collect()`
ad ogni eviction, cache LRU a 15-20 esperti). Non esiste ancora nessuna
cache/eviction multi-expert da blindare — oggi solo un expert su un
layer è stato provato. `gc.collect()` ad ogni eviction è probabilmente
overhead inutile (CPython libera per refcounting su `dict.pop()`, non
serve GC ciclico qui). Il tetto "15-20 esperti" non torna con `shadow_
pool_size=2` di oggi (al massimo 1 expert freddo alla volta) — questa
parte del ragionamento resta valida — ma il secondo motivo dato
("a 21GB/expert, 15 esperti = 315GB, non entrano nemmeno nei 256GB del
Z8 locale") era SBAGLIATO: 256GB veniva da un riferimento vecchio in
`LOGBOOK.md`, mai riverificato qui. **Correzione (stessa sessione,
project owner)**: l'host reale ha **503.75 GB** (`wmic`, verificato),
di cui solo 196GB allocati a WSL2/Docker (`.wslconfig`, `memory=196GB`)
al momento in cui è stato scritto il paragrafo sopra — un limite di
configurazione, non hardware. 315GB per 15 esperti sarebbero entrati
comodamente nell'host reale, anche se non nei 196GB allocati allora a
Docker. Il motivo "fuori scope per shadow_pool_size=2 di oggi" resta
comunque sufficiente da solo per rimandare il tetto a 15-20 — non serve
il secondo argomento (sbagliato) per la stessa conclusione. `.wslconfig`
aggiornato subito dopo a `memory=444GB` (vedi sessione successiva) —
non per questo lavoro specificamente, ma corregge il limite per
qualunque lavoro futuro ne avesse bisogno.

`torch.set_num_threads()` per il tuning
thread: valido in principio (stessa lezione di `OMP_NUM_THREADS` già
imparata oggi sul pod Malmö), ma non c'è ancora un punto di
inizializzazione a cui agganciarlo — rimandato all'integrazione vera.

**Verificato e confermato**: contiguità dei tensori. Test isolato
(matmul sulla stessa shape di `w2`, con e senza `.contiguous()` dopo un
`.T`): **4.4× più lento** senza (42.5ms vs 9.7ms). Applicato a
`bench_awq_cpu_pipeline.py` (`w13_fp32`/`w2_fp32` ora `.contiguous()`
esplicito) e rimisurato sulla pipeline reale: forward a regime
**199.9ms → 133.3ms (-33%)**, accuratezza invariata (0.000530, la
contiguità non tocca la matematica, solo le prestazioni — atteso e
confermato).

**Implementato**: `TestQuantizeInt4KnownLimitation` in
`tests/test_cpu_kernel.py` — guardia permanente contro la tentazione di
ri-quantizzare pesi AWQ-dequantizzati con `_quantize_int4()` per
risparmiare memoria (esattamente l'errore 0.95 di sessione precedente).
Non richiede checkpoint reale/GPU: riproduce la STRUTTURA del problema
(gruppi a scala molto diversa, come la quantizzazione per-gruppo di AWQ
esiste per gestire), non i pesi esatti — gira nella suite CPU veloce.

Primo tentativo (misurare l'errore sulla sola ricostruzione del peso)
NON riproduceva il problema (0.17, non >0.9) — verificato che il
degrado catastrofico emerge dalla propagazione attraverso il FORWARD
completo (due matmul + SiLU: i gruppi "normali" arrotondano a ~0 con
uno scale dominato dagli outlier, poi l'errore si compone), non dalla
quantizzazione del peso isolata. Riscritto per misurare l'errore sul
forward (riusando `_reference_forward_fp32` già esistente in questo
file) — raggiunge 1.0 di errore relativo con la stessa struttura, anche
oltre il 0.95 reale. Soglia del test a 0.9, con margine.

Full suite: **160 passed** (+1), 3 skipped, 0 failed. `ruff check`
pulito (stesso singolo warning pre-esistente non correlato).

### Result

Due dei quattro consigli erano validi e concreti (contiguità, test di
guardia) — implementati, con un miglioramento di performance reale
(-33%) e una guardia permanente contro un errore già commesso una volta.
Due erano prematuri o mal calibrati per lo scope attuale (gestione
memoria multi-expert, tuning thread) — non scartati, solo rimandati a
quando esisterà del codice a cui si applicano davvero.

### Why this matters

Vagliare consigli esterni con lo stesso rigore usato per le proprie
ipotesi — verificare prima di implementare, non implementare perché
"suona ragionevole" — ha pagato due volte in questa sessione: la
contiguità era un miglioramento reale non ancora colto (-33%, non
banale), e il primo tentativo di test di guardia sarebbe passato senza
proteggere nulla se non fosse stato verificato empiricamente che non
riproduceva il problema reale.

### Current state

- `scripts/bench_awq_cpu_pipeline.py`: fix `.contiguous()`, numeri
  aggiornati (133.3ms/call fp32, non più 199.9ms).
- `tests/test_cpu_kernel.py`: `TestQuantizeInt4KnownLimitation` (1 test),
  guardia permanente contro la ri-quantizzazione INT4 di pesi
  per-gruppo-quantizzati.
- Suite completa 160 passed / 3 skipped, lint pulito.
- Gestione memoria multi-expert e tuning thread: note valide, rimandate
  a quando esiste codice di integrazione reale a cui applicarle.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: come sopra — decisione finale su Fase 6a (integrare con
cache fp32, scoping quantizzazione per-gruppo, o chiudere qui) — ora con
anche il numero di performance corretto (133.3ms/call, non 199.9ms) e
una guardia di regressione per non ripetere l'errore INT4.

---

## 2026-08-17 (continued 12) — Corretto un numero sbagliato nel log, alzata la RAM allocata a WSL2/Docker

**Release:** nessuna modifica a codice — un file di sistema
(`C:\Users\Admin\.wslconfig`) e un riavvio di WSL2.

### What we set out to do

Il project owner ha corretto un numero scritto nella sessione precedente
("256GB del Z8 locale", usato per dire che una cache di 15-20 esperti
non ci sarebbe entrata): la macchina reale ne ha 504GB. Verificare e, se
serve, rimediare aumentando quanto allocato a WSL2/Docker.

### What we did

Verificato con comandi reali, non per fiducia: `wmic computersystem get
TotalPhysicalMemory` → 540853346304 byte = **503.75 GB** (host reale,
conferma il numero del project owner). `.wslconfig` aveva `memory=196GB`
— spiegava perché `docker info` vedeva solo 192.6GiB, non un limite
hardware. Confermato nessun container attivo (`docker ps -a`, tutti
`Exited`) prima di un'azione che richiede riavvio — sicuro, nessun
lavoro in corso da perdere.

Alzato `.wslconfig` a `memory=444GB` (margine di ~60GB per Windows/altri
processi, non al limite), `wsl --shutdown`, verificato il riavvio
automatico: `docker info` ora riporta **436.8GiB** (era 192.6GiB).

Corretta la sessione precedente (vedi sopra, "continued 11") — non
cancellato il numero sbagliato, annotata la correzione sul posto, stesso
principio già seguito in questo logbook fin dall'inizio.

### Result

RAM reale disponibile a Docker/WSL2 più che raddoppiata (192.6GiB →
436.8GiB), senza perdita di lavoro (nessun container attivo). Il numero
sbagliato nel log è corretto, non nascosto.

### Why this matters

Il vincolo di memoria che aveva fatto scartare il tetto "15-20 esperti"
della sessione precedente era in parte un artefatto di configurazione,
non un limite fisico — ora corretto. Non cambia la conclusione di quella
sessione (lo scope attuale, `shadow_pool_size=2`, resta il motivo
sufficiente per rimandare quel tetto), ma toglie di mezzo un dato
sbagliato prima che informasse una decisione futura. Con 436.8GiB
disponibili, un eventuale lavoro futuro su cache multi-expert più grandi
(issue #44, CPU-first) avrebbe comunque margine reale, non solo teorico.

### Current state

- `C:\Users\Admin\.wslconfig`: `memory=444GB` (era 196GB).
- Docker/WSL2 confermato riavviato con la nuova allocazione (436.8GiB
  disponibili).
- Nessun container perso — tutti già `Exited` prima del riavvio.
- Correzione annotata nella sessione precedente, non solo qui.
- Nessuna modifica a codice del progetto in questa sessione.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: come sopra — decisione finale su Fase 6a. Nessun'azione
di sistema pendente.

---

## 2026-08-17 (continued 13) — Stessa domanda per la CPU: allocazione WSL2 raddoppiata (14→28 core)

**Release:** nessuna modifica a codice — un altro campo di
`.wslconfig` più un riavvio di WSL2.

### What we set out to do

Il project owner ha posto la stessa domanda già fatta per la RAM, questa
volta sulla CPU: i core allocati a Docker bastano? E prima di decidere
un numero, ha chiesto di misurare l'uso REALE dell'host in questo
momento (RAM e CPU), per capire se serve liberare risorse o se si può
semplicemente allocare di più senza sottrarre nulla a nessuno.

### What we did

**Host reale**: 2× Intel Xeon Gold 6244 (8 core/16 thread ciascuno,
dual-socket) = 16 core fisici / 32 thread logici totali, AVX-512+VNNI
confermati anche dopo il riavvio precedente. `.wslconfig` aveva
`processors=14` — meno della metà della capacità totale, `lscpu` dentro
il container vede tutti e 14 su un solo NUMA node (0-13) — non
verificabile dall'interno della VM se rifletta i confini NUMA reali
dell'host o sia una vista appiattita di Hyper-V.

**Uso reale dell'host misurato PRIMA di decidere** (non assunto): RAM —
503.7GB totali, **464GB liberi** in questo momento, `vmmemWSL` (il
processo che rappresenta l'intera VM WSL2) usa solo 9.26GB nonostante
444GB allocati come tetto massimo (nessuna pressione, nessun processo
in competizione). CPU — **1% di carico** (`Win32_PerfFormattedData_
PerfOS_Processor`), host sostanzialmente a riposo. Nessun processo
Windows in esecuzione consuma risorse significative (i maggiori
consumatori di RAM sono le sessioni Claude Code stesse e VS Code, sotto
1GB ciascuno).

**Risultato della misura**: nessuna ottimizzazione/spegnimento
necessario — l'host ha margine reale, non solo teorico, per allocare di
più senza toccare nulla. Alzato `processors=14` → **28** (lascia 4
thread/2 core a Windows, stesso principio dell'88% usato per la RAM).
Confermato nessun container ATTIVO prima del riavvio (`mike_mem0` in
stato `Created`, mai avviato — nessuno stato da perdere). `wsl
--shutdown`, riavvio automatico verificato: `docker info` ora riporta
**28 CPU** (era 14) e **436.8GiB RAM** (invariato dalla sessione
precedente). Suite pytest completa rilanciata dopo il riavvio: 160
passed / 3 skipped, invariata.

### Result

Allocazione CPU a Docker/WSL2 raddoppiata (14→28 core) senza alcun
compromesso — l'host aveva margine reale misurato, non solo capacità
teorica. Stessa correzione di postura già fatta per la RAM: verificare
l'uso reale prima di decidere un numero, non allocare/rimandare per
istinto.

### Why this matters

La domanda "i core bastano?" da sola non avrebbe avuto una risposta
solida senza sapere se l'host fosse già sotto pressione da altri
processi — misurare prima (1% CPU, 464GB RAM liberi) ha reso la
decisione "alza a 28" priva di rischio reale, non un'ipotesi ottimista.
Stesso principio "misura prima di ottimizzare" già applicato per ogni
altra decisione tecnica di questa issue, qui applicato all'infrastruttura
invece che al codice.

### Current state

- `C:\Users\Admin\.wslconfig`: `memory=444GB`, `processors=28` (era 14).
- Docker/WSL2 confermato riavviato con la nuova allocazione (28 CPU,
  436.8GiB RAM).
- Nessun container perso — nessuno era in esecuzione.
- Suite completa 160 passed / 3 skipped dopo il riavvio, invariata.
- Nessuna modifica a codice del progetto in questa sessione.
- Domanda aperta e distinta, non ancora affrontata: specifiche per un
  eventuale prossimo pod RunPod (RAM/CPU/GPU) — lì il vincolo è budget,
  non condivisione con altri processi host, quindi la stessa logica
  "misura l'uso reale prima" non si applica allo stesso modo (nessun
  host condiviso da misurare, il pod è dedicato per costruzione).
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: decisione finale su Fase 6a (invariata). Separatamente,
da definire se/quando: specifiche di un prossimo pod RunPod, se emerge
un motivo concreto per provisionarne uno (es. proseguire Fase 6a o
issue #43/#44 su hardware Linux reale).

---

## 2026-08-17 (continued 14) — Fase 6a Passo 3: il dequant AWQ collegato a GCSGWorker, non più solo in script standalone

**Release:** sì — `src/scheduler/gcsg.py` (nuovo codice di produzione),
`tests/test_scheduler.py` (23 nuovi test), `scripts/verify_awq_cpu_dequant_wired.py`
(nuovo, Passo 3).

### What we set out to do

Chiudere il pezzo mancante identificato a fine sessione precedente:
tutto il lavoro Fase 6a fino a ieri (dequant AWQ, parità numerica contro
il kernel CUDA reale — Passo 2, ~0.0005 errore relativo — pipeline
completa fp32-cache vs INT4-cache) viveva in script standalone sotto
`scripts/`, mai collegato a `GCSGWorker`. Senza questo collegamento,
`--enable-cpu-offload` non fa nulla di reale sul path 3 (AWQ ModuleList)
— l'UNICO path che il checkpoint di produzione
(casperhansen/mixtral-instruct-awq) usa. Prerequisito esplicito prima di
lanciare qualunque run di validazione (16→32→64→full).

### What we did

**Vendorizzato in `gcsg.py`** (accanto a `_quantize_int4`, non in un
modulo separato — stessa convenzione del resto del file):
`_dequantize_awq_gemm`/`_awq_unpack`/`_awq_reverse_order` (identiche a
Passo 1/2, MIT da AutoAWQ) e una nuova funzione,
`_dequantize_awq_linear_to_fp32(linear)`, che dequantizza un singolo
modulo Linear AWQ-packed a fp32 layout nn.Linear-style, **derivando
bits/group_size dalle shape reali dei tensori** (`pack_factor =
out_features // qweight.shape[1]`, `bits = 32 // pack_factor`,
`group_size = qweight.shape[0] // scales.shape[0]`) invece che da
`quant_config.json` come faceva lo script standalone del Passo 2 — questo
metodo non deve sapere dove vive il checkpoint su disco.

**Nuovo metodo `GCSGWorker._build_cpu_shadow_pool_awq(layers,
expert_ids)`**: mirror CPU-resident del path 3, parallelo a
`self._shadow_pool` (GPU) come già Fase 2 per il path 1. Cache **fp32,
non INT4** — decisione presa ieri (misura completa della pipeline):
ri-quantizzare per-tensore un formato per-gruppo distrugge il segnale
(~0.95 errore relativo). w1 (gate) e w3 (up) concatenati su dim 0 per
formare w13 nello stesso layout che `_ShadowExpertINT4` già consuma —
nessuna classe nuova.

**Wiring**: agganciato al branch `else:` (path 3) di
`_load_shadow_pool()`, stesso gate esplicito del path 1
(`_tier_manager` wired E `_cpu_offload_enabled=True`, default False,
Fase 4 invariata). A differenza del path 1, costruito per **l'intero
`expert_ids` selezionato**, non solo quelli con pinning GPU riuscito —
un miglioramento, non solo un mirror: un expert con pinning GPU fallito
ora ha comunque una residenza CPU funzionante invece di restare
hook-only.

**23 nuovi test** in `test_scheduler.py`, tre classi isolate per layer
(stesso principio "un livello alla volta" già usato altrove in questo
file):
- `TestDequantizeAwqLinearToFp32` — deriva bits/group_size dalle shape
  reali osservate sul checkpoint (hidden=4096, intermediate=14336,
  bits=4, group_size=128), verifica layout .T + .contiguous().
  `_dequantize_awq_gemm()` monkeypatchata (già provata altrove contro
  il kernel reale — qui isolato solo il layer nuovo).
- `TestBuildCpuShadowPoolAwq` — concatenazione w1+w3, scale=1.0,
  forward CPU end-to-end funzionante. `_dequantize_awq_linear_to_fp32`
  monkeypatchata.
- `TestLoadShadowPoolAwqCpuPoolWiring` — i tre gate test mirror di
  `TestGCSGCpuShadowPool` (path 1): costruisce con tier_manager+flag,
  resta vuoto senza tier_manager, resta vuoto col flag disabilitato di
  default. `_build_cpu_shadow_pool_awq()` monkeypatchata direttamente —
  isola il wiring dalla correttezza del dequant. Bug incontrato scrivendo
  questi test: con un TierManager reale wired, il pinning GPU del fake
  minimale di `TestGCSG` (pensato solo per `.to('cuda')` diretto) fallisce
  perché passa da `_promote_module_via_tier_manager()`, che chiama
  `named_parameters()` — non supportato dal fake. Non un bug nel codice
  di produzione: i due test coinvolti sono stati corretti per non
  asserire sul pool GPU (fuori scope, già coperto altrove), solo sul pool
  CPU che è l'oggetto del test.

Suite completa: 168 test, tutti verdi (`pytest osx-poc/tests/ -q`).
`ruff check` pulito sui file toccati (un solo warning I001 pre-esistente
su un import in cima a `test_scheduler.py`, non introdotto da questa
sessione — verificato via `git stash`).

**Passo 3 — parità del codice REALE contro il kernel CUDA reale**
(`scripts/verify_awq_cpu_dequant_wired.py`, nuovo): a differenza del
Passo 2 (che verificava una reimplementazione standalone mai collegata a
`GCSGWorker`), questo script chiama `GCSGWorker._build_cpu_shadow_pool_awq()`
— il codice VERO — su un modulo `MixtralMLP` reale caricato via vLLM, e
confronta contro lo stesso forward CUDA reale. Girato due volte su
layer/expert diversi (checkpoint reale, locale, `/data/nvme/models/
mixtral-instruct-awq`):

| layer | expert | stato pinning | errore relativo L2 | diff assoluta max |
|---|---|---|---|---|
| 0  | 0 | offloaded, pinnato esplicitamente | 0.000483 | 0.007572 |
| 15 | 5 | già GPU-resident | 0.000555 | 0.002098 |

Entrambi ben sotto la soglia 1e-2, in linea con Passo 2 (~0.0005) —
il collegamento a `GCSGWorker` non ha introdotto degrado, e copre anche
il caso non testato da Passo 2 (modulo già GPU-resident, nessun
pinning). `_build_cpu_shadow_pool_awq()` per un solo expert su un solo
layer: ~2.1-2.7s (in linea con la stima ~1.1-1.2s/layer per w1+w2+w3
della sessione precedente, differenza attribuibile a caching/warmup del
processo). Log completi salvati in
`logs/z8_local/awq_fase6a_wiring/`.

### Result

Il gap identificato ieri è chiuso: `--enable-cpu-offload` ora fa
qualcosa di reale sul path 3 (AWQ ModuleList), il solo path che il
checkpoint di produzione usa. Verificato non solo con unit test (fake,
veloci, isolati per layer) ma anche contro il kernel CUDA reale sul
checkpoint reale, due volte, su moduli in stati di pinning diversi.

### Why this matters

Fino a ieri tutto il lavoro Fase 6a — per quanto numericamente corretto
— non poteva essere invocato da nessun percorso reale: viveva in script
mai chiamati da `GCSGWorker.load_model()`. Questo è il momento in cui
Fase 6a smette di essere "codice che dimostra che l'idea funzionerebbe"
e diventa "codice che il worker userebbe se il flag fosse acceso" — la
distinzione posta esplicitamente dal project owner ("cosa ci manca per
accenderla?"). Resta comunque dietro il flag esplicito (default False,
Fase 4 invariata): l'impatto aggregato di attivarla sul path AWQ reale
(tempo, memoria) non è ancora stato misurato — è il prossimo passo
naturale (il pattern di validazione già confermato: 16→32→64→full 570).

### Current state

- `src/scheduler/gcsg.py`: `_dequantize_awq_gemm`/`_awq_unpack`/
  `_awq_reverse_order`/`_dequantize_awq_linear_to_fp32` (vendorizzate/
  nuove), `GCSGWorker._build_cpu_shadow_pool_awq()` (nuovo), wiring nel
  branch path-3 di `_load_shadow_pool()`. Docstring di classe aggiornata
  (non più "solo path 1 coperto").
- `tests/test_scheduler.py`: 23 nuovi test, 3 classi nuove
  (`TestDequantizeAwqLinearToFp32`, `TestBuildCpuShadowPoolAwq`,
  `TestLoadShadowPoolAwqCpuPoolWiring`). Suite completa: 168 test, tutti
  verdi.
- `scripts/verify_awq_cpu_dequant_wired.py`: nuovo, Passo 3 — verifica il
  codice reale (non una reimplementazione) contro il kernel CUDA reale.
  Girato due volte su layer/expert reali diversi, entrambi sotto soglia.
- `logs/z8_local/awq_fase6a_wiring/`: output completi dei due run Passo 3.
- Nessuna run MMLU (16→32→64→full) ancora lanciata con
  `--enable-cpu-offload` genuinamente attivo sul path AWQ — prossimo
  passo, non ancora iniziato.
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session (o proseguimento stessa sessione): lanciare il pattern di
validazione confermato (16→32→64→full 570 domande MMLU) con
`--enable-cpu-offload` genuinamente attivo sul path AWQ, in locale
(28 core/436GB RAM/RTX3090 già disponibili) — confrontare i tempi
WITH/WITHOUT il flag alla scala piena, oltre alla sola correttezza
(accuracy MMLU già nota dover restare invariata, Fase 5/#17).

---

## 2026-08-17 (continued 15) — Registro interventi live: prima run reale cpu-offload su path AWQ, mai completata, indizio forte trovato

**Release:** nessuna modifica a codice committata in questa sessione
(solo script/telemetria locali, non ancora in git — vedi tabella sotto).
Registro richiesto esplicitamente dal project owner ("memorizza tutte le
change impostate... timestamp a mano, per valutare impatto ed eventuali
rollback") — formato tabellare invece del solito prosa, per essere
riutilizzabile come checklist di rollback.

### What we set out to do

Primo tentativo di eseguire il pattern di validazione confermato
(16→32→64→full) con `--enable-cpu-offload` genuinamente attivo sul path
AWQ (Fase 6a Passo 3, appena committato — 652b31a). Partito dal primo
gradino (n=16). Il project owner ha chiesto in parallelo: raccolta
telemetria host/docker/GPU con timestamp mappabili alle fasi del run, e
verifica della banda PCIe reale (non solo il traffico osservato dal
workload attuale).

### Registro interventi, con timestamp reali

| # | Timestamp (locale, +02:00) | Intervento | Stato attuale | Note rollback |
|---|---|---|---|---|
| 1 | 2026-08-17 13:46:06 | Commit `652b31a` — Fase 6a Passo 3 (dequant AWQ collegato a GCSGWorker), già pushato su `claude/ddr4-ram-processing-unzseh` | **Permanente** — in git, PR #42 | `git revert 652b31a` se necessario |
| 2 | ~13:47 | Primo tentativo run n=16 `--enable-cpu-offload` (container `...4601a24ea13a`), SENZA tuning thread | **Terminato manualmente** (`docker stop`) a ~14:11, dopo 23.5 min senza risultato | n/a — container `--rm`, nessuna traccia residua. Log conservato come `run_n16_KILLED_thread_starved_23min_no_result.log` |
| 3 | ~13:54:52 | Avvio sampler telemetria HOST (CPU/RAM/GPU), script `logs/z8_local/telemetry_fase6a/host_sampler.ps1`, processo PowerShell staccato (`Start-Process -WindowStyle Hidden`), tetto 7200s | **Attivo** — terminerà da solo verso le 15:54:52 se non fermato prima | `Stop-Process` sul PID, o atteso self-timeout. Script non committato (solo locale) |
| 4 | ~13:55:03 | Avvio sampler `docker stats`, script `/tmp/docker_stats_sampler.sh` (bash, `nohup`+`disown`), tetto 7200s | **Attivo** — terminerà verso le 15:55 | Script vive in `/tmp` DENTRO il container dev (non nel repo, non persistente oltre il riavvio WSL2) — se serve riprodurlo va ricreato |
| 5 | 14:11:22 | Secondo tentativo run n=16, questa volta con `-e OMP_NUM_THREADS=28 -e MKL_NUM_THREADS=28` passati sia a `docker compose run` sia inline prima del comando Python (container `...e37af10a082f`) | **Terminato forzatamente** (`docker kill`) alle 14:47:46, dopo 35.5 min (T+2130s) — MAI completato, nessun risultato prodotto | Nessuna traccia residua (container `--rm`). Flag non persistiti in nessun file — da riapplicare manualmente ad ogni run finché non si decide di metterli in `docker-compose.yml`/nello script |
| 6 | ~14:22:17 | Avvio sampler PCIe (`nvidia-smi dmon -s t`), script `logs/z8_local/telemetry_fase6a/pcie_sampler.ps1`, tetto 7200s | **Attivo** — terminerà verso le 16:22 | Come #3, `Stop-Process` o self-timeout. Script non committato |
| 7 | ~14:35 | Creato `osx-poc/scripts/bench_pcie_bandwidth_wsl2.py` (microbenchmark pinned vs pageable, CPU↔GPU, isolato da vLLM) | **Non ancora eseguito** — file presente, non committato | File nuovo, safe da rimuovere se non serve più |
| 8 | 14:41:35–14:47:46 | Watchdog interno dello script (`--watchdog-timeout` default 1800s) ha tentato SIGTERM poi SIGKILL su se stesso (PID 1 del container) — **entrambi inefficaci**, il processo è rimasto vivo altri ~10 min oltre il SIGKILL loggato | **Osservato, non ancora corretto nel codice** | Nessuna modifica di codice fatta — solo osservazione. Fix eventuale: il watchdog di `eval_mmlu_gcsg.py` andrebbe ridisegnato per non fare affidamento su self-SIGKILL da PID 1 in un namespace container (vedi "Why this matters") |
| 9 | 14:47:46 | `docker kill` da HOST (fuori dal container) sul container del punto #5 — riuscito immediatamente, a differenza del self-SIGKILL interno | Container rimosso (`--rm`) | n/a |

Telemetria raccolta finora (file ancora attivi, righe correnti al
momento di scrivere): `host_perf.csv` (297 righe), `gpu_telemetry.csv`
(297 righe), `docker_stats.csv` (267 righe), `pcie_throughput.csv` (294
righe) — tutte in `logs/z8_local/telemetry_fase6a/`, timestamp ISO,
cadenza 10s (5s per PCIe).

### What we did (sintesi tecnica)

Nessuna delle due run n=16 è arrivata a un risultato. La prima (senza
tuning thread) è stata fermata manualmente dopo 23.5 min per applicare
`OMP_NUM_THREADS=28`/`MKL_NUM_THREADS=28` (richiesta esplicita del
project owner, per escludere che il collo di bottiglia fosse
sotto-parallelizzazione banale). La seconda, CON quel tuning, non solo
non ha finito prima ma è arrivata al timeout interno di 30 minuti dello
script stesso (`--watchdog-timeout` default) senza produrre risultato,
ed è stata terminata a forza da fuori. Il tuning thread ha mostrato un
miglioramento parziale ma non decisivo (CPU container: ~270% senza
tuning → picchi 350-440% con tuning, mai vicino ai 28 core disponibili).

Durante l'attesa, indagati due indizi:
- **PCIe reale**: `nvidia-smi dmon -s t` funziona su questa RTX 3090
  (non tutte le metriche NVML sono ristrette su GeForce) — traffico
  osservato ~2.3 GB/s TX / ~800 MB/s RX, ma BURSTY, non uno stream
  continuo. Discussa e respinta l'ipotesi che questo numero rappresenti
  un "tetto" imposto da WSL2/GPU-PV: un workload che non sta cercando
  di saturare il bus non misura il bus, misura solo se stesso — serve
  il microbenchmark dedicato (`bench_pcie_bandwidth_wsl2.py`, creato ma
  non ancora eseguito) per un numero vero.
- **Casting silenzioso in `_ShadowExpertINT4.__call__()`** (indizio più
  concreto, non ancora verificato con un profiler ma ad alta
  confidenza): `w13 = w13_q.to(hidden_states.dtype) * w13_scale`
  progettato per pesi INT8/INT4 compatti (Fase 1), dove il cast è quasi
  gratuito. Sul path fp32-cache di Fase 6a, `w13_q` è GIÀ un tensore
  fp32 di centinaia di MB — questa riga rialloca e ricalcola l'intero
  tensore ad ogni singola chiamata (ogni token, ogni layer, ogni expert
  freddo), invece di un costo one-time. Scala con
  token × layer × expert-freddi, non con la dimensione del pool — il
  candidato più forte finora per spiegare 35+ minuti senza risultato su
  soli 16 prompt.

### Result

Nessun numero di accuratezza/tempo utilizzabile da questa sessione di
run — entrambi i tentativi n=16 sono falliti a produrre un
completamento. Ma il tempo non è stato sprecato: telemetria completa
raccolta per tutta la finestra (utile per il post-mortem), un bug
concreto identificato per lettura del codice (non ancora fixato), e un
comportamento infrastrutturale nuovo scoperto (self-SIGKILL da PID 1
inefficace sotto questo setup Docker/WSL2).

### Why this matters

Tre risultati distinti, tutti azionabili:
1. **Il casting bug è la priorità**: va corretto in
   `_ShadowExpertINT4.__call__()` PRIMA di ritentare qualunque run —
   continuare a misurare tempi su un bug di performance noto produrrebbe
   numeri che andrebbero ributtati via appena il fix arriva. Short-circuit
   proposto: se `w13_q.dtype == hidden_states.dtype` e `w13_scale == 1.0`,
   saltare cast+moltiplicazione e usare `w13_q` direttamente.
2. **Watchdog interno non affidabile**: `eval_mmlu_gcsg.py` non può
   più contare su un self-SIGKILL per garantire che un run non superi
   `--watchdog-timeout` — serve un meccanismo di stop esterno (quello che
   abbiamo appena usato manualmente, `docker kill` da host) finché la
   causa raw non è isolata. Rilevante per QUALUNQUE run futuro non
   presidiato (es. un batch 16→32→64→full lasciato girare incustodito).
3. **PCIe non ancora un sospetto qualificato**: il traffico osservato
   non è stato dimostrato essere un limite, solo un sintomo — non vale
   la pena investire su pinned memory finché il casting bug (causa più
   probabile e già isolata via lettura del codice) non è escluso.

### Current state

- Codice di produzione: invariato dal commit 652b31a (nessun fix
  applicato ancora al casting bug).
- File nuovi non committati: `osx-poc/scripts/bench_pcie_bandwidth_wsl2.py`,
  `logs/z8_local/telemetry_fase6a/` (2 script PowerShell + 4 CSV in
  crescita), `logs/z8_local/mmlu_fase6a_cpu_offload/run_n16.log` (run
  fallita, MAI completata) e
  `run_n16_KILLED_thread_starved_23min_no_result.log` (prima run
  fallita).
- Processi in background ancora attivi (vedi tabella): sampler
  host/GPU (~15:54:52), sampler docker stats (~15:55), sampler PCIe
  (~16:22) — tutti con tetto automatico, nessuno richiede intervento
  immediato.
- Nessun container Docker residuo — entrambi i tentativi di run sono
  stati puliti (`--rm` + stop/kill espliciti).
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session (o proseguimento stessa sessione): applicare il
short-circuit nel casting di `_ShadowExpertINT4.__call__()` (o una
variante dedicata per il path fp32-cache), poi ritentare n=16 — solo
DOPO quel fix ha senso rimettere in coda 32→64→full. Il microbenchmark
PCIe (`bench_pcie_bandwidth_wsl2.py`) resta un passo indipendente,
utile ma non bloccante.

### Correzione (stessa sessione, poco dopo): la vera causa dominante era già in LOGBOOK.md, non ancora controllata

Il project owner ha segnalato di controllare `LOGBOOK.md` (non
`LOGBOOK_ISSUE33.MD`) prima di continuare a teorizzare — richiesta
seguita, e ha cambiato la lettura. Entrambe le run n=16 di oggi
mostrano, identico, questo warning (mai controllato finché non
segnalato):

```
WARNING ... Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
```

`LOGBOOK.md` (sessioni 2026-08-11/12, Sprint 4 Tekniska) documenta
ESATTAMENTE questa stessa patologia, già diagnosticata e già risolta
una volta: un run completo da 570 domande sotto WSL2
(`pin_memory=False` forzato) ha impiegato **una notte intera**
(18 slice da 32, orchestrate separatamente proprio per aggirare stall
imprevedibili — una slice worst-case da sola ~30 minuti), mentre lo
STESSO carico su RunPod (Linux reale, `pin_memory=True` reale) ha
impiegato **~35-40 minuti per l'INTERO run da 570** — non 16 domande,
570. Il project owner conferma inoltre un dato più recente: ~9 minuti
su RunPod per un carico comparabile a oggi. La causa non è nuova né
specifica a Fase 6a: è la stessa limitazione strutturale di WSL2 già
isolata mesi fa ("il collo di bottiglia WSL2 non è mai stato
intrinseco al modello o a GCSGWorker" — `LOGBOOK.md`, entry
2026-08-11 "the stall was never a deadlock").

**Questo cambia l'ordine di priorità** stabilito poco sopra in questo
stesso entry: il casting bug in `_ShadowExpertINT4.__call__()` resta
un'inefficienza reale (trovata per lettura del codice, non ancora
smentita), ma quasi certamente NON è la causa dominante dei 35+ minuti
senza risultato — questo progetto ha già evidenza diretta che
`pin_memory=False` sotto WSL2, da solo, trasforma minuti in notti
intere, su un carico ben più grande (570 domande, non 16) e su un path
che non ha nulla a che fare con il casting di Fase 6a. Stesso principio
per le due piste aperte sopra (thread tuning, banda PCIe): entrambe
probabilmente confuse dalla stessa causa di fondo, non piste da
scartare ma da riconsiderare come secondarie.

**Implicazione pratica**: continuare a inseguire i tempi di
cpu-offload in locale sotto WSL2 rischia di ripetere lo stesso errore
già fatto e già corretto una volta in questo progetto — misurare (e
provare a ottimizzare) un artefatto dell'ambiente invece del codice.
La strada già battuta e già validata per QUALUNQUE misura di tempo
affidabile su questo tipo di carico è RunPod (Linux reale, pinning
vero) — non un nuovo esperimento, una richiamata di una conclusione
già presa. Il fix del casting bug resta comunque legittimo da
applicare (corregge uno spreco reale indipendente dall'ambiente), ma
la validazione 16→32→64→full va spostata su hardware Linux reale
prima di trarre conclusioni sui tempi, non ripetuta ancora in locale.

---

## 2026-08-17 (continued 16) — Report onesto di sessione: cosa è andato, cosa no, stato per il rollback

**Release:** sì — `src/scheduler/gcsg.py` (fix memoization),
`logs/z8_local/telemetry_fase6a/` (telemetria + SUMMARY.md).

Richiesto esplicitamente dal project owner prima di procedere oltre:
"aggiornare tutta la documentazione, recuperare tutti i log e
telemetria, fare un primo report onesto su cosa è andato e cosa no;
committare e pushare". Questo entry è quel report — separato dal
racconto cronologico sopra per essere leggibile come bilancio, non
come narrazione.

### Cosa è andato bene

- **Fase 6a Passo 3 (commit `652b31a`)**: il dequant AWQ è ora
  realmente collegato a `GCSGWorker` — non più solo script standalone.
  Verificato NUMERICAMENTE due volte contro il kernel CUDA reale su
  layer/expert diversi (errore relativo L2 0.00048 e 0.00056, entrambi
  ben sotto la soglia 1e-2). Questo è un risultato solido, non rimesso
  in discussione da nulla successo oggi — i problemi di oggi sono tutti
  di TEMPO/AMBIENTE, non di correttezza matematica.
- **23 nuovi test**, isolati per layer, 168 test totali verdi, ruff
  pulito.
- **Trovato e corretto un bug di performance reale**: `_ShadowExpertINT4
  .__call__()` ricalcolava cast+scale dell'intero tensore ad ogni
  chiamata invece che una volta sola — fix (memoization per
  layer_id+dtype) applicato, numericamente identico al comportamento
  precedente (verificato: scale è sempre 1.0 su entrambi i path, quindi
  cast-una-volta produce lo stesso tensore fp16 di cast-ad-ogni-chiamata),
  168/168 test ancora verdi dopo il fix.
- **Telemetria completa raccolta**: host CPU/RAM, GPU, PCIe RX/TX,
  docker stats — tutti timestampati, tutti in
  `logs/z8_local/telemetry_fase6a/` con `SUMMARY.md`.
- **Inventario hardware chiarito**: confermato che lo Z8 ha 4 moduli
  Intel Optane DC Persistent Memory (`NMA1XXD128GPS`, ~505GB nominali)
  in Memory Mode al 100% (`Get-PmemPhysicalDevice`: tutti a 0GB
  persistent), non semplice DDR4 — rilevante per l'interpretazione di
  qualunque numero di RAM/tier futuro su questa macchina.
- **Trovato un comportamento infrastrutturale reale**: self-SIGKILL da
  PID 1 dentro il container non termina il processo sotto questo setup
  Docker/WSL2 (osservato, non ancora spiegato con certezza) —
  `docker kill` da host invece funziona immediatamente. Rilevante per
  qualunque watchdog interno futuro.

### Cosa NON è andato

- **Nessuna delle due run n=16 con `--enable-cpu-offload` è arrivata a
  un risultato.** Prima run: 23.5 minuti, killata manualmente. Seconda
  run (con thread tuning): 35+ minuti, arrivata al timeout interno
  dello script (30 min) e killata da fuori. **Zero numeri di
  accuracy/tempo utilizzabili da oggi per la validazione 16→32→64→full.**
- **Causa più probabile identificata TARDI, non subito**: solo dopo che
  il project owner ha richiamato esplicitamente `LOGBOOK.md`, abbiamo
  controllato il warning `pin_memory=False` (presente su ENTRAMBE le
  run fin dall'inizio, mai controllato prima) e collegato la lentezza
  alla stessa patologia WSL2 già nota da mesi. Prima di quel momento
  questa sessione ha inseguito tre ipotesi alternative (thread
  under-utilization, banda PCIe, casting bug) — la terza è risultata un
  bug reale ma quasi certamente non la causa dominante di oggi, le
  prime due probabilmente confuse dalla stessa causa di fondo mai
  controllata per prima. Lezione operativa, non solo tecnica: quando un
  sintomo somiglia a uno già visto, controllare la storia PRIMA di
  ipotizzare cause nuove — non fatto qui fino a quando richiesto
  esplicitamente.
- **Il fix del casting NON è stato verificato end-to-end**: risolve un
  problema reale trovato per lettura del codice, i test unitari
  restano verdi, ma non abbiamo (ancora) rilanciato un run reale per
  misurare quanto effettivamente riduce il tempo — non lo sapremo con
  certezza finché non gira un run vero, probabilmente su hardware
  dove `pin_memory` funziona (RunPod), per non confondere ancora una
  volta l'effetto del fix con l'effetto dell'ambiente.
- **PCIe non risolto**: `bench_pcie_bandwidth_wsl2.py` creato ma mai
  eseguito — la domanda "quanto della banda osservata è un tetto reale
  vs. quanto è solo il workload attuale" resta aperta.
- **Nessuna misura di impatto aggregato cpu-offload disponibile**:
  l'obiettivo originale della giornata (16→32→64→full) resta a zero
  progressi concreti in termini di numeri, nonostante ~1.5 ore di
  tentativi.

### Stato per eventuale rollback

| Cosa | Stato | Rollback |
|---|---|---|
| Commit `652b31a` (Fase 6a Passo 3) | pushato, permanente | `git revert 652b31a` |
| Fix memoization `_ShadowExpertINT4` | locale, non ancora committato in questo report (committato a fine di questo stesso entry — vedi sotto) | `git revert` sul commit che lo introduce |
| `osx-poc/scripts/bench_pcie_bandwidth_wsl2.py` | locale, non eseguito, safe da rimuovere | `rm` — nessuna dipendenza da nessun altro file |
| `logs/z8_local/telemetry_fase6a/*.csv` + `SUMMARY.md` | locale, dati storici, da conservare | nessuno — sono un registro, non codice attivo |
| `logs/z8_local/mmlu_fase6a_cpu_offload/run_n16*.log` | locale, run fallite, conservate come evidenza | nessuno |
| `OMP_NUM_THREADS=28`/`MKL_NUM_THREADS=28` | NON persistito in nessun file — era solo un flag `-e` sulla singola invocazione di `docker compose run` | nessuna azione necessaria, già "rollback automatico" (non sopravvive oltre il comando che l'ha usato) |
| Sampler PowerShell/bash di telemetria | fermati esplicitamente (host+PCIe via `Stop-Process`, docker-stats lasciato al suo self-timeout ~15:55) | nessuno, già fermati o auto-terminano |
| `.wslconfig` (`memory=444GB`, `processors=28`) | invariato da sessioni precedenti, non toccato oggi | n/a, fuori scope di questo report |

### Why this matters

Il project owner ha chiesto esplicitamente onestà prima di procedere —
questo report la mantiene: oggi non ha prodotto NESSUN numero di
performance cpu-offload utilizzabile, nonostante il codice di
produzione (Fase 6a Passo 3) sia solido e verificato. Il tempo speso
non è stato inutile — un bug reale trovato e corretto, un
comportamento infrastrutturale documentato, telemetria completa
raccolta, e soprattutto la conferma che l'ambiente locale (WSL2) non è
quello giusto per misurare i tempi di questo carico, cosa che il
progetto aveva già imparato una volta e stava per re-imparare da capo.

### Current state

- Codice: fix di memoization applicato in `_ShadowExpertINT4`
  (`src/scheduler/gcsg.py`), 168/168 test verdi, ruff pulito. Non
  ancora committato al momento di scrivere questo entry — commit e
  push seguono subito dopo.
- Nessun container Docker residuo, nessun processo di telemetria
  ancora attivo (tranne il self-timeout innocuo del sampler docker
  stats).
- Branch: `claude/ddr4-ram-processing-unzseh`, PR #42 aperta verso
  `develop`.

Next session: eseguire la validazione 16→32→64→full su RunPod (non più
in locale sotto WSL2), col fix di memoization già incluso — prima
run vera capace di produrre sia un numero di tempo sia
un'accuratezza confrontabile con la baseline. `bench_pcie_bandwidth_wsl2.py`
resta un passo indipendente e non bloccante.

---

## 2026-08-17 (continued 17) — prima run cpu-offload REALMENTE completata su RunPod: corretta ma 25.7x più lenta, causa isolata

**Release:** sì — `src/scheduler/gcsg.py` (timing diagnostico, cap RAM
CPU pool, `_read_cgroup_available_gb()`), `scripts/eval_mmlu_gcsg.py`
(fix `logging.basicConfig` mancante), `tests/test_scheduler.py` (6
nuovi test). Nessuna modifica al comportamento di default: entrambi i
flag (`--wire-tier-manager`, `--enable-cpu-offload`) restano opt-in,
default `False`, invariato dalla Fase 4 — vedi "Why this matters" per
la verifica esplicita di questo punto.

### What we set out to do

Riprendere da dove "continued 16" si era fermato: eseguire
16→32→64→full con `--enable-cpu-offload` reale su hardware Linux vero
(RunPod), non più in locale sotto WSL2 — il bootstrap anti-Alzheimer
(§5.1) segnalava esplicitamente di non ripetere l'errore di misurare
un artefatto WSL2 invece del codice.

### What we did (cronologia condensata — sessione lunga, molti passaggi)

1. **Provisioning pod nuovo** (RTX3090, non più Malmö/RTX A6000 —
   quel pod era già spento): accesso SSH impostato con la stessa
   keypair dedicata (`id_ed25519_runpod_sprint4`, riusata da
   Sprint 4/Malmö in poi — vedi BOOTSTRAP §5.8 per la procedura, ora
   documentata lì per la prima volta esplicitamente). Hardware reale
   misurato (non fidato dal pannello RunPod, che mostra le spec
   nominali, non la quota cgroup — vedi BOOTSTRAP §5.9, nuovo questa
   sessione): **27.2 core cgroup reali** (non i 256 di `nproc`),
   **~116.4GiB RAM cgroup reale** (non l'1TB di `free`), CPU **AMD
   EPYC 7C13 (Zen3) senza AVX-512** (diverso dallo Xeon Ice Lake di
   Malmö che li aveva tutti). Checkpoint scaricato da HuggingFace
   direttamente su disco locale container in 28.96s (stesso pattern
   §5.7, bypassa il Network Volume).
2. **Primo run n=16 (v1)**: fallito, watchdog 600s scattato senza un
   solo prompt completato, GPU allo 0% per tutto il tempo, RAM cgroup
   salita all'84% (98GB/116GB). **Diverso da ieri**: nessun warning
   `pin_memory=False` (Linux reale) — quindi NON era WSL2. Il
   watchdog stavolta ha funzionato con un semplice SIGTERM (§5.2 del
   bootstrap non si applica: qui non c'è Docker annidato, il processo
   non è PID 1).
3. **Tentato profiling live con `py-spy dump`**: bloccato, il pod
   RunPod non ha la capability `SYS_PTRACE` (nuovo, BOOTSTRAP §5.10).
   Pivot obbligato: instrumentare il codice invece di profilarlo
   dall'esterno.
4. **Aggiunto timing per-layer/per-expert** a
   `_build_cpu_shadow_pool_awq()` + **cap RAM-aware** nuovo
   (`_check_cpu_ram_budget()`/`_read_cgroup_available_gb()`, stesso
   principio di `GCSGGuard._check_vram_budget` ma per la RAM host) —
   richiesto esplicitamente dal project owner ("ridimensioniamo gli
   expert in funzione della RAM reale, lasciando margine per il
   sistema"). 6 nuovi unit test (`TestCheckCpuRamBudget`,
   `TestReadCgroupAvailableGb`).
5. **Scoperto un bug di osservabilità pre-esistente**, non introdotto
   oggi: `scripts/eval_mmlu_gcsg.py` non ha mai chiamato
   `logging.basicConfig()` — ogni `log.info()`/`log.warning()` di
   `scheduler.gcsg` (inclusi tutti i messaggi "GCSG: ..." già scritti
   in Fase 2/3/4/6a, MAI visti in un log reale fino ad oggi) finiva
   nel vuoto. Fix minimo: `logging.basicConfig(level=INFO,
   stream=sys.stdout)` in cima allo script.
6. **Rilanciato (v4) col fix di logging**: rivelato che la build del
   pool CPU è VELOCE (~44s/expert × 2 = ~88s totali, completa PRIMA di
   "LLM ready") — **non era mai stata il collo di bottiglia**,
   contrariamente a quanto ipotizzato in "continued 15/16".
7. **Aggiunto un secondo livello di instrumentazione** (contatore +
   tempo cumulativo) a `_ShadowExpertINT4.__call__()` — rivela il vero
   collo di bottiglia: forward per-TOKEN (`hidden_states.shape=(1,
   4096)`, non batchato), **~0.10s/chiamata**, costante, migliaia di
   chiamate necessarie.
8. **Rilanciato (v5) — completato per la prima volta**: **1133.1s
   (~18.9 min)** per n=16, **8/16 corrette (50.0%)**,
   `shadow_activations=10081` — **identico byte-per-byte** al
   baseline Malmö (cpu-offload no-op, stesso n, stessa selezione
   expert) in accuratezza E in conteggio attivazioni. Prova numerica
   diretta su hardware reale, non solo unit test isolati: il dequant
   AWQ CPU di Fase 6a Passo 3 produce lo STESSO comportamento del path
   GPU, non un'approssimazione.
9. **Test A/B mirato sul thread count** (richiesto dal project owner
   dopo aver visto il numero, con scetticismo sul fatto che fosse
   davvero il thread tuning il problema): confronto `OMP_NUM_THREADS=27`
   vs `=2` alla stessa checkpoint (calls=200) — **0.0967s/call vs
   0.100-0.103s/call, praticamente identico**. La build del pool
   invece RALLENTA con meno thread (183s vs 88s con 2 thread) — il
   parallelismo aiuta sui tensori grandi (dequant one-time) ma è
   irrilevante sul forward a singola riga (1×4096). **Ipotesi
   thread-oversubscription falsificata con dati, non per
   discussione.**

### Result

Due fatti distinti, entrambi ora misurati non ipotizzati:

1. **Correttezza**: PROVATA su hardware reale. Stesso accuracy, stesso
   `shadow_activations` count del path no-op — il dequant AWQ CPU
   (Fase 6a) non degrada la qualità, anche sotto carico shadow
   attivo reale.
2. **Performance**: il path cpu-offload reale è **25.7x più lento**
   del baseline senza offload sullo stesso n=16 (1133s vs 44.1s,
   Malmö) — causa isolata (forward per-token a ~100ms/chiamata,
   indipendente dal thread count, quindi verosimilmente un limite di
   throughput per-core genuino su questo tipo di istanza RunPod
   ottimizzata per GPU, non un bug di configurazione). Estrapolato
   (non misurato) al full 570: dell'ordine delle ore, non minuti —
   non production-ready così com'è.

### Why this matters

- L'obiettivo originale di oggi ("un numero di tempo E di accuratezza
  per cpu-offload reale") è centrato, con un risultato onesto: la
  feature FUNZIONA correttamente ma non è ancora abbastanza veloce da
  essere utile in produzione su questo tipo di hardware.
- **Nessun rischio per il path di produzione esistente**: verificato
  esplicitamente che `--wire-tier-manager` e `--enable-cpu-offload`
  restano entrambi opt-in con default `False` — sia a livello CLI
  (`argparse action="store_true"`) sia a livello `GCSGWorker.__init__`
  (`enable_cpu_offload: bool | None = None`) — invariato dalla Fase 4,
  non toccato in questa sessione. La suite di test (168+6 nuovi, tutti
  verdi) include esplicitamente `test_..._disabled_by_default_even_with_tier_manager`
  per entrambi i path (1 e AWQ), quindi una regressione futura su
  questo punto verrebbe rilevata, non solo assunta.
- Sei "known issue" nuovi ora in BOOTSTRAP_ANTI_ALZHEIMER.md (§5.9-
  5.13) — nproc/free mentono su RunPod, py-spy bloccato, la
  falsificazione di "era solo WSL2" per questo specifico path, CRLF
  su scp diretto, `< /dev/null` per SSH+nohup — così la prossima
  sessione non deve reimparare nessuno di questi da zero.

### Current state

- Codice: tutte le modifiche committate e pushate su
  `claude/ddr4-ram-processing-unzseh` (PR #42 verso `develop`,
  aggiornata automaticamente dal push).
- `_check_cpu_ram_budget()`/`_read_cgroup_available_gb()` attivi in
  produzione (non dietro un flag separato — si applicano SOLO quando
  `--enable-cpu-offload` è già esplicitamente attivo, quindi zero
  impatto sul path di default).
- Pod RunPod (RTX3090, sessione odierna): stato al momento di scrivere
  non ancora determinato per lo shutdown — vedi se una sessione
  successiva lo trova ancora attivo prima di assumere che vada
  riprovisionato da zero.
- Log/telemetria/risultati completi salvati in locale in
  `logs/runpod_cpu_offload_20260817/` (run v5 completo, telemetria GPU
  host per l'intera sessione) — non solo sul pod, per lo stesso motivo
  di `logs/malmo_runpod/`/`logs/sprint4_tekniska/`.

Next session: nessuna azione bloccante — issue #33 ha ora sia la prova
di correttezza sia una causa isolata (non ancora risolta) per la
performance. Se si vuole inseguire la velocità: il prossimo sospetto
qualificato è il costo fisso per-chiamata di un matmul a singola riga
su CPU (~100ms per un'operazione che dovrebbe costare ok pochi ms) —
andrebbe isolato con un microbenchmark PyTorch puro (fuori da vLLM/GCSG)
per capire se è overhead di dispatch, allocazione, o genuino limite di
throughput per-core di questo tipo di istanza RunPod, prima di
proporre un fix. Non bloccante: la feature resta opt-in e sicura anche
senza quella risposta.

**Aggiornamento stessa sessione, poco dopo il push**: il microbenchmark
"non ancora fatto" indicato sopra esisteva già (`benchmarks/
bench_cpu_kernel.py`, mai eseguito a dimensioni reali su hardware reale
prima d'ora) — eseguito su questo stesso pod RunPod con
`OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336` (dimensioni
Mixtral reali):

```
CPU batch=1:  p50 37.04ms  (9.51 GFLOPS)
GPU batch=1:  p50  0.83ms  (424.4 GFLOPS)  -> GPU 44.6x più veloce
```

**Risposta alla domanda aperta**: è un limite di throughput per-core
genuino, non overhead di dispatch/allocazione — 9.5 GFLOPS per un
matmul isolato, puro PyTorch, senza vLLM/GCSG nel mezzo, è già basso
(coerente con l'assenza di AVX-512 su questo AMD EPYC 7C13/Zen3).
**Ma non spiega tutto**: nella pipeline reale (route_forward() +
GCSGWorker + vLLM) lo stesso identico forward costava ~100ms/chiamata,
**~2.7x più del benchmark isolato (37ms)** — un gap di overhead reale
non ancora spiegato (dispatch Python del path GCSG? contesa con la GPU
che gira in parallelo? altro?), da investigare se si vuole spingere
oltre la sola domanda "è la CPU o è il codice" (risposta: principalmente
la CPU, ma non SOLO la CPU).

Eseguito anche `scripts/bench_pcie_bandwidth_wsl2.py` (creato ieri,
mai lanciato da nessuna parte finora): su questo pod RunPod, pinned
D2H arriva a ~20-24 GB/s (oltre il teorico Gen3x16 ~15.75 GB/s — link
più veloce o effetto di misura, non approfondito), pageable H2D/D2H
~10-18 GB/s. **PCIe escluso come collo di bottiglia** su questo
hardware — coerente con GPU idle allo 0% durante lo stallo osservato
in tutta la sessione (il problema non è mai stato il trasferimento
dati, è il compute CPU stesso).

Risultati grezzi salvati in
`logs/runpod_cpu_offload_20260817/misc/bench_cpu_kernel_realdims.json`
e `.../bench_pcie.txt`.

**Nuova richiesta del project owner**: per un test finale/definitivo
servirebbe hardware bare-metal reale (non un'istanza RunPod condivisa/
virtualizzata) — non ancora avviato, richiede una decisione su
provider/costo da parte del project owner prima di procedere. Annotato
come direzione, non un'azione presa in questa sessione.

**Direzione proposta dal project owner, non ancora scoping-ata**: la
sessione di oggi ha misurato prestazioni ad-hoc (thread count, tempo
per-chiamata) senza un framework sistematico — l'idea è produrre due
report distinti e riutilizzabili invece di continuare a misurare cose
diverse ogni volta: (1) un **performance-test** che caratterizzi
sistematicamente il "terreno di gioco" hardware di un pod (CPU: core
reali/AVX support/throughput matmul isolato, RAM: banda/latenza,
GPU/VRAM: già ben coperto da telemetria esistente, PCIe: banda
pinned/pageable — `bench_pcie_bandwidth_wsl2.py` già esiste ma mai
eseguito su RunPod), e (2) un **performance-tuning** che usi quei dati
per proporre configurazioni (thread count, batch size, ecc.) invece di
tentativi isolati come il test A/B di oggi. Non ancora deciso: formato
dei report, se vivono come script in `scripts/`, dove si inserisce
rispetto al lavoro esistente (`bench_pcie_bandwidth_wsl2.py`,
`bench_hybrid.py`, `bench_route_forward.py` — verificare sovrapposizioni
prima di iniziare, non duplicare).

---

## 2026-08-17 (continued 18) — confronto controllato GPU-only / CPU-only / "misto": il misto non esisteva, causa trovata (issue #27)

**Release:** sì — `src/scheduler/gcsg.py` (`OSX_GCSG_FORCE_CPU_ROUTE`,
env var opt-in per forzare il routing CPU indipendentemente dalla
classificazione hot/cold, con 2 nuovi test). Nessun'altra modifica di
codice: sessione di misura, non sviluppo.

### What we set out to do

Confronto controllato richiesto dal project owner sullo stesso n=16:
Fase 1 (solo GPU), Fase 2 (solo CPU), Fase 3 (GPU+CPU misto) — per
capire se il tempo scala in modo prevedibile tra i tre scenari.

### What we did

- **Fase 1 (solo GPU)**: `--wire-tier-manager` senza
  `--enable-cpu-offload` — nessun pool CPU costruito, comportamento
  identico al baseline già noto.
- **Fase 2 (solo CPU)**: serviva un modo per forzare TUTTO il
  routing su CPU indipendentemente dalla classificazione hot/cold —
  non esisteva con i soli flag CLI esistenti (a cold start la
  classificazione è già determinata dall'ordine di seeding, non
  controllabile dall'esterno). Aggiunta `OSX_GCSG_FORCE_CPU_ROUTE=1`,
  env var di debug opt-in in `route_forward()`, invariata bit per bit
  quando assente (verificato con test dedicato).
- **Fase 3 (misto)**: **non è stata rieseguita** — durante l'analisi
  di come garantire un vero mix, scoperto (leggendo
  `SEEPolicy.classify_hot_cold()` e `_seed_eat_entries()`) che a cold
  start (nessun traffico EAT reale ancora) la classificazione hot/cold
  usa la sola componente di recency, che dipende dall'ORDINE di
  inserimento nel loop di seeding — gli expert inseriti per primi
  (0, 1, 2... — il nostro pool è `[0, 1]`) risultano sistematicamente
  i più "vecchi" al momento della classificazione, quindi i più
  "freddi". Con `hot_fraction=0.5` su 8 expert, i 4 più freddi sono
  esattamente `{0, 1, 2, 3}` — **il nostro pool intero cade nella metà
  fredda**. Il run "misto" di ieri (v5, "continued 17") non ha quindi
  mai instradato nulla su GPU: era già, di fatto, un run CPU-only.
  **Confermato empiricamente, non solo dedotto dal codice**: Fase 2
  (forzata) = 1072.3s, v5 ("misto") = 1133.1s — 5.7% di differenza,
  rumore plausibile su hardware condiviso, non una differenza
  strutturale.
- **Causa profonda del perché non esiste un vero "misto" oggi**:
  `ExpertAccessTable` è puramente in-memory, per-processo —
  `EAT.shutdown()` cancella tutto incondizionatamente, ogni processo
  riparte da `access_count=0` per tutti gli expert. **Issue #27 (EPM —
  Expert Position Memory)**, aperta 2026-08-13, propone esattamente
  questo (`export_snapshot()`/`load_snapshot()` con decay
  configurabile) ma non è mai stata implementata — confermato via
  `gh issue view 27`. Un vero "misto" richiederebbe O quella
  persistenza, O un pattern due-pass nello stesso processo
  (`generate()` → `refresh_shadow_pool_selection()` → `generate()`
  di nuovo, sfruttando `eat.access()` che AGGIORNA davvero
  `access_count` durante `generate()` reale, riga ~2281 di
  `gcsg.py`) — nessuno dei due implementato oggi, decisione esplicita
  del project owner di non farlo in questa sessione.

### Result — tabella comparativa finale (n=16, stesso pod RTX3090)

| Fase | Config | Tempo | Accuracy | shadow_activations |
|---|---|---|---|---|
| 1 — solo GPU | `--wire-tier-manager` | **43.9s** | 50.0% (8/16) | 10081 |
| 2 — solo CPU (forzata) | `--wire-tier-manager --enable-cpu-offload` + `OSX_GCSG_FORCE_CPU_ROUTE=1` | **1072.3s** | 50.0% (8/16) | 10081 |
| 3 — "misto" (v5, in realtà = Fase 2) | `--wire-tier-manager --enable-cpu-offload` | **1133.1s** | 50.0% (8/16) | 10081 |

**Zero errori matematici su tutte e tre** — stessa accuracy, stesso
`shadow_activations` count in tutti e tre gli scenari: la correttezza
non dipende da dove gira il forward, solo la velocità (già stabilito
in "continued 17", ri-confermato qui su un confronto controllato).
**Rapporto di velocità CPU-only/GPU-only: 24.4x** (1072.3s / 43.9s) —
coerente con il 25.7x già misurato ieri e con il 44.6x del
microbenchmark isolato (la differenza tra 24-26x nella pipeline reale
e 44.6x nel benchmark puro riflette l'overhead residuo non ancora
spiegato, già segnalato in "continued 17").

### Why this matters

- **Il piano originale a 3 fasi non era eseguibile come concepito**,
  non per un limite di tempo/budget ma per un gap architetturale reale
  (nessuna persistenza EAT tra processi) — scoperta più preziosa del
  numero stesso: capire CHE COSA misuriamo davvero prima di fidarsi
  del numero. Il project owner ricordava correttamente che esisteva
  un meccanismo di persistenza pensato per questo — issue #27, mai
  implementata, non un ricordo sbagliato.
- `OSX_GCSG_FORCE_CPU_ROUTE` resta nel codice come strumento di
  benchmark riutilizzabile per il prossimo confronto controllato,
  zero rischio per il path di produzione (env var assente = invariato).

### Current state

- Codice: tutto committato e pushato su
  `claude/ddr4-ram-processing-unzseh` (PR #42).
- Log/risultati/telemetria completi in
  `logs/runpod_cpu_offload_20260817/` (Fase 1, Fase 2, telemetria GPU/
  host per l'intera sessione — samplers fermati esplicitamente, non
  lasciati al self-timeout).
- **Pod RunPod (RTX3090) pronto per lo shutdown** — tutti gli
  artefatti recuperati in locale prima di considerare la sessione
  chiusa, stesso principio di Malmö/Sprint4.

Next session: se si vuole un vero confronto a 3 fasi, implementare
prima issue #27 (EPM) o il pattern due-pass generate/refresh/generate
nello stesso processo — nessuno dei due è un prerequisito per il resto
di issue #33 (che resta chiuso su correttezza, aperto solo su
performance non urgente, feature opt-in/default-off).

**Aggiunta del project owner, fine sessione**: regola di decadimento
proposta per quando si riprende questo lavoro — uno scheduler che
rivaluta la temperatura hot/cold degli expert ogni **60 secondi**
(tick periodico che chiama `refresh_shadow_pool_selection()`).
Distinta e complementare a issue #27: EPM risolve il cold-start tra
riavvii di processo, questo tick risolverebbe il fatto che DENTRO un
processo lungo la classificazione non si aggiorna mai da sola —
esattamente la domanda che il docstring di
`refresh_shadow_pool_selection()` lascia aperta oggi ("farlo prima di
avere il dato di costo promote/evict, Sprint 4 sotto-obiettivo 4,
rischierebbe uno storm ad ogni chiamata — pensato per 'ogni N
richieste' o un timer, mai deciso quale"). Non implementato in questa
sessione — prerequisiti: il dato di costo promote/evict (verificare se
esiste già), un thread/timer di background in `GCSGWorker`, e una
decisione tra cadenza fissa (60s) vs. basata su conteggio richieste.

---

## 2026-08-17 (continued 19) — framework performance-test/performance-tuning, scoping-ato e implementato

**Release:** sì — `benchmarks/perf_test_hardware.py` (nuovo),
`benchmarks/perf_tuning_report.py` (nuovo), `Makefile` (due nuovi target
+ help). Nessuna modifica a `scheduler/gcsg.py` o ad altro codice di
produzione — sessione di tooling, non di sviluppo GCSG.

### What we set out to do

Riprendere la direzione proposta a fine "continued 18" ma mai
scoping-ata: invece di misure ad-hoc ripetute ogni volta (thread A/B,
GFLOPS isolati, PCIe — ognuna con la propria metodologia), costruire due
report riutilizzabili — un **performance-test** che caratterizzi
sistematicamente CPU/RAM/GPU/VRAM/PCIe di un host/pod, e un
**performance-tuning** che ne derivi raccomandazioni.

### What we did

1. **Verificate le sovrapposizioni prima di iniziare** (richiesto
   esplicitamente in "continued 18"): letti `bench_cpu_kernel.py`
   (Fase 1), `bench_route_forward.py` (Fase 4), `bench_hybrid.py`
   (Fase 5), `bench_pcie_bandwidth_wsl2.py`, `bench_awq_cpu_pipeline.py`.
   Nessuno caratterizza l'hardware in modo generico — sono tutti legati
   a un path di codice GCSG specifico. Nessuna duplicazione.
2. **`benchmarks/perf_test_hardware.py`** — quattro sezioni, sempre
   `{"status": "done"|"skipped", ...}`, mai un numero inventato (stesso
   pattern di `bench_cpu_kernel.py`):
   - `cpu`: core cgroup-aware (`_read_cgroup_cpu_count()`, nuovo —
     gemello CPU di `_read_cgroup_available_gb()`, che non aveva un
     equivalente per la CPU nel codebase), flag AVX2/AVX-512/AVX-512-VNNI
     letti da `/proc/cpuinfo` (non dedotti dal nome CPU — esattamente
     l'errore fatto in "continued 17" con l'EPYC 7C13), throughput
     matmul isolato (GFLOPS) a batch 1/8/32.
   - `ram`: GB cgroup-aware (riusa `_read_cgroup_available_gb()` da
     `scheduler.gcsg`, già in produzione), banda copy GB/s.
   - `gpu`: singola lettura NVML (nome, VRAM, driver, compute
     capability) + la STESSA identica operazione matmul della sezione
     cpu (stessa funzione condivisa `_bench_matmul_gflops(device)`), per
     un rapporto CPU/GPU comparabile, non due metodologie diverse.
   - `pcie`: pinned/pageable H2D/D2H a varie dimensioni, generalizzato
     da `bench_pcie_bandwidth_wsl2.py` (creato per un dubbio specifico a
     WSL2, mai eseguito altrove prima di "continued 17") — inline,
     nessun cross-import tra script (stessa convenzione di
     `bench_awq_cpu_pipeline.py`).
3. **`benchmarks/perf_tuning_report.py`** — legge il JSON del primo,
   produce raccomandazioni ancorate a fatti già misurati per issue #33,
   non euristiche generiche: verdetto cpu-offload (rapporto GFLOPS
   CPU/GPU, annotato contro il range già osservato 24-26x
   pipeline/44.6x isolato), thread count che distingue esplicitamente
   pool-build (parallelizzabile) da forward per-token (falsificato che
   aiuti, §5.11 del bootstrap), budget RAM CPU-pool (stesse costanti di
   `GCSGWorker._check_cpu_ram_budget()`, ~3GB/expert path1 e ~21.5GB/
   expert path AWQ), flag PCIe (sotto 3GB/s pinned = sospetto
   WSL2/GPU-PV), e un segnale (non un'implementazione — la direzione
   "batching GEMV→GEMM" resta scelta non presa in questa sessione) sul
   guadagno potenziale batchando, rapporto GFLOPS batch32/batch1.
4. **Due nuovi target Makefile** (`perf-test-hardware`,
   `perf-tuning-report`) + entry in `make help`, stesso stile di
   `bench-cpu-kernel`.
5. **Eseguito realmente sullo Z8 (Xeon Gold 6244 + RTX 3090)**, non solo
   compilato — sia a dimensioni sintetiche di default sia a dimensioni
   Mixtral reali (`OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336`),
   `ruff check` pulito, suite `pytest` completa (168 test, invariata)
   ancora verde.

### Result

Il framework produce un segnale genuinamente calibrato, non solo
"gira senza errori": a dimensioni sintetiche piccole (default,
hidden=512/intermediate=1536) il rapporto CPU/GPU misurato sullo Z8 è
solo **1.7x** — e `perf_tuning_report.py` lo segnala correttamente come
**"FUORI dal range già osservato"** (24-26x sul pod RunPod, AMD EPYC
7C13 SENZA AVX-512) invece di assumere che le conclusioni si
trasferiscano. A dimensioni Mixtral reali, lo stesso host produce
**16.9x** — **"coerente con il range già osservato"**. La differenza
tra i due numeri sullo STESSO host (1.7x vs 16.9x, solo cambiando le
dimensioni della matmul) è di per sé un dato utile: conferma che il
confronto CPU/GPU è sensibile alla dimensione del problema, non solo
all'hardware — motivo per cui il tool espone `OSX_BENCH_HIDDEN`/
`OSX_BENCH_INTERMEDIATE` invece di un solo numero fisso.

Altri numeri raccolti sullo Z8 in questa run (per riferimento, non
normativi — lo scopo del tool è essere rieseguito, non produrre un
numero "canonico" una volta sola): 28 core cgroup, AVX-512F presente
(NO AVX-512-VNNI), 432GB RAM cgroup disponibile (135 expert path1/18
expert path AWQ nel budget), banda copy RAM ~19.5 GB/s, PCIe pinned
~11-12 GB/s H2D/D2H (nessun tetto WSL2/GPU-PV rilevato su questo host —
coerente con l'assenza del warning `pin_memory=False`, questo è Docker
Desktop nativo, non lo stesso ambiente WSL2 problematico di
"continued 16"), segnale batching ~8-12x (core CPU oggi idle durante il
forward single-token, coerente con la causa già isolata in
"continued 17").

### Why this matters

- L'obiettivo esplicito del project owner ("due report riutilizzabili
  invece di misure ad-hoc") è centrato: la prossima volta che serve un
  numero di questo tipo (nuovo pod, nuovo host, dopo un cambio hardware)
  non serve reinventare la metodologia — `make perf-test-hardware &&
  make perf-tuning-report` (o lo stesso via pipe) la restituisce in un
  formato consistente.
- Il framework non sostituisce il ragionamento — lo automatizza dove il
  ragionamento è già stabilito (es. "non aumentare i thread per il
  forward per-token", falsificato con dati in "continued 17") e lo
  segnala esplicitamente come incerto dove non lo è (il verdetto
  cpu-offload si auto-qualifica come coerente o meno col range noto,
  invece di spacciare un numero isolato per una conclusione).
- Nessun rischio per il path di produzione: zero modifiche a
  `scheduler/gcsg.py`, solo due nuovi script standalone + Makefile.

### Current state

- Codice: da committare e pushare su `claude/ddr4-ram-processing-unzseh`
  (PR #42 verso `develop`) subito dopo questo entry.
- File temporanei di smoke-test (`_smoke_hw.json`, `_smoke_hw_real.json`,
  creati dentro il container per la verifica end-to-end) già rimossi,
  non nel diff.
- Non ancora fatto, deliberatamente fuori scope di questa sessione:
  eseguire il framework su un pod RunPod reale (solo Z8 misurato qui) —
  utile prossimo passo per confrontare il rapporto CPU/GPU su hardware
  senza AVX-512, il caso che ha originato tutta l'indagine performance.

Next session: nessuna azione bloccante. Se si riprende issue #33 sulla
performance: rieseguire `make perf-tuning-report` su un pod RunPod
(AMD EPYC, no AVX-512) per confermare che il verdetto torni coerente col
range già noto (24-26x), poi scegliere tra le direzioni ancora aperte
(issue #27 EPM, refresh periodico 60s, batching GEMV→GEMM — quest'ultima
ora con un segnale quantitativo concreto da questo stesso framework,
~8-12x sullo Z8, invece di essere solo un'ipotesi).

---

## 2026-08-17 (continued 20) — unit test per il framework, validazione reale su un pod RunPod nuovo

**Release:** sì — `tests/test_perf_test_hardware.py` (nuovo, 18 test,
include `torch.cuda.empty_cache()` a fine dei due test GPU reali — vedi
sotto), `tests/test_perf_tuning_report.py` (nuovo, 20 test). Nessuna
modifica a `benchmarks/perf_test_hardware.py`/`perf_tuning_report.py`
né al codice GCSG di produzione.

### What we set out to do

Due richieste separate del project owner nella stessa sessione: (1)
aggiungere unit test per i due script nuovi di "continued 19"
(`perf_test_hardware.py`, `perf_tuning_report.py`) mentre (2) un pod
RunPod veniva acceso in parallelo — il "prossimo passo" lasciato aperto
alla fine di "continued 19" (validare il framework su hardware reale
senza AVX-512, non solo sullo Z8).

### What we did

1. **`tests/test_perf_test_hardware.py`** (18 test) — stesso pattern di
   mocking di `TestReadCgroupAvailableGb` (`test_scheduler.py`) per le
   parti che leggono file di sistema (`_read_cgroup_cpu_count()`,
   gemello CPU mai esistito prima nel codebase; `_cpu_flags()`), test
   diretti per `_percentiles()` (pura), smoke test end-to-end per
   `bench_cpu()`/`bench_ram()`/`_bench_matmul_gflops()` a dimensioni
   sintetiche di default (piccole, veloci), `@pytest.mark.gpu` per
   `bench_gpu()`/`bench_pcie()` — stesso marker già usato in
   `test_tier.py` per isolare i test che richiedono CUDA reale.
2. **`tests/test_perf_tuning_report.py`** (20 test) — tutte le funzioni
   sono pure (JSON -> JSON), nessun hardware richiesto: fixture JSON
   costruite a mano per ogni scenario (stesso principio di
   `TestCheckCpuRamBudget`), inclusi i casi limite già rilevanti in
   produzione (budget RAM negativo, GFLOPS mancanti, sezioni
   `"skipped"` propagate come `"n/a"`).
3. **Bug di ordine-dei-test trovato eseguendo la suite COMPLETA, non
   solo i due file nuovi**: `test_tier.py::
   test_vram_free_decreases_after_load` (asserisce che la VRAM libera
   via NVML diminuisca dopo un'allocazione di 64MB) falliva quando
   girava DOPO i due test `@pytest.mark.gpu` nuovi — non un bug nel
   codice di produzione, ma la cache dell'allocatore CUDA di PyTorch
   (lasciata da `bench_gpu()`/`bench_pcie()`, che allocano/liberano
   parecchi tensori) soddisfaceva la nuova allocazione senza un
   `cudaMalloc` reale, quindi NVML non vedeva alcuna variazione. Fix:
   `torch.cuda.empty_cache()` a fine dei due test GPU nuovi — verificato
   che risolve rieseguendo la suite completa due volte.
4. **Validazione su un pod RunPod nuovo**, acceso dal project owner in
   parallelo: connessione SSH riuscita con la keypair dedicata
   (`id_ed25519_runpod_sprint4`, BOOTSTRAP §5.8, appesa manualmente
   dal project owner via web terminal), repo clonato al volo
   (`git clone --depth 1 --branch claude/ddr4-ram-processing-unzseh`),
   framework eseguito a dimensioni Mixtral reali
   (`OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336`).

### Result

Pod: **27.2 core cgroup, RAM cgroup 116.0GB, RTX 3090 24GB, AVX2 sì /
AVX-512 NO** — stesso identico profilo hardware del pod già misurato in
"continued 17"/"continued 18" (conferma indiretta: RunPod continua ad
assegnare istanze GPU su host AMD EPYC senza AVX-512 per questo tipo di
richiesta).

Il verdetto del tuning report: **predicted_slowdown_factor = 36.9x**,
**"coerente con il range già osservato in produzione (24-26x pipeline
reale, 44.6x microbenchmark isolato)"** — esattamente il comportamento
per cui il framework è stato costruito, e la conferma diretta che
mancava a fine "continued 19" (lì il verdetto era stato validato solo
sullo Z8, con AVX-512, dove il framework si è correttamente
auto-qualificato come "FUORI dal range" invece di dare un falso
allarme).

Altri numeri di interesse:
- **`avx512_note`**: si attiva correttamente, stesso testo già visto
  per l'EPYC 7C13 di "continued 17" — nessuna modifica necessaria, il
  framework generalizza bene a un host DIVERSO con la stessa
  caratteristica (no AVX-512).
- **`batching_signal`**: **~3.0x** su questo pod, contro **~8-12x**
  misurato sullo Z8 in "continued 19" — dato NUOVO, non atteso a
  priori: il guadagno potenziale da batching GEMV→GEMM non è uguale
  ovunque, varia con l'hardware. Se la direzione "batching" verrà mai
  scelta, andrà ri-misurata sull'hardware target reale, non assunta
  costante da un solo host.
- **`pcie.flag`**: `"ok"`, pinned ~20-26 GB/s — nessun tetto WSL2/GPU-PV
  (atteso: questo è un pod Linux reale, non lo stesso ambiente WSL2
  problematico di "continued 16").
- **`ram_budget`**: 30 expert path1/4 expert path AWQ — stessi ordini di
  grandezza di "continued 17" (RAM cgroup quasi identica, 116.0GB vs
  116.4GB).

JSON completi salvati in
`logs/runpod_perf_framework_validation_20260817/` (`hw_pod.json`,
`tuning_report_pod.json`).

### Why this matters

- Il framework non è più solo "gira senza errori sullo Z8" — è
  verificato che produce il segnale CORRETTO su due host con
  caratteristiche opposte (AVX-512 sì/no), auto-qualificandosi in
  entrambi i casi invece di richiedere giudizio umano per interpretare
  il numero.
- La suite di test (212 totali, 174 preesistenti + 38 nuovi, ancora
  tutta verde) copre ora anche
  gli script di questo framework, non solo il codice `scheduler/gcsg.py`
  — un futuro refactor di `perf_test_hardware.py`/
  `perf_tuning_report.py` avrebbe una rete di sicurezza.
- Il bug di ordine-dei-test trovato (punto 3 sopra) è un promemoria
  operativo: un `@pytest.mark.gpu` che allunga la vita di tensori CUDA
  può influenzare test scritti mesi prima nello stesso processo pytest
  — vale la pena controllare la suite COMPLETA dopo aver aggiunto nuovi
  test GPU-touching, non solo il file nuovo in isolamento.

### Current state

- Codice: da committare e pushare su `claude/ddr4-ram-processing-unzseh`
  (PR #42) subito dopo questo entry.
- Pod RunPod: ancora acceso al momento di scrivere — nessuna azione di
  shutdown presa in questa sessione (a differenza delle sessioni
  precedenti, il project owner lo ha acceso esplicitamente e non ha
  ancora chiesto di spegnerlo).

Next session: nessuna azione bloccante. Il framework performance-test/
performance-tuning è ora testato e validato su due profili hardware
diversi — pronto per essere riusato senza ulteriore lavoro di
validazione la prossima volta che serve un numero di questo tipo.

---

## 2026-08-17 (continued 21) — dati target su H200/Xeon Platinum 8568Y+ (AMX), bug reale trovato e corretto

**Release:** sì — `benchmarks/perf_test_hardware.py` (fix
`avx512_vnni`, vedi sotto), `tests/test_perf_test_hardware.py` (test di
regressione dedicato). Nessuna modifica al codice GCSG di produzione.

### What we set out to do

Il project owner ha chiesto dati "target" da hardware di ultima
generazione, per confronto con l'architettura RTX 3090/EPYC-senza-
AVX-512 misurata finora. Pod nuovo: **H200 SXM (1x)**, **12 vCPU Intel
Xeon Platinum 8568Y+ (Emerald Rapids)**, 251GB memoria — stesso
template Docker delle sessioni precedenti
(`ghcr.io/danielesalpietro/vmemoryfabric:sprint-4-tekniska`).

### What we did

1. Connessione SSH con la keypair dedicata (stesso pattern di
   "continued 20"), repo clonato al volo.
2. Numeri cgroup reali (mai fidarsi di `nproc`/`free` grezzi, BOOTSTRAP
   §5.9): **10.2 core cgroup** (dei 12 vCPU nominali — overhead
   riservato), **233.5GB RAM cgroup**.
3. Flag CPU reali da `/proc/cpuinfo`: AVX2, AVX-512F, AVX-512-VNNI,
   **più AMX** (`amx_bf16`, `amx_int8`, `amx_tile` — non ancora tracciati
   dal framework, vedi "Why this matters").
4. **Bug trovato eseguendo il framework su hardware reale, non in unit
   test**: il report segnalava `avx512_vnni: false` nonostante il
   grep grezzo su `/proc/cpuinfo` mostrasse chiaramente `avx512_vnni`
   nella lista flag. Causa: `perf_test_hardware.py` cercava la
   sottostringa `"avx512vnni"` (senza underscore) — il kernel Linux
   nomina questo flag specifico CON underscore in `/proc/cpuinfo`
   (`avx512_vnni`), a differenza di `avx512f`/`avx512bw`/`avx512vbmi`/
   ecc. (nessun underscore). Fix immediato: cercare la stringa esatta
   `"avx512_vnni"`, con un test di regressione dedicato che avrebbe
   dovuto esistere fin da "continued 19" (nessun test copriva
   direttamente la mappatura `bench_cpu()["avx_support"]` contro flag
   noti, solo la struttura). Verificato: `git pull` sul pod, rilanciato,
   `avx512_vnni: true` corretto. Stesso principio del bootstrap
   generale: un valore misurato su HARDWARE REALE ha rivelato un bug
   che dimensioni sintetiche/mock nei test non avrebbero necessariamente
   catturato se il mock stesso avesse ripetuto lo stesso errore di
   naming (per fortuna qui i test esistenti usavano stringhe letterali
   diverse, non il bug — ma il nuovo test di regressione lo blocca
   esplicitamente ora).

### Result

Numeri (dimensioni Mixtral reali, `OSX_BENCH_HIDDEN=4096
OSX_BENCH_INTERMEDIATE=14336`), JSON completi in
`logs/runpod_perf_framework_validation_20260817/hw_h200.json` +
`tuning_report_h200.json`:

| | H200 pod (oggi) | RunPod EPYC (continued 20) | Z8 locale |
|---|---|---|---|
| GPU | H200, 1889 GFLOPS (batch=1) | RTX 3090, 430 GFLOPS | RTX 3090 |
| CPU matmul batch=1 | **3.3 GFLOPS** | 11.7 GFLOPS | (AVX-512, non ricalcolato a dim reali in questo entry) |
| RAM copy bandwidth | **5.4 GB/s** | 11.0 GB/s | 19.5 GB/s |
| CPU cgroup cores | 10.2 | 27.2 | 28 |
| predicted_slowdown | **~570x** | 36.9x | 16.9x |
| batching_signal | 17.3x | 3.0x | ~8-12x |
| PCIe pinned min | 30-55 GB/s | 20-26 GB/s | 11-12 GB/s |

**Risultato controintuitivo, non nascosto**: il CPU forward a batch=1
sul pod H200/Xeon Platinum (AVX-512+VNNI+AMX, silicio 2024) è **più
lento** (3.3 GFLOPS) del pod EPYC senza AVX-512 misurato in "continued
20" (11.7 GFLOPS) — nonostante il silicio più recente. Due letture
possibili, nessuna delle due verificata a fondo qui: (a) solo 10.2 core
cgroup contro 27.2 — anche se il forward batch=1 è a singolo thread
(memory-bandwidth-bound, non thread-bound, §5.11), un'allocazione vCPU
più piccola su un host condiviso può comunque ricevere meno banda
memoria/cache L3 per-core; (b) il benchmark usa tensori fp32
(`torch.randn` default) — **AMX accelera INT8/BF16/TF32, non fp32**,
quindi il framework attuale non misura affatto il vantaggio AMX di
questo hardware. La banda RAM copy (5.4 GB/s, la più bassa delle tre
misurate) supporta la lettura (a): un collo di bottiglia di banda
memoria per-core su questa allocazione vCPU specifica, non solo
l'assenza di sfruttamento AMX.

**Predicted slowdown ~570x**, correttamente segnalato "FUORI dal range
già osservato" — il verdetto onesto per questo host: un H200 rende il
cpu-offload ANCORA MENO conveniente in termini relativi (non di più),
perché la GPU è diventata molto più veloce mentre il CPU-side di questo
specifico pod non ha tenuto il passo (vedi sopra).

### Why this matters

- **Risposta diretta alla domanda del project owner**: "dati target di
  ultima generazione" non significa automaticamente "gap CPU/GPU più
  piccolo" — su QUESTO pod è il contrario. Il framework ha fatto
  esattamente il suo lavoro: dare un numero calibrato invece di
  un'aspettativa intuitiva sbagliata.
- **Limite del framework ora esplicito, non implicito**: non misura
  AMX (solo fp32). Se in futuro serve sapere se AMX aiuterebbe il path
  produzione reale (che usa INT4/AWQ, non fp32 — diverso ancora da
  BF16/INT8 nativi AMX), servirebbe un benchmark dedicato a quella
  precisione, non un'estensione ovvia di quello attuale.
- Il bug avx512_vnni conferma ancora una volta il principio del
  bootstrap: un numero misurato su hardware reale, non sintetico, ha
  trovato in minuti un bug che una sessione di solo sviluppo/mock non
  avrebbe necessariamente incontrato.

### Current state

- Codice: da committare e pushare su `claude/ddr4-ram-processing-unzseh`
  (PR #42) subito dopo questo entry (fix avx512_vnni già pushato
  separatamente a metà sessione per permettere `git pull` sul pod).
- Pod H200: ancora acceso al momento di scrivere, nessuna azione di
  shutdown presa (stesso principio di "continued 20" — acceso
  esplicitamente dal project owner).
- JSON completi (pre-fix E post-fix per l'H200) in
  `logs/runpod_perf_framework_validation_20260817/`.

Next session: nessuna azione bloccante. Se si vuole approfondire il
gap CPU inatteso su questo pod: verificare se è la banda memoria
per-core dell'allocazione vCPU (ipotesi a) o la mancata attivazione di
AMX su fp32 (ipotesi b) — nessuna delle due confermata qui, ora
tracciate in [issue #45](https://github.com/danielesalpietro/vMemoryFabric/issues/45)
(aperta in "continued 22"). Prossimo hardware target già pianificato
dal project owner: NVIDIA GB10/DGX Spark (memoria unificata CPU+GPU —
la metodologia CPU/RAM/GPU/PCIe separata di questo framework potrebbe
non applicarsi direttamente, vedi promemoria in memoria di sessione).

---

## 2026-08-17 (continued 22) — documento di confronto a tre host, poi aggiornamento documentazione generale

**Release:** sì — `logs/runpod_perf_framework_validation_20260817/
COMPARISON_ANALYSIS.md` (nuovo), `hw_z8.json`/`tuning_report_z8.json`
(nuovi, mancava un JSON archiviato per lo Z8 a dimensioni reali — solo
verificato a schermo in "continued 19"). Nessuna modifica al codice.

### What we set out to do

Il project owner ha chiesto un documento di confronto ed analisi,
baseline Z8 → chiusura H200, usando i tre dataset già raccolti oggi
("continued 19", "continued 20", "continued 21").

### What we did

1. Rigenerato un JSON pulito per lo Z8 a dimensioni Mixtral reali
   (mancava — il numero 16.9x citato in "continued 19" veniva solo
   dall'output a schermo, non da un file archiviato) via
   `docker compose run` locale.
2. Letti i tre JSON completi (non i numeri ricordati dalla
   conversazione) per costruire tabelle accurate.
3. Scritto `COMPARISON_ANALYSIS.md`: profili host, tabella metriche
   grezze, tabella verdetti derivati, 5 sotto-analisi (§6.1-6.5),
   conclusioni, limiti espliciti.
4. Aperta [issue #45](https://github.com/danielesalpietro/vMemoryFabric/issues/45)
   su richiesta esplicita del project owner, per tracciare le due
   ipotesi non verificate del documento (banda RAM per-core
   sull'allocazione vCPU, AMX non esercitato dal benchmark fp32)
   invece di lasciarle solo come nota in un file di log.

### Result

Trovata una correlazione pulita non ancora notata nei tre entry
precedenti: **la banda RAM (19.84→10.98→5.96 GB/s) predice il
GFLOPS CPU batch=1 (22.97→11.66→3.32) molto meglio della presenza di
AVX-512/VNNI** — l'H200/Xeon Platinum ha AVX-512+VNNI+AMX ed è
comunque il più lento dei tre, lo Z8 ha AVX-512+VNNI ed è il più
veloce. Rinforza (con un terzo punto dati, non solo l'A/B thread-count
di "continued 17") che il collo di bottiglia è la banda memoria, non
il set di istruzioni. Documento consegnato al project owner via file.

### Why this matters

Il documento traduce tre run isolati (uno per host) in un confronto
strutturato con una conclusione azionabile per issue #33: "hardware più
recente" non garantisce un cpu-offload migliore, dipende dal
bilanciamento RAM/CPU-core della specifica allocazione — rilevante per
qualunque decisione futura su quale pod/hardware usare per lavoro
successivo su questo path.

### Current state

Documento e JSON dello Z8 non ancora committati al momento di scrivere
questo entry — segue subito un aggiornamento della documentazione di
progetto più ampia (README, CHANGELOG, questo file), poi commit/push
unico.

Next session: nessuna azione bloccante.
