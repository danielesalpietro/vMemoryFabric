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
