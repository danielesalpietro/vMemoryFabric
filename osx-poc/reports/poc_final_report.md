# OSX-PoC Final Report — EAT (M1), Tier Manager (M2), GCSG/MMLU (M3) Results

**Last updated:** 2026-08-13 (Sprint 5 / Berg)
**Supersedes:** `osx-poc/mmlu_final_report.md` (renamed and folded in as §3 below —
that file covered M3/MMLU only and had drifted out of date; see LOGBOOK for
the rename). This is now the single consolidated results document the
Sprint 5 plan's §A2 called for — M1/M2/M3 in one place with the 4
non-functional acceptance criteria evaluated together, replacing the
README-roadmap-table reconstruction and this file's own previous
MMLU-only scope.

---

## 0. Non-functional acceptance criteria — final status

These are the 4 targets stated in the README ("Development roadmap"
section). Presented together here for the first time with the actual
supporting numbers next to each, rather than as a checklist elsewhere with
the evidence scattered across README/LOGBOOK/two separate reports.

| # | Target | Result | Status | Margin |
|---|---|---|---|---|
| 1 | PT-PEP latency < 3ms p99 (CPU) | `tests/test_scheduler.py::test_latency_under_3ms` passes in the full suite; the only concrete number ever recorded against it is a single **isolated-run failure at 3.71ms** (2026-08-11), which then passed cleanly when the full suite ran right after — read as container CPU jitter, not re-investigated | 🟡 nominally met, thin evidence | No comfortable margin on record — unlike the other three targets, this one has never had a clean passing p99 number actually written down, only a pass/fail test outcome |
| 2 | PT-PEP hit rate > 70% | 87.2% on held-out validation (TF-IDF+centroid classifier, 8 HuggingFace datasets, 200 train / 50 held-out per domain) | ✅ met | +17.2pp, but scope caveat: same-distribution held-out set, not out-of-distribution, and the softmax temperature was calibrated on the same set — a documented deviation from the originally-planned BERT-small classifier |
| 3 | GCSG quality degradation < 2% (MMLU-5shot) | -0.19pp to +0.7pp across 5 independent full-570-question runs spanning platform/hardware/quantization/data-path (§3 below) | ✅ met | Wide margin, and unusually well-corroborated — five independent axes of variation, not one run |
| 4 | Shard promotion latency within 1.5× theoretical bandwidth | `pin=True` P50 194.1µs (round 1) / 207.8µs (round 2) vs. 732.4µs threshold (§2.2) | ✅ met (P50) | ~3.5–3.7× margin at P50; P95/P99 miss the target in both rounds, but n=20 synthetic shards is too small a sample for tail statistics to be meaningful — not investigated further, consistent with prior reporting |

Three of four targets are solidly met. The PT-PEP latency target (#1) is
the one place this report is *not* claiming a clean result — it's a test
that passes, not a number that was measured and found comfortably under
budget, and that distinction matters for anyone citing this in the paper
(Sprint 5 plan §B2/§B3: state it as "test-suite-verified," not "measured
with margin," unlike the other three).

---

## 1. M1 — Expert Access Table (EAT)

All numbers in this section are from `benchmarks/bench_eat.py`, run twice
(`logs/sprint4_tekniska/regression/bench_eat_20260812_224334.log`,
`..._round2_20260812_230701.log`) on the same RunPod RTX A5000 pod as the
Sprint 4 MMLU/tier-manager work (real Linux, not WSL2) — **not** the same
environment as the sandbox re-measurement cited in §1.2, a distinction
that matters below.

### 1.1 Lookup/insert vs. plain `dict` (issues #1/#4, both closed — Bloom filter removed)

| | EAT (dict + `RLock`) | plain `dict` | delta |
|---|---|---|---|
| Insert throughput, round 1 | 741,799 ops/s | 2,638,627 ops/s | dict ~3.6× faster |
| Insert throughput, round 2 | 744,877 ops/s | 2,588,522 ops/s | dict ~3.5× faster |
| Lookup p50 (overall), round 1 | 0.420µs | 0.410µs | +0.019µs (noise) |
| Lookup p50 (overall), round 2 | 0.420µs | 0.410µs | +0.019µs (noise) |
| Lookup p99 (overall), round 1 | 0.980µs | 0.500µs | ~2.0× |
| Lookup p99 (overall), round 2 | 1.050µs | 0.710µs | ~1.5× |

The p50 delta (~0.02µs) is consistent with the "~0.07µs, noise, not a gap
to justify" conclusion recorded on removal day — both are effectively
measurement noise at this timescale, just from two different sessions
(removal-day vs. this Sprint 4 regression pass), so the exact figure
differs but the conclusion doesn't. **New observation from consolidating
these two regression runs, not previously stated anywhere:** the p50
convergence masks a real, consistent p99 gap (~1.5–2.0× worse for EAT vs.
a bare `dict`), plausibly the residual cost of `RLock` acquisition on the
uncontended path even after the Bloom filter's removal. Small in absolute
terms (sub-microsecond) and not close to threatening any acceptance
target, but worth recording precisely rather than folding into the same
"noise" bucket as the p50 number, since it isn't the same claim.

Insert throughput is a real, unavoidable cost of the `dict`+`RLock`
design (~3.5× slower than a bare `dict`) — this was never the subject of
issues #1/#4 (both were about lookup latency, i.e., the Bloom filter's
actual job) and isn't flagged as a problem; it's the expected price of a
threadsafe insert path.

### 1.2 `RLock` contention (issue #2 — open, deliberately, not redesigned)

Three independent measurements of the same synthetic scenario (4 readers +
1 writer, 20,000 writes), from three different environments, none of them
overwriting the others per this project's own convention of recording
unexplained discrepancies rather than picking one:

| Source | Environment | Uncontended p99 | Contended reader p99 | Implied degradation |
|---|---|---|---|---|
| Original issue #2 filing | unrecorded/unknown | single-digit µs | several ms | **~1360×** |
| Sandbox re-measurement (2026-08-12, 4 runs) | pure Python/threading, no GPU, agent sandbox | 25.6–33.8µs | 2062–2324µs | **~61–91×** |
| This report's pod regression, round 1 | RunPod RTX A5000, real Linux | 0.98µs | 341.4µs | **~348×** (computed here for the first time) |
| This report's pod regression, round 2 | RunPod RTX A5000, real Linux | 1.05µs | 433.8µs | **~413×** (computed here for the first time) |

None of the four numbers agree with each other, and the gap has never
been root-caused (per the 2026-08-12 decision to leave issue #2 open
rather than redesign the `RLock` against a scenario real traffic doesn't
yet produce). The two new pod-regression ratios computed here sit between
the sandbox and original figures but don't match either — if anything,
this *weakens* any single-cause hypothesis (e.g. "it's just a sandbox vs.
real-hardware difference") rather than resolving it, since the pod numbers
aren't close to the original 1360× either. Recorded as a fourth data point
in an unresolved question, not a resolution.

**Reminder, unchanged from the original decision:** real `GCSGWorker`
traffic today is single-threaded, so this 4-reader/1-writer scenario isn't
produced by anything running in production yet. All four numbers above
describe a synthetic stress scenario, not observed behavior.

### 1.3 Bloom filter (issues #1/#4, both closed)

Removed entirely from `EAT`'s hot path (commit 64f6bdc) after the Counting
Bloom Filter fix for #4 made it possible to re-measure #1 fairly and found
lookup latency got *worse*, not better, under the new implementation
(~6.8–8.1× vs. the old ~4.7–4.9× slower than a plain `dict`). Full
narrative: `LOGBOOK.md`, 2026-08-12 "issue #4 actually fixed" and "Bloom
filter removed" entries. §1.1 above is the post-removal state.

### 1.4 Slab allocator — incidental data, not a target

Not one of the 4 acceptance criteria, but present in both regression runs
and not surfaced in any prior report:

| | 4 slots (1GB pool) alloc p50 | 32 slots (8GB pool) alloc p50 |
|---|---|---|
| Round 1 | 2.78µs | 0.63µs |
| Round 2 | 4.10µs | 0.80µs |

Alloc latency drops as pool size grows (more slots to search/pick from is
apparently not the bottleneck at this scale — likely allocator warm-up /
page caching effects, consistent with the soak test's own pin-alloc drift
observation in §2.3). Free latency is sub-microsecond in both configs.
Not investigated further — included here only because it was sitting in
data already collected for other purposes and no report had mentioned it.

---

## 2. M2 — EMH Tier Manager

### 2.1 NVMe → DDR4 → VRAM raw pipeline (Sprint 2 benchmarks)

From the same two regression-pod runs (`logs/sprint4_tekniska/bench_tier_pod/bench_tier_pod_20260812_212555.log`,
`..._rerun2_20260812_221103.log`), synthetic 4MB shards (not the 256MB
production shard size — see the benchmark module's own docstring):

| | NVMe→DDR4 p50 | NVMe→DDR4 p95/p99 | DDR4→VRAM p50 | DDR4→VRAM p95/p99 |
|---|---|---|---|---|
| Round 1 | 1,568.8µs | 6,834.7µs | 13,269.3µs | 213,210.7µs |
| Round 2 | 1,514.0µs | 6,909.7µs | 16,636.5µs | 207,081.5µs |

p95/p99 dominated by CUDA cold-start on the first shard of each run — this
is issue #3 (`bench_tier.py` needs a warm-up iteration before timing),
still open, flagged for the A1 fix batch in the Sprint 5 plan precisely
because these numbers are unusable for the paper's Evaluation section
until it's fixed. Not re-measured here — recorded as-is, with the caveat
attached rather than silently presenting a skewed number as clean.

### 2.2 `promote_live_tensor` — pinned vs. unpinned (issue #17 sub-goal 4, closed)

Both regression rounds, not just the one previously cited in README/the
old MMLU report:

| | `pin=False` P50 | `pin=False` P95/P99 | `pin=True` P50 | `pin=True` P95/P99 | 1.5× threshold (732.4µs) |
|---|---|---|---|---|---|
| Round 1 | 684.2µs | 29,831.9µs | 194.1µs | 82,668.1µs | met at P50 (~3.5–3.8× margin), missed at tail |
| Round 2 | 620.7µs | 1,321.9µs | 207.8µs | 83,774.0µs | met at P50 (~3.0–3.5× margin), missed at tail |

`pin=True` is consistently faster at P50 across both rounds (the metric
the acceptance target is actually defined against). The tail behavior is
inconsistent between the two rounds in an interesting way: round 1's
`pin=False` P95/P99 (29,831.9µs) is dramatically worse than round 2's
(1,321.9µs) — a ~22× difference in the *unpinned* tail between two runs of
the same benchmark, bigger than the `pin=True` vs `pin=False` difference
itself in round 2. At n=20 synthetic shards this reads as noise/cold-start
artifacts rather than a real characterization of tail behavior, consistent
with the existing "not investigated further" position — but the magnitude
of that round-to-round swing is worth recording precisely rather than
averaging away, since it's a caution against reading too much into any
single P95/P99 number from this benchmark until issue #3's warm-up fix
lands.

### 2.3 Soak test — pinned memory under sustained load (issue #17 sub-goal 2, closed)

`logs/sprint4_tekniska/logbook_20260812_1344/soak_test_pinning/soak_test.log` — 1000 cycles,
256MB shard (production size, unlike §2.1/§2.2's synthetic 4MB shards),
real pinned `H2D`+`D2H`+byte-comparison each cycle:

| | min | mean | max | stddev | drift (first 10% vs. last 10%) |
|---|---|---|---|---|---|
| Cycle total | 439.4ms | 513.5ms | 949.1ms | 30.0ms | -1% |
| Pin alloc | 5.99ms | 6.99ms | 337.4ms | 10.5ms | -31% (faster after warm-up) |
| H2D+D2H transfer | 40.4ms | 79.8ms | 225.0ms | 22.2ms | -3% |

**0/1000 byte-exact mismatches.** No corruption, no degradation under
sustained load — the pin-alloc allocator actually gets *faster* after
warm-up (-31% drift), not slower, which is the basis for the "manually
pinned transfer is safe and stable" conclusion `GPUTransfer.to_vram(pin=True)`
now ships with (opt-in, default `False`).

---

## 3. M3 — GCSG / MMLU-5shot Evaluation

*(Folded in from the former `osx-poc/mmlu_final_report.md`, 2026-08-13 — content unchanged from that file's last update.)*

### 3.0 Consolidated summary (all full 570-question runs)

| # | Run | Date | Platform | Path | Correct/570 | Accuracy | `tier_manager_wired` |
|---|---|---|---|---|---|---|---|
| 1 | Overnight baseline | 2026-08-11 | WSL2, RTX 3090 | `cpu_offload_gb` (hook-only path, real shadow execution) | 411 | 72.11% | — (field absent, pre-#17) |
| 2 | Cross-hardware repro, sliced (18×) | 2026-08-12 | Real Linux, RTX A5000 (RunPod) | `cpu_offload_gb` (not wired) | 412 | 72.28% | `false` |
| 3 | Cross-hardware repro, single-process | 2026-08-12 | Real Linux, RTX A5000 (RunPod) | `cpu_offload_gb` (not wired) | 412 | 72.3%¹ | `false` |
| 4 | TierManager-wired, AWQ (path 3) | 2026-08-12 | Real Linux, RTX A5000 (RunPod) | `TierManager`/`EAT`, `--wire-tier-manager` | 411 | 72.11% | `true` |
| 5 | TierManager-wired, Marlin (path 2) | 2026-08-13 | Real Linux, RTX 3090-class (`eu-cz-1`) | `TierManager`/`EAT`, `--wire-tier-manager --quantization awq_marlin` | 412 | 72.3%¹ | `true` |

¹ 72.3% is 412/570 = 72.2807%, rounded to 1 decimal instead of the 2
decimals used elsewhere in this table — same underlying count as row 2/3,
not a different result.

**Row 4 was run three independent times** (`logs/sprint4_tekniska/mmlu/mmlu_tier_manager_pod_singleshot_20260812_195140.jsonl`,
`..._rerun_20260812_210821.jsonl`, `..._rerun3_20260812_215416.jsonl`) —
byte-identical result: 411/570, 562,354 shadow activations, every time.

All five numbers are within 0.7 percentage points of each other and inside
the <2% GCSG quality-degradation target against the 72.3% hook-only
baseline, regardless of platform, hardware generation, quantization
backend, or data path. That invariance across five independent axes of
variation is the headline result of this section, not any single accuracy
number in isolation, and directly supports target #3 in §0 above.

### 3.1 Sprint 3 baseline (2026-08-11) — full detail

**Model:** `casperhansen/mixtral-instruct-awq` (Mixtral-8x7B-Instruct, AWQ, `quantization="awq_marlin"`)
**Runtime:** vLLM 0.6.6.post1, `GCSGWorker` (real shadow execution — issues #10/#16), `cpu_offload_gb=4`, `gpu_memory_utilization=0.95`, `max_num_seqs=16`, `enforce_eager=True`, `max_model_len=3328`
**Results file:** `logs/sprint4_tekniska/mmlu/mmlu_results_overnight_20260811.jsonl`
**Orchestration:** `scripts/run_mmlu_in_slices.sh` — 18 slices of 32 prompts each

| Metric | Value |
|---|---|
| Prompts evaluated | 570 / 570 (100%) |
| Subjects covered | 57 / 57 (100%) |
| Correct | 411 |
| **Overall accuracy** | **72.11%** |
| Shadow activations (sum across all 18 slices) | 562,338 |
| Total `generate()` time (summed across slices) | 3,690s (~1h 03m) |
| Slowest slice | `[32:64)` — 1,784.7s (~29.7 min) |

Quality target: -0.19pp vs. the 72.3% hook-only baseline. Timing note: the
slowest slice was root-caused to `maybe_offload_to_cpu()`'s pageable-memory
CPU→GPU swap-in under WSL2 (`pin_memory=False`) — structural,
upstream-acknowledged (vllm-project/vllm#1084, #37883), not a project bug.
Full per-subject accuracy table: Appendix A.

**Data-quality correction preserved from the original report:** the
shadow-activations figure originally cited here was 13,756 (the last
slice's own counter, mistaken for a run-wide total, since each slice runs
in its own fresh container). Correct total, recomputed by summing all 18
rows: **562,338**.

### 3.2 Sprint 4 addendum (2026-08-12/13) — cross-hardware repro + TierManager-wired reruns

**Cross-hardware reproduction, `cpu_offload_gb` path (not yet wired):**

| Run | Results file | Correct | Accuracy | Wall time |
|---|---|---|---|---|
| Sliced (18×) | `logs/sprint4_tekniska/logbook_20260812_1344/mmlu_sliced_run/mmlu_results.jsonl` | 412/570 | 72.28% | ~38–40 min |
| Single-process | `logs/sprint4_tekniska/logbook_20260812_1344/mmlu_burn_singleshot/burn_singleshot.log` | 412/570 | 72.3%¹ | 9.5 min (~4.2× faster, same quality) |

No stall reproduced on real Linux — the historical WSL2 stall (~request
27-31) is structural to WSL2's pageable-memory path, not a deadlock, so it
doesn't reproduce off WSL2.

**New finding — shadow-execution activation rate correlates with latency,
not accuracy** (full detail: `gcsg_shadow_execution_report.md` §6.1).
Per-slice `shadow_activations` vs. accuracy: r=0.04 (none). Per-slice
`shadow_activations` vs. wall time: r=0.95, tightening to **r=0.993** once
isolated to `generate()` duration alone. A real, previously-unmeasured
*performance* cost of shadow execution — worth citing in the paper's
Evaluation section as a latency-vs-quality tradeoff finding.

**TierManager-wired reruns (issue #17 — the actual point of Sprint 4):**

AWQ path (path 3): `logs/sprint4_tekniska/mmlu/mmlu_tier_manager_pod_singleshot_20260812_195140.jsonl`
(+2 identical reruns) — **411/570 (72.11%)**, `tier_manager_wired: true`.
First MMLU number generated via the real `TierManager`/`EAT` data path
instead of vLLM's native `cpu_offload_gb`.

Marlin path (path 2, the path every previously-published MMLU number
actually used): `logs/sprint4_tekniska/marlin_mmlu/mmlu_marlin_singleshot_20260813_005201.jsonl`
— **412/570 (72.28%/72.3%)**, `tier_manager_wired: true`,
`quantization: awq_marlin`. Closes the last open gap of issue #17.

**Methodology note:** `eval_mmlu_gcsg.py`'s `--wire-tier-manager` flag
silently forced `quantization=awq` (stale from before Marlin was wired),
fixed by adding an explicit `--quantization` flag.

### 3.3 Exact per-subject diffs (recomputed from raw `.jsonl`, not from prose)

The original README/LOGBOOK prose described "4 individual answers flipped
(2 up, 2 down)" without specifying which baseline. Confirmed by diffing
`per_subject_correct` directly:

**Row 1 (WSL2 baseline, 411) vs. Row 4 (AWQ TierManager-wired, 411):**

| Subject | Baseline | AWQ (wired) |
|---|---|---|
| college_mathematics | 5 | 4 |
| elementary_mathematics | 5 | 6 |
| high_school_european_history | 9 | 8 |
| high_school_world_history | 8 | 9 |

**Row 2 (A5000 unwired, 412) vs. Row 5 (Marlin TierManager-wired, 412):**

| Subject | Unwired | Marlin (wired) |
|---|---|---|
| college_physics | 5 | 4 |
| electrical_engineering | 3 | 4 |
| high_school_macroeconomics | 8 | 7 |
| machine_learning | 5 | 6 |

Matches the 4 subjects named in `LOGBOOK.md`'s 2026-08-13 entry exactly.
No root cause claimed beyond floating-point non-determinism and/or
run-topology differences — every pairing nets to ±0 or ±1 out of 570.

---

## 4. Open items for the paper (Sprint 5, §B2/§B3)

- §0's acceptance-criteria table is the single best summary artifact for
  the paper's Evaluation section opening — four targets, one table, honest
  about which one (#1) is weaker evidence than the others.
- §3.0's five-way MMLU invariance is the strongest single Evaluation claim.
- §1.2's three-way (now four-way) `RLock` contention discrepancy is a
  genuine open question — cite it as a limitation, not a solved problem.
- §3.2's shadow-activation/latency correlation (r=0.993) belongs in
  Evaluation as a real latency-vs-quality tradeoff finding.
- No equivalent MMLU number exists yet for path 1 (`_ShadowExpertINT4`) —
  verified mechanically under real offload (Sprint 4 sub-goal 6) but never
  run against MMLU. Flagged as a real gap.

---

## Appendix A: Per-subject accuracy, Sprint 3 baseline run (411/570, 72.11%)

| Subject | Correct/10 | Accuracy |
|---|---|---|
| abstract_algebra | 4 | 40.0% |
| anatomy | 7 | 70.0% |
| astronomy | 9 | 90.0% |
| business_ethics | 7 | 70.0% |
| clinical_knowledge | 7 | 70.0% |
| college_biology | 9 | 90.0% |
| college_chemistry | 6 | 60.0% |
| college_computer_science | 6 | 60.0% |
| college_mathematics | 5 | 50.0% |
| college_medicine | 9 | 90.0% |
| college_physics | 4 | 40.0% |
| computer_security | 8 | 80.0% |
| conceptual_physics | 8 | 80.0% |
| econometrics | 5 | 50.0% |
| electrical_engineering | 4 | 40.0% |
| elementary_mathematics | 5 | 50.0% |
| formal_logic | 4 | 40.0% |
| global_facts | 6 | 60.0% |
| high_school_biology | 9 | 90.0% |
| high_school_chemistry | 6 | 60.0% |
| high_school_computer_science | 8 | 80.0% |
| high_school_european_history | 9 | 90.0% |
| high_school_geography | 8 | 80.0% |
| high_school_government_and_politics | 10 | 100.0% |
| high_school_macroeconomics | 7 | 70.0% |
| high_school_mathematics | 4 | 40.0% |
| high_school_microeconomics | 7 | 70.0% |
| high_school_physics | 6 | 60.0% |
| high_school_psychology | 7 | 70.0% |
| high_school_statistics | 9 | 90.0% |
| high_school_us_history | 7 | 70.0% |
| high_school_world_history | 8 | 80.0% |
| human_aging | 10 | 100.0% |
| human_sexuality | 8 | 80.0% |
| international_law | 10 | 100.0% |
| jurisprudence | 8 | 80.0% |
| logical_fallacies | 10 | 100.0% |
| machine_learning | 5 | 50.0% |
| management | 9 | 90.0% |
| marketing | 7 | 70.0% |
| medical_genetics | 10 | 100.0% |
| miscellaneous | 8 | 80.0% |
| moral_disputes | 7 | 70.0% |
| moral_scenarios | 6 | 60.0% |
| nutrition | 7 | 70.0% |
| philosophy | 9 | 90.0% |
| prehistory | 8 | 80.0% |
| professional_accounting | 6 | 60.0% |
| professional_law | 8 | 80.0% |
| professional_medicine | 6 | 60.0% |
| professional_psychology | 8 | 80.0% |
| public_relations | 5 | 50.0% |
| security_studies | 7 | 70.0% |
| sociology | 7 | 70.0% |
| us_foreign_policy | 8 | 80.0% |
| virology | 6 | 60.0% |
| world_religions | 10 | 100.0% |

Lowest-scoring subjects (STEM-heavy, expected for a 4-bit quantized model
at this size): `abstract_algebra`, `college_physics`,
`electrical_engineering`, `formal_logic`, `high_school_mathematics` (all
40.0%). Highest: `high_school_government_and_politics`, `human_aging`,
`international_law`, `logical_fallacies`, `medical_genetics`,
`world_religions` (all 100.0%).

---

## References

- **M1 (EAT):** `logs/sprint4_tekniska/regression/bench_eat_20260812_224334.log`, `..._round2_20260812_230701.log`, `LOGBOOK.md` 2026-08-12 "sub-goal 5 started" entry (sandbox re-measurement), "issue #2 ... decided" entry (decision)
- **M2 (Tier Manager):** `logs/sprint4_tekniska/bench_tier_pod/bench_tier_pod_20260812_212555.log`, `..._rerun2_20260812_221103.log`, `logs/sprint4_tekniska/logbook_20260812_1344/soak_test_pinning/soak_test.log`
- **M3 (GCSG/MMLU):** `logs/sprint4_tekniska/mmlu/mmlu_results_overnight_20260811.jsonl`, `logs/sprint4_tekniska/logbook_20260812_1344/mmlu_sliced_run/`, `logs/sprint4_tekniska/logbook_20260812_1344/mmlu_burn_singleshot/burn_singleshot.log`, `mmlu_tier_manager_*.jsonl`, `logs/sprint4_tekniska/marlin_mmlu/*.jsonl`
- Full narrative/investigation trail: `LOGBOOK.md`, `osx-poc/reports/gcsg_shadow_execution_report.md` §6/§9
- Issues: [#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1), [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2) (open, deliberately), [#3](https://github.com/danielesalpietro/vMemoryFabric/issues/3) (open), [#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4), [#10](https://github.com/danielesalpietro/vMemoryFabric/issues/10), [#16](https://github.com/danielesalpietro/vMemoryFabric/issues/16), [#17](https://github.com/danielesalpietro/vMemoryFabric/issues/17) (all others closed)
- Upstream vLLM/WSL2 confirmation: [vllm-project/vllm#1084](https://github.com/vllm-project/vllm/issues/1084), [vllm-project/vllm#37883](https://github.com/vllm-project/vllm/issues/37883)
