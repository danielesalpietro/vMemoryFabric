# MMLU Final Report — Sprint 3 (Oskarshamn) baseline through Sprint 4 (Tekniska) TierManager-wired reruns

**Last updated:** 2026-08-12 (Sprint 5 kickoff) — consolidates every full/partial
MMLU-5shot run recorded in this repository across Sprint 3 and Sprint 4,
recomputed directly from the raw `.jsonl` result files rather than carried
over from prior prose summaries. Two prior discrepancies in how README/
LOGBOOK described these numbers are resolved below (§3) by recomputing
per-subject diffs from source instead of re-describing them.

---

## 0. Consolidated summary (all full 570-question runs)

| # | Run | Date | Platform | Path | Correct/570 | Accuracy | `tier_manager_wired` |
|---|---|---|---|---|---|---|---|
| 1 | Overnight baseline | 2026-08-11 | WSL2, RTX 3090 | `cpu_offload_gb` (hook-only path, real shadow execution) | 411 | 72.11% | — (field absent, pre-#17) |
| 2 | Cross-hardware repro, sliced (18×) | 2026-08-12 | Real Linux, RTX A5000 (RunPod) | `cpu_offload_gb` (not wired) | 412 | 72.28% | `false` |
| 3 | Cross-hardware repro, single-process | 2026-08-12 | Real Linux, RTX A5000 (RunPod) | `cpu_offload_gb` (not wired) | 412 | 72.3%¹ | `false` |
| 4 | TierManager-wired, AWQ (path 3) | 2026-08-12 | Real Linux, RTX A5000 (RunPod) | `TierManager`/`EAT`, `--wire-tier-manager` | 411 | 72.11% | `true` |
| 5 | TierManager-wired, Marlin (path 2) | 2026-08-13 | Real Linux, RTX 3090-class (`eu-cz-1`) | `TierManager`/`EAT`, `--wire-tier-manager --quantization awq_marlin` | 412 | 72.3%¹ | `true` |

¹ 72.3% is 412/570 = 72.2807%, rounded to 1 decimal instead of the 2
decimals used elsewhere in this table — same underlying count as row 2/3,
not a different result. Verified by recomputing the ratio directly rather
than treating the two decimal displays as different numbers.

**Row 4 was run three independent times** (`mmlu_tier_manager_pod_singleshot_20260812_195140.jsonl`,
`..._rerun_20260812_210821.jsonl`, `..._rerun3_20260812_215416.jsonl`) —
byte-identical result: 411/570, 562,354 shadow activations, every time.
Deterministic, not a fluke of one run.

All five numbers are within 0.7 percentage points of each other and inside
the README's <2% GCSG quality-degradation target against the 72.3%
hook-only baseline, regardless of platform (WSL2 vs. real Linux), hardware
generation (3090 vs. A5000), quantization backend (AWQ vs. Marlin), or data
path (`cpu_offload_gb` vs. `TierManager`/`EAT`). That invariance across five
independent axes of variation is the headline result of this report, not
any single accuracy number in isolation.

---

## 1. Sprint 3 baseline (2026-08-11) — full detail

**Model:** `casperhansen/mixtral-instruct-awq` (Mixtral-8x7B-Instruct, AWQ, `quantization="awq_marlin"`)
**Runtime:** vLLM 0.6.6.post1, `GCSGWorker` (real shadow execution — issues #10/#16), `cpu_offload_gb=4`, `gpu_memory_utilization=0.95`, `max_num_seqs=16`, `enforce_eager=True`, `max_model_len=3328`
**Config file:** `scripts/eval_mmlu_gcsg.py` (unmodified defaults)
**Results file:** `mmlu_results_overnight_20260811.jsonl`
**Orchestration:** `scripts/run_mmlu_in_slices.sh` — 18 slices of 32 prompts each, one fresh Docker container/`GCSGWorker` per slice, `--watchdog-timeout 2700`, external `timeout 3000` (raised 2026-08-11, see `LOGBOOK.md`)

| Metric | Value |
|---|---|
| Prompts evaluated | 570 / 570 (100%) |
| Subjects covered | 57 / 57 (100%) |
| Correct | 411 |
| Unresolved (no valid answer-letter logprob) | 0 |
| **Overall accuracy** | **72.11%** |
| Shadow activations (sum across all 18 slices) | 562,338 |
| Slices attempted / failed | 18 / 0 |
| Total `generate()` time (summed across slices) | 3,690s (~1h 03m) |
| Average slice time | 205.0s |
| Slowest slice | `[32:64)` — 1,784.7s (~29.7 min) |

This was the first complete, defensible MMLU-5shot accuracy number this
project measured with real shadow execution active end-to-end. Quality
target: **-0.19pp** vs. the 72.3% hook-only baseline (target: <2pp) — met
with a wide margin. Full per-subject accuracy table (unchanged from the
original report, still accurate for this run): see Appendix A.

**Data-quality correction preserved from the original report:** the
shadow-activations figure originally cited here was 13,756 — the value of
the *last* of the 18 rows' cumulative counter, mistaken for a run-wide
total, since each slice runs in its own fresh container and that counter
resets to 0 per slice. Correct run-wide total, recomputed by summing all 18
rows: **562,338**.

**Timing note:** the slowest slice (`[32:64)`) took ~30 min, root-caused in
`LOGBOOK.md` (2026-08-11) to `maybe_offload_to_cpu()`'s pageable-memory
CPU→GPU swap-in under WSL2 (`pin_memory=False`) — a structural,
upstream-acknowledged platform limitation (vllm-project/vllm#1084, #37883),
not a bug in this project's code and not a deadlock.

---

## 2. Sprint 4 addendum (2026-08-12/13) — cross-hardware repro + TierManager-wired reruns

### 2.1 Cross-hardware reproduction, `cpu_offload_gb` path (not yet wired)

Same evaluation repeated on real Linux (RunPod, RTX A5000, no WSL2), both
sliced and single-process, still on vLLM's native `cpu_offload_gb` —
`TierManager`/`EAT` (issue #17) not yet in the loop for this pair of runs:

| Run | Results file | Correct | Accuracy | Wall time |
|---|---|---|---|---|
| Sliced (18×) | `LogBook_20260812_1344/mmlu_sliced_run/mmlu_results.jsonl` | 412/570 | 72.28% | ~38–40 min |
| Single-process | `LogBook_20260812_1344/mmlu_burn_singleshot/burn_singleshot.log` | 412/570 | 72.3%¹ | 9.5 min (~4.2× faster, same quality) |

No stall reproduced on real Linux — the historical WSL2 stall around
request ~27-31 (§5 of `gcsg_shadow_execution_report.md`) is structural to
WSL2's pageable-memory path, not a deadlock in this project's code, so it
doesn't reproduce off WSL2.

**New finding — shadow-execution activation rate correlates with latency,
not accuracy** (full detail: `gcsg_shadow_execution_report.md` §6.1).
Per-slice `shadow_activations` vs. accuracy: r=0.04 (no relationship — GCSG
contamination doesn't selectively degrade high-activation slices). Per-slice
`shadow_activations` vs. wall time: r=0.95, tightening to **r=0.993** once
isolated to just `generate()` duration (excluding the ~91.6–106.8s
near-constant model-load cost). This is a real, previously-unmeasured
*performance* cost of shadow execution — each activation is an extra
forward pass through the INT4 verification expert — distinct from the
quality-degradation numbers this report otherwise focuses on. Worth citing
in the paper's Evaluation section (Sprint 5 plan, §B2) as a latency-vs-
quality tradeoff finding, not just a quality-parity result.

### 2.2 TierManager-wired reruns (issue #17 — the actual point of Sprint 4)

**AWQ path (path 3):** `mmlu_tier_manager_pod_singleshot_20260812_195140.jsonl`
(+ 2 identical reruns) — **411/570 (72.11%)**, `tier_manager_wired: true`,
562,354 shadow activations. This is the first MMLU number generated via the
real `TierManager`/`EAT` data path (`GCSGWorker.configure_tier_manager()` →
`TierManager.promote_live_tensor()`) instead of vLLM's native
`cpu_offload_gb`. Two 32-question slices (`mmlu_tier_manager_fetta0_...`,
`mmlu_tier_manager_pod_20260812_194757.jsonl`) matched the corresponding
Sprint 3 baseline slices exactly (21/32 and 24/32).

**Marlin path (path 2, the path every previously-published MMLU number
actually used):** `marlin_mmlu_20260812/mmlu_marlin_singleshot_20260813_005201.jsonl`
— **412/570 (72.28%/72.3%)**, `tier_manager_wired: true`,
`quantization: awq_marlin`, 562,350 shadow activations. This closes the
last open gap of issue #17 — a Marlin-specific MMLU number on the
TierManager-routed path — matching the historical single-shot Marlin
baseline (row 3 above) exactly in aggregate. A 32-question slice
(`mmlu_marlin_fetta1_20260813_004902.jsonl`, 24/32, 75.0%) matched the same
slice's Sprint 3/AWQ-wired results exactly, per subject.

**Methodology note, recorded here so it isn't lost:** `eval_mmlu_gcsg.py`'s
`--wire-tier-manager` flag silently forced `quantization=awq` (stale from
before the Marlin path was wired), which would have made a Marlin-specific
TierManager-routed run impossible without the fix. Resolved by adding an
explicit `--quantization` flag before this section's Marlin run.

### 2.3 Exact per-subject diffs (recomputed 2026-08-12 from raw `.jsonl`, not from prose)

The original README/LOGBOOK prose described "4 individual answers flipped
(2 up, 2 down)" between the AWQ-wired rerun and "the baseline" without
specifying *which* baseline, and separately described a 4-subject Marlin
divergence. Both are confirmed here by actually diffing the
`per_subject_correct` dictionaries, not re-describing the earlier text:

**Row 1 (WSL2 baseline, 411) vs. Row 4 (AWQ TierManager-wired, 411)** — same
aggregate count, but *not* byte-identical per subject (net-zero, 2 up/2 down):

| Subject | Baseline | AWQ (wired) |
|---|---|---|
| college_mathematics | 5 | 4 |
| elementary_mathematics | 5 | 6 |
| high_school_european_history | 9 | 8 |
| high_school_world_history | 8 | 9 |

This is the pair the original "4 flipped, 2 up 2 down" description
referred to — confirmed by direct computation, not assumed.

**Row 2 (A5000 unwired, 412) vs. Row 5 (Marlin TierManager-wired, 412)** —
also same aggregate, 4 subjects differ:

| Subject | Unwired | Marlin (wired) |
|---|---|---|
| college_physics | 5 | 4 |
| electrical_engineering | 3 | 4 |
| high_school_macroeconomics | 8 | 7 |
| machine_learning | 5 | 6 |

Matches the 4 subjects named in `LOGBOOK.md`'s 2026-08-13 entry exactly —
independently re-derived here rather than copied.

No root cause is claimed for either divergence beyond what's already in
`gcsg_shadow_execution_report.md` §6.1: floating-point non-determinism
and/or run-topology differences (single continuous process vs. sliced
per-process runs), not a quality regression — every pairing above nets to
±0 or ±1 out of 570, an order of magnitude below the <2% target.

### 2.4 Related non-MMLU number: promotion latency

Not an MMLU result, but the other half of "does TierManager actually do
something real" — `benchmarks/bench_tier.py::bench_promote_live_tensor`,
`bench_tier_pod_20260812_212555.log`, n=20 synthetic 4MB shards:

| | P50 | P95/P99 |
|---|---|---|
| `pin=False` | 684.2µs | 29,831.9µs |
| `pin=True` | 194.1µs | 82,668.1µs |

`pin=True` is ~3.5× faster at P50, meeting the "within 1.5× theoretical
bandwidth" acceptance target (`0.000732s` threshold) on the metric that
target is actually defined against. P95/P99 go the other way for
`pin=True`, but at n=20 that tail isn't a statistically meaningful
measurement (noted in README, not re-litigated here).

---

## 3. Open items for the paper / PoC Final Report (Sprint 5, §A2/§B2)

- The five-way invariance in §0 (platform × hardware × quantization × data
  path, all within 0.7pp) is the strongest single evidence for the paper's
  Evaluation section — stronger framed as one consolidated claim than as
  five separate percentages scattered across README/LOGBOOK as they were
  before this update.
- §2.1's shadow-activation/latency correlation (r=0.993) is a genuine,
  previously-uncited performance-cost finding — belongs in the paper, not
  just in the GCSG technical report's §6.1 where it currently lives alone.
- No equivalent MMLU number exists yet for path 1 (`_ShadowExpertINT4`) —
  Sprint 4 sub-goal 6 verified it mechanically under real offload but never
  ran MMLU against it (see README, Sprint 4 sub-goal 6). Out of scope for
  this update; flagged as a real gap, not silently omitted.

---

## Appendix A: Per-subject accuracy, Sprint 3 baseline run (411/570, 72.11%)

Unchanged from the original report — still the correct table for this
specific run (row 1 in §0).

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

- Sprint 3 baseline: `mmlu_results_overnight_20260811.jsonl`, `LOGBOOK.md` 2026-08-10/11 entries
- Sprint 4 cross-hardware: `LogBook_20260812_1344/SUMMARY.md`, `LogBook_20260812_1344/mmlu_sliced_run/`, `LogBook_20260812_1344/mmlu_burn_singleshot/burn_singleshot.log`
- Sprint 4 TierManager-wired (AWQ): `mmlu_tier_manager_fetta0_20260812_183539.jsonl`, `mmlu_tier_manager_pod_20260812_194757.jsonl`, `mmlu_tier_manager_pod_fetta1_rerun2_20260812_215033.jsonl`, `mmlu_tier_manager_pod_singleshot_20260812_195140.jsonl` (+ `..._rerun_20260812_210821.jsonl`, `..._rerun3_20260812_215416.jsonl`)
- Sprint 4 TierManager-wired (Marlin): `marlin_mmlu_20260812/mmlu_marlin_fetta1_20260813_004902.jsonl`, `marlin_mmlu_20260812/mmlu_marlin_singleshot_20260813_005201.jsonl`, `LOGBOOK.md` 2026-08-13 "Tekniska, continued" entry
- Promotion latency: `bench_tier_pod_20260812_212555.log`
- Full narrative/investigation trail: `LOGBOOK.md`, `osx-poc/reports/gcsg_shadow_execution_report.md` §6/§9
- Issues: [#10](https://github.com/danielesalpietro/vMemoryFabric/issues/10), [#16](https://github.com/danielesalpietro/vMemoryFabric/issues/16), [#17](https://github.com/danielesalpietro/vMemoryFabric/issues/17) (all closed)
- Upstream vLLM/WSL2 confirmation: [vllm-project/vllm#1084](https://github.com/vllm-project/vllm/issues/1084), [vllm-project/vllm#37883](https://github.com/vllm-project/vllm/issues/37883)
