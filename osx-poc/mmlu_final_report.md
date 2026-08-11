# MMLU Final Report — Sprint 3 (Oskarshamn), full coverage with real shadow execution

**Date:** 2026-08-11
**Model:** `casperhansen/mixtral-instruct-awq` (Mixtral-8x7B-Instruct, AWQ, `quantization="awq_marlin"`)
**Runtime:** vLLM 0.6.6.post1, `GCSGWorker` (real shadow execution — issues #10/#16), `cpu_offload_gb=4`, `gpu_memory_utilization=0.95`, `max_num_seqs=16`, `enforce_eager=True`, `max_model_len=3328`
**Config file:** `scripts/eval_mmlu_gcsg.py` (unmodified defaults)
**Results file:** `mmlu_results_overnight_20260811.jsonl`
**Orchestration:** `scripts/run_mmlu_in_slices.sh` — 18 slices of 32 prompts each, one fresh Docker container/`GCSGWorker` per slice, `--watchdog-timeout 2700`, external `timeout 3000` (raised 2026-08-11, see `LOGBOOK.md`)

---

## Summary

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

This is the first complete, defensible MMLU-5shot accuracy number this project has measured with real shadow execution active end-to-end (not hook-only, not partial/skip-and-continue coverage). Every one of the 570 questions across all 57 MMLU subjects was evaluated in a single overnight run, with zero slice failures.

**Correction (2026-08-11, post-review):** the shadow-activations figure originally reported here was 13,756 — the value of `shadow_activations_cumulative` on the *last* of the 18 result rows, mistaken for a run-wide total. Each slice runs in its own fresh Docker container/`GCSGWorker`, so that counter resets to 0 at the start of every slice and only ever reflects *that slice's* own count — it never accumulates across slices. The correct run-wide total is the **sum across all 18 rows: 562,338**, recomputed directly from `mmlu_results_overnight_20260811.jsonl`. Note this total has no companion `total_tokens_evaluated` per slice in the results file, so an aggregate shadow-activation *rate* (activations / tokens evaluated) cannot be derived from this file alone — recompute from a rerun with that field logged if the rate is needed.

## Quality target

README's non-functional target: **GCSG quality degradation < 2% (MMLU-5shot)**, measured against a hook-only baseline of **72.3%** (recorded in earlier `LOGBOOK.md` sessions).

```
72.3%  (hook-only baseline)
72.11% (real shadow execution, this run)
------
-0.19 percentage points
```

**Target met**, with a wide margin (~0.19pp actual vs. 2pp allowed).

## Timing note

The slowest slice (`[32:64)`, containing the previously-flagged position 33) took ~30 minutes — far above the old 250s/300s watchdog/timeout that would have killed it, comfortably inside the 2700s/3000s values raised this session. This is expected and understood: `LOGBOOK.md` (2026-08-11 entry) root-causes the variable slowdown to `maybe_offload_to_cpu()`'s pageable-memory CPU→GPU swap-in under WSL2 (`pin_memory=False`), a structural, upstream-acknowledged platform limitation (vllm-project/vllm#1084, #37883) — not a bug in this project's code, and not a deadlock. No slice came close to actually hanging.

## Per-subject accuracy (10 questions each)

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

Lowest-scoring subjects (STEM-heavy, expected for a 4-bit quantized model at this size): `abstract_algebra`, `college_physics`, `electrical_engineering`, `formal_logic`, `high_school_mathematics` (all 40.0%). Highest: `high_school_government_and_politics`, `human_aging`, `international_law`, `logical_fallacies`, `medical_genetics`, `world_religions` (all 100.0%).

## References

- Full investigation trail: `LOGBOOK.md`, 2026-08-10 and 2026-08-11 entries
- Root-cause analysis of the batch-slowdown mechanism: `LOGBOOK.md`, 2026-08-11
- Issues: [danielesalpietro/vMemoryFabric#10](https://github.com/danielesalpietro/vMemoryFabric/issues/10), [#16](https://github.com/danielesalpietro/vMemoryFabric/issues/16)
- Upstream vLLM/WSL2 confirmation: [vllm-project/vllm#1084](https://github.com/vllm-project/vllm/issues/1084), [vllm-project/vllm#37883](https://github.com/vllm-project/vllm/issues/37883)
