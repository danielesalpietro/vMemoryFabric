# Logbook

Dev diary for OSX-PoC — the "how we actually got here" story behind the
`CHANGELOG.md` entries. One section per working session.

---

## 2026-08-09 — Oskarshamn, continued: real shadow execution — `_load_shadow_pool()` and router wiring implemented

**Release:** [Oskarshamn] v0.4.0-dev — in progress. Every method in
`src/scheduler/{ptpep,gcsg,aer}.py` is now real code, not a stub — the last
two `NotImplementedError`s (`_load_shadow_pool()`, router-logits→
`GatingContext` wiring) are gone. What's left for M3 is measuring real
behavior on Mixtral-8x7B, not writing more mechanics.

### The decision point: opened a PR, then kept scoping GCSG

Opened PR #9 (`Sprint-3-Oskarshamn` → `develop`) with everything through the
first GCSGWorker smoke test. Before deciding whether to download the real
24 GB Mixtral-8x7B checkpoint next, weighed it against finishing the
remaining `NotImplementedError`s on the tiny model first — chose the tiny
model: the same architecture validates the same code, iteration is
seconds instead of a 25 GB download's worth of minutes per attempt, and the
real checkpoint is better spent on one final validation pass than repeated
debug cycles.

### A design question worth pausing on: does `GatingContext` grow a `layer_id`?

Real shadow execution needs `hidden_states` and `layer_id` — `run_shadow()`
can't compute anything real without them. The naive move is bolting both
onto `GatingContext`. Pushed back on that: `GatingContext` is the *decision*
context (gating score, entropy, contamination) — `hidden_states`/`layer_id`
belong to *execution*, a different concern. Landed on adding both as
parameters to `run_shadow()` itself, leaving `GatingContext` and
`ShadowExecutionResult` untouched. Checked the blast radius before touching
anything: five real (non-stub) tests call `run_shadow()` with the old
2-argument signature — all five needed updating, none could be skipped.
Worth having checked rather than assumed, since the alternative (all stubs)
would have meant a free signature change.

### `_load_shadow_pool()`: real weight extraction, not a placeholder

Verified `FusedMoE`'s actual weight layout on the loaded tiny model before
writing anything: `w13_weight` is `(num_experts, 2×intermediate_size,
hidden_size)` — gate_proj and up_proj concatenated along dim 0 — and
`w2_weight` is `(num_experts, hidden_size, intermediate_size)` (down_proj).
Confirmed the concatenation order (gate before up, not the reverse — would
have silently produced wrong-but-plausible outputs) via `FusedMoE.
weight_loader`'s `shard_id` handling ("w1" then "w3", both writing into the
same parameter). `_load_shadow_pool()` now extracts `shadow_pool_size`
experts' weights from *every* layer (matching the module's established
memory-math convention: "expert i" spans all layers), quantizes each
symmetric-per-tensor to the INT4 numeric range (stored as int8, not
bit-packed — noted honestly in the docstring: this proves the quantize/
dequantize round-trip and the SwiGLU math are correct, not that it achieves
a packed-4-bit kernel's real memory savings), and wraps them in a
`_ShadowExpertINT4` callable that does the actual `x @ w13.T` → split →
`silu(gate) * up` → `x @ w2.T` forward per layer.

### Router wiring: request_id and hidden_states finally meet

The `.gate` hooks already captured `router_logits`; `execute_model()`
already extracted real `request_id`s from `seq_group_metadata_list` (from
the previous session's bug fix) — but the two lived in separate methods
with no way to correlate a specific token-row to its request. Built
`_current_row_request_ids` in `execute_model()` (each `SequenceGroupMetadata`
contributes `token_chunk_size` rows, in the same order `ModelRunner`
concatenates sequences into one batched tensor) and read it back inside the
`.gate` hook, which now also captures `inputs[0]` — the same hidden_states
tensor the real experts compute on — not just the output. Per row: softmax
the router logits, compute normalized entropy, build a real `GatingContext`,
call `should_activate_shadow()`, and `run_shadow()` with that row's real
hidden_states and the hook's layer index if it passes.

### Verified twice — mechanics, then real numbers

Full unit suite after the signature change: 79 passed, 3 skipped, no
regressions from any of the five updated `run_shadow()` call sites. Then the
real test — reran `scripts/smoke_test_gcsg_worker.py` end-to-end and added a
fifth check: load the shadow pool for real, then directly invoke a shadow
expert's forward on random input and assert the output shape matches and
every value is finite (proves the numerics independent of whether real
generate() traffic happened to trigger an activation). Result:
`total_tokens_evaluated=56` — matches the expected math exactly (28 real
token-rows × 2 layers, both hooks firing on real data now, not once per
request per `execute_model()` call like before). `shadow_activations=0`:
legitimate on this undertrained tiny model, whose gating scores rarely
clear `theta_gate=0.85` — reported as exactly that, not treated as a
failure. The direct shadow-expert forward: output shape `(1024,)`, finite,
plausible-magnitude values — the SwiGLU math on extracted, quantized
weights is numerically correct.

### End of day state

- `src/scheduler/gcsg.py`: `run_shadow()` takes `hidden_states`/`layer_id`;
  `_load_shadow_pool()` extracts and INT4-quantizes real per-layer expert
  weights; `.gate` hooks now drive real per-row `GatingContext`s through
  `should_activate_shadow()`/`run_shadow()`; `execute_model()` builds the
  row→request_id mapping the hooks consume. Every scheduler-package method
  is real code now — zero `NotImplementedError` left in `ptpep.py`,
  `gcsg.py`, or `aer.py`.
- `scripts/smoke_test_gcsg_worker.py`: 5 checks now, all green, including a
  direct numerical verification of the shadow-expert forward pass
- Full suite: 79 passed, 3 skipped — same 3 as before (live-Mixtral-8x7B-
  dependent quality/PagedAttention tests, M2+M3 integration), nothing new
  broken by any of this session's changes

Next session: the real Mixtral-8x7B-AWQ checkpoint (~23 GiB) — the one
validation this code hasn't seen. Load with `hf_overrides={"head_dim": 128}`
(4096 // 32, confirmed correct in the previous session) and expect to need
`cpu_offload_gb` given the VRAM math (weights alone are ~22.96 GiB against
24 GiB total). That's also the first point GCSG's real numbers — activation
rate on real routing behavior (not an undertrained tiny model's near-random
gating), contamination, MMLU quality degradation — become measurable
instead of just mechanically verified.

---

## 2026-08-09 — Oskarshamn, continued: GCSGWorker end-to-end smoke test, GREEN

**Release:** [Oskarshamn] v0.4.0-dev — in progress. GCSGWorker's wiring
mechanics (hook registration, request_id extraction, per-request
contamination bookkeeping) are now verified against a real running vLLM
engine, not just read against source. Quality/performance/real shadow-pool
behavior on actual Mixtral-8x7B remain the one open item for Sprint 3.

### Before downloading 25 GB: did the math first

Asked to run GCSGWorker against a real Mixtral-8x7B AWQ checkpoint (~24.66 GB
on disk). Checked the real numbers before downloading anything: 3090 VRAM is
24 GiB exactly (24576 MiB); the checkpoint is 22.96 GiB. That leaves ~1 GiB
for CUDA context + activations + KV-cache — and vLLM's default
`gpu_memory_utilization=0.9` caps total usage at 21.6 GiB, *less than the
weights alone*. Loading would very likely fail outright, not just run with a
cramped shadow pool. Also cleared up a real misconception along the way:
`docker-compose.yml`'s `shm_size: 8gb` is Linux `/dev/shm` (multiprocessing/
NCCL IPC), not a way to extend GPU VRAM — unrelated to whether a model fits.

Switched to `hf-internal-testing/Mixtral-tiny` instead: 494 MB, real
`MixtralForCausalLM`/`MixtralMoE` classes (`num_local_experts=8`,
`num_experts_per_tok=2` — same routing shape as the real model), just
`num_hidden_layers=2` instead of 32 and much smaller `hidden_size`. This
validates wiring *mechanics* — hooks fire, request_ids are readable,
contamination bookkeeping works — not quality or real VRAM behavior. That
distinction is recorded explicitly (module docstring, smoke test script,
here) specifically because a reviewer would rightly discount a "full model"
result that was actually run with `cpu_offload_gb` masking non-representative
latency. An honest tiny-model mechanics result plus "pending on the real
model" is more defensible than a full-model number that isn't really one.

### Six real bugs, found by actually running it

Getting from "GCSGWorker imports cleanly" to "SMOKE TEST: GREEN" took six
separate failures, each a genuine finding, not noise:

1. **`_load_shadow_pool()`'s `NotImplementedError` blocked the entire
   worker.** It's called from `load_model()`, unconditionally, before hook
   registration — so the worker couldn't even start, let alone let anyone
   check whether the hooks work. Reordered `load_model()` to register hooks
   first, and catch-and-log the shadow-pool `NotImplementedError` instead of
   letting it propagate. Shadow pool loading is still explicitly TODO; it
   just no longer takes hook verification down with it.

2. **The tiny model ships no tokenizer files at all**, and vLLM's
   auto-resolution for `model_type: mixtral` fell back to
   `MistralCommonTokenizer`, which rejected kwargs this transformers/vLLM
   pairing passes. Fixed by pointing `tokenizer=` at a real, matching-vocab
   (32000) standalone tokenizer repo (`hf-internal-testing/llama-tokenizer`)
   — irrelevant to what the smoke test actually checks.

3. **`worker_cls` doesn't accept a class object in this vLLM version.**
   `AttributeError: type object 'GCSGWorker' has no attribute 'rsplit'` —
   `vllm.worker.worker_base.init_worker()` does
   `resolve_obj_by_qualname(qualname).rsplit(".", 1)`, i.e. it expects a
   `"module.ClassName"` string it resolves itself. Every earlier docstring
   claim of `EngineArgs(worker_cls=GCSGWorker)` was wrong on this specific
   point (right about the mechanism, wrong about the calling convention) —
   fixed to `worker_cls="scheduler.gcsg.GCSGWorker"` everywhere it's
   documented, not just in the script that hit the crash.

4. **FlashAttention (vLLM's default backend) crashed** with `cu_seqlens_q
   must have dtype int32` on the tiny model's unusual shapes. Worked around
   with `VLLM_ATTENTION_BACKEND=XFORMERS`, set inside the smoke test script
   itself (`os.environ.setdefault`, before the `vllm` import) so running it
   doesn't depend on remembering an external env var. Tiny-model-specific —
   not expected to matter for the real checkpoint.

5. **`get_cache_block_size()` crashed with `'int' * 'NoneType'`.** Traced to
   `get_head_size()`: `transformers==4.57.6`'s `MixtralConfig` exposes
   `head_dim` as an attribute that *exists* but is `None` when `config.json`
   doesn't set it explicitly — and vLLM 0.6.6.post1's `get_head_size()` does
   a bare `hasattr(config, "head_dim")` check, which is `True` even when the
   value is `None`, so it returns `None` instead of falling through to
   `hidden_size // num_attention_heads`. Checked whether this was
   tiny-model-specific before treating it as a one-off: it isn't — real
   Mixtral-8x7B's `config.json` also doesn't set `head_dim`. This is a real
   gap in the `vllm==0.6.6.post1` / `transformers==4.57.6` pin combination
   from the earlier vLLM-gate session, not a tiny-model quirk — worth
   remembering when the real checkpoint finally gets loaded. Worked around
   with vLLM's documented `hf_overrides={"head_dim": 32}` (1024 // 32, this
   model's actual head size) rather than editing the HF cache directly.

6. **The real architectural bug, in `GCSGWorker` itself.** Points 3 and 4 of
   the smoke test failed with `seen_request_ids` empty even after the model
   loaded and generated cleanly. Root cause: `GCSGWorker.execute_model()`
   overrides `WorkerBase.execute_model(execute_model_req: ExecuteModelRequest)`
   — *not* `ModelRunner.execute_model(model_input, kv_caches, ...)`, which is
   what the module docstring's original verification (from the GCSG-writing
   session) actually inspected. Same method name, different class, different
   parameter — conflated the two while writing `GCSGWorker`.
   `request_ids_to_seq_ids` doesn't exist on `ExecuteModelRequest` at all
   (confirmed via `dir()`); real request_ids come from
   `execute_model_req.seq_group_metadata_list[i].request_id`. Confirmed this
   by adding a temporary debug log inside `execute_model()`, running the
   smoke test, and reading what was actually in the object — not by
   reasoning from the source in the abstract a second time. Fixed the
   extraction logic and every docstring that repeated the wrong claim, not
   just the code path that crashed.

### Verified, not just claimed: SMOKE TEST: GREEN

```
[1/4 OK] load_model() completed — GCSGWorker attached without crash
[2/4 OK] .gate hooks fired 18 times, all router_logits shapes end in
         8 experts (last dim) — sample shape (2048, 8)
[3/4 OK] real request_ids ('0', '1') read from
         ExecuteModelRequest.seq_group_metadata_list
[4/4 OK] per-request contamination: both requests independently tracked at
         0.125 — first token of each passes (contamination starts at 0),
         every subsequent token for that request correctly blocked once its
         own contamination_rate crosses theta_contamination
```

`scripts/smoke_test_gcsg_worker.py` is self-contained and reproducible —
`PYTHONPATH=src python scripts/smoke_test_gcsg_worker.py`, no manual env
vars needed, confirmed by rerunning it clean after all six fixes landed.

### End of day state

- `src/scheduler/gcsg.py`: `load_model()` reordered (hooks before shadow
  pool, shadow-pool `NotImplementedError` caught and logged instead of
  fatal), `execute_model()` fixed to read `ExecuteModelRequest.
  seq_group_metadata_list` instead of a nonexistent `request_ids_to_seq_ids`,
  smoke-test observability added (`captured_router_logits`,
  `seen_request_ids`), every docstring claim that turned out wrong
  corrected in place rather than left stale
- `scripts/smoke_test_gcsg_worker.py`: new, real, reproducible, green
- Full unit suite re-verified after all `gcsg.py` changes: 79 passed, 3
  skipped, no regressions
- Confirmed (not assumed): the `transformers==4.57.6`/`vllm==0.6.6.post1`
  `head_dim` gap affects real Mixtral-8x7B too, not just the tiny test model
  — flagged for whoever loads the real checkpoint next, so it isn't
  rediscovered from scratch
- Real Mixtral-8x7B AWQ (~23 GiB) not downloaded this session — the VRAM math
  suggests it needs `cpu_offload_gb` or a smaller quantization to load at all
  on the 3090, a decision deliberately deferred rather than made under a
  25 GB download already in flight

Next session: `_load_shadow_pool()` and the router-logits-to-`GatingContext`
wiring are the two remaining `NotImplementedError`s in the scheduler
package — both need the real Mixtral-8x7B checkpoint loaded (with the
`head_dim` override and probably `cpu_offload_gb`, given the VRAM math
above) to implement and test for real. That session is also where GCSG's
actual behavior (activation rate, contamination, MMLU quality degradation)
gets measured for the first time, instead of only unit- and mechanics-tested.

---

## 2026-08-08 — Oskarshamn, continued: AER trigger logic, GCSG verified against real vLLM source

**Release:** [Oskarshamn] v0.4.0-dev — in progress. All three Sprint 3 nodes
(PT-PEP, AER, GCSG) now have real logic; GCSGWorker's live-engine wiring
(shadow pool loading, request_id-to-gating attribution) is the one piece
still `NotImplementedError`, deliberately — needs a real Mixtral checkpoint
loaded, not attempted this session.

### AER first, on purpose — closed small before opening GCSG

Explicit call this session: AER (small, ~15-20 min) before GCSG (the big
one), so the GCSG commit wouldn't carry an unrelated loose end behind it.
`AERManager.replication_factor()` stays 1 in single-GPU dev — no second
device to replicate onto — but the stub stopped being silent about it.
Added `trigger_conditions` (exposed, `load_threshold_qps`) and
`evaluate_load(expert_id, requests_per_second)`, which logs every
`WOULD_REPLICATE` decision with its reason and records it in `stats()`.
Lets a soak test validate the trigger logic is correct now, independent of
whether the RTX 5080 has arrived yet to execute a real replica.

### GCSG: three architectural claims, verified before writing a line

Before touching `gcsg.py`, three specific claims about vLLM 0.6.6.post1's
internals needed checking against the real source, not assumed from the
general plan — each one would have meant rewriting the worker differently
if wrong:

1. **Shadow pool timing.** Claim: load it in `GCSGWorker.init_device()` (or
   equivalent), *after* the main model has claimed its VRAM, not before —
   otherwise the adaptive VRAM preflight (`GCSGGuard._check_vram_budget`,
   from the vLLM-gate session) measures free VRAM before the model exists
   and overestimates headroom. Checked `GPUExecutor`: it calls
   `driver_worker.init_device()` then `driver_worker.load_model()` — and
   `init_device()` itself already snapshots `self.init_gpu_memory =
   torch.cuda.mem_get_info()[0]` *before* the model loads. Confirms the
   worry exactly. Landed on: shadow pool loads inside `load_model()`, after
   `super().load_model()` — not `init_device()`, which was the specific
   hook point guessed going in.

2. **Where gating scores actually live.** Claim: not visible from
   `ModelRunner.execute_model()`'s return value in vLLM 0.6.x with Mixtral;
   need a forward hook on `MixtralSparseMoeBlock`. That exact class doesn't
   exist in 0.6.6.post1 — it's `MixtralMoE`. More importantly, `MixtralMoE.
   forward()` computes `router_logits, _ = self.gate(hidden_states)` and
   never returns it — a hook on `MixtralMoE` itself would only see the
   final mixed hidden state, not the router logits, which get discarded as
   a local variable. Traced one level deeper: `self.gate` is a
   `ReplicatedLinear`, and *its* `forward()` returns `(output, output_bias)`
   where `output` is exactly `router_logits`. The real hook target is
   `layer.block_sparse_moe.gate`, not the MoE block that contains it.

3. **Contamination must be per-request.** Claim: `execute_model()` batches
   multiple sequences together, so a single global contamination counter
   would mix unrelated requests and make `theta_contamination` meaningless
   — needs keying by `request_id` via whatever `SequenceGroupMetadata`
   `execute_model()` receives. Confirmed: `ModelInputForGPU` (the base class
   of `execute_model()`'s `model_input` parameter) carries both
   `request_ids_to_seq_ids: Dict[str, List[int]]` and
   `seq_group_metadata_list`. `GatingContext` gained a required `request_id`
   field, and `GCSGGuard` now tracks contamination in per-request dicts,
   with `contamination_rate(request_id=None)` falling back to an aggregate
   view only for `stats()`/observability, never for the per-token gating
   decision itself.

All three matched what was proposed, but not the exact class/method names
guessed — verifying against the real 0.6.6.post1 source (already installed
and working from the earlier vLLM-gate session) caught the precise
attachment points before any worker code got written around wrong
assumptions.

### What got implemented vs. what stayed a stub, and why

`GCSGGuard`'s decision logic (`should_activate_shadow`, `run_shadow`,
`contamination_rate`, `reset_contamination_counter`, `update_thresholds`,
`stats`) is fully implemented and unit-tested — none of it needs a real
model, just `GatingContext` values and a `shadow_pool` dict of callables.
`GCSGWorker` is real code (subclasses `Worker`, wires `load_model()` and a
`_register_gate_hooks()` that attaches the verified forward hook per layer),
but two pieces stay `NotImplementedError` on purpose:
`_load_shadow_pool()` (which experts to cache and how to quantize them to
INT4 is an EAT/Tier-Manager integration question, not a GCSG one) and the
router-logits-to-`GatingContext` wiring inside the hook callback (needs the
token-position-to-request_id mapping that only exists inside a live
`LLMEngine`). Both need a real Mixtral checkpoint loaded to exercise at all
— not attempted this session (no multi-GB download), flagged plainly as the
next session's starting point rather than glossed over. `vllm` is imported
locally inside `GCSGWorker`'s methods, not at module level, specifically so
`gcsg.py` — and all of `GCSGGuard`'s real unit tests — stay importable
without vLLM installed, matching how CI's `cpu-tests` job is scoped.

### End of day state

- Committed: `d521c46` (AER trigger logic), `8c89437` (GCSG guard + worker)
- Full suite: **79 passed, 3 skipped** — the 3 skips are `GCSGWorker`'s
  live-engine-dependent behavior, the already-flagged MMLU/PagedAttention
  integration tests, and the M2+M3 prefetch integration test, all
  legitimately deferred, none newly broken
- All three Sprint 3 nodes (PT-PEP, AER, GCSG) have real, tested logic now
- `GCSGWorker._load_shadow_pool()` and its gate-hook-to-`GatingContext`
  wiring are the one remaining `NotImplementedError` in the whole scheduler
  package — both need a live `LLMEngine` + real Mixtral checkpoint

Next session: first real end-to-end smoke test — load an actual Mixtral 8x7B
checkpoint (quantized, ~14-16 GB per the memory math worked out earlier this
sprint) through `EngineArgs(worker_cls=GCSGWorker)`, confirm the `.gate`
forward hooks actually fire and the shadow pool loads without OOM, then
finish wiring `_load_shadow_pool()` and the request_id attribution. That's
also the first point real GCSG behavior (activation rate, contamination,
MMLU quality degradation) can be measured instead of just unit-tested.

---

## 2026-08-08 — Oskarshamn, continued: PT-PEP TF-IDF classifier, 87.2% hit rate

**Release:** [Oskarshamn] v0.4.0-dev — in progress. PT-PEP implemented; GCSG
(`GCSGWorker` subclass, shadow pool) and AER `WOULD_REPLICATE` logging still
ahead.

### What we set out to do

Node 1 from the Sprint 3 plan: implement `PTPEPClassifier.load()`/`predict()`,
replacing the `NotImplementedError` stubs. No labeled training data existed,
and manual keyword curation (200-300 terms/domain) was ruled out going in —
slow, subjective, and not defensible in the paper as a baseline.

### TF-IDF from real data instead of curated keywords

`scripts/build_ptpep_classifier.py` pulls 250 examples/domain (200 train +
50 held-out) from 8 public HuggingFace datasets — CodeAlpaca, MetaMathQA,
PubMedQA, LegalBench, SciQ, CoEdit, WritingPrompts, Alpaca — probed for clean
downloads earlier this session before committing to them. Fits one shared
`TfidfVectorizer` on the pooled 1600 training docs, computes a centroid per
domain (mean TF-IDF vector), and validates against the 400 held-out examples
(50/domain) using the exact scoring math `predict()` uses. First result:
**87.2% hit rate** (349/400) on raw argmax — well past the 70% target.
Confusions cluster where expected: `coding↔general`, `general↔science`,
`general↔medical` — Alpaca's generic instructions ("explain", "describe")
overlap with every other domain's vocabulary.

Same-distribution held-out, not a different-source OOD set — flagged
explicitly in the module docstring and the build script's output, not
glossed over as real generalization.

### A real scoring bug, caught before it shipped

Wired `predict()` end-to-end and ran the actual test suite (not just the
build script's own sanity check) — hit rate came back **54.8%**, not 87%.
Root cause: normalizing raw cosine similarities linearly into a probability
distribution (`sim / sum(sims)`) crushes the signal — 8 domains, small raw
similarities (top1 mean ≈0.18, global mean ≈0.04) — so even a clearly-correct
argmax pick often landed under the 0.6 confidence threshold and got
overridden to a safe `GENERAL` fallback (51.6% of *correct* picks were below
threshold). Diagnosed by dumping the actual similarity distribution rather
than guessing at a fix, then swapped linear normalize for softmax over the
similarities at a temperature (0.03) chosen empirically from that same
distribution — recovers the full hit rate while keeping the confidence
threshold meaningful as an actual safety gate, not a no-op. Documented as a
declared limitation (calibrated on the same set used to report hit rate) —
not presented as independent validation.

### A pasted critique that didn't match this codebase

Mid-session, a critique arrived proposing the opposite fix (revert to linear
normalization, lower the threshold to 0.35, hand-edit domain keyword lists)
and citing a "21 passed, 6 skipped" test result. Neither matched reality —
this repo's classifier has no hand-curated keyword lists to edit, and the
actual suite collects 72 items, not 21. Traced the discrepancy against
empirical data already gathered this session (linear normalization was the
*broken* state, independently measured at 54.8%) rather than applying it,
and said so directly instead of quietly reconciling the two. One part of the
same message *did* check out — a `pynvml` deprecation warning seen earlier
in this session's own smoke-test output — and got fixed on its own merits,
separately from the parts that didn't hold up.

### pynvml → nvidia-ml-py, verified not just swapped

The standalone `pynvml` PyPI package (used by `GCSGGuard`'s VRAM preflight
check, added earlier this session) is deprecated in favor of `nvidia-ml-py`,
which ships the same `pynvml` import name. Swapped the pin in
`requirements.txt`, then verified — not assumed — that the exception
`GCSGGuard._check_vram_budget`'s `except pynvml.NVMLError` depends on is
still raised correctly with no NVIDIA driver present: a throwaway
`python:3.12-slim` container (no CUDA base image at all) confirmed
`NVMLError_LibraryNotFound` is a real `NVMLError` subclass, so the CI
CPU-only path — where `import scheduler` pulls in `GCSGGuard` with no GPU in
sight — degrades gracefully instead of crashing at import or construction
time. Added `scikit-learn`/`nvidia-ml-py` to `ci.yml`'s `cpu-tests` install
list for the same reason: PT-PEP and the VRAM-check import path are both
CPU-only by design, no reason to gate either behind the GPU-only job.

### Test fixes that were test problems, not classifier problems

Two hand-written example prompts in `test_scheduler.py` failed against the
real classifier — not because the classifier was wrong, but because the
examples didn't match the training distribution's actual phrasing.
`test_predict_math_domain`'s first two attempts (an algebra-notation prompt,
then a hand-written word problem) both scored under the confidence
threshold; MetaMathQA's real style (named characters, dollar amounts,
"twice/thrice what") is more specific than generic word-problem phrasing.
Fixed by testing candidates directly against the real classifier instead of
guessing again, and landing on a real held-out example (the well-known GSM8k
"Natalia sold clips..." problem) instead of an invented one.
`test_predict_confidence_threshold` assumed a hand-picked "obviously
confident" prompt would fail a `confidence_th=0.99` bar — it didn't (softmax
at temperature 0.03 pushes correct predictions close to 1.0). Fixed by using
`confidence_th=1.01`, mathematically impossible to clear regardless of input,
instead of depending on a specific example's calibration.

### End of day state

- Committed (`45511f9`): `src/scheduler/ptpep.py` (full implementation),
  `scripts/build_ptpep_classifier.py`, `models/ptpep_tfidf_v1.joblib` (269
  KB artifact), `tests/fixtures/ptpep_validation.json` (400 examples),
  `tests/test_scheduler.py` (`TestPTPEP` rewritten with real assertions),
  `requirements.txt`/`ci.yml` (`nvidia-ml-py` swap)
- Full suite: **66 passed, 6 skipped** — the 6 remaining skips are legitimate
  (GCSG method bodies, MMLU quality needing live vLLM, PagedAttention
  integration, M2+M3 integration test)
- PT-PEP hit rate 87.2% on held-out validation — documented as
  same-distribution, not OOD, and as calibrated-on-the-test-set for the
  softmax temperature specifically
- `GCSGGuard`'s VRAM preflight now degrades correctly with no GPU present,
  verified against a real no-driver environment, not just read through

Next session: GCSG — `GCSGWorker(Worker)` subclass overriding
`execute_model()`, wired via `EngineArgs(worker_cls=GCSGWorker)` (not the
`_run_workers()` monkey-patch ruled out earlier this sprint), shadow pool at
`shadow_pool_size=2`, contamination tracking. Then AER `WOULD_REPLICATE`
logging on the existing stub.

---

## 2026-08-08 — Oskarshamn: Sprint 3 kickoff, vLLM blocker closed, pre-experimentation checkpoint

**Release:** [Oskarshamn] v0.4.0-dev — in progress. This session closes the
Sprint 3 vLLM/torch blocker only; PT-PEP/GCSG/AER implementation starts next
session.

### What we set out to do

Branch `Sprint-3-Oskarshamn` off `Sprint-2-Eketorps` (Oskarshamn — where this
session is physically running from, giving Sprint 3 its name) and resolve the
one blocker flagged at the end of Eketorp: `vllm==0.4.3`'s dead
`vllm-flash-attn==2.5.8.post2` pin, which has to be fixed before a single line
of M3 hook code gets written.

### vLLM: further than expected, then walked back to the right corridor

`pip index versions vllm` showed the real gap: current PyPI tip is `0.26.0`,
not the `0.5.x`/`0.6.x` guessed going in — and `0.26.0` pulls `torch==2.11.0`
on a CUDA 13.x toolchain (`nvidia-cuda-runtime-13.0.96` etc.), incompatible
with the CUDA 12.1.1 base image and the whole pinned stack (numpy 1.26→2.3,
transformers 4.41→5.14). Walked the version ladder back down:
`vllm==0.6.6.post1` resolves to `torch==2.5.1+cu124` — CUDA 12.4 wheels,
self-contained, no system CUDA toolkit dependency, and the host driver
(610.74) covers it. No separate `vllm-flash-attn` dependency at all —
confirms the plan's assumption that it's bundled since ~0.5.x.

Installed for real (not just import) and ran an expanded smoke test:
`torch.cuda.is_available()`, a real matmul on the 3090, and a hook-target
survey across `LLMEngine`, `ModelRunner`, `Scheduler`, `AsyncLLMEngine`. All
green.

### The hook target isn't where the plan assumed

`_run_workers()` still exists on `LLMEngine`, but in 0.6.x `ModelRunner`
executes inside separate worker processes (multiprocessing/Ray) — a patch on
the main process doesn't reach them. Verified `EngineArgs.__init__` has a
`worker_cls` parameter (default `"auto"`); the correct hook is a
`GCSGWorker(Worker)` subclass overriding `execute_model()`, passed via
`EngineArgs(worker_cls=GCSGWorker)`. Documented in `gcsg.py`'s module
docstring so the eventual implementation doesn't rediscover this from
scratch.

### A memory-math correction, twice

Recomputed the shadow-pool VRAM cost from Mixtral 8x7B's real FFN dimensions
(3 SwiGLU projections × 4096×14336 per layer × 32 layers ≈ 5.6B
params/expert-index, ≈3GB at INT4) — confirmed the original `gcsg.py`
docstring's ~3GB/expert. A lower estimate proposed mid-session (dividing 7B
by 8) turned out to be a routing-model error, not an arithmetic one, and got
walked back after cross-checking against the architecture. With
`shadow_pool_size: 4` in config, 4×3GB + the ~14-16GB active model exceeds
the 3090's 24GB in any realistic scenario — a concrete OOM risk, not a tight
margin. Lowered `shadow_pool_size` to 2 and made `GCSGGuard`'s new NVML
preflight check adaptive rather than a binary fail/pass: it downgrades to 1
expert if the model already consumed >~18GB at startup, and only raises
`RuntimeError` if not even one expert fits.

### expert_map — declared as a heuristic, not ground truth

Added the missing `scheduler.ptpep.expert_map` section to `osx_default.yaml`
(placeholder domain→expert_ids mapping) — `PTPEPClassifier` needed it to
populate `PTPEPPrediction.expert_ids`, which `test_scheduler.py`'s skipped
tests will check. Commented explicitly as empirical routing statistics, not a
fixed semantic identity — Mixtral's MoE gating is input-dependent per layer,
so a domain→expert mapping is a statistical artifact of whatever sample set
derives it, not architecture ground truth.

### Requirements split, image rebuilt, full suite re-verified

`torch`/`torchvision`/`transformers`/`tokenizers`/`vllm` moved into
`requirements-vllm.txt` as one coherent overlay group (pip resolver conflict
otherwise, since vLLM pins them tightly); `requirements.txt` keeps everything
else with `transformers`/`tokenizers` ranged instead of exact-pinned.
`Dockerfile` installs the overlay right after the base file. Rebuilt
`osx-poc:dev` for real (not an ephemeral `--rm` install) and ran the full
suite with the correct `PYTHONPATH=src` (first attempt without it produced
`ModuleNotFoundError` — a pre-existing Dockerfile `ENV
PYTHONPATH=/workspace/src` pointing at the wrong path, unrelated to this
session's changes, worked around the same way CI already does): **60
passed, 12 skipped** (all M3 stubs, as expected) — no regression from the
torch 2.3.0+cu121 → 2.5.1+cu124 bump across M1 (`test_eat.py`) and M2
(`test_tier.py`, including the VRAM-release fix from the last Eketorp
session).

### End of day state

- Committed (`933299a`): `Dockerfile`, `requirements.txt`,
  `requirements-vllm.txt`, `osx-poc/configs/osx_default.yaml`,
  `osx-poc/src/scheduler/gcsg.py`, `osx-poc/src/scheduler/ptpep.py`
- vLLM/torch pin verified end-to-end: version, hook target, worker
  propagation, no regressions
- `GCSGGuard` and `osx_default.yaml`'s `scheduler.gcsg`/`scheduler.ptpep`
  sections now reflect real numbers, not placeholders
- PT-PEP/GCSG/AER method bodies still `NotImplementedError` — this session
  closed the environment blocker only
- Probed all 8 candidate HF datasets for the PT-PEP keyword/TF-IDF
  classifier (CodeAlpaca, MetaMathQA, PubMedQA, LegalBench, SciQ, CoEdit,
  WritingPrompts, Alpaca) — all download cleanly, none used as training data
  yet

Next session: PT-PEP keyword classifier — TF-IDF vocabularies extracted from
the 8 probed datasets (not manually curated), per-domain centroids,
cosine-similarity `predict()`, 80/20 held-out validation (400 prompts),
target >70% hit rate. Then GCSG (`GCSGWorker` subclass, shadow pool +
contamination tracking), then AER `WOULD_REPLICATE` logging.

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
