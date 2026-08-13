# Changelog



All notable changes to OSX-PoC are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning follows [Semantic Versioning](https://semver.org/).



Releases are named after cities where significant project milestones were reached.



---



## [Oskarshamn] — v0.4.0-dev — 2026-08-10 (in progress, not yet released)

*Named after Oskarshamn, Sweden, where M3 — Expert Scheduler work is happening.*

### Added

**M3 — GCSG shadow execution, real (not stub)**

- `src/scheduler/gcsg.py` — `GCSGWorker._load_shadow_pool()` now actually populates the shadow pool against the real Mixtral-8x7B-Instruct checkpoint, for both quantization paths: AWQ ModuleList (`_pin_awq_expert_to_gpu()`, direct `.to('cuda')` per expert) and Marlin-packed AWQ (`_build_marlin_shadow_pool()` + new `_PinnedMarlinExperts` proxy class, GPU-resident slices of the shared `w13_qweight`/`w2_qweight`/etc. tensors, built only for layers actually offloaded to CPU — resident layers reuse the original module, zero extra allocation)
- `_MarlinFusedShadowExpert` — constructor changed from one `(expert_id, num_experts)` per instance to per-layer `(fused, expert_id, num_experts)` entries, since offloaded layers now route through the 2-expert pinned proxy (local index) while resident layers use the original 8-expert tensor (global id)
- First real, non-zero `GCSGGuard.stats()["shadow_activations"]` this project has ever measured — every prior run had shadow execution stubbed to hook-only

### Fixed

**GCSGWorker — shadow-execution call on a CPU-offloaded expert crashes (issues [#10](https://github.com/danielesalpietro/vMemoryFabric/issues/10), [#16](https://github.com/danielesalpietro/vMemoryFabric/issues/16))**

- Root cause: `_load_shadow_pool()` registered shadow experts by reference into vLLM's normal `cpu_offload_gb` mechanism, which wraps only whole decoder layers' `.forward` in an async (`non_blocking=True`) H2D copy — never individual expert submodules. A direct, out-of-sequence call into an expert that happens to be CPU-resident at that moment bypasses the real swap-in path entirely and crashes with `CUDA error: illegal memory access`. Reproduced identically on both the Marlin path (original issue #10) and the AWQ ModuleList path (new issue #16, found while investigating #10) — one shared mechanism, two code paths
- Fix: pin every shadow-pool expert explicitly GPU-resident at load time (see Added above), verified numerically (`scripts/verify_marlin_pinned_proxy.py`, proxy vs. same-layer original, rtol=1e-2/atol=1e-3, PASS) and end-to-end (`scripts/verify_shadow_pool_pinning_e2e.py`, both quantization paths, PASS)
- A real allocator-fragmentation hang was hit and fixed mid-implementation: the first attempt built the pinned Marlin proxy for all 32 layers instead of only the 6 offloaded ones (192 small allocations vs. 36, on an already-near-full 24GiB model) — GPU pegged at 100%, 9 minutes with zero progress, confirmed via `nvidia-smi` and killed rather than waited out. Fixed by scoping proxy construction to `w13_qweight.device.type == "cpu"` layers only

**GCSGWorker — 1D vs. 2D `hidden_states` in `_evaluate_gcsg_for_rows()`**

- `hidden_states[row_idx]` collapsed the row dimension to a 1D tensor; `_MarlinFusedShadowExpert` assumes 2D and misreads `hidden_dim` as the row count, crashing several calls later inside vLLM's own `fused_topk` — the error surfaced far from its actual cause. Fix: `hidden_states[row_idx : row_idx + 1]` (slice, preserves 2D shape). Only reachable once shadow execution actually ran past `should_activate_shadow()` for the first time this session — predates this session's other changes entirely, just never exercised before

**`_evaluate_gcsg_for_rows()` — per-row CUDA sync batched into a single per-tensor sync**

- `gating_scores=probs[row_idx].tolist()` / `token_entropy=float(entropy[row_idx])` previously ran inside the per-row loop, forcing a blocking GPU→CPU sync twice per row per layer (32 layers × batch_size per token). Hoisted `.tolist()` above the loop — same data, same downstream logic, 2 syncs per hook call instead of 2×batch_size. Tested directly as a candidate cause of the stall below (a plausible, code-verified hypothesis, not a guess) and **disproven** — `[32:48)` at batch=16 stalled identically with the fix in place. Kept regardless as a real, independent efficiency improvement

**`captured_router_logits` unbounded growth**

- Fired on every `.gate` hook call, every layer, every forward pass, holding live GPU tensor references for the full process lifetime — a real bug (tens of thousands of un-freeable tensors over a full 570-prompt run), fixed with a hard cap (`_MAX_CAPTURED_ROUTER_LOGITS = 1000`). Hypothesized as the cause of a separate reproducible stall (below); re-tested with the cap in place and the stall recurred identically — kept as a legitimate independent fix, explicitly **not** claimed as the stall's resolution

### Known issue — not fixed this release

**Reproducible stall under certain concurrent batch compositions (issues #10/#16, reopened scope)**

Full 570-question MMLU runs, and several smaller batched runs, stall
reproducibly (GPU pinned near 100%, or 15-25% depending on the run, no
forward progress for 100s+, in-process `SIGKILL` sometimes doesn't land —
`docker kill` required) at specific points. Extensively isolated this
session and **not** explained by: prompt content in isolation (a known
stall position run alone, batch=1, completes cleanly with hundreds of
real shadow activations), a generic batch-size ceiling (batch=16 is clean
for some slices, stalls for others), process reuse (fresh-process re-runs
stall identically), determinism/race (4/4 identical repeats of a known-
clean slice), prompt length, subject boundaries, or host memory pressure
(all directly tested and ruled out — see `LOGBOOK.md`, 2026-08-10
entries), CUDA Graph interaction (already disabled, `enforce_eager=True`),
chunked prefill (already disabled), or per-row CPU-GPU sync pressure in
the GCSG hook (a well-grounded, code-verified candidate — tested directly
with a real fix in place, disproven). Current best explanation: an
interaction effect between specific concurrently-scheduled prompts,
mechanism unidentified. Tracked
as an open item on issues #10/#16, which stay open on purpose rather than
being closed against a partial fix.

### Status

**In progress — not released.** Shadow execution is real and verified for
what it targets (the offload/pin_memory crash class is fixed on both
quantization paths, numerically and end-to-end). Full-dataset MMLU
quality comparison against the 72.3% hook-only baseline (Möllstorp-era
measurement) is blocked by the stall above — partial coverage only via a
skip-and-continue strategy, with an acknowledged representativeness
caveat (skipped slices could bias subject coverage). Not marked Released
like Karlshamn/Möllstorp/Eketorp above, on purpose — the module works,
but the sprint's own acceptance criterion (GCSG quality degradation < 2%,
MMLU-5shot) can't be evaluated yet without full coverage.

### Tracking

- Issues #10 and #16 kept open, cross-referenced, updated with the shared-
  mechanism finding and the real fix — not closed, since the stall above
  is a distinct, still-open failure surfaced by the same code path
- Board [OSX-PoC Roadmap](https://github.com/users/danielesalpietro/projects/1) updated: Sprint 3 card moved to In Progress with the real state (not the original stub-scope description), Sprint 6 (Stockholm, telemetry — added mid-sprint, see README) card body filled in, issues #10/#12/#16 added

---

*Next: root-cause the batch-composition stall (issues #10/#16), resume skip-and-continue MMLU coverage, then close out Sprint 3 for real once the acceptance criterion can actually be measured.*

---

## [Eketorp] — v0.3.0-dev — 2026-08-08



*Named after Eketorp, Öland, Sweden, where M2 — EMH Tier Manager was implemented.*



### Added



**M2 — EMH Tier Manager (real implementation)**



- `src/tier/policies.py` — `LRUPolicy.rank` (ordina per `last_access_ts` crescente) e `SEEPolicy.score`/`rank`: quando `context_vec` è `None`, gamma viene ridistribuito su alpha/beta (LRU pesata, entrambi i termini normalizzati a (0, 1] per restare comparabili a prescindere dalla scala assoluta di access_count/age); quando `context_vec` è fornito, σ resta uno stub a 0.0 senza ridistribuire i pesi — comportamento deliberatamente diverso dal caso `None`, non un fallback silenzioso

- `src/tier/io.py` — `AsyncNVMeIO.read_shard`/`write_shard`/`exists`/`delete` via `aiofiles`/`aiofiles.os`. dtype esplicito `uint8` in `read_shard` (non il default implicito di `np.frombuffer`), coerente col dtype del pool di `SlabAllocator`

- `src/tier/gpu.py` — `GPUTransfer.to_vram`/`to_ddr4`/`vram_free_bytes`/`vram_total_bytes`/`create_stream` via `torch.cuda`. Nessuna pipeline multi-stream in M2 — deferred, come già indicato nel docstring del modulo

- `src/tier/manager.py` — `TierManager` completo: `promote` (NVME→DDR4, DDR4→VRAM, e NVME→VRAM incatenando i due hop sotto lo stesso lock), `evict` (single-hop soltanto — asimmetria intenzionale rispetto a `promote`, documentata esplicitamente nel docstring), `evict_to_free_vram` (ranking SEE/LRU), `prefetch` (`asyncio.gather` con `return_exceptions=True`: un fallimento su un singolo shard viene loggato, non propagato, così non abortisce il batch), `stats`

- `src/eat/eat.py` — `ExpertAccessTable.slab` (nuova property read-only): espone la `SlabAllocator` privata la cui lifecycle è già guidata da `initialize()`/`shutdown()`. Unico touch a un file M1 in questo sprint — un accessor puro, nessun cambio a locking, layout o `EATEntry`



**Benchmarks**



- `benchmarks/bench_tier.py` — P50/P95/P99 reali per NVMe→DDR4 e DDR4→VRAM (quest'ultimo con guardia esplicita su `torch.cuda.is_available()`: riporta `status: skipped` invece di inventare numeri da un path che non ha toccato GPU reale), JSON output — rimpiazza il placeholder `"status": "pending"`



### Design decisions



- **Concorrenza: lock per-key, non un redesign dell'RLock di EAT.** Il finding di Möllstorp sulla tail latency dell'RLock sotto contention resta un limite noto e non risolto qui — la regola di Sprint 2 è invece un vincolo di design: EAT viene toccata solo per operazioni sincrone rapide (`lookup`, `update_tier`), mai mentre un transfer NVMe/GPU è in corso. Questo da solo non bastava: due `promote()` concorrenti sulla stessa key vedrebbero entrambi lo stesso tier di partenza prima dell'`await` sull'I/O, allocando due slot slab per un solo shard logico — un leak reale sotto asyncio, non un edge case teorico, individuato in fase di planning prima di scrivere il codice. Risolto con un `dict[key, asyncio.Lock]` per-key in `TierManager`, creato lazy, che serializza l'intera transizione di tier per una data key. Un test di regressione dedicato (`test_concurrent_double_promote_no_slab_leak`) verifica che due `promote()` concorrenti sulla stessa key producano un solo slot slab allocato

- **Bloom filter: lasciato invariato.** Il finding di Möllstorp (Bloom filter ~5-14x più lento di un dict semplice) resta parcheggiato — nessuna modifica a `src/eat/bloom.py` o al fast-path di `EAT.lookup()` in questo sprint

- **Slot ownership: bookkeeping nel TierManager, non in EATEntry.** Nessun campo `slot_idx` aggiunto a `EATEntry` — `TierManager` traccia da sé quale slot slab (`self._slots`) e quale tensor VRAM (`self._vram`) appartengono a quale shard, rispettando il layout a 28 byte documentato in M1

- **DDR4 come staging-only.** Una volta promosso a VRAM, lo slot DDR4 viene liberato immediatamente — nessuna doppia copia residente, bookkeeping più semplice



### Fixed



**`GPUTransfer`/`TierManager` — VRAM non liberata a livello driver dopo evict, trovato solo su hardware reale**



- Il primo tentativo di verifica GPU (via `workflow_dispatch` su `Z8-G4-RTX3090`) ha fatto fallire `test_evict_to_free_vram_evicts_candidate`: `evict_to_free_vram()` continuava a ciclare e sollevava `MemoryError` anche dopo un'eviction riuscita. Causa: `torch.cuda.mem_get_info()` (usato da `vram_free_bytes()`) riporta la memoria libera a livello driver, non quella che il caching allocator di PyTorch trattiene per riuso dopo un `del tensor` — senza un `torch.cuda.empty_cache()` esplicito, `vram_free_bytes()` restava invariato dopo ogni eviction

- Il primo fix (aggiungere `GPUTransfer.empty_cache()` e chiamarlo da `TierManager.evict()`) non bastava: `torch.cuda.empty_cache()` restituisce al driver solo i blocchi SENZA reference vive, e la variabile locale `tensor` dentro `evict()` continuava a referenziare il tensor anche dopo averlo rimosso da `self._vram` — un `del self._vram[key]` non è un `del tensor`. Fix definitivo: `self._vram.pop(key)` seguito da un `del tensor` esplicito prima di `empty_cache()`, così l'ultima reference viene rilasciata prima che l'allocator possa restituire il blocco al driver

- Aggiunto un test di regressione dedicato (`test_evict_frees_vram_visible_to_mem_get_info`) che asserisce `vram_free_bytes()` aumenti davvero dopo un'eviction, non solo che l'eviction non sollevi eccezioni

- Bug non riproducibile in nessun test CPU-only o mock — esiste solo con CUDA reale, trovato exattamente eseguendo la verifica GPU che il Möllstorp/Eketorp precedente aveva lasciato in sospeso



### Tests



- `tests/test_tier.py` — ogni stub (`pytest.raises(NotImplementedError)` / `pytest.skip("TODO Sprint 2")`) rimpiazzato con asserzioni reali, stesso pattern di riscrittura di `test_eat.py` in M1. Split CPU/GPU che rispecchia i job CI `cpu-tests`/`full-gpu-tests`: I/O NVMe, policy LRU/SEE, promozione NVME→DDR4, `evict()` single-hop-only, prefetch, e il test di regressione sul double-promote girano ovunque torch sia importabile; tutto ciò che tocca CUDA reale (`GPUTransfer`, DDR4→VRAM, NVME→VRAM incatenato, `evict_to_free_vram`, l'integration test lato VRAM) resta `@pytest.mark.gpu`

- **Verificato per davvero su hardware reale in questa sessione**, non solo scritto: 28/28 in `test_tier.py` (inclusi tutti i `@pytest.mark.gpu`), 60/72 nell'intera suite (12 skip sono stub M3 non correlati) — girato due volte, prima direttamente via `docker compose run` contro l'RTX 3090 locale (questa macchina è anche il runner self-hosted `Z8-G4-RTX3090`: nessun round-trip push/workflow_dispatch necessario per iterare sul bug sopra), poi confermato via un `workflow_dispatch` reale su CI (`31263503614`, entrambi i job verdi: CPU Tests 35s, Full GPU Tests 45s)

- 18/18 test non-GPU passati anche via il container leggero `python:3.12-slim` (dependency set identico a `cpu-tests`)



### CI/CD



- `.github/workflows/ci.yml` — `full-gpu-tests` esegue ora anche `benchmarks/bench_tier.py` dopo la test suite, caricando il risultato come artifact `bench-tier-result` — stesso schema del passo `bench-eat-result` di M1. **Verificato con un run reale**: `workflow_dispatch` `31263503614`, entrambi gli artifact scaricati e confrontati (vedi Evaluation sotto)



### Evaluation — numeri reali su Z8-G4-RTX3090



- **EAT (M1, in-memory) vs TierManager (M2, I/O reale)**: lookup EAT p50≈2.6µs/p95≈2.9µs/p99≈4.7-4.9µs (in linea con Möllstorp, run-to-run entro il ~10% già documentato) contro promote NVMe→DDR4 p50≈5.2ms/p95≈p99≈20.6ms — circa 2000× più lento, atteso: I/O di file reale + overhead asyncio contro un dict in memoria

- **DDR4→VRAM**: p50≈77ms, ma p95≈p99≈1.3-1.4 secondi — riprodotto identico (stesso ordine di grandezza) sia nel run locale che nel run CI, quindi un effetto sistematico, non rumore. Causa quasi certa: costo one-time di inizializzazione del contesto/allocator CUDA sulla prima vera chiamata GPU del processo — `bench_tier.py` non ha un'iterazione di warm-up prima di misurare, e con soli 20 campioni quella singola chiamata fredda domina p95/p99. Segnalato onestamente come limite del benchmark attuale, non del codice di `TierManager` — candidato per un refinement futuro (warm-up prima delle misure), non affrontato in questa sessione



### Status



**Verificato su hardware reale.** M2 (Tier Manager) è implementato e verificato per davvero, non solo scritto: 28/28 test di `test_tier.py` (inclusi tutti i `@pytest.mark.gpu`) passati sia via `docker compose run` diretto contro l'RTX 3090 locale sia via `workflow_dispatch` su CI (`31263503614`, entrambi i job verdi). 60/72 nell'intera suite (12 skip M3, non correlati). Un bug reale — VRAM non liberata a livello driver dopo evict, vedi Fixed sopra — è stato trovato e corretto proprio grazie a questa verifica: invisibile in ogni test CPU-only/mock, avrebbe fatto ciclare all'infinito `evict_to_free_vram()` in produzione. `benchmarks/bench_tier.py` ha girato per davvero su hardware target, artifact `bench-tier-result` caricato e confrontato con `bench-eat-result` (vedi Evaluation)



### Tracking



- Aperte 8 GitHub Issue per i finding lasciati deliberatamente irrisolti tra M1 ed M2, invece di lasciarli solo come note in LOGBOOK.md: [#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1) destino Bloom filter, [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2) tail latency RLock sotto contention, [#3](https://github.com/danielesalpietro/vMemoryFabric/issues/3) manca warm-up in `bench_tier.py`, [#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4) `BloomFilter.remove_expert()` non implementato, [#5](https://github.com/danielesalpietro/vMemoryFabric/issues/5) pipeline CUDA stream mancante, [#6](https://github.com/danielesalpietro/vMemoryFabric/issues/6) manca config ruff/pyproject.toml, [#7](https://github.com/danielesalpietro/vMemoryFabric/issues/7) integrazione PMEM (bloccata da hardware), [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) dual-GPU/AER (bloccata da hardware)

- Board [OSX-PoC Roadmap](https://github.com/users/danielesalpietro/projects/1) aggiornata: card Sprint 2 spostata a Done col riepilogo reale, card Sprint 3 arricchita coi riferimenti alle issue #1/#2/#5 rilevanti per il kickoff, tutte le 8 issue aggiunte alla board



---



*Next release: Sprint 3 — M3 (Expert Scheduler): PT-PEP classifier, GCSG guard, AER stub.*



---



## [Möllstorp] — v0.2.0-dev — 2026-08-08



*Named after Möllstorp, Öland, Sweden, where M1 — Expert Access Table was implemented.*



### Added



**M1 — Expert Access Table (real implementation)**



- `src/eat/bloom.py` — `BloomFilter.add/may_contain_expert/may_contain_shard/__len__` implemented on top of `pybloom_live`, string-encoded keys (`e:{expert_id}`, `s:{expert_id}:{shard_idx}`) for deterministic hashing. `remove_expert` still raises `NotImplementedError` — standard Bloom filters can't delete; deferred to a future Counting Bloom Filter evaluation, unused by `EAT.evict()` by design (see Fixed/Dev constraints below)

- `src/eat/slab.py` — `SlabAllocator` pool lifecycle (`initialize`/`shutdown`), `alloc`/`free`/`get_buffer`, `free_slots`/`used_slots`. Pool backed by `np.empty((n_slots, shard_size), dtype=np.uint8)` — `empty` not `zeros`, content is a placeholder for the future PMEM/DDR4 payload so there's no reason to zero-fill pages

- `src/eat/eat.py` — `ExpertAccessTable` full CRUD: `insert`, `lookup` (Bloom fast-negative before touching the dict), `update_tier` (bumps `version`), `evict`, `access` (`touch()`), `get_tier`, `eviction_candidates` (LRU by `last_access_ts`, SEE score deferred to M2 per the existing docstring), `initialize`/`shutdown`, `stats()`



**Benchmarks**



- `benchmarks/bench_eat.py` — real P50/P95/P99 lookup latency + insert throughput, JSON output (replaces the `"status": "pending"` placeholder)



### Design decisions



- **SlabAllocator stays standalone for M1** — `EAT.insert()`/`evict()` operate only on the dict + Bloom filter; they do not call `slab.alloc()`/`free()`. `EATEntry`'s documented 28-byte layout has no `slot_idx` field, and physical DDR4 promotion/eviction is the Tier Manager's responsibility (M2, Sprint 2). `EAT.initialize()`/`shutdown()` still drive the `SlabAllocator`'s lifecycle in lockstep so the two stay wired together for M2 to build on

- Bloom filter keys are explicit strings rather than raw tuples/ints, to not depend on `pybloom_live`'s internal key-hashing behavior for compound keys



### Tests



- `tests/test_eat.py` — every stub (`pytest.raises(NotImplementedError)` / `pytest.skip("TODO Sprint 1")`) replaced with real assertions: Bloom add/contains/false-positive-rate, Slab alloc/free/exhaustion/variable-tail, EAT CRUD + LRU eviction ordering, 8-thread concurrent insert/read-write/version-counter races, and a real lookup-latency smoke timing. Added `test_initialize_shutdown` (previously uncovered) and a couple of extra edge-case tests (`test_insert_duplicate_raises`, `test_update_tier_missing_raises`, `test_evict_missing_returns_none`, `test_access_missing_returns_none`) not present in the Sprint 0 skeleton

- `make test-eat` — 24/24 passed (verified via a lightweight `python:3.12-slim` container mirroring CI's `cpu-tests` dependency set, not the full CUDA dev image — nothing in M1 touches torch/CUDA)



### CI/CD



- `.github/workflows/ci.yml` — `full-gpu-tests` now also runs `benchmarks/bench_eat.py` on the self-hosted `Z8-G4-RTX3090` runner after the test suite, uploading the result as the `bench-eat-result` workflow artifact. Every manual `workflow_dispatch` re-measures M1 on real target hardware instead of the numbers staying container-host-only



### Evaluation — extended benchmark + technical report



- `benchmarks/bench_eat.py` — extended past the initial P50/P95/P99 + insert-throughput numbers with three additional measurements, requested during report review to close the gaps an OSDI-style reviewer would flag first:

  - **Baseline comparison** — a plain-dict, no-Bloom-filter baseline (`_PlainDictBaseline`, benchmark-only, not part of the EAT's public API) run against the identical workload. Result: the Bloom filter measured ~5× slower on lookups and ~14.6× slower on inserts than the plain dict — it costs latency rather than saving it, because the structure it guards is already an O(1) in-memory dict. Flagged in the report as an open question for M2/M3 rather than a justified design choice

  - **Contention profile** — 4 concurrent reader threads + 1 writer thread against a shared EAT. Reader p99 lookup latency rose ~1,360× under contention (single-digit µs → several ms), exposing a real tail-latency cost from the `RLock`-based concurrency model in `eat.py` (see the "CAS counter" design note in Möllstorp's Design decisions above)

  - **Slab allocator at scale** — alloc/free timing at 4 vs. 32 slots (1 GB vs. 8 GB). No measurable latency growth, consistent with the free-list's claimed O(1) behavior across that 8× increase (still ~2 orders of magnitude below the ~1,000-slot production target)

- *OSX Technical Report — M1: Expert Access Table* (internal document, not committed to this repository) — formal design/implementation/evaluation writeup including the baseline, contention, and slab-scale results above, and an explicit limitations section. Re-run against real target hardware twice via CI (`workflow_dispatch` runs `31259547453` and `31260457679`), after an initial container-host-only draft — the two same-hardware runs differed by ~32% in measured insert throughput, noted in the report as unquantified run-to-run variance rather than smoothed over



### Status



**Released.** M1 (Expert Access Table) is fully implemented, tested, and benchmarked: 24/24 tests passing in `tests/test_eat.py`; real hardware numbers from `benchmarks/bench_eat.py` on `Z8-G4-RTX3090` (p50 ≈ 2.6 µs / p95 ≈ 3.8 µs / p99 ≈ 4.6 µs lookup latency, ≈ 177k inserts/sec); and, per the extended evaluation above, two open design questions with measured costs attached — the Bloom filter's net benefit in the current single-process design, and the RLock's tail-latency behavior under concurrent read/write load — both recorded as Sprint 2/M2 candidates rather than fixed here, per the decision to leave `EATEntry`'s layout and M1's core design untouched until end-to-end numbers exist (Sprint 4). M2/M3 remain Sprint 0 skeletons (`NotImplementedError`), as planned.



---



*Next release: Sprint 2 — “Eketorp” — M2 (Tier Manager) implementation complete, `make test-tier` passing green.*



---



## [Karlshamn] — v0.1.0-dev — 2026-08-07



*Named after Karlshamn, Sweden, where the PoC development environment was established.*



### Added



**Environment**

- `Dockerfile` — CUDA 12.1.1 + Python 3.12 + full M1/M2/M3 stack

- `requirements.txt` — pinned dependencies (PyTorch 2.3.0+cu121, transformers 4.41.2, onnxruntime 1.18.0; vLLM deferred — see `requirements-vllm.txt`)

- `docker-compose.yml` — single RTX 3090, NVMe volume, optional Prometheus sidecar

- `Makefile` — targets: `build`, `smoke`, `test`, `test-eat`, `test-tier`, `test-scheduler`, `bench`, `lint`, `typecheck`, `shell`, `metrics-up`, `clean`

- `pytest.ini` — asyncio auto mode, short traceback



**M1 — Expert Access Table (skeleton)**

- `src/eat/types.py` — `EATEntry`, `Tier` (VRAM/DDR4/NVME), `ExpertID`, `ShardID`

- `src/eat/bloom.py` — `BloomFilter` 2-level interface (expert-level + shard-level)

- `src/eat/slab.py` — `SlabAllocator` interface (DDR4 backend, PMEM-ready)

- `src/eat/eat.py` — `ExpertAccessTable` interface: insert, lookup, update_tier, evict, access, get_tier, eviction_candidates



**M2 — EMH Tier Manager (skeleton)**

- `src/tier/io.py` — `AsyncNVMeIO` interface (asyncio+aiofiles proxy for io_uring)

- `src/tier/gpu.py` — `GPUTransfer` interface (standard cudaMemcpy, no pinned memory)

- `src/tier/policies.py` — `SEEPolicy` (α=0.3, β=0.3, γ=0.4) + `LRUPolicy` interfaces

- `src/tier/manager.py` — `TierManager` interface: promote, evict, evict_to_free_vram, prefetch



**M3 — Expert Scheduler (skeleton)**

- `src/scheduler/ptpep.py` — `PTPEPClassifier` interface: 8 domains, ONNX runtime, confidence threshold

- `src/scheduler/gcsg.py` — `GCSGGuard` interface: θ_gate=0.85, θ_entropy=0.70, θ_contamination=0.05

- `src/scheduler/aer.py` — `AERManager` stub: replication_factor=1 (single GPU dev), sync_lora_delta no-op



**Tests**

- `tests/test_eat.py` — BloomFilter, SlabAllocator, EAT CRUD, concurrency, latency smoke

- `tests/test_tier.py` — AsyncNVMeIO, GPUTransfer, SEE/LRU policies, TierManager, EAT↔Tier integration

- `tests/test_scheduler.py` — PT-PEP, GCSG, AER, Scheduler↔Tier integration



**Benchmarks (placeholder)**

- `benchmarks/bench_eat.py` — Sprint 1 target

- `benchmarks/bench_tier.py` — Sprint 2 target



**Config**

- `configs/osx_default.yaml` — full runtime config with dev constraints documented inline

- `configs/prometheus.yml` — Prometheus scrape config (port 9090)



**Scripts**

- `scripts/smoke_test.py` — 13 independent checks: Python version, PyTorch, CUDA, VRAM ≥22 GB, tensor roundtrip, NVMe volume, vLLM, transformers, onnxruntime, aiofiles, prometheus_client, src/ importable, pinned memory info



### Added — CI/CD (extra scope)



Not part of the original Sprint 0 plan; added opportunistically while closing out the environment.



- `.github/workflows/ci.yml` — two jobs. `cpu-tests` runs on every push/PR (`ubuntu-latest`, `pytest -m "not gpu"`, CPU-only dep subset — no torch/vLLM/CUDA). `full-gpu-tests` runs only on manual `workflow_dispatch` (`runs-on: [self-hosted, gpu]`), since GitHub Actions has no per-job trigger — the workflow listens for push/PR/workflow_dispatch and the GPU job gates itself with `if: github.event_name == 'workflow_dispatch'`

- `.gitignore` (repo root) — excludes `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`

- `osx-poc/SELF-HOSTED.MD` (new) — setup guide for the self-hosted GPU runner: hardware/software prerequisites, pre-registration GPU-in-Docker check, registration steps with the required `gpu` label, security note on why the job stays gated to `workflow_dispatch`

- Self-hosted runner **Z8-G4-RTX3090** registered against the repo, `gpu` custom label attached, running as a dedicated `GitRunner` local service account (member of `docker-users`, granted "Log on as a service")



### Fixed



**Docker build**

- `Dockerfile` — Python 3.12 requires the deadsnakes PPA on Ubuntu 22.04 (jammy ships 3.10 only); bootstrap pip via `ensurepip` since apt's `python3-pip` targets the default interpreter, not deadsnakes'

- `requirements.txt` — pinned `torch==2.3.0+cu121` / `torchvision==0.18.0+cu121` (was `2.3.1`, an incompatible pair once vLLM entered dependency resolution)

- `requirements.txt` — removed `vllm==0.4.3` from the base install: it pins `vllm-flash-attn==2.5.8.post2`, no longer published on PyPI, and the GCSG hooks that need it are still `NotImplementedError` stubs (Sprint 3)

- `requirements-vllm.txt` (new) — parks the vLLM pin separately; install manually once the GCSG hook code exists and the right vLLM version can be verified against it



**Smoke test**

- `scripts/smoke_test.py` — `check_vllm()` no longer fails the suite when vLLM is absent; not installed is the expected state until Sprint 3

- `scripts/smoke_test.py` — `check_aiofiles()` / `check_prometheus_client()` read the version via `importlib.metadata` instead of the module's `__version__` attribute, which neither package exposes



**M1 — Expert Access Table**

- `src/eat/types.py` — added the `SHARD_SIZE_MB` / `SHARD_SIZE_BYTES` constants, which `src/tier/io.py` was already importing from here but that were only ever defined in `src/eat/slab.py`; `slab.py` now imports them from `.types` instead of redefining them



**CI (found on the first real run, not by inspection)**

- `.github/workflows/ci.yml` — `cpu-tests` now installs `torch==2.3.0+cpu` (~190 MB, no bundled CUDA libs) instead of skipping torch entirely: `TierManager.__init__` unconditionally constructs a `GPUTransfer`, which needs `torch` importable even for tests that never touch real CUDA — `TestTierManager` was failing with 4 errors until this was added



### Dev constraints (documented, not bugs)



- Pinned CUDA memory (`cudaMallocHost`) not used — Docker/Windows overhead; standard `cudaMemcpy` path instead

- `io_uring` replaced by `asyncio + aiofiles` — not available on WSL2/Docker Windows

- PMEM (EMH-2) absent — DDR4 acts as proxy; `SlabAllocator` is PMEM-ready via `libpmem2` mmap

- Single GPU (RTX 3090) — AER disabled, dual-GPU benchmarks deferred to RTX 5080 arrival

- M4 (RecursiveMAS LED Bridge) — out of PoC scope, deferred



### Status



**Released.** All module interfaces are defined; all methods raise `NotImplementedError` — the intended baseline for Sprint 1, `make test` shows all tests failing deterministically as expected. The dev environment itself is verified end-to-end: `make build` produces a working image and `make smoke` passes 13/13 checks (Python 3.12, PyTorch 2.3.0+cu121, CUDA, RTX 3090 24 GB VRAM, GPU tensor roundtrip, NVMe volume, all pinned packages, `src/` importable).

Both CI jobs verified green through an actual run, not just by reading the YAML: `cpu-tests` on a real push, `full-gpu-tests` via manual `workflow_dispatch` against the newly-registered `Z8-G4-RTX3090` runner (54 collected, 27 passed / 27 skipped, 0 failed).



---



*Next release: Sprint 1 — M1 (EAT) implementation complete, `make test-eat` passing green.*

