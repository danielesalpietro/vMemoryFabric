# Changelog



All notable changes to OSX-PoC are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning follows [Semantic Versioning](https://semver.org/).



Releases are named after cities where significant project milestones were reached.



---

## [Berg] — v0.6.0-dev — 2026-08-13 (in progress, not yet released)

*Named after Berg, Stockholm, where the Sprint 5 planning conversation happened — PoC delivery + paper, two parallel tracks.*

### Added — 2026-09-02 (bis) — FineMoE e MoE-Infinity letti sul codice: i differenziatori di EMH si restringono a GCSG; H2 alzata a 0.25; bake-off dei predittori

- `reports/related_work_finemoe.md` (nuovo) — FineMoE (EuroSys '26) verificata sul **repository ufficiale** (`IntelliSys-Lab/FineMoE-EuroSys26`, `5c58468`) e su MoE-Infinity HEAD (`b766f8f`); **il PDF è irraggiungibile** dal sandbox (arXiv, sito autori, HF, Semantic Scholar: tutti bloccati dal proxy) e i numeri del paper restano non verificati. Meccanismo: traccia per **token** (embedding di input + probs del router per layer), store `(embed → traiettoria)`, kNN sull'embedding **prima del primo MoE** + **correzione a ogni layer** dalla traiettoria parziale, selezione a massa cumulativa `1 − similarità`, eviction LFU × probabilità del router (`task_scheduler.cpp:296`, contributo FineMoE rispetto a MoE-Infinity). Tre tier GPU/host/SSD, esperto intero per layer. Demo: solo Qwen1.5-MoE, ≥ 48 GB, batch 1
- **Verdetto scomodo, scritto per intero**: granularità di esperto, prefetch predittivo e hotness persistente **non sono differenziatori** di EMH in questo cluster; PT-PEP è meno sofisticato del meccanismo FineMoE (nessuna correzione in-flight) e la sua sola proprietà distintiva è il lead time pre-tokenizzazione, da misurare; **GCSG resta l'unico differenziatore forte**. MoE-Infinity a HEAD è un terzo concorrente vivo (serving engine, KV paginata, Mixtral, quantizzazione, sm_120) — l'unico del cluster che può girare sul banco: braccio **A5** nel test plan
- `reports/test_plan_emh_vs_vllm_offloader.md` — **H2 alzata da 0.5 a 0.25** (decisione del project owner, 2026-09-02): a routing uniforme e hit rate zero il rapporto β_E/β_P è già k/E = 0.25 per geometria, 0.5 non testava EMH ma la sparsità del MoE; aggiunta **H2b** a 0.15 ("PT-PEP aggiunge qualcosa"). Nuovo livello **L0b**: bake-off dei predittori (PT-PEP, kNN-embedding, +correzione per layer, EAM, speculativo dai logit, oracolo, frequenza) sulle **stesse tracce**, senza GPU, metrica recall@k vs lead time; ipotesi **H8/H9**. Metrica **M9** (entropia coarse/fine del routing, definizione di FineMoE `process_data.py`) e M8-bis (hit rate per esperto come `NodeBody` di MoE-Infinity)
- `logbook_paper.md` (entry 2026-09-02 bis) — conseguenza sull'outline: EMH non può essere presentata come *contributo* (è MoE-Infinity 2024 / FineMoE 2026) ma come **substrato** di GCSG; bozza in inglese del paragrafo "Expert offloading systems" per §8; sotto-obiettivo 2 resta 🟡 con rischio cambiato di natura


### Added — 2026-09-02 — il vero vicino di EMH dentro vLLM è l'offloader nativo, non EEP; piano dei test pre-registrato; paragrafo per il paper

- `reports/related_work_vllm_offloader.md` (nuovo, S0-bis di #68) — `vllm/model_executor/offloader/` (backend `uva` zero-copy e `prefetch` a gruppi di layer), presente da v0.25.0, assente in v0.13.0, verificato sui tag. Fatto decisivo, di layout non di policy: nel `FusedMoE` gli esperti di un layer sono **un solo tensore** `w13_weight (num_experts, …)` (`unquantized_fused_moe_method.py:70-79`) e l'offloader lavora per parametro (`start_onload_to_static` copia i whitelisted per intero) — non può muovere un esperto singolo. Zero occorrenze di `topk`/`router`/`gate`/`hot`/`lru`/`evict` in `prefetch.py`: schedule statico `module_index % group_size`. Distanza da EMH misurata in quattro assi: routing-aware, granularità di esperto, > 2 tier, eviction dinamica. Prezzo che EMH paga, detto con la stessa chiarezza: ΔQ ≠ 0 (shadow INT4) e nessuna integrazione con CUDA graph
- `reports/test_plan_emh_vs_vllm_offloader.md` (nuovo) — piano **pre-registrato** (H1–H7 con soglie e significato del fallimento) per confrontare EMH con `uva`/`prefetch` nonostante il confondimento di versione (GCSG su V0 0.10.1, offloader su V1 ≥ 0.25): modello di costo comune (β_P = S·E·L_off costante; β_U = S·k·L_off; β_E = S·k·L·μ con μ funzione dello skew del routing), normalizzazione per versione (ρ_X = M(A_X)/M(A0 stessa versione), IC bootstrap), livello L0 senza vLLM per calibrare le costanti. Predizioni falsificabili P1–P5, fra cui **P3: a routing uniforme EMH non guadagna nulla oltre la capacità** — se vince lì, la misura è sbagliata (H5). L0 e L1 eseguibili ora sulla 3090 (la 3090 regge `torch>=2.7`; #8 blocca solo la 5060 Ti); L2 dopo #8. EEP non è un braccio: controllo negativo documentato
- `logbook_paper.md` (entry 2026-09-02) — il vicino *ingegneristico* accanto a quello *accademico* (FineMoE); bozza in inglese del paragrafo "Production engines" per §8 e della frase per §7 Limitations; riga nuova in bibliografia Cluster A. Il claim "nessuno fa offload di esperti in vLLM" non è più vero e il paper non deve farlo


### Added — 2026-09-01 — related work: Elastic Expert Parallelism di vLLM (issue #68, deliverable S0)

- `reports/related_work_elastic_ep.md` (nuovo) — analisi di **Elastic Expert Parallelism** (EEP) di vLLM e del suo rapporto con OSX/EMH, verificata sul codice sorgente (shallow clone `vllm-project/vllm`, HEAD `18c5372` del 2026-09-01) con la stessa disciplina di `related_work_petals_exllama.md`: anchor `file:riga`, no-list con condizioni di rivalutazione, sezione di correzioni esplicite. Nessun benchmark eseguito, nessun pin modificato
- **Tesi**: EEP e OSX sono ortogonali — EEP scala gli esperti sull'asse orizzontale (quanti rank DP/EP, chi ospita cosa, deciso da EPLB), OSX sull'asse verticale (in quale tier EMH vive uno shard). Si contendono però la stessa VRAM e descrivono lo stesso oggetto con due chiavi che non si parlano: `(logical_expert, physical_slot, ep_rank)` contro `(expert_id, shard_idx, tier)`
- **Tre affermazioni di issue #68 smentite dal codice** (§5 del report), la #68 essendo stata scritta da fonti secondarie: (1) `tensor_parallel_size=1` non è un vincolo validato — l'aritmetica EP è `dp × tp`, TP=1 è solo ciò che i test esercitano; (2) il trasferimento pesi a scale-up **esclude esplicitamente** i pesi degli esperti (`elastic_execute.py:84`) — densi via P2P `isend`/`irecv`, esperti solo via reshuffle EPLB, per giunta asincrono; (3) **un nuovo rank non legge mai da disco** (`load_model(load_dummy_weights=True)`, `elastic_execute.py:242-243`)
- **Il vincolo su Ray invece è reale**, solo dichiarato dove non lo si cerca: non in `config/parallel.py` ma sul path di scale, `assert data_parallel_backend == "ray"` in `core_client.py:1636-1638` ("Only ray DP backend supports scaling elastic EP"). Che i tre executor implementino `elastic_ep_execute` non lo rende opzionale: lo scale non arriva mai fino a loro
- **Contabilità esperti fisici/logici** (§1.5 del report, la parte più controintuitiva): lo scale-up tiene **costante** il numero di slot fisici per rank e fa crescere il totale, riempiendo il surplus di **repliche** degli esperti caldi — `num_redundant_experts = num_physical × new_dp // cur_dp − num_experts` (`core_client.py:1641-1646`). Su Mixtral 8x7B (8 esperti per layer × 32 layer) da DP=2 a DP=8: 8 → 32 slot fisici per layer, di cui 24 repliche, sempre 4 per rank. Le GPU aggiunte comprano copie degli esperti caldi, non frammenti più piccoli. Ne segue il pavimento dello scale-down (`ValueError` sotto `ceil(num_experts × cur_dp / num_physical)`)
- **Conseguenza sull'ipotesi di lavoro**: #68 proponeva di idratare i nuovi rank dal tier warm DDR4/PMEM assumendo che l'alternativa fosse un caricamento da disco. Non lo è: il confronto reale è PCIe locale contro NCCL peer-to-peer già sovrapposto al servizio, e su nodo singolo perdiamo quasi certamente. L'opportunità che sopravvive è lo **scale-down** — gli esperti dei rank rimossi vengono compressi sui superstiti (`rank_mapping` a `-1`) e vLLM non ha nessun tier dove metterli
- **Vincoli EEP a HEAD**, distinti fra *hard* (validati: richiede `--enable-eplb`, vieta PP>1, incompatibile con external/hybrid LB, EPLB async richiede NIXL, non supportato dal model runner V2) e *de facto* (Ray e TP=1: solo ciò che i test esercitano). Distinzione assente dai sommari divulgativi
- Decisioni aperte che ne derivano (D1–D6: chiave EAT, invalidazione al cambio di mapping, confine AER/EPLB) tracciate in [discussion #70](https://github.com/danielesalpietro/vMemoryFabric/discussions/70), non in questo file


### Added — 2026-08-30/31 — seconda GPU (RTX 5060 Ti), analisi compatibilità vLLM/torch≥2.7, evidenze archiviate

*Narrativa completa: `LOGBOOK.md`, entry 2026-08-30/31. **Nessun pin modificato** — solo verifica e analisi, come richiesto da `handoff_rtx5060ti16gb.md` §2.2.*

- `logs/new_z8_bare_metal/` — recuperati i dati grezzi della sessione 2026-08-24, citati in prosa da `LOGBOOK_NEW_Z8.md` ma mai committati: `perf_test_hardware_20260824/`, `pmem_tier_20260824/` (PR precedenti), più `baseline_e32981d_20260824/`, `bench_20260824/`, `telemetry_20260824_1253/`. Identità verificata contro i numeri già citati nel logbook (`gflops_at_p50 = 31.604908778112684` ↔ "31.60"), non dedotta dal nome della cartella
- `logs/new_z8_bare_metal/dual_gpu_sm120_20260830/` (nuovo) — issue [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8): output grezzi della verifica toolchain sulla **RTX 5060 Ti 16GB** realmente installata (non la RTX 5080 attesa). `torch==2.5.1+cu124` → `sm_120` assente da `arch_list`, `torch.randn(10, device='cuda:1')` fallisce con `no kernel image is available` — stesso errore della RTX PRO 6000 Blackwell (entry 2026-08-12/13). Controfattuale sullo stesso hardware: `torch==2.7.0+cu128` → `sm_120` presente, operazione **OK** su entrambe le GPU
- `logs/new_z8_bare_metal/vllm_torch27_compat_20260831/` (nuovo) + `reports/vllm_torch27_compat_analysis.md` (nuovo) — analisi di compatibilità vLLM ↔ `torch>=2.7`/`cu128`: **minimo `vllm==0.9.0`**, finestra utilizzabile **`0.9.0`–`0.10.1`**, chiusa non da torch ma dalla rimozione di `mixtral_quant.py` (0.10.2, rompe `_AWQShadowExpert`) e `vllm/worker/worker.py` (0.11.0, sparisce la base di `GCSGWorker`). Trappola documentata: `torch==2.7.0` puro da PyPI è una build **cu126** senza `sm_120`, quindi `pip install vllm==0.9.0` liscio riproduce l'errore
- **Convenzione di archiviazione applicata a sé stessa:** i check delle due sessioni precedenti erano stati eseguiti in container poi rimossi, lasciando i numeri solo in prosa sull'issue #8 — lo stesso pattern che `handoff_rtx5060ti16gb.md` §0 elenca come i due incidenti già capitati. Ri-eseguiti il 31 agosto e archiviati: ogni file porta in testa comando, fonte, timestamp e host, e gli script che li rigenerano (`check_sm120.py`, `collect_vllm_torch_pins.py`) sono committati accanto
- **Un errore di analisi corretto in corsa e lasciato agli atti** (stessa convenzione dell'entry PPR 2026-08-25): una prima passata concludeva che `0.9.0` fosse la prima release vLLM con arch `12.0` nei kernel, avendo campionato solo 4 tag; la bisezione completa mostra che è `0.8.0`. Conclusione invariata, motivo diverso — vincola `torch` (0.8.x pinna `torch==2.6.0`, che non ha wheel cu128), non i kernel vLLM
- **Resta aperto**: nessun run end-to-end (`vllm` mai installato), baseline M1/M2 non eseguita, `NVIDIA_VISIBLE_DEVICES=0` in `docker-compose.yml` ancora a bloccare la seconda GPU indipendentemente da torch, e due riferimenti obsoleti nel docstring di `gcsg.py` (`vllm.executor.gpu_executor.GPUExecutor` rimosso in 0.7.0; firma `num_bits=` di `fused_marlin_moe`)

### Added — 2026-08-26 — Gate A5 (PoC delivery exit gate) at 3/4, first beta tag prepared

- `reports/sprint5_berg_plan.md` §A5 updated: conditions 1–3 (issue-tracker/README consistency, clean-clone `make smoke && make test`, PoC Final Report) verified and marked closed with evidence, not just asserted — condition 2 confirmed 2026-08-26 (fresh clone, build from scratch, 13/13 smoke, 224 passed/3 skipped)
- `v0.6.0-beta.1` prepared as an annotated tag on `develop`'s HEAD (`6a6cdfe`) — the first externally-consumable snapshot, once condition 4 (release tagged) closes. Release notes drafted, covering the four non-functional targets, known limitations (PMEM/#57 not production-wired, #33 CPU-offload tier performance, dual-GPU/#8), and environment
- **Condition 4 stays open on purpose:** the tag couldn't actually be pushed to GitHub from the agent session preparing this update — `git push origin v0.6.0-beta.1` was rejected with HTTP 403 by that session's git proxy (scoped to push only its own working branch, not arbitrary refs like tags), and no available GitHub API tool creates tags or releases directly. Recorded as pending rather than closed, per this file's own §0 issue-tracker-hygiene rule extended to gates: a condition is not "done" until it's verifiably true, not until the surrounding work is
- **Resolved same day, 15:58 UTC:** `v0.6.0-beta.1` published on GitHub Releases (`prerelease: true`) using the prepared tag message and release notes, on commit `31f526a` (the merge of this very gate-closure change, so the beta already includes it) — confirmed via `get_release_by_tag`/`get_tag`, not assumed from a "done" in chat. Gate A5 is now **4/4 closed**; see `reports/sprint5_berg_plan.md` §A5. First public release: <https://github.com/danielesalpietro/vMemoryFabric/releases/tag/v0.6.0-beta.1>
- Full narrative and follow-up thread: `LOGBOOK.md`, 2026-08-26 entry

### Added — 2026-08-25 — first Project Plan Review (PPR)

- `osx-poc/reports/ppr_20260825_sprint5_review.html` / `.docx` — an independent, external status audit of the Sprint 5 (Berg) plan itself (not a code review), cross-reading README, this file, `reports/sprint5_berg_plan.md`, `reports/poc_final_report.md`, `logbook_paper.md`, and all 39 GitHub issues (26 open, 13 closed) as of 25 August, six days before the plan's self-declared 31 August target. Full narrative: `LOGBOOK.md`, 2026-08-25 entry
- Headline findings: Filone B (the arXiv paper) has produced no artifact beyond the outline since 13 August — B3's related-work survey is stuck on its own stated blocker (FineMoE not yet read in full), B4 (LaTeX infra) hasn't started (`osx-poc/paper/` doesn't exist), B5 can't start without a draft; meanwhile 16 of the 26 open issues were filed 16–24 August, inside the window the plan assigned to closing A1 triage and starting B3, none of them belonging to the paper track (Z8 bare-metal migration, the #33 DDR4/CPU-offload follow-on cluster #43–#49/#53/#55, and Marstrand #32/#34–#37 absorbed that time instead)
- Full triage of all 26 open issues by category, and nine prioritized recommendations, are in the report — not duplicated here
- **A documentation-lag bug this review caught in itself before publishing, worth recording explicitly:** the draft initially repeated README's claim that PR #42 was still open. Verified against the GitHub API instead of trusting the prose: PR #42 (issue #33) was actually merged **2026-08-24 19:59**, together with four more same-day merges — [#51](https://github.com/danielesalpietro/vMemoryFabric/pull/51) (fixes #48), [#50](https://github.com/danielesalpietro/vMemoryFabric/pull/50) (closes #12), [#52](https://github.com/danielesalpietro/vMemoryFabric/pull/52) (closes #7 / PMEM), [#54](https://github.com/danielesalpietro/vMemoryFabric/pull/54) (host telemetry samplers). **None of these five merges is reflected yet in README** (PMEM still shown `❌ deferred` in the environment-constraints table; #12 still listed as an open item in "Known limitations") **or in this file** (no entries below for #42/#48/#50/#52/#54 — see Tracking further down, still describing PR #42 as "not yet merged"). This is the mirror image of the 2026-08-14 branch-consolidation incident recorded in `reports/sprint5_berg_plan.md` §0 (there, code lagged the docs; here, the docs lag already-merged code) — flagged here rather than silently fixed, since backfilling README/CHANGELOG for those five PRs is itself the report's #2 recommendation, left for a deliberate follow-up rather than folded into this entry. **Addressed same day**: README and the two sections immediately below now reflect all five PRs — this paragraph is kept as-is rather than rewritten, per this project's own convention of recording corrections instead of erasing the finding that prompted them

### Added — 2026-08-24 — Z8 G4 bare-metal migration, EMH-2 (PMEM), two dormant bugs fixed

*Full session narrative: `osx-poc/LOGBOOK_NEW_Z8.md`. Five PRs merged same day, filling the exact gap the PPR entry above caught in itself: [#42](https://github.com/danielesalpietro/vMemoryFabric/pull/42) (issue #33, below), [#50](https://github.com/danielesalpietro/vMemoryFabric/pull/50), [#51](https://github.com/danielesalpietro/vMemoryFabric/pull/51), [#52](https://github.com/danielesalpietro/vMemoryFabric/pull/52), [#54](https://github.com/danielesalpietro/vMemoryFabric/pull/54).*

**New dev host: Z8 G4 bare-metal (Ubuntu 24.04), replacing Docker-on-Windows/WSL2 as the reference environment for this hardware line**

- `/dev/pmem1` (one of two Optane regions) reconfigured `sector`→`fsdax` via `ndctl` — the other region hosts the OS (an installer defect, out of scope here, tracked for the next ISO release)
- Confirmed unlocked vs. the WSL2/Docker-on-Windows constraints table: pinned CUDA memory now auto-active (`GCSGWorker._should_pin_transfers()`'s `in_wsl()` check), `io_uring` available at the kernel level (not yet wired into `AsyncNVMeIO`, which still uses `asyncio`+`aiofiles`)
- Same-code-version re-run (checked out at the exact commit behind `logs/runpod_perf_framework_validation_20260817/COMPARISON_ANALYSIS.md`) of the three-host `perf-test-hardware`/`perf-tuning-report` comparison: this Z8 bare-metal host came out fastest of the four hosts measured on RAM bandwidth and CPU GFLOPS, though the underlying issue #33 conclusion (CPU-offload is memory-bandwidth-bound, not thread-bound) is unchanged
- Self-hosted GitHub Actions runner (`berlin-3eie`) migrated from the wrong repo (`kickstart-berlin`, where it had been mistakenly registered) to `vMemoryFabric`; the stale offline `Z8-G4-RTX3090` (Windows) runner entry removed
- `.github/CODEOWNERS` added (no branch-protection enforcement) — visibility for overlapping PRs across parallel work, not person-routing (single owner today)

**EMH-2 (Optane PMEM) tier, closes issue [#7](https://github.com/danielesalpietro/vMemoryFabric/issues/7)**

- `tier/pmem.py` (new): `PMEMTransfer` — fixed-slot pool allocator backed by a DAX-mounted mmap file (`numpy.memmap` on a `posix_fallocate`'d pool file) instead of anonymous DRAM. Uses `numpy.memmap`, not `libpmem2`/`MAP_SYNC` — real byte-addressable DAX access, not the fsync-free durability guarantee `MAP_SYNC` would give; stated explicitly in the module docstring
- `eat/types.py`: `Tier.PMEM = 3` (appended, not renumbered between `DDR4`/`NVME`)
- `tier/manager.py`: `TierManager(pmem_path=..., pmem_n_slots=...)`, both optional (`None` = tier disabled, zero behavior change for every other environment). New `promote()` hops NVME→PMEM and PMEM→DDR4; new `evict()` hop PMEM→NVME. `NVME→DDR4` direct stays available unconditionally — PMEM is an additional route, not a replacement; no `NVME→PMEM→VRAM` chaining yet (see issue #57)
- `benchmarks/bench_pmem_tier.py` (new, `make bench-pmem-tier`): raw read/write bandwidth + hop promotion latency; skips cleanly (not a hard failure) when no DAX mount is present
- 14 new tests (`TestPMEMTransfer`, `TestTierManagerPMEM`), all portable — mmap on a regular file behaves identically to DAX for correctness, no `@pytest.mark.gpu`
- A bug caught and fixed in this same change, same root cause as issue #48 below: `PMEMTransfer.read()` initially returned the whole fixed 256 MB slab slot instead of the real payload size — caught by a suspicious `pmem_to_ddr4` latency (116 ms for a 4 MB synthetic shard) before it was ever reported as a measured number
- **Not yet wired into the production GCSG/EAT path** — `GCSGWorker`'s `promote_live_tensor()` and `evict_to_free_vram()` have no PMEM awareness, and `eval_mmlu_gcsg.py` constructs `TierManager` without `pmem_path`. Tracked as issue [#57](https://github.com/danielesalpietro/vMemoryFabric/issues/57), including the larger open question (a real multi-tier placement policy — `SEEPolicy`/`LRUPolicy` today only reason about VRAM eviction)
- Measured on the Z8 bare-metal host: raw read 1.85 GB/s / write 0.61 GB/s (single-shot, no warm-up — flagged as a first-pass characterization, not a validated number), `nvme_to_pmem` P50 5.41 ms, `pmem_to_ddr4` P50 2.73 ms (4 MB synthetic shards)

**Linux host telemetry samplers**

- `osx-poc/scripts/telemetry/{host_sampler,pcie_sampler,docker_stats_sampler}.sh` — bash rewrite of `logs/z8_local/telemetry_fase6a/host_sampler.ps1`/`pcie_sampler.ps1` (PowerShell, Docker-on-Windows only) for Linux bare-metal hosts; same CSV schema for continuity with existing logs. Detachable (`setsid`/`nohup`), configurable output dir and `DURATION_S`/`INTERVAL_S`

### Fixed — 2026-08-24

**Issue [#12](https://github.com/danielesalpietro/vMemoryFabric/issues/12) — `make lint`/`test`/`bench`/`smoke` fail on relative paths**

- `docker-compose.yml`: `working_dir: /workspace/osx-poc` added to the `osx-dev` service — `docker compose run` always uses the image's `WORKDIR` (`/workspace`, repo root), never the host cwd `make` is invoked from, even from inside `osx-poc/`. No Makefile/Dockerfile changes needed (`PYTHONPATH` was already correct). Verified: `make smoke` 13/13 PASS on the Z8 bare-metal host, previously failing with `No such file or directory`

**Issue [#48](https://github.com/danielesalpietro/vMemoryFabric/issues/48) — `bench_ddr4_to_vram()` guaranteed CUDA OOM on any single ≤24 GB GPU**

- `SlabAllocator.get_buffer()` always returns the fixed `SHARD_SIZE_BYTES` (256 MB) slot, not the 4 MB synthetic payload the benchmark's own docstring describes — every `to_vram()` call transferred the full 256 MB slot. `_ddr4_to_vram()` never freed the promoted tensor (no `evict()` in the loop); with `N_SHARDS=100` (raised from 20 in the issue #3 fix above, *after* the two 2026-08-12 pod runs that produced the promotion-latency numbers already cited for target 4 in `poc_final_report.md` — those numbers come from the separate `promote_live_tensor` section, untouched by this fix) this accumulated 100 × 256 MB = 25.6 GB of permanently-resident VRAM — exceeds a single RTX 3090's 24 GB, guaranteed OOM, not host-specific. As far as the repo's own logs show, this section had never been run to completion on real single-GPU hardware since `N_SHARDS` was raised
- Fix: `mgr.evict()` after each timed promote, outside the `t0`/`latencies_us` window (doesn't skew the measured latency) — steady-state VRAM usage ~1 shard instead of accumulating all `N_SHARDS`. Verified: `ddr4_to_vram` now completes (P50 84.0 ms for 100 shards), `promote_live_tensor` pin=True P50 196.99 µs still matches the previously-reported 194.1/207.8 µs range

### Added

**EAT locking — selectable strategy, issue [#23](https://github.com/danielesalpietro/vMemoryFabric/issues/23)**

- `ExpertAccessTable(locking_strategy=...)` — `"single"` (Option A: `RLock` → `Lock`, no method re-enters another while holding the lock, so `RLock`'s owner/reentrancy bookkeeping was pure overhead), `"striped"` (Option B: table partitioned into `n_shards` independent shards, each with its own `Lock`; bulk methods — `get_tier`/`eviction_candidates`/`hottest_candidates`/`stats`/`__len__`/`__iter__` — read one shard at a time under its own lock, a deliberate relaxed snapshot rather than holding all shard locks together), or `"lockfree_read"` (Option C: `lookup()` skips locking entirely, relying on `dict.get()`'s atomicity under the GIL). `"single"`/`"lockfree_read"` are internally a `"striped"` special case with one shard — no behavior change for existing callers beyond the new default below
- `EATEntry.seqlock_write()` / `EATEntry.write_in_progress` (Option D — the `version` field already existed as an unused "CAS counter" comment; now a real seqlock) — `version` increments before (odd = write in progress) and after (even = stable) every mutation (`touch()`, `update_tier()`). Lets a `lockfree_read` caller detect a `touch()` caught mid-write (its two non-atomic field writes, `access_count` then `last_access_ts`) instead of silently trusting a possibly-torn read; the caller decides the policy (retry/discard/accept), no automatic enforcement
- `benchmarks/bench_eat.py::bench_contention_by_strategy()` — the existing `contention` benchmark (disjoint reader/writer keys) repeated across all three strategies
- `benchmarks/bench_eat.py::bench_churn_by_strategy()` (new scenario) — writer calls `access()` repeatedly on the *same* keys the readers query (the pattern flagged as more realistic of M2/M3 traffic in the PR #15 comment on issue #2, vs. the disjoint-key `contention` scenario). Readers apply the seqlock check and count detected torn reads directly, instead of leaving the race theoretical

**GCSG CPU-offload tier (DDR4-resident shadow experts), issue [#33](https://github.com/danielesalpietro/vMemoryFabric/issues/33)**

- `GCSGWorker(enable_cpu_offload=...)` / `configure_cpu_offload()` — opt-in, default `False`, byte-for-byte unchanged behavior unless enabled (a dedicated regression test, `..._disabled_by_default_even_with_tier_manager`, covers this for both quantization paths so a future regression here would fail loudly, not silently). Two CPU-resident pool paths: `_build_cpu_shadow_pool()` (path 1, INT4-simulated, ~3GB/expert) and `_build_cpu_shadow_pool_awq()` (path AWQ, fp32-cached AWQ dequant, ~21.5GB/expert) — `_ShadowExpertINT4` needed no new CPU kernel, confirmed device-agnostic by construction (the same class already runs the GPU path in production)
- `GCSGWorker._check_cpu_ram_budget()` / `_read_cgroup_available_gb()` — caps the CPU pool to RAM actually available in the process's cgroup (not `/proc/meminfo`'s host-wide view, which lies under Docker/RunPod — confirmed on a real pod: `free` reported 1TB, the real cgroup limit was 116GB), degrading the pool size rather than risking an OOM mid-build
- `OSX_GCSG_FORCE_CPU_ROUTE` (env var, debug/benchmark opt-in) — forces all shadow routing to the CPU pool regardless of hot/cold classification, for controlled GPU-only/CPU-only/mixed comparisons; env var absent leaves `route_forward()` byte-for-byte unchanged (dedicated test)
- `benchmarks/perf_test_hardware.py` + `benchmarks/perf_tuning_report.py` (`make perf-test-hardware` / `make perf-tuning-report`) — reusable hardware characterization (CPU cgroup cores/AVX-512-VNNI/isolated matmul GFLOPS, RAM cgroup GB/bandwidth, GPU/VRAM, PCIe pinned/pageable bandwidth) and tuning-recommendation framework, replacing one-off ad-hoc measurements (a prior session's manual `OMP_NUM_THREADS` A/B test). 38 unit tests (`tests/test_perf_test_hardware.py`, `tests/test_perf_tuning_report.py`); validated end-to-end on three real hosts (self-hosted `Z8-G4-RTX3090`, a RunPod RTX 3090/EPYC pod, a RunPod H200/Xeon Platinum 8568Y+ pod) — see `logs/runpod_perf_framework_validation_20260817/COMPARISON_ANALYSIS.md`

### Changed

**`ExpertAccessTable` default locking strategy: `"single"` → `"lockfree_read"`**

- Verified via grep that no production caller (`TierManager`, `GCSGWorker`, `eval_mmlu_gcsg.py`) passed `locking_strategy` explicitly, and none read `access_count`/`last_access_ts` off a raw `lookup()` result — hotness consumers (`gcsg.py`, `tier/policies.py`) go through the bulk methods, which stay locked regardless of strategy. On that basis `lockfree_read` was promoted to the default rather than left opt-in

### Fixed

**Issue [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2) — `RLock` p99 degradation under contention**

- Superseded the 2026-08-12 (Tekniska) decision to leave this open as a "known limitation" pending real production contention — implementing the locking-strategy options turned out to be cheap enough not to wait. `bench_contention_by_strategy()` (disjoint keys): p99 reader latency single→striped→lockfree_read = ~1119µs → ~611µs → ~1.3µs in sandbox CI, ~758µs → ~442µs → ~1.1µs on real hardware (`Z8-G4-RTX3090`, `full-gpu-tests` run [#150](https://github.com/danielesalpietro/vMemoryFabric/actions/runs/31726685030)) — ~700-900× improvement, confirmed on both environments, unlike the four-way sandbox/pod discrepancy this issue previously accumulated (see Tekniska entry, and `reports/poc_final_report.md` §1.2)
- `bench_churn_by_strategy()` (same-key writer/reader traffic): the torn read `lockfree_read` deliberately accepts is real but rare — 10/119,038 (~0.0084%) in sandbox, 10/410,829 (~0.0024%) on real hardware; zero on `single`/`striped` in both environments

**Issue [#3](https://github.com/danielesalpietro/vMemoryFabric/issues/3) — `bench_tier.py` DDR4→VRAM p95/p99 dominated by CUDA cold-start**

- `bench_ddr4_to_vram()`: added a dedicated warm-up shard (`_WARMUP_SHARD_IDX = N_SHARDS`, outside the measured range), promoted NVMe→DDR4→VRAM once and discarded before the timed loop — pays the one-time CUDA context/allocator init cost (`torch.from_numpy(...).to(device)`) that was previously absorbed by the first measured sample and dominating p95/p99 together (both landing on the same outlier at `N_SHARDS=20`)
- Bumped `N_SHARDS` 20 → 100 so p95/p99 land on distinct indices; `_seed_nvme_shards()` now takes `shard_indices` explicitly so the extra warm-up shard doesn't leak into `bench_nvme_to_ddr4()`, which doesn't need it (no CUDA on that hop)
- Originally opened against `Sprint-3-Oskarshamn` (the real `bench_tier.py` this fix targets didn't exist on `develop` yet); reapplied onto `develop`'s current file — which had grown a third section (`promote_live_tensor`, issue #17) in the meantime — once that branch merged

**Issue [#33](https://github.com/danielesalpietro/vMemoryFabric/issues/33) — CPU-offload tier: a real memoization bug and a false "mixed" run**

- `_ShadowExpertINT4.__call__()` recalculated `w13_q.to(dtype) * scale` on every call (every token, every layer, every cold expert) instead of once — negligible on the original INT4 path (small int8 source) but costly on the fp32-cache AWQ path (source already fp32, hundreds of MB). Fixed with per-`(layer_id, dtype)` memoization in `__init__`/`__call__`, numerically verified identical to the pre-fix behavior (scale is always 1.0 on both paths)
- A three-phase GPU-only/CPU-only/mixed comparison found the "mixed" scenario had never actually existed in any prior run: `ExpertAccessTable` is purely in-process and non-persistent (`EAT.shutdown()` clears it unconditionally), so at cold start every process re-derives hot/cold purely from expert-insertion order — with this project's 2-expert shadow pool (`[0, 1]`) and `hot_fraction=0.5` on 8 experts, the pool always lands entirely in the "cold" half. Every "mixed" run measured to date was actually CPU-only, confirmed empirically (forced-CPU vs. "mixed": 1072.3s vs. 1133.1s, a 5.7% difference consistent with run-to-run noise, not a structural one). Root cause of the missing "mixed" path is architectural, not a bug — tracked as issue [#27](https://github.com/danielesalpietro/vMemoryFabric/issues/27) (EPM, cross-process EAT persistence), not fixed here
- `benchmarks/perf_test_hardware.py` — `avx_support["avx512_vnni"]` false-negatived on hardware that has it: the check looked for `"avx512vnni"` (no underscore), but the Linux kernel's `/proc/cpuinfo` naming for this specific flag has one (`avx512_vnni`), unlike `avx512f`/`avx512bw`/etc. Found running the framework on a real Xeon Platinum 8568Y+ pod, not in a unit test — a dedicated regression test now pins the real kernel naming

### Status — issue #33 (CPU-offload tier)

**Correctness closed, performance open (opt-in feature, default off — zero risk to the existing production path).** Proven identical across three independent real-hardware comparisons on the same n=16 MMLU slice (GPU-only, CPU-only forced via `OSX_GCSG_FORCE_CPU_ROUTE`, and the "mixed" run that turned out to be CPU-only) — same 50.0% accuracy, same `shadow_activations=10081` count in every case: the CPU AWQ-dequant path produces the same result as GPU, not an approximation. Performance is not production-ready: **17-570× slower than GPU-only, depending on host** (measured across three real hosts — self-hosted `Z8-G4-RTX3090`, a RunPod RTX 3090/EPYC pod, a RunPod H200/Xeon Platinum 8568Y+ pod). Root cause isolated, not assumed: the per-token forward is a single-row GEMV, memory-bandwidth-bound by construction — falsified as a thread-count problem with a direct A/B test (`OMP_NUM_THREADS=27` vs. `2`, same latency), and confirmed by a clean cross-host correlation between RAM bandwidth and CPU throughput that holds independently of AVX-512/AMX presence (the AMX-capable Xeon Platinum pod was the *slowest* of the three, not the fastest — see `logs/runpod_perf_framework_validation_20260817/COMPARISON_ANALYSIS.md`). Open, non-blocking directions: issue #27 (EPM) or a periodic hot/cold refresh tick for a genuine mixed-routing comparison; batching the per-token GEMV into a GEMM to use otherwise-idle CPU cores (host-dependent potential gain, 3-17× measured); issue [#45](https://github.com/danielesalpietro/vMemoryFabric/issues/45) (whether AMX/BF16 or a per-core RAM-bandwidth constraint explains the H200 pod's CPU throughput).

### Tracking

- Issues [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2), [#3](https://github.com/danielesalpietro/vMemoryFabric/issues/3), and [#23](https://github.com/danielesalpietro/vMemoryFabric/issues/23) closed (`state_reason: completed`); #2/#23 confirmed on real hardware before closing, not just sandbox CI
- `reports/sprint5_berg_plan.md` §2/A1 updated — #2 was scoped there as "document as known limitation, not fix this sprint" one day before this fix landed; corrected in place rather than left inconsistent
- Issue [#33](https://github.com/danielesalpietro/vMemoryFabric/issues/33) stays open (performance, not correctness — see Status above); issue [#45](https://github.com/danielesalpietro/vMemoryFabric/issues/45) opened as a follow-on to today's three-host comparison. Full narrative across ~22 working sessions: `LOGBOOK_ISSUE33.MD`; branch `claude/ddr4-ram-processing-unzseh`, PR [#42](https://github.com/danielesalpietro/vMemoryFabric/pull/42) into `develop`, **merged 2026-08-24** (see the 2026-08-24 entries above for what landed alongside it)
- **Branch consolidation (2026-08-14):** `Sprint-3-Oskarshamn`, `claude/sprint-5-berg-plan-e4dsc3` (Tekniska), `claude/log-folder-organization-uh5f1a`, and this branch (`claude/rlock-data-quality-9pcxzq`) were all merged into `develop` in dependency order (PRs #9, #29, #25, #26) — `develop` had none of Sprint 3/4's real M2/M3 implementation until this point, still running the Sprint 0/1 `NotImplementedError` stubs. PR [#13](https://github.com/danielesalpietro/vMemoryFabric/pull/13) ("fix: bound Bloom filter FP leak") was closed rather than merged: superseded by this branch's own, independent removal of the Bloom filter entirely (issue #1). PR [#14](https://github.com/danielesalpietro/vMemoryFabric/pull/14) (issue #3, above) was retargeted from the now-merged `Sprint-3-Oskarshamn` to `develop` and reapplied. See `reports/sprint5_berg_plan.md` §0 for the issue-tracker-hygiene principle this extends.

---

## [Tekniska] — v0.5.0-dev — 2026-08-12

*Named after the Tekniska museet, Stockholm, where the Sprint 4 kickoff conversation happened.*

### Added

**Sub-goal 1 — GCSGWorker wired to TierManager/EAT for real, both quantization paths (issue [#17](https://github.com/danielesalpietro/vMemoryFabric/issues/17))**

- `GCSGWorker(tier_manager=...)` — opt-in, default `None`, byte-for-byte unchanged behavior unless wired. Since vLLM constructs the worker itself via a string `worker_cls` (no path for an extra kwarg to reach it), added `GCSGWorker.configure_tier_manager()` as a classmethod set *before* `LLM(...)`/`EngineArgs(...)` construction — `__init__` falls back to the pending value when `tier_manager=` isn't passed directly
- `TierManager.promote_live_tensor()` (new) — a live-tensor promotion bridge, since `TierManager.promote()`'s NVMe→DDR4→VRAM chain expects shard *files* on NVMe and GCSG's shadow experts are live in-process tensors, not files. Registers/updates the tensor in EAT (seeded at `Tier.DDR4`) and moves it to VRAM via `GPUTransfer`, skipping the NVMe hop that doesn't apply here. Idempotent by key
- `EAT.hottest_candidates(tier, n)` (new) — the promotion-side complement to `eviction_candidates()` (LRU/coldest-first); ranks by `(access_count, last_access_ts)` descending
- `GPUTransfer.to_vram()` — now accepts `pin: bool` (default `False`) and a `torch.Tensor` directly, not just numpy
- `_seed_eat_entries()` / `_select_shadow_expert_ids()` — seeds one EAT entry per `(expert_id, layer_id)` at load time for *all* experts, and replaces the round-robin `range(shadow_pool_size)` placeholder with EAT-hotness-driven selection when a `TierManager` is wired. Real per-token EAT traffic now flows through `_evaluate_gcsg_for_rows()` → `EAT.access()` independent of shadow activation — the "no real concurrent EAT traffic" precondition Sprint 1/2 were waiting on before touching issues #1/#2/#4
- **AWQ ModuleList path (path 3)** — `_pin_awq_expert_to_gpu()` routes its GPU-residency transfer through `TierManager.promote_live_tensor()` instead of a direct `.to('cuda')` when wired. Verified 4/5 checklist items on a real RTX 3090 (WSL2 — `pin=True` correctly disabled there by design) and 5/5 on a second real GPU (real Linux, `pin=True` confirmed end-to-end)
- **Marlin path (path 2)** — wired only after AWQ was validated twice with perfect determinism, since `_PinnedMarlinExperts` is the most fragile mechanism in the file (its own docstring documents the 2026-08-10 CUDA-allocator-fragmentation hang from an earlier attempt at pinning all 32 layers instead of only the offloaded ones). New `GCSGWorker._build_marlin_tensor_promoter(layer_id, expert_ids)` routes one dominant tensor per layer (`w13_qweight`) through `promote_live_tensor()`; EAT key uses a sentinel (`expert_id=-1`) with `shard_idx = _marlin_pool_shard_key(layer_id, expert_ids)`, encoding both `layer_id` and `hash(tuple(sorted(expert_ids)))` — caught before writing it, not after: a key varying only by `layer_id` would let a future `refresh_shadow_pool_selection()` silently reuse stale VRAM data across a changed pool composition
- `scripts/smoke_test_gcsg_tier_manager.py` (new) — hardware verification checklist, extended with `--quantization {awq,awq_marlin}`
- 98 unit tests passing by end of sub-goal 1 work (up from 91 pre-sprint)

**Sub-goal 2 — real pinning wired in, soak-tested safe on real Linux**

- `torch.Tensor.pin_memory()`, bypassing vLLM's own `is_pin_memory_available()`/WSL2 gate, now wired into `GPUTransfer.to_vram(pin=True)` — opt-in, default `False`
- `scripts/verify_pin_memory_soak.py` (new) — **1000/1000 cycles, 0 byte-exact mismatches**, real pinned 256MB shards (the same unit `TierManager`/`EAT` operate on), on the RTX A5000 pod. Total-cycle drift -1%, H2D+D2H transfer-time drift -3%, pin-alloc-specific drift **-31%** (pinning got *faster* after the first several hundred cycles — consistent with host-allocator warm-up, no sign of the fragmentation-style degradation seen once before on 2026-08-10). 1000 cycles in 589s (~513ms/cycle average, including a fresh pin + H2D + D2H + byte comparison each time)

**Sub-goal 3 — `--wire-tier-manager` on `eval_mmlu_gcsg.py`**

- New opt-in flag (default off, zero change to the existing baseline path); forces `quantization="awq"` on the AWQ path, `awq_marlin` once the Marlin path was wired

**Sub-goal 4 — promotion latency benchmark**

- `benchmarks/bench_tier.py::bench_promote_live_tensor` (new), `pin=True` vs. `pin=False`

**Sub-goal 6 — path 1 real-offload harness**

- `scripts/probe_kv_blocks.py` extended: `--model-path`, `--cpu-offload-gb`, `--quantization none`
- `scripts/smoke_test_gcsg_path1_real_offload.py` (new) — same watchdog+heartbeat+checklist idiom as every other smoke test in this project, including a direct numerical check of the INT4 quantize/dequantize/SwiGLU math at real `hidden_size=4096` (only ever verified before at the tiny test model's `hidden_size=1024`)

### Fixed

**`_build_marlin_shadow_pool()` — inferred a whole layer's offload state from one tensor (first real-hardware Marlin failure)**

- First hardware run of the Marlin+TierManager wiring failed: `cannot pin 'torch.cuda.HalfTensor' — only dense CPU tensors can be pinned`, `worker._shadow_pool` empty. Root cause: the function decided "offloaded" by checking only `w13_qweight.data.device.type`, but vLLM's `cpu_offload_gb` doesn't offload a Marlin-packed layer's six tensors (`w13_qweight`/`w2_qweight`/`w13_scales`/`w2_scales`/`w13_qzeros`/`w2_qzeros`) as one indivisible unit — one of the other five could stay GPU-resident while `w13_qweight` was correctly offloaded, and the non-dominant-tensor promotion branch called `.pin_memory()` unconditionally on whatever it was handed. The AWQ path's equivalent function already guarded this (`p.device.type != "cuda"` filter); the Marlin promoter never got the same guard. Notably **not** the sentinel-key/`_marlin_pool_shard_key` risk both sessions initially suspected — one layer more basic
- Fix: `_promoter()` now checks `cpu_slice.device.type == "cuda"` first and short-circuits before either the dominant-tensor `promote_live_tensor()` branch or the non-dominant `.pin_memory()` branch. 2 new regression tests (`TestMarlinTensorPromoterDeviceCheck`) reproduce the exact scenario. Reran on real hardware afterward, twice independently: green both times, "Marlin-path sentinel entries (expert_id=-1) reached Tier.VRAM: 6 confirmed"

**Path 1 (`_ShadowExpertINT4`) — device mismatch under real offload on A100 SXM 80GB**

- First real-checkpoint, real-offload test of path 1 (`mistralai/Mixtral-8x7B-Instruct-v0.1`, unquantized, ~93.4GB, `cpu_offload_gb=28`) failed at `generate()`: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (mat2 in wrapper_CUDA_mm)`, traced to `gcsg.py:452`, `_ShadowExpertINT4.__call__`'s `hidden_states @ w13.T`. Root cause: the fused-FusedMoE build loop constructed the INT4-dequantized shadow weight `w13` directly from the source `w13_weight`, which under real `cpu_offload_gb` can itself be CPU-resident — nothing moved the dequantized result to GPU, because the only prior test of this path (a tiny, never-offloaded model) had every source weight on GPU already, so the gap never had a chance to show. Fixed with the same device-check-then-pin pattern used on paths 2/3. Retest green: `generate()` completed (`' The sum of 2 + 2'`), `.gate` hooks fired 288 times, `shadow_activations=35` (5.0%), direct shadow-expert forward at `hidden_size=4096` finite and correctly shaped

**`GCSGWorker.__getattr__` — `RecursionError` instead of a clean `AttributeError` on test doubles**

- `__getattr__` delegated unknown attributes to `self._base` unconditionally; on a worker built via `__new__()` with `_base` never set (the established test pattern bypassing vLLM's constructor), any missing-attribute access recursed infinitely instead of raising cleanly. Pre-existing latent bug, surfaced only once the new `_tier_manager` read in `_evaluate_gcsg_for_rows()` exercised it. Fixed with an explicit `_base` presence check via `__dict__` before delegating

**Cold-start expert selection — silent insertion-order bias, caught by a unit test before hardware**

- First draft of `_select_shadow_expert_ids()` used `last_access_ts` as a tie-break for cold-start selection ("prefer fresher"), which turned out to silently bias toward whichever expert `_seed_eat_entries()` happened to insert *last* — an insertion-order artifact, not real hotness. Caught writing the cold-start unit test, not on hardware; dropped the tie-break entirely rather than patch around it. At true cold start (all `access_count == 0`), selection is now provably equivalent to the prior round-robin placeholder, since Python's `sorted(..., reverse=True)` is stable

**Issue [#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4) — `BloomFilter.remove_expert()` fixed, then made moot the same day**

- `remove_expert()` had raised `NotImplementedError` since Sprint 1 and was never called from anywhere, not even `EAT.evict()` — two gaps, not one. A classic Bloom filter can't support deletion at all, so the original `pybloom_live`-backed implementation could never have supported it; replaced with a self-contained Counting Bloom Filter (`_CountingBloomFilter`, 8-bit counters, double hashing via two `blake2b` digests, no new dependency) and wired `EAT.evict()` to call the new `remove_shard()`/`remove_expert()` correctly (expert-level entry only cleared when it was the *last* remaining shard). New regression test: 20 cycles of insert-then-evict 500 entries (10,000 cumulative "ghosts") — false-positive rate stays under 2%; would have failed loudly against the old stub. Made moot hours later when issue #1 was closed by removing the Bloom filter entirely (see Changed)

### Changed

**Issue [#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1) — Bloom filter removed from EAT's hot path entirely, not tuned**

- The Counting Bloom Filter fix above also produced fresh numbers: hit-lookup ratio vs. a plain dict went from ~4.7-4.9× (old `pybloom_live`) to ~6.8-8.1× (new Counting BF) — worse, not better; insert throughput improved (~50,715 → ~74,000-98,000 ops/sec). Net effect on the open question ("does the Bloom filter belong in the hot path at all") was a wash. Decided with the project owner: the structure it guards (`self._table`, a dict at ~16k-entry scale) is already O(1) and doesn't need a fast-negative layer in front of it regardless of which implementation backs it. `src/eat/bloom.py` deleted entirely (not bypassed); `EAT.insert()`/`lookup()`/`evict()` simplified back to direct dict operations under the existing `RLock`; `pybloom-live==4.0.0` removed from `requirements.txt`. `benchmarks/bench_eat.py` repurposed from a Bloom-vs-dict comparison into a regression check — `eat` and a plain-dict baseline are now expected to converge, and do: delta dropped from ~4-5µs to **~0.019-0.07µs**

**Issue [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2) — RLock contention: decided, left open deliberately, not redesigned**

- Re-measured `bench_eat.py::bench_contention` (4 readers + 1 writer) under this sprint's real EAT traffic: p99 degradation landed at **~61-91×** across 4 runs, 15-20× smaller than the issue's originally-cited **~1360×**, gap unexplained (plausible causes — different machine, Python/GIL scheduling, an unrepeated original measurement — none confirmed). More decisive than the gap itself: real `GCSGWorker` traffic today is single-threaded (`_evaluate_gcsg_for_rows()` runs synchronously inside vLLM's own forward-pass hook), so the multi-threaded contention scenario issue #2 measures isn't produced by anything running in this project yet. Redesigning the `RLock` now would be speculative work against a scenario that doesn't exist. Decision, requested and given explicitly by the project owner: leave the `RLock` as-is; issue #2 stays open, re-scoped from "in progress" to "blocked on real concurrent EAT access actually existing" (a future multi-worker/async-server setup), not closed and not silently forgotten. Both the original ~1360× and today's ~61-91× figures stay recorded side by side, neither overwriting the other. **Superseded 2026-08-13 — see the Berg entry above: implementing selectable locking turned out to be cheap enough not to wait for real contention to exist first.**

### Status

**Complete — all 7 sub-goals of issue [#17](https://github.com/danielesalpietro/vMemoryFabric/issues/17) done or closed.**

- **Sub-goal 1 (wiring):** both quantization paths hardware-verified — AWQ 5/5, Marlin 6/6 sentinel `Tier.VRAM` promotions confirmed, reproduced on a second independent pod run
- **Sub-goal 2 (pinning):** closed positively — 1000-cycle soak test, 0 mismatches, no degradation, real pinning now the defensible default on real Linux (not WSL2, where vLLM disables it structurally)
- **Sub-goal 3 (integrated MMLU rerun):** AWQ+TierManager landed at **411/570 (72.1%)**, same total as the historical baseline with 4 individual answers flipped (0.7%, inside the <2% target), confirmed byte-identical on a determinism rerun. Marlin+TierManager later matched **412/570 (72.3%)** exactly against the historical Marlin baseline, plus an exact 24/32 per-subject match on a spot-check slice — both quantization paths now have real MMLU numbers on the integrated path, not just mechanical checklists
- **Sub-goal 4 (promotion latency):** `pin=True` P50 **194.1µs** vs. `pin=False` P50 **684.2µs** (~3.5× faster) — meets the README's "within 1.5× theoretical bandwidth" criterion at P50 (threshold 732.4µs), reproduced on a second run (207.8µs/620.7µs). P95/P99 move the other way under `pin=True`, but at n=20 synthetic 4MB shards that isn't a statistically meaningful tail measurement — not investigated further
- **Sub-goal 5 (M1 debt):** #1 closed (Bloom filter removed), #4 moot (same removal), #2 decided-and-deferred at the time (see Changed — resolved a sprint later, in Berg)
- **Sub-goal 6 (path 1 parity):** verified end-to-end on A100 SXM 80GB against the real 93GB unquantized checkpoint under `cpu_offload_gb=28`; one real device-mismatch bug found and fixed (see Fixed). Mechanics/correctness only — no MMLU number on this path, no production-viability claim for this offload configuration
- **Sub-goal 7 (close-out):** `reports/gcsg_shadow_execution_report.md` EN/IT `.docx` exports regenerated via `pandoc --reference-doc=...`, several generic pandoc-template OOXML ordering bugs hand-fixed (`nsid` zero-padding, `pStyle`-before-`numPr`, `jc`-before-`tblLook`, `settings.xml`'s `CT_Settings` sequence), both pass full XSD schema validation with zero errors, 19 heading styles matching the markdown's 19 `#` headings

**Environment note:** most of this sprint ran on a RunPod RTX A5000 (GA102, CC 8.6, 24GB) rather than the RTX 3090 reference hardware every prior measurement is calibrated against — a substitution, not a deviation, since it's the same architecture/VRAM class and was the first choice among WSL2-escape candidates for exactly that reason. An RTX 4090 (Ada Lovelace, CC 8.9 — a different generation) was used briefly for architecture-independent SSH/`pin_memory` checks only, explicitly flagged as not pre-approved for MMLU numbers. Sub-goal 6 required an A100 SXM 80GB for VRAM headroom; an RTX PRO 6000 Blackwell (sm_120) attempt was retired first — the project's pinned `torch==2.5.1+cu124` reports no `sm_120` in `get_arch_list()`, a hardware/library mismatch, not a project bug.

### Tracking

- Issue [#17](https://github.com/danielesalpietro/vMemoryFabric/issues/17) closed — all 7 sub-goals done
- Issue [#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1) closed (Bloom filter removed); issue [#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4) closed as moot by the same removal; issue [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2) stays open at this point, re-scoped to blocked-on-real-concurrency (closed the following sprint — see Berg)
- Known open item, not blocking sprint close: no root cause identified yet for the small, stable per-subject divergence pattern (4-5 subjects, ±1 question each, net near zero) seen on both the AWQ+TierManager and Marlin+TierManager reruns against their respective baselines — an early hypothesis (the `awq`-vs-`awq_marlin` kernel switch) was tested directly and ruled out, since the Marlin+TierManager rerun uses the identical kernel as its baseline and shows the same class of divergence anyway

---

## [Oskarshamn] — v0.4.0-dev — 2026-08-10 (merged into `develop` 2026-08-14, PR #9)

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

