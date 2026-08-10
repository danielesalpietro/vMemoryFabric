# OSX — Low-Level Design v1.0

> Status: **scaffold**. Le intestazioni e l'ordine delle sezioni riflettono la struttura
> comunicata per OSX LLD v1.0. Il contenuto di ogni sezione è popolato solo dove
> corrisponde a qualcosa di verificabile nel repo (`osx-poc/src`, `configs/osx_default.yaml`,
> `CHANGELOG.MD`) al 2026-08-10. Le sezioni senza corrispondenza nel codice sono marcate
> `☐ TODO` — contenuto da fornire, non inventato.

---

## 1. Application layer

☐ TODO — nessun riferimento nel repo. Da definire: superficie di integrazione con
l'application layer (hook vLLM esistenti sono `pre-tokenization` per PT-PEP e
`post-gating, pre-expert-execution` per GCSG — vedi §2).

---

## 2. Expert Scheduler — M3 (PT-PEP / GCSG / AER, solo Base+LoRA-Delta)

Stato implementativo: **skeleton, tutti i metodi `NotImplementedError`** (Sprint 3, in corso).

### PT-PEP — Pre-Tokenization Prompt-Expert Predictor
- Modello: BERT-small fine-tuned, 8 domini (`coding, math, language, science, medical, legal, creative, general`)
- Runtime: ONNX su onnxruntime CPU, target < 3 ms p99 (Xeon 6244)
- Pipeline: dominio → expert_ids → prefetch_queue (EAT + Tier Manager) → promozione shard pre-forward-pass
- Training: ~2.000 prompt (OpenHermes, MetaMathQA, CodeAlpaca, PubMedQA, LegalBench) + labeling automatico
- Config: `confidence_threshold = 0.6` (`configs/osx_default.yaml`)
- Non implementato: `load`, `unload`, `predict`, `predict_batch`, `stats`

### GCSG — Gating Confidence Shadow Guard
- Soglie: `θ_gate=0.85`, `θ_entropy=0.70`, `θ_contamination=0.05`
- Shadow execution attiva solo se: BF16 non disponibile/budget insufficiente **e** gating_score > θ_gate **e** token_entropy < θ_entropy **e** contamination < θ_contamination
- Vincolo dev (single RTX 3090 24GB): shadow pool teorico top-8 INT4 (~3GB cad. ≈ 24GB) → ridotto a **top-4** per headroom al modello BF16 attivo (`shadow_pool_size: 4` in config)
- Hook: monkey-patch su `_run_workers()` vLLM, post-gating pre-expert-execution
- Non implementato: `should_activate_shadow`, `run_shadow`, `contamination_rate`, `reset_contamination_counter`, `update_thresholds`, `stats`

### AER — Adaptive Expert Replication (Base+LoRA-Delta)
- Design confermato nel codice: **base weights immutabili**, solo il **LoRA Delta** viene sincronizzato tra repliche (via PCIe; NVLink deferred, no RTX 5080 in dev) — coerente con "solo Base+LoRA-Delta" della LLD
- Stato dev: **stub puro** — `replication_factor()` ritorna sempre `1`, `sync_lora_delta()` è no-op (`pass`)
- Attivazione reale: deferred a setup dual-GPU (arrivo RTX 5080)

---

## 3. EMM — M1+M2 (EAT entry 28B + EMH tier con latenze e flussi)

### EAT — Expert Access Table (M1, **implementato e benchmarkato**, release Möllstorp v0.2.0-dev)
- Layout target dichiarato nel codice: **28 bytes/entry** (`EATEntry` docstring, `types.py:36`)
  - ⚠️ **non ancora vero a runtime** — vedi documento di confronto, §EAT layout
- Bloom filter 2 livelli (expert-level + shard-level), false positive rate 1%/livello
- Slab allocator: pool DDR4, slot fissi 256MB, variable-tail sull'ultimo shard
- CRUD completo: `insert`, `lookup`, `update_tier` (version bump/CAS), `evict`, `access`, `get_tier`, `eviction_candidates` (LRU)
- **Numeri reali misurati** (Z8-G4-RTX3090, CI `full-gpu-tests`): lookup p50 ≈ 2.6 µs / p95 ≈ 3.8 µs / p99 ≈ 4.6 µs; ≈177k insert/sec
- 24/24 test passing (`tests/test_eat.py`)

### EMH — Expert Memory Hierarchy tier
Tier **documentati nel repo**: 4 (non 5 — vedi documento di confronto)

| Tier | Nome repo | Ruolo | Stato dev |
|---|---|---|---|
| EMH-1a | VRAM (RTX 3090, 24GB) | hot | ✅ attivo |
| EMH-1c | DDR4 (256GB) | warm buffer | ✅ attivo |
| EMH-2 | PMEM Optane | — | ❌ deferred (richiede bare-metal Linux ≥5.1) |
| EMH-3 | NVMe/volume (1TB) | cold | ✅ attivo |

Flusso di promozione dichiarato in `TierManager`: `NVMe (EMH-3) → DDR4 (EMH-1c) → VRAM (EMH-1a)`.
Latenze di transizione: **non ancora misurate** — `promote()`, `_nvme_to_ddr4()`, `_ddr4_to_vram()`, `evict()`, `evict_to_free_vram()`, `prefetch()` sono tutti `NotImplementedError` (Sprint 2, non ancora chiuso).

☐ TODO — la LLD parla di **5 tier**: manca un tier rispetto a quanto documentato in README/config. Da chiarire quale sia il quinto (vedi documento di confronto).

---

## 4. ICP — single-layer Commodity vs two-layer NVL72

☐ TODO — nessun riferimento nel repo (nessun modulo, nessuna menzione in README/config/changelog). Componente interamente da specificare: cosa significa "ICP", cosa distingue il layout single-layer commodity dal two-layer NVL72, e come si interfaccia con EMH/Tier Manager.

---

## 5. IoX Fabric

☐ TODO — nessun riferimento nel repo. Da specificare: protocollo, topologia, relazione con `AsyncNVMeIO`/`GPUTransfer` esistenti (che oggi sono interfacce locali single-node, non fabric).

---

## 6. Metrics

Base esistente: Prometheus, porta 9090, scrape interval 5s (`configs/osx_default.yaml`), sidecar opzionale in `docker-compose.yml`. `EAT.stats()` espone `total_entries`, `by_tier`, `bloom_shard_count`; `TierManager.stats()` e gli `stats()` di PT-PEP/GCSG/AER sono ancora stub o parziali (AER espone solo `replication_enabled`/`reason`).

☐ TODO — metriche specifiche richieste dalla LLD v1.0 (oltre a quelle già esposte) da elencare esplicitamente.

---

## 7. Open Problems

Dal codice/changelog, problemi aperti già noti e misurati (non ipotetici):
- **Bloom filter costa più di quanto risparmia** nel design single-process attuale: ~5× più lento in lookup, ~14.6× in insert vs. baseline plain-dict (misurato, non stimato) — perché la struttura che protegge è già un dict O(1) in-memory
- **Tail latency sotto contesa**: p99 di lookup concorrente (4 reader + 1 writer) sale di ~1.360× per via del RW lock (`RLock`) in `eat.py`
- Bloom filter non supporta cancellazione — `remove_expert()` non implementato, richiederebbe Counting Bloom Filter o rebuild periodico
- Varianza run-to-run non quantificata: due run identici su hardware reale differiscono ~32% nel throughput di insert

☐ TODO — eventuali "Open Problems" aggiuntivi specifici della LLD v1.0 (es. relativi a ICP/IoX Fabric) da aggiungere.

---

## 8. Benchmark Suite

| Bench | Stato |
|---|---|
| `bench_eat.py` (M1) | ✅ implementato — P50/P95/P99 lookup, insert throughput, baseline plain-dict, contention profile, slab-at-scale |
| `bench_tier.py` (M2) | 🔲 target Sprint 2, non ancora implementato (Tier Manager stesso è stub) |
| B4 | 🔲 **marcato come rimandato**, per come indicato — nessun riferimento nel repo a cosa sia B4 specificamente; ☐ TODO chiarire scope

---

## Riferimenti verificati
`osx-poc/src/eat/*.py`, `osx-poc/src/tier/*.py`, `osx-poc/src/scheduler/*.py`,
`osx-poc/configs/osx_default.yaml`, `osx-poc/CHANGELOG.MD`, `osx-poc/README.md` — letti 2026-08-10.
