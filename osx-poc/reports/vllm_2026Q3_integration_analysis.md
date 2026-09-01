# vLLM giugno–agosto 2026 (0.22.1 → 0.28.0) — cosa cambia per vMemoryFabric e dove innestarlo

**Data:** 2026-09-01
**Ruolo:** analisi da solutions architect, sullo stesso modello di
[`component_reuse_analysis.md`](component_reuse_analysis.md) — ogni affermazione
confrontata col codice, non con le release notes, e ogni novità confrontata con
ciò che `osx-poc/src/` fa **oggi**, non con la roadmap.
**Prerequisito:** [`vllm_torch27_compat_analysis.md`](vllm_torch27_compat_analysis.md)
(2026-08-31), di cui questa è la continuazione: quella si fermava a 0.10.1 perché
lì si chiudeva la finestra di `GCSGWorker`; questa guarda che cosa c'è dall'altra
parte di quella chiusura.

**Perimetro.** Sono state scaricate ed estratte le sdist ufficiali
`vllm-0.22.1` (2026-06-05) e `vllm-0.28.0` (2026-08-26) e confrontate riga per
riga. **Nessun `pip install vllm`, nessun run, nessun benchmark.** Dati grezzi e
comandi: [`logs/new_z8_bare_metal/vllm_0280_integration_20260901/`](../../logs/new_z8_bare_metal/vllm_0280_integration_20260901/).

---

## Conclusione in tre righe

1. **La lista di partenza è quasi tutta vera** (2 voci su ~20 non esistono col
   nome citato) **ma è la lista sbagliata**: nessuna di quelle voci è la novità
   che conta per questo progetto.
2. **Quella che conta è che vLLM, in questa finestra, ha triplicato
   `v1/kv_offload/` (3 556 → 11 459 LOC) e ci ha costruito dentro una EMH** —
   tier multipli, promotion/cascade, policy di eviction sostituibili, metriche —
   **per la KV cache**. È la stessa architettura di M1+M2, per un oggetto diverso.
3. Questo **non svuota vMemoryFabric, gli dà tre prese di corrente**: vLLM 0.28.0
   espone tre registry che caricano classi **out-of-tree via `module_path`, senza
   fork né patch** (tier secondario, cache policy, KV connector). Il costo di
   integrazione crolla; quello che va riscritto è l'innesto di M3, che non ha più
   una classe base.

---

## 1. Verifica della lista di partenza

Ogni nome cercato nell'albero `.py` di 0.28.0
(`vllm_0280_feature_verification.txt`).

| Voce della richiesta | Esito | Dove |
|---|---|---|
| DFlash | ✅ esiste | `DFlashModelTypes = Literal["dflash"]`, `config/speculative.py:64` |
| DSpark | ✅ esiste | `DSparkModelTypes = Literal["dspark"]`, idem `:65`; c'è anche `models/gemma4_dspark.py` |
| **P-EAGLE** | ❌ **non esiste** | i 5 file che matchano `p_eagle` sono falsi positivi da sottostringa (`_setup_eagle3_aux_hidden_state_outputs`) |
| **EAGLE 3.1** | ❌ **non esiste con questo nome** | il metodo dichiarato è `eagle3`, senza minor version |
| Elastic Expert Parallelism | ✅ esiste | `vllm/distributed/elastic_ep/` (3 moduli + `standby_state.py`) |
| Decode Context Parallelism | ✅ esiste | `decode_context_parallel_size`, `config/parallel.py:342` |
| Kernel Blackwell / GDN | ✅ esiste | `v1/attention/backends/gdn_attn.py`; `CUDA_SUPPORTED_ARCHS` include `12.0` e `12.1` |
| DiffusionGemma / dLLM | ✅ esiste | `model_executor/models/diffusion_gemma.py` |
| Kimi K3, DeepSeek V4 (MTP), GLM, MiniMax M3, Nemotron, Qwen3.5 | ✅ esistono | `MTPModelTypes` elenca `kimi_k3_mtp`, `minimax_m3_mtp`, `qwen3_5_mtp`, `nemotron_h_mtp`, `glm4_moe_mtp`, `gemma4_mtp`, … |
| **PegaFlow / Novita** | ⚠️ **non nel core** | zero occorrenze nell'albero vLLM. Se esiste è un connector esterno, e allora vive dietro `KVConnectorFactory` come tutti gli altri (§3.C) |
| **Semantic Router / Themis** | ⚠️ **non nel core** | zero occorrenze: è un repo separato dell'organizzazione, non un modulo di `vllm` |
| 25 000 TPS/GPU su Qwen3.5 397B | ⛔ **non verificabile qui** | è un numero di benchmark, non un simbolo: nessun sorgente lo può confermare o smentire. Trattarlo come claim di marketing finché non c'è un run |

Le due voci inesistenti sono registrate anche in
[`README.md`](../../logs/new_z8_bare_metal/vllm_0280_integration_20260901/README.md)
dei dati grezzi, per la stessa ragione della correzione analoga di agosto: un
dettaglio non verificato si propaga per mesi.

**Nessuna delle voci verificate sopra tocca vMemoryFabric.** Speculative decoding,
DCP, i modelli day-0 e i kernel Blackwell agiscono tutti *sopra* o *accanto* al
livello in cui vive questo progetto. L'unica con adiacenza reale è Elastic EP —
e non è quella che pensa (§4).

---

## 2. La novità vera: vLLM ha costruito la sua EMH

Delta misurato sull'esatta finestra richiesta
(`vllm_subsystem_delta_0221_0280.txt`):

| Sottosistema | LOC 0.22.1 → 0.28.0 | Lettura |
|---|---|---|
| **`v1/kv_offload/`** | **3 556 → 11 459 (+222 %)** | è **qui** che è successo tutto |
| `distributed/kv_transfer/kv_connector/v1/` | 25 465 → 31 837 (+25 %) | crescita del parco connector (NIXL push/pull, mooncake, moriio) |
| `distributed/eplb/` | 3 253 → 3 508 (+8 %) | manutenzione |
| `distributed/elastic_ep/` | 1 385 → 1 273 (−8 %) | consolidamento, non espansione |
| `model_executor/offloader/` | **1 163 → 1 163 (invariato)** | offload dei **pesi**: fermo da prima della finestra |

Cosa è comparso dentro `v1/kv_offload/` in tre mesi: `tiering/obj/` (object
store), `tiering/p2p/` (tier remoto completo — data plane NIXL, control plane
ZMQ, session server/client: 4 543 LOC da sole), `tiering/async_lookup.py`,
`tiering/metrics.py`, `cpu/policies/factory.py`, `cpu/swap_blocks_triton.py`.

Messo accanto a `osx-poc/src/`, il parallelo è imbarazzante nella sua precisione:

| Concetto vMemoryFabric | Equivalente vLLM 0.28.0 | Oggetto gestito |
|---|---|---|
| `TierManager` (`tier/manager.py`) | `TieringOffloadingManager` (`tiering/manager.py`, 894 LOC) | vMF: **shard di expert** — vLLM: **blocchi KV** |
| `promote()` / eviction | `submit_load()` (promotion) / `submit_store()` (cascade) | idem |
| `SEEPolicy` / `LRUPolicy` (`tier/policies.py`) | `CachePolicy` → `LRUCachePolicy`, `ARCCachePolicy` | idem |
| `EAT` + `SlabAllocator` | `OffloadKey` + `BlockStatus` (`ref_cnt`, `block_id`) con ref-counting | idem |
| `AsyncNVMeIO` (`tier/io.py`) | `tiering/fs/` (`io.py` + `thread_pool.py`) | idem |
| Tier EMH-1c / EMH-3 | `Medium.CPU` / `Medium.STORAGE` | idem |
| `AERManager` (stub) | `AbstractEplbPolicy.rebalance_experts()` | **entrambi: expert** |
| — | `tiering/p2p/` (tier remoto, 4 543 LOC) | vMF non ha nulla di equivalente |

Va detto senza attenuanti, perché è il punto su cui il posizionamento del
progetto va rifatto: **la parte di vMemoryFabric che vLLM ha ora anche lui è M1+M2
applicati alla KV cache**, ed è più avanti (ARC oltre a LRU, tier remoto, lookup
asincrono, 15 metriche già nominate in `TieringOffloadingMetrics`).

**La parte che vLLM non ha, e che nessun modulo qui sopra tocca, è:**

- **il tiering dei pesi guidato dal gating.** `model_executor/offloader/` fa
  offload di pesi ma è **statico**: `cpu_offload_gb` a soglia, oppure
  `offload_group_size=8, num_in_group=2` cioè "scarica i layer 6,7,14,15,22,23…".
  Non c'è access frequency, non c'è predizione, non c'è promozione a caldo. Il
  campo `cpu_offload_params={"experts"}` esiste ed è documentato con l'esempio
  `mlp.experts.w2_weight`, ma seleziona *quali parametri*, non *quali expert*, e
  non cambia mai idea a runtime. È esattamente la lacuna che PT-PEP + GCSG
  esistono per riempire — e nella finestra giugno–agosto **non è stata toccata di
  una riga** (1 163 → 1 163 LOC).
- **qualunque tier sotto la DDR per i pesi.** L'offloader dei pesi conosce solo
  `mode == "cpu"` (`_BaseParamOffloader.create()`, `prefetch.py:557`). PMEM e NVMe
  non esistono come destinazione dei pesi.
- **PMEM come medium.** `Medium` è `Enum{CPU, STORAGE}`: nessun terzo valore. Il
  tier EMH-2 (issue #7/#57) si presenterebbe come `STORAGE`, perdendo in
  configurazione la distinzione che è metà del punto del tier.

---

## 3. I punti di innesto, verificati sul sorgente

Sorgente letterale di tutti e sei in `vllm_0280_extension_points.txt`. Il
criterio adottato: **conta come plugin solo se il sorgente dice esplicitamente
che una classe esterna è caricabile senza fork.**

### A. `SecondaryTierFactory` — ✅ out-of-tree, nessun fork

`vllm/v1/kv_offload/tiering/factory.py`, docstring letterale:

> *"External tiers can either `register_tier()` a friendly short name up front,
> or skip registration entirely and pass a `module_path` at lookup time
> (out-of-tree, no vLLM fork/patch required)"*

Il contratto è `SecondaryTierManager` (`tiering/base.py`, 336 LOC): 4 metodi
astratti principali (`lookup`, `submit_store`, `submit_load`,
`get_finished_jobs`), un `medium: ClassVar[Medium]`, e due vincoli non negoziabili
scritti nella docstring della classe:

> *"Secondary tiers cannot directly access GPU memory. All data transfers must go
> through the CPU (primary) tier: Store: GPU → CPU → secondary; Load: secondary →
> CPU → GPU"*
> *"All methods run in the Scheduler process and must be lightweight and
> non-blocking."*

Entrambi combaciano con la forma che `TierManager` ha già: NVMe→DDR4→VRAM è
letteralmente la stessa catena, e `AsyncNVMeIO` è già `asyncio`. **Questa è la
presa a costo più basso del repo.**

### B. `CachePolicyFactory` — ✅ out-of-tree, nessun fork

`vllm/v1/kv_offload/cpu/policies/factory.py`, stessa identica meccanica
`name` + `cache_policy_module_path`. Built-in: `lru`, `arc`. Il contratto
`CachePolicy` (`policies/base.py`) unifica organizzazione dei blocchi e decisione
di eviction. **`SEEPolicy` può essere registrata qui as-is come terza policy**, ed
è l'unico modo per far girare la SEE su traffico vLLM reale senza scrivere altro.

### C. `KVConnectorFactory` — ✅ out-of-tree via `kv_connector_module_path`

Il livello sopra: `KVConnectorBase_V1` con split scheduler/worker
(`get_num_new_matched_tokens`, `build_connector_meta` lato scheduler;
`start_load_kv`, `save_kv_layer`, `wait_for_save` lato worker). Serve solo se si
vuole sostituire l'intero connector invece di aggiungere un tier sotto quello
esistente — **più superficie, stesso risultato: non è la strada da prendere per
primo.**

### D. `set_offloader()` — ⚠️ tecnicamente aperto, praticamente stretto

`vllm/model_executor/offloader/base.py` espone `set_offloader(instance:
BaseOffloader)` pubblica, e `BaseOffloader` è una ABC con un solo metodo
obbligatorio (`wrap_modules`). Un `VMemoryFabricOffloader` è quindi scrivibile.

Il problema è **quando**: `gpu_model_runner.py:988` fa
`set_offloader(create_offloader(self.offload_config))` con il commento *"Make sure
this is called before any get_offloader call"*, e i modelli chiamano
`get_offloader().wrap_modules(...)` durante la costruzione. Sovrascrivere significa
infilarsi fra quelle due righe — fattibile da un plugin `vllm.general_plugins` che
patcha `create_offloader`, ma è di nuovo dipendenza da un dettaglio interno, cioè
esattamente il tipo di legame che ha già bruciato `GCSGWorker`. Il livello
sotto (`_BaseParamOffloader.create()`, che conosce solo `"cpu"`) è privato e
hardcoded: nessuna presa lì.

### E. `AbstractEplbPolicy` — ❌ registry chiuso, serve una PR upstream

`vllm/distributed/eplb/policy/abstract.py` è una ABC pulita con un solo metodo:

```python
def rebalance_experts(cls, weight, num_replicas, num_groups,
                      num_nodes, num_ranks, old_global_expert_indices) -> torch.Tensor
```

`weight` è `[layers, num_logical_experts]` di statistiche di carico, il ritorno è
`physical_to_logical_map [layers, num_replicas]`. **È la firma che AER vorrebbe**:
`num_replicas` è il fattore di replica che `AERManager.replication_factor()`
ritorna sempre `1` (`aer.py:46-53`).

Ma: `EPLB_POLICIES = {"default": DefaultEplbPolicy}` con
`assert set(EPLB_POLICIES.keys()) == set(get_args(EPLBPolicyOption))` e
`EPLBPolicyOption = Literal["default"]` (`config/parallel.py:39`). **Nessun
`module_path`, nessun `register`.** A differenza di A/B/C, una policy AER esterna
qui **non** è caricabile: va aperta una PR upstream (o aggiungendo il
`module_path` mancante — che, essendo il pattern già usato da tre altri factory
nello stesso repo, è una PR piccola e difendibile).

### F. `Medium` — ❌ nessun PMEM

`Enum{CPU, STORAGE}` (`v1/kv_offload/base.py:47`). Un tier PMEM si dichiara
`STORAGE`. Aggiungere `Medium.PMEM` è una PR upstream di poche righe, e sarebbe
il contributo più naturale che questo progetto possa fare a vLLM: è l'unico dei
tre partecipanti (vLLM, SGLang, questo repo) che ha hardware Optane in casa.

---

## 4. Il problema d'ingresso: M3 non ha più una base

`vllm_torch27_compat_analysis.md` aveva già misurato dove si chiudeva la
finestra. Alla luce di 0.28.0 (`gcsg_hooks_vs_0280.txt`):

| Dipendenza di `gcsg.py` | 0.6.6.post1 (pin attuale) | 0.28.0 |
|---|---|---|
| `vllm/worker/worker.py` (base di `GCSGWorker`) | presente | **rimosso** — `vllm/worker/` non esiste più |
| `vllm/worker/worker_base.py` (`resolve_obj_by_qualname`) | presente | **rimosso** |
| `mixtral_quant.py` (`_AWQShadowExpert`) | presente | **rimosso** |
| `EngineArgs(worker_cls=...)` come qualname | presente | non più questo meccanismo |
| **`MixtralMoE.gate` + `router_logits` scartato** | presente | **presente** (`mixtral.py:109,139`) |
| `in_wsl()` | presente | **presente** (`platforms/interface.py:62`) |
| `_ShadowExpertINT4` (puro torch) | — | **immune, zero dipendenze vLLM** |

La distinzione che salva il lavoro fatto: **il meccanismo dell'hook è intatto, il
veicolo di installazione no.** Il punto 3 del docstring di `gcsg.py` — forward
hook PyTorch su `layer_i.block_sparse_moe.gate`, perché `router_logits` è una
locale scartata e il blocco MoE intero vedrebbe solo l'output già mescolato —
è ancora **letteralmente vero in 0.28.0**, riga per riga. Quello che non c'è più
è `GCSGWorker(Worker)` come posto dove registrarlo. Il sostituto supportato è un
entry point `vllm.general_plugins` (`vllm/plugins/__init__.py:18`), che gira in
ogni worker process — cioè risolve lo stesso problema che nel 2026-08 aveva
portato a scartare il monkey-patch su `_run_workers()`, e lo risolve nel modo
per cui vLLM lo ha previsto.

Il punto 4 invece va riscritto e basta: `execute_model(scheduler_output:
SchedulerOutput)` (`v1/worker/gpu_worker.py:1053`) non ha
`seq_group_metadata_list`, quindi l'estrazione dei `request_id` per la
contaminazione per-request va rifatta contro l'API V1.

**Nota sul debito accumulato.** Il pin attuale (`vllm==0.6.6.post1`,
dicembre 2024) è indietro di **49 release** e di 8 minor version di torch
(2.5.1 → 2.13.0). Nella sola finestra giugno–agosto vLLM ha pubblicato 8 release
e ha spostato il pin torch da 2.11.0 a 2.13.0 fra 0.26.0 e 0.27.0
(`vllm_torch_pins_window.txt`). Un buon lato c'è: **`CUDA_SUPPORTED_ARCHS`
include `12.0` incondizionatamente** (e `12.1` con CUDA 12.8), quindi la RTX
5060 Ti dell'issue #8 è coperta senza flag speciali da qualunque release recente.

---

## 5. Cosa proporre di fare, in ordine

Tre percorsi, indipendenti, dal più economico al più caro. **Nessuno dei tre è
stato eseguito**: sono proposte, non risultati.

### P1 — `SEEPolicy` come `CachePolicy` out-of-tree (§3.B)

Il più economico e il più informativo per il paper (Filone B). Non tocca
`gcsg.py`, non tocca il pin di produzione, non richiede la seconda GPU: si scrive
un adattatore di `tier/policies.py` al contratto `CachePolicy` e si misura SEE
contro LRU e **contro ARC** su traffico KV reale. Oggi la SEE è misurata solo
contro se stessa nei benchmark interni; un confronto con ARC su un carico non
scelto da noi è il tipo di numero che un referee chiede. Rischio: la SEE è
progettata su expert (oggetti da 256 MB, riuso guidato dal gating), non su
blocchi KV (oggetti piccoli, riuso guidato dal prefisso) — **può benissimo
perdere**, ed è comunque un risultato pubblicabile.

### P2 — tier PMEM come `SecondaryTierManager` out-of-tree (§3.A)

È la via che sblocca l'issue **#57** (PMEM implementato ma non cablato alla via
di produzione) senza dover prima cablare la via di produzione: il tier EMH-2
diventa un tier secondario di vLLM, e il "chiamante reale" che a
`src/tier/pmem.py` manca da agosto lo fornisce vLLM. Costi: `submit_store`/
`submit_load` devono essere non bloccanti nel processo scheduler (oggi
`AsyncNVMeIO` è `asyncio`+`aiofiles`, va verificato che regga il vincolo), e
il medium va dichiarato `STORAGE` finché §3.F non è risolto upstream. Prerequisito
hardware: la Z8 bare-metal, non il setup Docker-on-Windows.

### P3 — reinnesto di M3 (GCSG) su V1

Il più caro e l'unico che va davvero fatto se M3 deve sopravvivere. Non è un
adattamento di firma: `GCSGWorker` va sostituito da un plugin
`vllm.general_plugins` che registra i forward hook sui gate (il §4 dice che gli
hook reggono), e la contaminazione per-request va riportata sull'API V1. Da
fare **dopo** P1/P2, perché quelli producono valore senza dipendere da questo, e
perché il pin va comunque spostato prima.

**Regola trasversale, dal precedente di agosto:** ogni conclusione qui sopra è
statica. Nessuna delle due sdist è stata installata; il prossimo passo di
ciascun percorso è un `pip install` reale in un container usa-e-getta sulla Z8,
non altra analisi.

---

## 6. Cosa NON è stato verificato

- **Nessun run, nessun `pip install`.** Che un simbolo esista al tag con quella
  firma non dimostra che l'innesto proposto compili o funzioni.
- **Nessun numero di performance.** I 25 000 TPS/GPU e ogni altra cifra della
  richiesta non sono confermabili da sorgente. Qui non si misura nulla.
- **PegaFlow e Themis non sono stati cercati fuori dal repo `vllm`.** Si è
  verificato solo che non sono nel core; se esistono come progetti separati, il
  loro rapporto con vMemoryFabric va valutato a parte.
- **La finestra 0.22.1 ↔ 0.28.0 è stata confrontata agli estremi**, non release
  per release: un modulo comparso e sparito dentro la finestra non risulterebbe.
- **Le sdist ≠ le wheel.** Il confronto è sul sorgente Python; kernel compilati e
  dipendenze binarie (flashinfer, tilelang) non sono stati toccati.
