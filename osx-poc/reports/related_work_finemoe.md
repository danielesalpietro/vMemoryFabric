# Related Work — FineMoE (EuroSys '26) e MoE-Infinity: il vicino accademico letto sul codice, e cosa resta davvero a OSX

**Status:** Nota di ricerca / posizionamento, **verificata sul codice sorgente** — non sul
PDF. Il testo del paper (`intellisys.haow.us`, arXiv 2502.05370 e tutti i mirror provati)
è irraggiungibile dal sandbox di questa sessione per policy di egress; **il repository
ufficiale degli autori** (`IntelliSys-Lab/FineMoE-EuroSys26`, commit `5c58468` del
2025-09-25, "demo implementation of the paper") sì, e con esso **MoE-Infinity** su cui
FineMoE è costruito (`EfficientMoE/MoE-Infinity`, HEAD `b766f8f` del 2026-08-17). Il paper
va comunque letto quando raggiungibile: numeri, baseline e claim quantitativi qui **non
sono verificati**. I meccanismi sì, riga per riga.
**Data:** 2026-09-02
**Progetto:** OSX — Operating System for Experts (repo: `vMemoryFabric`)
**Scope:** cosa fa FineMoE, cosa aggiunge a MoE-Infinity, cosa è oggi MoE-Infinity a HEAD,
e — la parte scomoda — quali differenziatori di OSX sopravvivono al confronto.

**Vedi anche:** [`related_work_vllm_offloader.md`](related_work_vllm_offloader.md) (vicino
ingegneristico dentro vLLM), [`related_work_elastic_ep.md`](related_work_elastic_ep.md),
`logbook_paper.md` (entry 2026-08-13 che ha identificato FineMoE come priorità; entry
2026-09-02 per le conseguenze).

> **Verdetto in una riga.** FineMoE fa, a granularità di esperto e con più sofisticazione
> di PT-PEP, esattamente il prefetch predittivo che OSX rivendica; MoE-Infinity a HEAD ha
> tre tier (GPU/host/SSD), un serving engine con continuous batching e KV paginata, e
> supporta Mixtral quantizzato. **I differenziatori di OSX che sopravvivono alla lettura
> sono due, e uno solo è forte: GCSG.** Il resto va riscritto come "diverso", non come
> "in più".

---

## 1. FineMoE: il meccanismo, dal codice

Il demo è costruito su HuggingFace Transformers, supporta **un solo modello**
(`Qwen1.5-MoE-A2.7B-Chat`: 24 layer, 60 esperti, top-4), richiede **≥ 48 GB di VRAM**
(`README.md`), gira a `eval_batch_size = 1` con prompt e generazione di 16 token
(`demo/configs/`). Non è un sistema di serving: è l'apparato sperimentale del paper.

### 1.1 Cosa traccia — "fine-grained" significa per token, non per sequenza

`ExpertTracer` (`finemoe/memory/expert_tracer.py`) mantiene per ogni sequenza due livelli:

- **coarse**: una matrice `[L, E]` di conteggi cumulativi degli esperti attivati
  (`update_entry`, righe 103-124) — è l'*Expert Activation Matrix* di MoE-Infinity;
- **fine**: per ogni token (`iters`, `update_embed` righe 61-82), l'**embedding di input**
  del token, gli hidden state per layer, gli esperti scelti (`nodes`), le **probabilità
  del router** per layer (`probs`) e le predizioni fatte (`preds`).

`demo/process_data.py` calcola l'entropia della distribuzione degli esperti a livello
coarse (per sequenza) e fine (per token): è la Figura 3 del paper. La tesi implicita —
verificabile con quel codice — è che il routing per token è molto più prevedibile di quanto
non appaia aggregando per sequenza.

### 1.2 Cosa predice — kNN su embedding e su traiettorie parziali, non un classificatore

`ExpertMapStore` (`finemoe/runtime/model_offload.py:61-190`) è una memoria di coppie
`(embed [d], traj [L, E])`: l'embedding di input di un token e la traiettoria completa di
probabilità del router che quel token ha poi seguito. Capacità 1000 (default); quando è
piena, viene sostituita la voce **più ridondante**, con punteggio
`w·cos(embed) + (1−w)·cos(traj)` e `w = prefetch_distance / L` (righe 143-159). Lo store si
costruisce offline (`demo/prepare_data.py`, `eval_mode="online"` raccoglie e poi
`export_store_data`) e si ricarica a inferenza (`eval.py`, `import_store_data`).

`ExpertMapMatcher` (righe 193-290) ha **due trigger**, agganciati nel modello:

1. **`embed_prefetch`** (riga 255; chiamato in `modeling_qwen2_moe.py:1237`, subito dopo
   `embed_tokens`): cerca l'embedding del token nello store per similarità coseno
   (`match_embed`, riga 161), prende la traiettoria della voce più simile e la usa per
   prefetchare i layer `[0, prefetch_distance)`. È la parte che il paper chiama *semantic
   hints*: **non c'è un modello addestrato**, è nearest-neighbour sugli embedding del
   modello stesso.
2. **`traj_prefetch`** (riga 272; chiamato in **ogni** MoE block, `modeling_qwen2_moe.py:887`):
   dopo il layer *l*, confronta le probabilità osservate sui layer `[0, l]` con i prefissi
   delle traiettorie nello store (`match_traj`, riga 175) e prefetcha i layer da
   `l + prefetch_distance` in poi. È la **correzione online**: la predizione si aggiorna a
   ogni layer con il routing reale.

`process_expert_map` (riga 238) trasforma la traiettoria matchata in priorità: per layer
tiene gli esperti che coprono una **massa cumulativa** `1 − similarità` (`_select_by_cumsum`,
riga 214) — meno simile la voce, più esperti vengono prefetchati, mai meno di top-k — e pesa
con un **decadimento lineare sulla distanza di layer** (`_layer_decay_weights`, riga 229).
`ExpertPrefetcher.prefetch_experts` (`finemoe/memory/expert_prefetcher.py:53-98`) ordina
per priorità, marca i candidati da proteggere (`replace_cache_candidates`) e accoda ogni
esperto con la sua probabilità di router come hint (`enqueue_prefetch(tid, gpu, p)`).

### 1.3 Cosa evict — LFU pesata dalla probabilità del router, con protezione dei candidati

Nel motore C++ (ereditato da MoE-Infinity, modificato) `RemoveCachedSparseNode`
(`core/prefetch/task_scheduler.cpp:231-317`) ordina gli esperti residenti in GPU per
`prob × incache_visit_count` crescente (riga 296) e sfratta dal basso, **saltando** i
candidati protetti dall'ultimo prefetch (riga 306) e i nodi in esecuzione. `prob` è la
probabilità del router dell'ultimo prefetch (`archer_prefetch_handle.cpp:199-206`). Il diff
con MoE-Infinity a HEAD mostra che lì il punteggio è il solo `incache_visit_count`: **la
ponderazione per probabilità è un contributo di FineMoE**.

### 1.4 Tier e granularità

Tre tier: GPU, host pinned (`host_memory_ratio`), **disco** (`initial_host = DISK_DEVICE`,
`core/model/model_topology.h`; AIO a due code di priorità, `core/aio/archer_prio_aio_handle.h`:
on-demand alto, prefetch basso). L'unità è **l'esperto intero per layer**
(`expert_tensors` per esperto in `setup_archer_hooks`, `model_offload.py:~940-960`), cioè
la granularità che OSX rivendica come propria — qui c'è già.

### 1.5 Cosa FineMoE aggiunge a MoE-Infinity (delta di meccanismo)

| | MoE-Infinity (`ExpertTracer.find_most_similar`, `ExpertPredictor.predict`) | FineMoE |
|---|---|---|
| Unità di traccia | sequenza (matrice `[L,E]` cumulativa) | **token** (`iters`: embed, probs, nodes per layer) |
| Chiave di lookup | traiettoria cumulativa della sequenza corrente | **embedding di input** (pre-MoE) *e* traiettoria parziale del token corrente |
| Quando predice | a ogni layer, dalla matrice cumulativa | a `embed_tokens` (prima di qualsiasi MoE) *e* a ogni layer |
| Selezione | tutti gli esperti visti nella voce più simile, con decadimento lineare | massa cumulativa `1 − similarità`, min top-k, decadimento lineare |
| Eviction | LFU (`incache_visit_count`) | LFU × probabilità del router |
| Store | traiettorie, rimpiazzo LFU | coppie (embed, traj), rimpiazzo per ridondanza combinata |

Il salto concettuale è uno: **usare l'embedding del token come chiave predittiva prima che
il modello tocchi un MoE**. È la stessa intuizione di PT-PEP, realizzata con un kNN sugli
embedding del modello invece che con un classificatore esterno sul testo.

## 2. MoE-Infinity a HEAD: un terzo concorrente, non un antenato

Il repository da cui FineMoE è partita non è fermo. A `b766f8f` (2026-08-17) MoE-Infinity è:

- **un serving engine** (`moe_infinity/serving/`: continuous batching, KV paginata,
  scheduler con preemption, CUDA graph, server OpenAI-compatibile, streaming);
- con **tre tier** (GPU / host / SSD, stesso motore AIO) e cache "activation-aware" —
  `expert_priority_score.py` combina frequenza, posizione nel layer e traccia della
  sequenza; `offloading_policy.py` ha LRU e **ARC** come policy;
- con **prefetch speculativo dai logit del router** del layer corrente per il layer
  successivo e **correzione** sui miss (`expert_prefetcher.py`, `speculative_prefetch` /
  `correct_prefetch`), più un path "route-ahead" legato allo speculative decoding;
- con **Mixtral 8x7B/8x22B supportato** (`models/mixtral.py`), path quantizzati GPTQ/AWQ/
  Marlin/FP8/MXFP4, multi-GPU single-node round-robin, build per **sm_120**
  (`MOE_ENABLE_SM120=1`).

Per la nostra evaluation questo conta più di FineMoE: FineMoE non gira su Mixtral né su
24 GB; **MoE-Infinity potrebbe** — è l'unico sistema di questo cluster che si può mettere
come braccio sul banco reale. Se Mixtral-8x7B quantizzato entri davvero con
`device_memory_ratio` sulla 3090 è da provare, non da assumere.

## 3. Cosa resta a OSX — la lettura scomoda

| Differenziatore rivendicato | vs FineMoE / MoE-Infinity | Verdetto |
|---|---|---|
| Granularità di esperto | ce l'hanno entrambi | **non è un differenziatore** in questo cluster (lo è solo vs l'offloader vLLM) |
| Prefetch predittivo | FineMoE: kNN su embedding + correzione online per layer; MoE-Infinity: EAM + speculativo dai logit | **PT-PEP è meno sofisticato**: predice una volta, dal testo, senza correzione in-flight |
| Tier > 2 | 3 tier con disco e AIO prioritizzato | 4 contro 3; **PMEM come tier distinto** è l'unica differenza, e va dimostrata utile (#57) |
| Eviction dinamica | LFU × prob (FineMoE); priority score, LRU/ARC (MoE-Infinity) | SEE policy: da confrontare, non da dichiarare superiore |
| Hotness persistente (#27) | store esportabile/importabile in entrambi | **non è un differenziatore** |
| Eterogeneità host (DDR/PMEM/NVMe come spazio unico) | tier fissi, ordinati | **differenza reale ma debole** finché resta single-GPU |
| **GCSG** — verifica shadow della qualità sotto quantizzazione | **assente** in entrambi (caricano GPTQ/AWQ, nessuna guardia) | **unico differenziatore forte**, e già così in `logbook_paper.md` 2026-08-13 |
| Integrazione con un engine di produzione (vLLM) | MoE-Infinity ha il *suo* engine con continuous batching; FineMoE nessuno | **debole**: OSX gira nel worker V0, `--enforce-eager`, finestra di pin chiusa |
| Lead time del prefetch (pre-tokenizzazione, CPU, fuori dal forward) | FineMoE predice al primo embedding, dentro il forward | **differenza architetturale reale**: PT-PEP può partire prima che la richiesta arrivi al modello — ma va misurata in millisecondi guadagnati, non affermata |

Tre conseguenze:

1. Il paper **non può** presentare EMH come "gerarchia di memoria per esperti con prefetch
   predittivo" come contributo: è il contributo di MoE-Infinity (2024) e FineMoE (2026).
   Può presentarla come *substrato* su cui GCSG opera, e lì FineMoE non c'è.
2. **PT-PEP va o rafforzato o ridimensionato.** La domanda giusta non è "PT-PEP funziona"
   ma "cosa guadagna PT-PEP rispetto a un kNN sugli embedding con correzione per layer,
   a parità di tracce". È un esperimento **eseguibile ora, senza GPU**, sulle tracce di
   routing già raccolte dai hook GCSG: recall@k per lead time. Va nel test plan (L0b).
3. **D5 si chiude da sola**: AER, se mai fosse "replica gli esperti caldi", sarebbe il
   terzo sistema a farlo. Sopravvive solo come replica *fra tier* su hardware asimmetrico.

## 4. Cosa prendere in prestito (con attribuzione), cosa no

| Da FineMoE / MoE-Infinity | Uso in OSX | Perché |
|---|---|---|
| Entropia coarse vs fine del routing (`process_data.py`) | metrica **M9** nel test plan | è il modo giusto di caratterizzare lo skew di W3; e rende i nostri numeri confrontabili con i loro |
| Definizione di hit rate per esperto (`NodeBody`: `gpu_hit_cnt / visit_cnt`) | allineare M8 | stessa definizione, stesso confronto |
| Selezione a massa cumulativa `1 − similarità` | candidato per SEE/PT-PEP (#21) | è la risposta pulita a "quanti esperti prefetchare quando la predizione è incerta" |
| Correzione online per layer (`traj_prefetch`) | **manca in OSX** | senza, PT-PEP resta una scommessa fatta una volta sola |
| Motore C++ AIO / task scheduler | no | OSX ha `AsyncNVMeIO`; riscriverlo non è nello scope |
| Serving engine di MoE-Infinity | no, ma come **braccio A5** | è l'unico del cluster che gira su Mixtral in 24 GB |

## 5. Non verificato, da non affermare

- **Tutti i numeri del paper** (47 % latenza, 39 % hit rate): mai visti nel testo, solo
  nell'abstract via README. Le baseline con cui li ottiene, ignote.
- Se il demo rappresenti il sistema completo "su testbed a sei GPU" citato dall'abstract:
  il codice distribuito (`finemoe/distributed/`) esiste, il demo non lo esercita.
- Se MoE-Infinity a HEAD faccia entrare Mixtral-8x7B quantizzato in 24 GB con throughput
  utile: da provare (braccio A5, test plan).
- Che cosa FineMoE abbia cambiato nel C++ oltre alla ponderazione per probabilità: il diff
  con HEAD è dominato dall'evoluzione di MoE-Infinity, non isolabile senza il commit base.

## 6. Fonti

- `IntelliSys-Lab/FineMoE-EuroSys26`, commit `5c584686` (2025-09-25): `finemoe/memory/
  expert_tracer.py`, `expert_prefetcher.py`, `finemoe/runtime/model_offload.py`
  (`ExpertMapStore`, `ExpertMapMatcher`), `finemoe/models/modeling_qwen/modeling_qwen2_moe.py`
  (hook a 863-887, 1233-1237), `core/prefetch/task_scheduler.cpp` (231-317),
  `core/prefetch/archer_prefetch_handle.cpp` (199-206), `core/model/model_topology.h`,
  `core/aio/archer_prio_aio_handle.h`, `demo/process_data.py`, `demo/configs/*`, `README.md`.
- `EfficientMoE/MoE-Infinity`, HEAD `b766f8f1` (2026-08-17): `README.md`,
  `moe_infinity/memory/expert_tracer.py`, `expert_predictor.py`, `expert_prefetcher.py`,
  `expert_priority_score.py`, `offloading_policy.py`, `moe_infinity/models/mixtral.py`,
  `moe_infinity/utils/config.py`, `moe_infinity/serving/`.
- Paper: Yu, Cui et al., *Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via
  Fine-Grained Expert Offloading*, EuroSys 2026, arXiv 2502.05370 — **non letto**, vedi
  Status.
