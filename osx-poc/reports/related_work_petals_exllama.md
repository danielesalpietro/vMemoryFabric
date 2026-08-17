# Related Work — Petals vs ExLlamaV2/V3/tabbyAPI: gestione dell'eterogeneità e confronto con vMemoryFabric + grastorp

**Status:** Nota di ricerca / posizionamento competitivo, **verificata sul codice sorgente**
(v2, 2026-08-16). La prima stesura (2026-08-16, v1) era qualitativa, basata su conoscenza
generale dei progetti; questa revisione è stata prodotta leggendo i repository reali
(shallow clone, HEAD al momento della verifica — vedi §5 per commit e percorsi). Non è un
report sperimentale: nessun benchmark è stato eseguito in questo repository.

**Data:** 2026-08-16 (v1), aggiornato 2026-08-16 (v2, verifica codice)
**Progetto:** OSX — Operating System for Experts (repo: `vMemoryFabric`)
**Scope:** confronto architetturale tra due approcci esistenti alla distribuzione di modelli
su hardware eterogeneo (Petals, ExLlamaV2/V3 + tabbyAPI) e l'approccio di `vMemoryFabric` +
`grastorp` (progetto correlato, non incluso in questo repository).

**Vedi anche:** [`component_reuse_analysis.md`](component_reuse_analysis.md) — quali
componenti di questi progetti sono concretamente riusabili nel codice attuale di
`vMemoryFabric`, e perché sì o perché no.

---

## 1. Petals (P2P Swarm & Pipeline Parallelism)

**Petals** (BigScience / Yandex, repo `bigscience-workshop/petals`) è un layer applicativo
sottile sopra `hivemind` (libreria separata che fornisce DHT, networking P2P, trasporto RPC e
compressione tensori). Il codice proprio di Petals vive in `src/petals/client/` (routing e
sessioni lato utente), `src/petals/server/` (hosting dei blocchi, cache, handler RPC) e
`src/petals/models/*` (wrapper compatibili HuggingFace per BLOOM/Falcon/LLaMA/Mixtral).

### Come gestisce l'eterogeneità

- **Assegnazione dei blocchi — dinamica, non statica.** Se l'operatore non specifica
  `--num_blocks`/`--block_indices`, `Server._choose_num_blocks()`
  (`src/petals/server/server.py:275-326`) stima la memoria disponibile (VRAM via
  `torch.cuda.get_device_properties`, o RAM via `psutil`), sottrae un buffer riservato e
  calcola quanti blocchi il nodo può ospitare in base al costo per blocco (pesi + riserva
  KV-cache). *Quali* indici di blocco ospitare è deciso da
  `block_selection.choose_best_blocks` (`src/petals/server/block_selection.py`), che modella il
  "throughput" per posizione lungo l'intero stack e sceglie la posizione che minimizza il
  collo di bottiglia dello swarm; `Server.run()` richiama periodicamente
  `_should_choose_other_blocks()` per un **self-rebalancing** continuo (soglia
  `--balance_quality`, default 0.75).
- **Routing via DHT.** `RemoteSequenceManager._update()`
  (`src/petals/client/routing/sequence_manager.py:340`) interroga la DHT per sapere quali peer
  servono quali blocchi. Per l'inferenza, `_make_sequence_with_min_latency` (linea 177)
  costruisce un grafo (peer, blocco) pesato per RTT + throughput inverso e trova il cammino
  più breve con Dijkstra (libreria `dijkstar`); per training/forward batch,
  `_make_sequence_with_max_throughput` (linea 302) sceglie greedily lo span disponibile più
  lungo a ogni hop.
- **Protocollo sul wire.** `rpc_forward`/`rpc_backward`/`rpc_inference`
  (`src/petals/server/handler.py`) scambiano tensori hidden-state 3D `[batch, seq_len,
  hidden]` serializzati da hivemind. Di default `compression=CompressionType.NONE` (server.py,
  righe 74/350): i tensori viaggiano fp16/bf16 non compressi salvo flag esplicito. Esiste anche
  un path server→server diretto (`rpc_push`/`_push_outputs`, handler.py:310-350) che inoltra
  gli hidden state al nodo successivo senza tornare dal client.
- **Fault tolerance — reale, non aspirazionale.** `InferenceSession.step()`
  (`inference_session.py:284-362`) avvolge ogni hop in un retry loop con backoff; in caso di
  fallimento, `sequence_manager.on_request_failure()` mette in blacklist temporanea il peer e,
  se un blocco diventa irraggiungibile, forza un re-routing DHT che ricostruisce il path
  (`_update_sequence()`, linea 364) e ripristina la history del KV-cache sui nuovi server.

### Limiti strutturali (confermati nel codice)

1. **Nessun paging del KV-cache oltre la VRAM.** `MemoryCache`
   (`src/petals/server/memory_cache.py`) è un allocatore a budget fisso
   (`--attn_cache_bytes`): quando è pieno, `_wait_until_available`/`AllocationFailed` blocca o
   rifiuta nuove sessioni — **non esiste eviction verso RAM host o spill su disco**. I tensori
   sono allocati direttamente sul device configurato (`backend.py`,
   `get_inference_cache_descriptors`). `src/petals/utils/disk_cache.py` esiste ma cachea solo i
   **pesi del modello** scaricati, non il KV-cache runtime.
2. **Speculative decoding — presente ma marginale, non un meccanismo di swarm.**
   `DistributedLlamaForSpeculativeGeneration`
   (`src/petals/models/llama/speculative_model.py`) è l'unica implementazione: solo Llama, usa
   un piccolo modello locale come "draft" e valida in un'unica chiamata distribuita; asserisce
   esplicitamente `do_sample=False` (il sampling non è supportato) e non è collegato al path di
   generazione di default (`remote_generation.py`). Va corretta l'impressione — anche nostra,
   nella v1 di questa nota — che sia un'ottimizzazione generale dello swarm: è un add-on per un
   singolo modello.
3. **Latenza di rete per ogni hop.** Confermato architetturalmente: ogni passaggio di
   attivatori tra server è un round-trip RPC (compresso o meno), quindi la latenza di swarm
   dipende linearmente dal numero di hop e dalla rete sottostante.
4. **Nessun memory fabric sottostante.** Il passaggio di testimone resta a livello applicativo
   (tensor passing su RPC), non a livello di memoria o di pagina — non c'è condivisione di
   VRAM tra nodi, solo trasferimento sequenziale di attivatori.

---

## 2. ExLlamaV2 / ExLlamaV3 + tabbyAPI

**ExLlamaV2** (`turboderp-org/exllamav2`) è il motore di inferenza C++/CUDA per modelli
quantizzati in formato EXL2. **tabbyAPI** (`theroyallab/tabbyapi`) è il server API ufficiale.

**Nota di verifica importante:** lo snapshot corrente di tabbyAPI (verificato da
`pyproject.toml`) dipende da **`exllamav3`** (non più `exllamav2`) — `backends/exllamav2/` non
esiste più nel repo, solo `backends/exllamav3/` e `backends/infinity/` (embeddings). Questa
sezione copre quindi entrambi: le proprietà strutturali generali (split multi-GPU,
niente rete) valgono per entrambi i motori; le funzionalità di CPU-offload descritte al punto
4 sono specifiche di **exllamav3** e non sono state verificate nel codice di exllamav2.

### Come gestisce l'eterogeneità

- **Split multi-GPU — GB assoluti, statico, calcolato al load.** `gpu_split` è una lista di GB
  assoluti per device (non un rapporto), documentato in `exllamav2/model.py:266-330`.
  `set_device_map()` (`model.py:176-263`) fa un **bin-pack greedy calcolato una sola volta al
  caricamento**: itera i moduli in ordine e passa alla GPU successiva quando quella corrente si
  riempie — **non è dinamico a runtime**, a differenza di quanto scritto nella v1 di questa
  nota. Esiste anche `load_autosplit`/`load_autosplit_gen` (probing automatico della VRAM
  libera) e `load_tp`/`TPContext` (`exllamav2/tensor_p.py`) per tensor-parallelism, ma anche il
  TP bilancia contro budget GB per-GPU fissati al load.
- **Nessun componente di rete — confermato per assenza.** Una ricerca su tutto l'albero
  ExLlamaV2 per `socket`, `rpc`, `grpc`, `torch.distributed`, `nccl`, `zmq`, `ray.remote` non
  ha prodotto risultati. Il multi-GPU (incluso il TP) passa per CUDA P2P/stream nello stesso
  processo (`exllamav2/device.py`, `tensor_p.py`) — tutto intra-host, PCIe/NVLink, nessun RDMA.
- **Paged attention del KV-cache — locale, capacità fissa.** `exllamav2/generator/dynamic.py`
  implementa paging in stile vLLM (`PAGED_PAGE_SIZE = 256`, classe `CachePage`); il numero
  massimo di pagine è fissato alla costruzione del generator
  (`max_pages = cache.max_seq_len // page_size`, linea 391) — **non elastico tra tier di
  memoria**, con deduplica/prefix-caching via hash delle pagine. Non pagina i pesi del modello
  e non fa spill su RAM/NVMe (confermato dall'assenza di path host-memory/mmap in
  `cache.py`).
- **EXL2 — quantizzazione a bit-budget variabile per layer.** `exllamav2/conversion/measure.py`
  profila l'errore di quantizzazione per layer; `optimize.py` alloca bit/peso diversi per
  layer per centrare un target medio di bit-per-peso minimizzando l'errore complessivo.

### exllamav3: CPU offload — la scoperta più rilevante per il confronto

Verificando `turboderp-org/exllamav3` (il motore effettivamente usato dalla tabbyAPI attuale)
sono emerse due funzionalità assenti in ExLlamaV2 e rilevanti per il confronto con
vMemoryFabric:

1. **Offload degli esperti MoE su RAM host (`cpu_moe_offload_layers`).**
   `exllamav3/model/moe_cpu_host.py` (classe `MoeCpuHost`) implementa un meccanismo reale, non
   banale: un **processo figlio separato** possiede i pesi degli esperti offloadati in una
   "huge arena" di memoria host (mmap anonimo, promosso a huge page), comunica con il processo
   GPU tramite un **job ring in memoria condivisa pinned** (`multiprocessing.shared_memory` +
   `cuda_host_register`). Per ogni layer offloadato, gli esperti "caldi" (assegnati a molti
   token) vengono streammati verso la GPU via DMA su un ring di slot VRAM e calcolati lì
   (`submit_prefill`/`_submit_prefill_streamed`), mentre la "coda fredda" viene calcolata
   direttamente su CPU con kernel AVX-512 (VBMI/VNNI se disponibili). Esposto in tabbyAPI come
   `cpu_moe_offload_layers` in `config_sample.yml:130-132` ("Number of mixture-of-expert layers
   to offload to CPU inference").
2. **KV-cache a due livelli (`sysmem_kv_cache`).** `exllamav3/exllamav3/generator/cpu_cache.py`
   (classe `CPUPageCache`, esplicitamente documentata come "Second-tier page cache in pinned
   system memory") mantiene in RAM pinned le pagine K/V complete evitate dalla cache GPU,
   indicizzate per hash di catena; al bisogno vengono ripristinate in VRAM invece di essere
   ricalcolate dal prefill. Ha una propria politica di eviction (catene orfane prima, poi
   alberi interi, LRU sulla radice). Esposto in tabbyAPI come `sysmem_kv_cache` (passato come
   `cpu_cache_size` a `ExLlamaV3DynamicGenerator`, vedi `backends/exllamav3/model.py:748-749`).

**Come si posiziona rispetto a vMemoryFabric:** questi due meccanismi sono concettualmente il
più vicino che Petals/ExLlama arrivino all'idea di "tiering di memoria" di vMemoryFabric — ma
restano **strettamente single-host**: usano memoria condivisa POSIX e `cudaHostRegister`, non
rete. **Non esiste un terzo tier NVMe** (confermato per assenza di riferimenti a
`mmap`/file su disco nel path del generator) e **non esiste alcun meccanismo per estendere
questo tiering su più nodi** (nessun socket/RPC trovato in `model_tp_cuda.py`, lo stesso file
che implementa il tensor-parallelism intra-host). Restano quindi due tier fissi (VRAM + RAM
pinned) su un solo host, non uno spazio di indirizzamento unificato multinodo.

### Limiti strutturali

1. **Vincolato al singolo host.** Nessun layer RDMA o di rete in nessuno dei tre repository
   (ExLlamaV2, ExLlamaV3, tabbyAPI) — confermato per assenza di codice, non per assunzione.
2. **Split statico calcolato al load** (GB assoluti o autosplit), non ribilanciato a runtime
   come invece fa Petals per i suoi blocchi.
3. **Il tiering RAM (CPU MoE offload, second-tier KV-cache) è per-processo e locale**: due
   soli tier fissi, nessuna estensione a NVMe o ad altri nodi.

---

## 3. Confronto con l'architettura vMemoryFabric + grastorp

| Caratteristica              | Petals                                          | ExLlamaV2/V3 + tabbyAPI                                              | vMemoryFabric + grastorp                                   |
|------------------------------|--------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------|
| **Topologia**                | Multinodo P2P su Internet/WAN (via hivemind DHT). | Multi-GPU eterogeneo **solo locale** (PCIe/NVLink, un solo processo).   | Multinodo eterogeneo su LAN/fabric (RDMA, PCIe).             |
| **Livello di operazione**    | Applicativo / pipeline (RPC su hivemind).         | Tensor engine / driver C++ CUDA + processo CPU-helper locale (v3).       | **Hypervisor / layer di memoria (paged fabric)**.             |
| **Assegnazione al device**   | Dinamica, con rebalancing periodico (§1).         | Statica, bin-pack calcolato una volta al load (§2) — salvo autosplit al load. | **A livello di singola pagina di memoria logica**, dinamica.  |
| **Paging / tiering memoria** | Assente (hard cap VRAM, nessuna eviction).        | **Due tier fissi, single-host**: VRAM + RAM pinned, per KV-cache (v2/v3) e per esperti MoE (solo v3). Nessun tier NVMe. | **Paging distribuito continuo multi-tier (VRAM/RAM/NVMe) su più nodi, via grastorp**. |
| **Fault tolerance**          | Retry/reroute per-hop implementato (§1).          | N/A (nessun concetto di nodo remoto).                                   | Da verificare in questo repo.                                 |

Il punto di rottura di `vMemoryFabric` rispetto a Petals e ExLlamaV2/V3 non è più, dopo questa
verifica, "nessuno dei due fa tiering di memoria" — **ExLlamaV3 lo fa**, in modo sofisticato,
per due tipi di dato specifici (pagine KV-cache ed esperti MoE). La differenza strutturale
resta però netta su due assi:

1. **Ambito**: ExLlamaV3 pagina *dati specifici* (KV-cache, pesi MoE) con meccanismi dedicati
   e non riusabili l'uno per l'altro; `vMemoryFabric` mira a uno spazio di indirizzamento
   unificato a livello di pagina, indipendente dal tipo di oggetto AI.
2. **Topologia**: ExLlamaV3 è vincolato a un singolo host (memoria condivisa POSIX,
   `cudaHostRegister`) e a due soli tier (VRAM + RAM); Petals è multinodo ma senza alcun
   concetto di memoria condivisa o paginata (solo passaggio sequenziale di attivatori via RPC).
   `vMemoryFabric` + `grastorp` combina multinodo *e* paging multi-tier (RAM/NVMe), la
   combinazione che né l'uno né l'altro progetto realizza.

---

## 4. Correzioni rispetto alla v1 di questa nota

La prima stesura (basata su conoscenza generale, non sul codice) conteneva due imprecisioni
corrette in questa revisione:

- ExLlamaV2 non fa "layer allocation manuale/**dinamico**" — è uno split **statico**,
  calcolato una sola volta al caricamento del modello (salvo il probing una-tantum
  dell'autosplit).
- Non era menzionato che la tabbyAPI attuale gira su **exllamav3**, non exllamav2, e che
  exllamav3 introduce un vero tiering RAM (CPU MoE offload + second-tier KV-cache) assente in
  ExLlamaV2 — la scoperta più rilevante di questa verifica.

## 5. Fonti (repository verificati, shallow clone HEAD al 2026-08-16)

- Petals — `bigscience-workshop/petals` (commit `22afba6`).
- ExLlamaV2 — `turboderp-org/exllamav2` (commit `7dc12af`).
- ExLlamaV3 — `turboderp-org/exllamav3` (verificato per la scoperta del §2, CPU offload).
- tabbyAPI — `theroyallab/tabbyapi` (commit `e632af4`).

File più rilevanti per verifiche future — Petals: `src/petals/server/server.py`,
`block_selection.py`, `throughput.py`, `client/routing/sequence_manager.py`,
`client/inference_session.py`, `server/handler.py`, `server/memory_cache.py`,
`models/llama/speculative_model.py`. ExLlamaV2: `exllamav2/model.py`, `tensor_p.py`,
`cache.py`, `generator/dynamic.py`, `conversion/optimize.py`. ExLlamaV3:
`exllamav3/model/moe_cpu_host.py`, `exllamav3/generator/cpu_cache.py`,
`exllamav3/generator/generator.py`. tabbyAPI: `backends/exllamav3/model.py`,
`config_sample.yml`, `pyproject.toml`.
