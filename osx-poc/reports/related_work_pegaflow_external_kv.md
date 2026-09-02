# Related Work — vLLM × Novita AI "PegaFlow" (External KV Cache): valutazione critica e impatto su vMemoryFabric

**Status:** Nota di ricerca / posizionamento competitivo, **verificata sul codice sorgente**.
Non è un report sperimentale: nessun benchmark è stato eseguito né riprodotto in questo
repository (l'hardware necessario — cluster RDMA multi-nodo — non è disponibile, vedi §5).

**Data:** 2026-09-01
**Progetto:** OSX — Operating System for Experts (repo: `vMemoryFabric`)
**Oggetto:** `novitalabs/pegaflow`, annunciato sul blog vLLM il 2026-05-18 come
"production-grade external KV cache" in collaborazione con Novita AI.

**Livello di verifica delle fonti (importante — le affermazioni qui sotto non hanno tutte lo stesso peso):**

| Fonte | Accesso | Uso in questo documento |
|---|---|---|
| `novitalabs/pegaflow` @ `b5d4acf` (v0.24.0, 2026-08-29), clone shallow | ✅ letto direttamente | §1, §4 — affermazioni marcate **[codice]** |
| `docs/*.md`, `README.md`, issue tracker del repo | ✅ letti | §1, §4 — marcate **[docs]** / **[issue]** |
| Blog post vLLM `vllm.ai/blog/2026-05-18-pegaflow` | ❌ **dominio bloccato dal proxy di rete di questa sessione** | §2 — numeri ripresi da indice di ricerca, marcati **[blog, non verificato in originale]** |

> ⚠️ Le quattro cifre di performance discusse in §2 **non sono state lette nel post originale**:
> provengono da estratti indicizzati. Il ragionamento di §2 è su *cosa misurano* quelle metriche,
> ed è valido a prescindere dal decimale esatto; ma prima di citarle in un paper vanno riprese
> dalla fonte primaria.

**Vedi anche:** [`related_work_petals_exllama.md`](related_work_petals_exllama.md) (stesso formato,
issue [#32](https://github.com/danielesalpietro/vMemoryFabric/issues/32)) e
[`component_reuse_analysis.md`](component_reuse_analysis.md).

---

## 1. Che cosa è realmente PegaFlow

Sintesi in una riga: **non è una feature di vLLM**. È un progetto esterno (Apache-2.0, Rust,
di Novita AI — un cloud provider GPU) che si aggancia a vLLM tramite l'interfaccia
`KVConnector` e che il blog vLLM ha ospitato. La distinzione conta: non è codice mantenuto
dal progetto vLLM, non segue il suo ciclo di release, e non ha le sue garanzie di
compatibilità.

**Architettura** [codice + docs]:

```
   vLLM worker (Python)                    pegaflow-server (Rust, 1 per host)
   ├── PegaKVConnector  ──gRPC───────────► control plane (registrazione, lookup, lease)
   │   (scheduler + worker role)                 │
   └── KV blocks in VRAM ──CUDA IPC──────────────┤ host KV pool (pinned, NUMA-aware)
                                                 ├── SSD cache (io_uring)
                                                 └── RDMA ──► altri nodi (MetaServer per l'indice)
```

- **Daemon per host, non libreria in-process.** Il pool pinned, l'indice, le risorse RDMA,
  la cache SSD e i task di background vivono nel processo Rust; il worker vLLM mappa la
  memoria via CUDA IPC (`python/pegaflow/ipc_wrapper.py`, `connector/worker.py`) [codice].
  Conseguenza vera e importante: **la cache sopravvive al riavvio dell'engine**.
- **Gerarchia a tre livelli**: pinned host DRAM → RDMA remoto → SSD
  (`pegaflow-core/src/backing/{rdma,ssd,uring}.rs`) [codice]. Eviction LRU
  (`pegaflow-core/src/cache.rs`, `hashlink::LruCache`) [codice] — semplice, non pesata per
  costo di ricostruzione né per tier.
- **Multi-tenant sullo stesso pool**: più engine, più modelli, più configurazioni TP sotto
  un solo server, separati da "namespace".
- **Deduplica logica tra rank TP**: il KV viene memorizzato una volta invece di una per rank —
  è da qui che viene il numero più alto (§2.3).

---

## 2. I numeri: che cosa misurano davvero

Le quattro cifre pubblicizzate **[blog, non verificato in originale]** e la loro lettura critica.

### 2.1 "2.15× avvio di vLLM più veloce" (host KV pool da 500 GiB)

È una misura di **tempo di startup**, non di inferenza. Il guadagno è quasi interamente
l'allocazione + pinning di 500 GiB di host memory, che il daemon ha già fatto una volta per
tutte e che vLLM altrimenti rifà a ogni boot. Detto altrimenti: *il costo non sparisce, cambia
proprietario*. È rilevante per chi fa rolling restart di decine di repliche (un cloud provider —
cioè Novita), irrilevante per chi avvia un engine e lo lascia su. Non dice nulla sul throughput
a regime.

### 2.2 "+56% throughput con 8 istanze Qwen3-8B su una cache condivisa"

Confronto **8 cache isolate vs 1 cache condivisa a parità di RAM totale**. Il guadagno è
reale ma è il guadagno del *pooling*, non della tecnologia di trasporto: N pool statici
sotto-utilizzati contro un pool unico elastico. Dipende interamente dal fatto che le 8 istanze
condividano prefissi (stesso system prompt, stesso modello) — cioè lo scenario multi-tenant
di un provider. Con 8 modelli diversi e prefissi disgiunti il vantaggio collassa alla sola
elasticità di allocazione.

### 2.3 "+72% throughput su DeepSeek-V3.2 MLA con TP8"

Questo è il numero **architetturalmente più interessante e il meno trasferibile**. Nasce dal
memorizzare il KV logico una volta invece di 8 (una per rank TP). Ma la ridondanza tra rank
**esiste solo quando il KV è replicato, non quando è partizionato** (derivazione in §7.2):
con MHA/GQA vLLM sharda le teste KV fra i rank, quindi con `n_kv ≥ TP` ogni rank possiede
una fetta *diversa* e il fattore di dedup è **1 — zero beneficio**; la replica compare solo
per `TP > n_kv` (fattore `TP/n_kv`) oppure con MLA, dove il latente compresso non è shardato
per testa e ogni rank ne tiene una copia identica (fattore `= TP`). "MLA + TP8" è quindi il
caso di massima ridondanza possibile nel baseline: il numero è misurato esattamente nel punto
in cui la vecchia soluzione sprecava di più, e per Llama-70B a TP8 (`n_kv = 8`) sarebbe 0%.

### 2.4 "~9× TTFT" (572 ms cold → 61 ms warm, H800, Llama-3.1-8B)

Questo è semplicemente **cache hit vs cache miss**. Qualunque prefix cache — compresa quella
già dentro vLLM — produce un rapporto di questo ordine. Non è un confronto contro LMCache,
Mooncake o l'offload CPU nativo di vLLM: **in tutto il materiale pubblicato non esiste un
head-to-head con alternative dirette**, che è l'unico confronto che deciderebbe l'adozione.

### 2.5 Il contesto hardware che rende i numeri poco portabili

Il throughput RDMA citato (~194 GB/s in lettura remota) è misurato su un cluster interno con
**8 × 400 Gbps per nodo** [blog, non verificato in originale]. È una configurazione da
datacenter di un provider GPU. Chi non ha quel fabric non ottiene quella curva, e il tier RDMA —
il differenziatore principale rispetto a un semplice offload su host — diventa inaccessibile.

---

## 3. Che cosa c'è di realmente valido

Va detto con chiarezza, perché la critica che segue non deve nascondere il merito:

1. **Disaccoppiare il ciclo di vita della cache dal processo dell'engine è la scelta giusta.**
   Sopravvivere a crash, upgrade e rolling restart è un requisito operativo che nessuna
   soluzione in-process può soddisfare. Questa è l'idea forte del progetto.
2. **Un pool per host invece di N pool per N engine.** Corretto sia in termini di
   utilizzazione sia di prevedibilità.
3. **Deduplica cross-rank TP.** Elimina una ridondanza strutturale che nessuna cache
   in-process può vedere, perché nessun rank conosce gli altri.
4. **Hot path fuori dal GIL** e DMA layer-wise: ingegneria seria, non marketing.
5. **Apache-2.0, codice leggibile, osservabilità (Prometheus/OTLP) presente dal giorno uno.**

---

## 4. Criticità

### 4.1 "Production-grade" è un'affermazione più forte di quanto il tracker sostenga

Issue aperte al 2026-09-01 [issue]:

| # | Titolo | Perché conta |
|---|---|---|
| 403 | *Transfer timeout can DMA into freed pinned memory (use-after-free window)* | Corruzione di memoria potenziale sul data path |
| 402 | *`rdma_accept_handshake` can panic via `unreachable!()`* | Crash del daemon → tutti gli engine dell'host perdono la cache |
| 401 | *RDMA fetch tears down a healthy connection on local allocation failure* | Fragilità sotto pressione di memoria |
| 408 | *Kernel transfer backend causes decode TPOT tail spikes* | Interferenza sulla latenza di decode, il KPI che si vuole proteggere |
| 353 | *gRPC/tonic hardening: per-RPC deadlines, keepalive, message-size limits* | Il control plane **non ha ancora** timeout né limiti di messaggio |
| 339 | *`http_cleanup_hang_repro` aborts with 'Fatal Python error'* | Shutdown non pulito lato client |
| 338 | *E2E harness: fail fast on client/server version mismatch* | Oggi un mismatch di versione fallisce in modo oscuro |

La roadmap (#314) dichiara il path P/D "functional end-to-end (small-model validated)", che
richiede ancora "production router path, formal benchmarking, and reliability hardening", e
marca il path RDMA v2 come **experimental** [issue]. Il divario tra questo e l'etichetta
"production-grade" del titolo del blog è la cosa più importante da registrare.

### 4.2 Sicurezza: assente, e non per svista

- **Nessuna autenticazione, nessun TLS.** Grep su `pegaflow-server/src` e
  `pegaflow-common/src` per `tls|mtls|bearer|auth_token|authenticat`: **zero occorrenze**
  [codice]. Il control plane gRPC e il piano dati RDMA sono in chiaro e non autenticati; #353
  conferma che l'hardening è ancora da fare.
- **Il "namespace" non è un confine di tenant.** `derive_namespace()`
  (`python/pegaflow/connector/common.py:519-565`) calcola uno SHA-256 troncato a 8 hex su
  `{model, dtype, tp_size, pp_size, num_kv_heads, head_size, num_hidden_layers, cache_dtype,
  is_hma_enabled, dcp/pcp_world_size, cross_layer_blocks, mla_layer_split_kv_cache}` [codice].
  Sono **tutti e soli fattori di compatibilità di layout**: nessun tenant, nessun utente,
  nessuna chiave. Il commento nel codice lo dice esplicitamente — serve a evitare che due
  layout incompatibili condividano un namespace. **Due tenant che girano lo stesso modello con
  la stessa configurazione condividono la cache by design.** Con una cache di prefissi
  condivisa, il tempo di risposta diventa un oracolo: si può testare se un prefisso è già
  stato visto da qualcun altro. È il classico side-channel delle prefix cache condivise, qui
  senza alcuna mitigazione.
- **CUDA IPC**: qualunque processo sull'host che possa parlare al daemon e ottenere gli handle
  mappa memoria altrui. Accettabile in un container isolato, non in un host multi-utente.
- **373 blocchi `unsafe`** nel codice Rust [codice]. La promessa "GIL-free Rust core" è vera
  sul piano delle performance; sulla memory safety, il data path (DMA, pinned memory, IPC) è
  per costruzione `unsafe`, e #403 è esattamente il tipo di bug che ne consegue. "È in Rust"
  non è, qui, un argomento di affidabilità.

### 4.3 Accoppiamento profondo (e non versionato) a vLLM

- Il connector importa da **~28 moduli vLLM distinti**, di cui la maggioranza interni:
  `vllm.v1.kv_cache_interface` (13 import), `vllm.config` (12), `vllm.v1.request`,
  `vllm.v1.core.kv_cache_manager`, `vllm.v1.core.sched.output`, `vllm.v1.worker.block_table`,
  `vllm.v1.attention.backends.utils`, … [codice]. Non è un plugin che tocca un'API pubblica:
  è codice che vive dentro le viscere dello scheduler V1.
- **`pyproject.toml` non pinna né vLLM né torch** [codice]. Con una superficie di
  accoppiamento del genere, l'assenza di un vincolo di versione non è flessibilità: è debito
  che si scarica sull'operatore, e #338 (fail fast su version mismatch) è la conferma che il
  problema si è già manifestato.
- **`docs/vllm-patch.md` chiede di patchare il sorgente di vLLM a mano**
  (`blocks.sort(key=lambda x: x.block_id)` in `vllm/v1/core/kv_cache_utils.py`) per ottenere
  le performance dichiarate. L'RFC upstream corrispondente,
  [vllm-project/vllm#31371](https://github.com/vllm-project/vllm/issues/31371), è stata
  **chiusa come `not planned` per inattività** [verificato]. Quindi: la patch non è upstream,
  non lo sarà, e va riapplicata a ogni aggiornamento di vLLM. "Drop-in connector" è
  un'etichetta generosa.
- Il deployment P/D richiede `PYTHONHASHSEED=42` coerente su tutti i nodi [docs] — un
  invariante globale, critico per la correttezza delle chiavi, imposto via variabile
  d'ambiente e non verificato a runtime.

### 4.4 Copertura funzionale ancora stretta

Stabile solo su layout **FlashAttention HND**; MLA parziale; esplicitamente non ancora
supportati: layout KV cross-layer, hybrid KV cache manager (HMA) per modelli ad attenzione
mista (sliding-window, Mamba/SSM), pipeline parallel per il P/D handoff [issue #314]. Nel 2026
la famiglia dei modelli ad attenzione ibrida non è un caso di nicchia: è una fetta crescente
del campo.

### 4.5 Nuovo dominio di guasto e nuovo lavoro operativo

Un daemon per host è un **single point of failure per tutti gli engine dell'host**: se muore
(#402), nessuno serve dalla cache. In cambio si accettano: un processo in più da monitorare,
versionare e aggiornare in lock-step col connector; NUMA e pinned memory da configurare;
capacità SSD e RDMA da dimensionare. Il costo operativo è reale e va confrontato con
l'alternativa banale — offload CPU nativo di vLLM, zero componenti aggiuntivi.

### 4.6 Governance e allineamento di incentivi

200 star, 32 issue aperte, un solo vendor a mantenerlo. Il design ottimizza precisamente
l'economia di Novita AI: molte istanze piccole per host, fabric RDMA denso, restart frequenti.
La licenza Apache-2.0 protegge dal lock-in legale, non dal rischio di manutenzione. L'ospitata
sul blog vLLM conferisce un'autorevolezza che il livello di maturità del codice non ha ancora
guadagnato — ed è esattamente su questo che va esercitata cautela.

---

## 5. Applicabilità al nostro setup — oggi: nessuna

| Requisito PegaFlow | Stato su Z8 G4 |
|---|---|
| Wheel CUDA 12.8 / 13 | ❌ siamo su `torch==2.5.1+cu124`, bloccati su issue [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) |
| NIC RDMA (il valore vero è il tier remoto) | ❌ nessuna |
| Più engine/host che condividono prefissi | ❌ un engine, un modello |
| Restart frequenti su pool pinned enormi | ❌ non è il nostro pattern |
| Pinned host memory | ⚠️ disponibile solo su bare-metal Linux, non nel dev Docker-on-Windows |

Il vincolo CUDA 12.8+ coincide però con il lavoro già in corso su #8 (`torch>=2.7`/`cu128`):
se e quando quel blocco si sblocca, PegaFlow diventa *installabile*. Resterebbe comunque
**inutile** senza fabric RDMA e senza multi-istanza: si comprerebbe il costo operativo di §4.5
per il solo tier host, che vLLM offre già nativamente.

---

## 6. Impatto su vMemoryFabric / OSX

### 6.1 Sovrapposizione: quasi nulla sul piano funzionale

Sono **oggetti diversi con economie diverse**:

| | PegaFlow | OSX / vMemoryFabric |
|---|---|---|
| Oggetto gestito | Blocchi KV cache | Shard di esperti (pesi) |
| Ciclo di vita | Effimero, per-richiesta, cresce col traffico | Statico, read-only, dimensione nota a priori |
| Indirizzamento | Content-addressed su hash di prefisso | Identità dell'esperto (`ExpertID`) |
| Riuso | Dipende dai prefissi condivisi tra richieste | Dipende dal routing del gating |
| Predizione | Nessuna: si scopre l'hit al lookup | **Il cuore del sistema**: PT-PEP + GCSG predicono *prima* |
| Miss | Ricalcolo (costoso ma corretto) | Stallo del forward pass |
| Hook vLLM | `KVConnector` | Hook di gating / pre-tokenizzazione |

La differenza decisiva è l'ultima riga della colonna "Predizione": una cache KV è **reattiva**
(indicizza quello che è già successo), OSX è **predittivo** (anticipa quello che il gating
farà). Questa è, e resta, la nostra tesi.

### 6.2 Dove invece la sovrapposizione è reale: il substrato

Sullo stesso host, PegaFlow e OSX **competono per le stesse risorse**: DDR pinned, banda PCIe,
NVMe, corsie NUMA. Un pool pinned da 500 GiB gestito da un daemon terzo è esattamente la
memoria che il nostro EMH-1c vuole per il warm buffer. Se un giorno si co-installano, serve
una politica di partizionamento esplicita — non è un dettaglio, è un vincolo di progetto.

### 6.3 Il rischio vero: erosione della novelty (Filone B, paper)

**Questo è l'impatto che conta.** PegaFlow costituisce prior art pubblicata e visibile su:
daemon di sistema sotto lo stack di inferenza; gerarchia a tre livelli DRAM/remoto/SSD;
pinned pool NUMA-aware; DMA layer-wise; ciclo di vita disaccoppiato dal processo engine;
pooling multi-istanza. Sono **sei elementi del nostro racconto architetturale** che dal
2026-05-18 non si possono più presentare come nuovi in sé, indipendentemente dal fatto che
noi li applichiamo a oggetti diversi.

Il paper va quindi riformulato per rivendicare ciò che PegaFlow **non** fa:

1. **Granularità a esperto** anziché a blocco KV — con un'economia di riuso e di eviction
   diversa (oggetti read-only, riutilizzati tra richieste indipendenti, con dimensione fissa
   nota: la SEEPolicy ha informazione che un LRU su blocchi KV non ha).
2. **Prefetch gating-aware**: PT-PEP/GCSG usano un segnale semantico interno al modello.
   PegaFlow non ha nulla di analogo e non può averlo: non vede il routing.
3. **Tiering eterogeneo con PMEM/CXL** (issue [#57](https://github.com/danielesalpietro/vMemoryFabric/issues/57)):
   PegaFlow ha DRAM/RDMA/SSD, non un tier a latenza intermedia persistente.
4. **Replica adattiva (AER)** guidata dalla popolarità degli esperti — non esiste l'equivalente
   in una cache content-addressed.

Consiglio concreto: **citare PegaFlow come related work nel paper**, non ignorarlo. Un
revisore lo conosce. Presentarlo come "il caso KV cache dello stesso principio, applicato a
oggetti reattivi" rafforza il posizionamento invece di indebolirlo.

### 6.4 Cosa vale la pena copiare (idee, non codice)

- **`io_uring` reale sul tier SSD.** `pegaflow-core/src/backing/uring.rs` è la prova che
  l'approccio è praticabile in produzione. Il nostro `AsyncNVMeIO` (`osx-poc/src/tier/io.py`)
  usa ancora `asyncio` + `aiofiles` anche su bare-metal Linux — un gap dichiarato nel README e
  mai chiuso.
- **Pattern "control plane / data plane separati"**: gRPC per il controllo, IPC/DMA per i dati.
  Applicabile al confine tra GCSG e worker.
- **Namespace come hash dei fattori di layout** per invalidare la cache quando cambia la
  configurazione: pattern piccolo, corretto, direttamente riusabile per lo shard store.
- **Osservabilità dal giorno uno** (Prometheus/OTLP nativi nel daemon) — noi abbiamo già il
  sidecar Prometheus, il modello di metriche per-tier è però più maturo dalla loro parte.

Da **non** copiare: l'assenza di autenticazione, il pin di versione mancante, la patch manuale
al sorgente dell'engine.

---

## 7. Approfondimento quantitativo

Questa sezione mette in formule ciò che §2 dice a parole, così che ogni cifra pubblicata
diventi una conseguenza calcolabile di parametri noti — e si veda esattamente *da quali*
parametri dipende. Le costanti dei modelli vengono dai config HuggingFace pubblici; il
resto è derivato.

### 7.1 Notazione e costanti

Per un modello con `L` layer, `n_kv` teste KV, dimensione di testa `d_h` e `s` byte per
elemento, il KV di **un token in un layer** occupa

```
b_KV = 2 · n_kv · d_h · s          (K e V)
B_tok = L · b_KV                    (un token, tutti i layer)
```

Per MLA (DeepSeek-V2/V3) K e V non esistono separati: per token e layer si memorizza il
latente compresso `c_kv` (`kv_lora_rank = 512`) più la componente RoPE disaccoppiata
(`qk_rope_head_dim = 64`), quindi `b_KV = 576 · s`.

| Modello | `L` | `n_kv` | `d_h` | dtype KV | `b_KV` | `B_tok` | blocco 16 token |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 32 | 8 | 128 | bf16 | 4 KiB | **128 KiB** | 2 MiB |
| Llama-3.1-70B | 80 | 8 | 128 | bf16 | 4 KiB | **320 KiB** | 5 MiB |
| DeepSeek-V3 (MLA) | 61 | — | 576 | FP8 | 576 B | **34.3 KiB** | 549 KiB |

Capacità di un pool host da 500 GiB: **4.0 M token** (Llama-8B), **1.6 M** (Llama-70B),
**15.2 M** (DeepSeek-V3 dedup) contro **1.9 M** se il latente è memorizzato una volta per
rank a TP8. L'ordine di grandezza è già la spiegazione di §2.3.

### 7.2 Fattore di replica tra rank TP

Con tensor parallelism vLLM sharda le teste KV: ogni rank possiede `n_kv / TP` teste. Se
`TP > n_kv` le teste vengono **replicate** per far lavorare ogni rank. Il fattore di
replica `r` — cioè quante copie identiche dello stesso KV logico esistono nel cluster — è

```
r_MHA/GQA = max(1, TP / n_kv)
r_MLA     = TP                       (il latente non è shardato per testa)
```

e il risparmio di memoria della deduplica è `1 − 1/r`:

| Configurazione | `r` | risparmio |
|---|---|---|
| Llama-3.1-70B, TP8 (`n_kv = 8`) | 1 | **0 %** |
| Llama-3.1-70B, TP16 | 2 | 50 % |
| Qwen3-8B, TP1 | 1 | 0 % |
| DeepSeek-V3, TP8 (MLA) | 8 | **87.5 %** |

Il +72 % di §2.3 è la manifestazione di `r = 8`. Su tutta la famiglia GQA con `TP ≤ n_kv`
— che è il caso normale del deployment single-node — la deduplica cross-rank **non fa
nulla**. È un beneficio reale ma confinato a MLA e a TP estremi.

### 7.3 Startup: Amdahl sul pinning

Se `T_start = T_pin + T_rest` e l'esternalizzazione azzera solo `T_pin`, lo speedup è
`S = (T_pin + T_rest) / T_rest`, da cui la frazione di startup spesa in pinning è

```
f_pin = 1 − 1/S = 1 − 1/2.15 ≈ 0.53
```

cioè il post dice, in altre parole, che **oltre metà dell'avvio di vLLM con un pool da
500 GiB è `cudaHostRegister`**. È plausibile? Il nostro soak test (`poc_final_report.md`
§2.3, 1000 cicli su shard da 256 MiB) dà due numeri utili: pin-alloc **cold 337 ms**
(≈ 0.8 GB/s: page fault first-touch + pinning su pagine da 4 KiB) e **warm 6–7 ms**
(≈ 37 GB/s: allocator caldo, pagine già residenti). Un processo fresco è nel caso cold; con
hugepages da 2 MiB (che `pinned_pool.rs` supporta come opzione) le PTE da bloccare calano di
512× e il rate sale a qualche GB/s. Quindi 500 GiB costano fra **~1.5 e ~10 minuti** a
seconda della configurazione — contro i 30–90 s tipici del resto dell'avvio (pesi da NVMe,
CUDA graph, warm-up). `f_pin ≈ 0.5` è coerente. Il punto resta quello di §2.1: il costo è
`O(dimensione del pool)` e viene **ammortizzato** sul numero di restart dell'engine durante
la vita del daemon — vale `n_restart × T_pin`, che per noi è ≈ `1 × T_pin` = niente.

Corollario sistemistico che il post non dice: memoria pinnata è memoria `mlock`ata,
**sottratta permanentemente al page cache**. 500 GiB pinnati su un host da 1 TB dimezzano il
page cache di tutto il resto. PegaFlow ne è immune perché il suo tier SSD usa `O_DIRECT`
(§8.4); il nostro `AsyncNVMeIO` no (§8.4, §6.2).

### 7.4 Hit vs miss: il rapporto TTFT è una proprietà del modello, non della cache

Prefill di `n` token costa `TTFT_cold ≈ n · c_tok` con `c_tok = 2P / (MFU · F_peak)`
(due FLOP per parametro per token). Caricare il KV degli stessi `n` token costa
`TTFT_warm ≈ n · B_tok / BW + c_ultimo_blocco`. Il rapporto, per `n` grande, è

```
ρ = TTFT_cold / TTFT_warm ≈ (2P · BW) / (MFU · F_peak · B_tok)
```

Per Llama-3.1-8B su H800 (`F_peak ≈ 989 TFLOPS` bf16 dense, `MFU ≈ 0.5`):
`c_tok ≈ 16 GFLOP / 495 TFLOPS ≈ 32 µs/token`; con PCIe Gen5 x16 a `BW ≈ 40–50 GB/s`
effettivi: `128 KiB / 45 GB/s ≈ 2.9 µs/token`. **`ρ ≈ 11`**, contro il 9.3× misurato
(572.5 / 61.5) — la differenza è l'overhead fisso (ultimo blocco ricalcolato, lookup,
scheduling) che pesa di più a prompt corti. Il numero è riprodotto senza alcun parametro di
PegaFlow: dipende da `P`, `F_peak`, `BW`, `B_tok`. E **cresce con la taglia del modello**:
per Llama-70B `c_tok ≈ 283 µs`, `B_tok/BW ≈ 7 µs`, `ρ ≈ 40`.

Il corollario più utile è la **banda di pareggio** sotto cui ricalcolare batte ricaricare:

```
BW* = B_tok / c_tok
```

| Modello | `c_tok` (H800, MFU 0.5) | `B_tok` | `BW*` |
|---|---|---|---|
| Llama-3.1-8B | 32 µs | 128 KiB | **4.1 GB/s** |
| Llama-3.1-70B | 283 µs | 320 KiB | 1.2 GB/s |
| DeepSeek-V3 (37B attivi, MLA) | ~150 µs | 34.3 KiB | **0.23 GB/s** |

Per un MoE con MLA anche un SSD SATA o un link 10 GbE battono il ricalcolo: ecco perché a
Novita conviene un tier SSD e un tier RDMA, e perché il loro caso d'uso (MoE grandi, MLA,
FP8) è quello in cui l'external KV cache rende di più. Per un modello denso da 8B il
margine sul tier SSD è di un fattore ~2: molto meno ovvio.

### 7.5 Pooling: dove sta il +56 %

Sia `h` l'hit rate sui prefissi, `T_pre` il tempo di prefill di una richiesta, `T_dec` il
suo decode. Tempo per richiesta:

```
T(h) = T_dec + T_pre · [(1 − h) + h/ρ]
```

Il guadagno di throughput passando da `h₁` a `h₂` è `T(h₁)/T(h₂)`. Con `φ = T_pre/T(0)`
frazione di prefill e `ρ ≈ 9`:

```
gain ≈ 1 / (1 − φ · Δh · (1 − 1/ρ))
```

Per ottenere **+56 %** serve `φ · Δh ≈ 0.40`: ad esempio `φ = 0.6` (workload
prefill-heavy, prompt lunghi e risposte corte) e `Δh = 0.67` (due terzi dei prefissi
diventano hit grazie alla condivisione). Il numero è consistente solo con un workload in
cui (a) il prefill domina e (b) le otto istanze condividono la maggior parte dei prefissi.
Con `φ = 0.2` (chat tipica, decode-heavy) lo stesso `Δh` dà +14 %. Il +56 % non è falso: è
**il valore in un angolo specifico dello spazio dei workload**, e quell'angolo è quello di
chi serve lo stesso system prompt su otto repliche.

Il perché *strutturale* del guadagno è la concavità della curva di hit rate `h(C)`: con
popolarità Zipf e richieste indipendenti, `h` è concava crescente nella capacità, quindi un
pool unico da `C` batte sempre `N` pool da `C/N` sulla stessa distribuzione
(`h(C) ≥ h(C/N)` per monotonia; il guadagno è massimo quando `C/N` cade sulla parte ripida
della curva). Con workload **disgiunti** (modelli o prefissi diversi) la condivisione dei
contenuti sparisce e resta solo il multiplexing statistico sulla domanda: il guadagno si
riduce all'elasticità di allocazione.

Un dettaglio che il pooling non aggiusta: l'eviction LRU **per blocco** (`cache.rs`) rompe
la contiguità dei prefissi — l'hit utile è il *prefisso cached più lungo*, non la *frazione*
di blocchi cached. La loro issue #396 ("preserve contiguous KV prefixes") lo riconosce.

### 7.6 Eviction: perché per il KV basta LRU e per gli esperti no

Il valore di tenere in cache un oggetto `o` è `V_o = p_o · Δ_o`, con `p_o` probabilità di
riuso nell'orizzonte utile e `Δ_o` il tempo risparmiato in caso di hit.

- **Blocco KV.** `Δ_b = c_blocco · (1 − 1/ρ)`: il costo di ricalcolo di un blocco è quasi
  costante per un dato modello, quindi `Δ_b ≈ const` e massimizzare `Σ V_b` equivale a
  massimizzare `Σ p_b`. LRU/LFU sono stimatori ragionevoli di `p_b` stazionaria: **il
  valore è uniforme, conta solo la probabilità**. Un LRU non è una scelta pigra, è una
  scelta quasi ottima per questo oggetto.
- **Shard di esperto.** Non esiste ricalcolo: un miss è uno **stallo** di `T_load(e)` sul
  path critico del forward. Lo stallo atteso per token è

  ```
  E[S | x] = Σ_{e ∉ cache} p(e | x) · T_load(e)
  ```

  dove `x` è il contesto corrente. La differenza decisiva rispetto al KV: `p(e | x)` è
  **condizionale al contesto** e osservabile *prima* del forward (dal gating, o da un suo
  predittore come PT-PEP), mentre per il KV `p_b` si scopre solo al lookup.

Questo permette di enunciare la tesi di OSX in una riga verificabile: il vantaggio di una
policy predittiva su LRU è limitato superiormente dall'**informazione mutua `I(E; X)`** tra
routing e contesto. Se `I(E; X) = 0` (routing indipendente dal prompt), SEE informata da
PT-PEP degenera in LRU e non c'è nulla da guadagnare; se `I(E; X) > 0`, il gap di hit rate
a pari capacità tra LRU e SEE+PT-PEP **è la misura empirica di quell'informazione** — ed è
il numero che #19/#21 devono produrre. La formula SEE (`score = α·freq + β·recency + γ·σ`,
`policies.py`) va letta come un proxy lineare di `log p̂(e | x)`: `α·freq + β·recency`
stima la componente stazionaria, `γ·σ` la componente condizionale. Il paper è forte
esattamente nella misura in cui `γ·σ` sposta il risultato.

### 7.7 Il side-channel in numeri

`TTFT_warm p99 = 77 ms < TTFT_cold mean = 572 ms`: le due distribuzioni sono disgiunte, un
singolo campione classifica hit/miss con errore ≈ 0. Granularità dell'oracolo: un blocco
(16 token). Un attaccante *cieco* dovrebbe indovinare `|V|^16` continuazioni — infattibile;
ma l'oracolo risponde in `O(1)` query per *candidato noto* (un documento, un system prompt,
un template), e la risposta è "**qualcun altro nel namespace l'ha inviato di recente**". Con
il tier RDMA l'oracolo è cluster-wide. Non è estrazione di contenuto; è membership su
contenuti noti — sufficiente per un'inferenza su chi usa quale prompt, e senza mitigazioni
(salt per tenant nell'hash, o namespace per tenant) nel codice attuale.

### 7.8 La finestra use-after-free (#403) come problema di code

Timeline: DMA sottomesso a `t₀`; timeout del trasferimento a `t₀ + τ`; il buffer pinned viene
liberato e può essere riassegnato; il copy engine completa a `t₀ + δ`. Corruzione se
`δ > τ` **e** il buffer è stato riassegnato nell'intervallo `(τ, δ)`. Sotto carico la coda
del copy engine cresce, la coda di `δ` si allunga ed è **esattamente sotto carico** che il
bug si attiva. La correzione corretta non è aumentare `τ`: è rilasciare i buffer **a
completamento** (`cudaEvent`) e non a timer, o metterli in quarantena finché l'evento non
arriva. Il nostro `GPUTransfer.to_vram(pin=True)` è oggi immune perché sincrono; la
pipeline a stream di **#5** introdurrà lo stesso identico rischio, e va progettata con
rilascio completion-driven fin dal primo commit.

---

## 8. Approfondimento sistemistico

### 8.1 Il data path: chi fa davvero la copia

Il meccanismo che rende PegaFlow "GIL-free" non è il Rust in sé, è **chi emette la copia**.
vLLM alloca i tensori KV per layer; il connector esporta per ciascuno un handle
`cudaIpcGetMemHandle` (`ipc_wrapper.py`, con mappatura per UUID della GPU così da
sopravvivere a `CUDA_VISIBLE_DEVICES` diversi fra processi); il daemon apre gli handle nel
proprio contesto CUDA e da lì emette `cuMemcpyAsync` H2D/D2H **sui propri stream**
(`gpu_worker.rs`: un batch di descrittori per layer, uno stream, una sola sincronizzazione).
Python non tocca il path della copia. Il prezzo: i copy engine (CE) della GPU sono condivisi
con vLLM, e il daemon compete sullo stesso PCIe con qualunque altro traffico host↔device
dell'engine. Per un modello denso è irrilevante; **per un MoE con esperti in offload — cioè
per noi — è la stessa corsia PCIe** (§6.2).

La scelta fra i due backend di trasferimento è un compromesso classico: `memcpy` usa i CE e
non consuma SM, ma paga un costo di emissione per ogni range; `kernel` (usato per MLA, dove
gli slot sono piccoli e frammentati) lancia un solo kernel di copia ma **occupa SM** e
interferisce col decode — la loro issue #408 (picchi di TPOT) è la conseguenza visibile.

### 8.2 Granularità delle copie: perché la patch a vLLM non è opzionale

Da `layout.rs`, ogni blocco di un layer è 1 range contiguo (MLA, o K/V adiacenti) oppure 2
(K e V in regioni separate, distanti `kv_stride_bytes`). Una richiesta di `n` token su `L`
layer produce

```
N_copie = L · seg · n / 16
```

Per Llama-3.1-8B, 4 096 token: `32 · 2 · 256 = 16 384` `cuMemcpyAsync` da **32 KiB** se i
blocchi non sono contigui in memoria. A 2–5 µs di costo di emissione ciascuna, sono
**33–80 ms di CPU solo per accodare le copie** — più del TTFT warm pubblicato (61 ms). Con
i block ID ordinati, `memcpy.rs::merge()` fonde i range adiacenti — ma solo se sono contigui
**sia lato device sia lato host** — e nel caso ideale le 16 384 copie diventano `L · 2 = 64`
da 8 MiB. Due conseguenze:

1. La patch di `docs/vllm-patch.md` (ordinare i blocchi liberi per `block_id`) **non è
   un'ottimizzazione, è una precondizione** dei numeri pubblicati; e l'RFC upstream è morta.
2. Il requisito di contiguità *host* vincola il loro allocatore pinned a replicare l'ordine
   di allocazione dei blocchi di vLLM: il pool non è indipendente dall'engine come la
   narrativa "sidecar" lascia intendere.

Il confronto col nostro caso è istruttivo per differenza: uno shard da **256 MiB** è una
copia sola. Siamo **bandwidth-bound, non issue-bound**, e il problema che PegaFlow risolve
con patch + coalescenza per noi non esiste. I nostri numeri lo confermano: 4 MiB in 194 µs
pinned (≈ 21.6 GB/s, vicino al pratico di PCIe Gen4 x16) contro 684 µs pageable
(≈ 6.1 GB/s) — il fattore 3.5× che ha chiuso il sotto-obiettivo 2 di #17.

### 8.3 Il pool pinned: NUMA, hugepages, `mlock`

`pinned_pool.rs`: `mmap` + `cudaHostRegister` (non `cudaHostAlloc`), first-touch eseguito
da un thread **bound al nodo NUMA** della GPU (`run_on_numa`), hugepages opzionali, **un
pool per nodo NUMA**, e fail-fast se l'affinità NUMA di una GPU non è determinabile. È il
pattern giusto e vale la pena spiegare perché: una H2D che attraversa il socket passa
per UPI/Infinity Fabric (≈ 20–40 GB/s per direzione, condivisi) invece che per il root
complex locale, e su un dual-socket la banda si dimezza. Le hugepages spiegano anche il
nostro cold/warm di §7.3: 2 MiB per pagina = 512× meno PTE da bloccare.

Per noi è direttamente rilevante: il Z8 G4 è dual-socket, la PMEM (EMH-2, #57) è sui
canali di *un* socket, la RTX 3090 è su *un* root complex, e #49 ha già rilevato che il
framework di perf-test è cieco a `numactl`. Adottare "alloca dal thread pinnato al nodo
giusto + un pool per nodo" costa poche righe (`os.sched_setaffinity` + first touch) e vale
potenzialmente un fattore 2 sul tier caldo.

Costo nascosto: `mlock` richiede `ulimit -l` adeguato, e la memoria pinnata **non è
swappabile né usabile come page cache**. Il tier SSD di PegaFlow è progettato per non
dipendere dal page cache (§8.4); il nostro EMH-3, che legge con `aiofiles` attraverso il
page cache, in co-locazione con un daemon che pinna la maggior parte della DDR diventerebbe
davvero "cold" a ogni lettura.

### 8.4 SSD e `io_uring`: quando serve davvero

`uring.rs`: `cfg.threads` ring, ciascuno con `io_depth` voci, `SQPOLL` opzionale,
`O_DIRECT` con allineamento imposto a `SSD_ALIGNMENT` e stride dei segmenti host arrotondati
di conseguenza (gli `iovec` puntano direttamente nel pool pinned). Perché a loro serve:
per la legge di Little, la concorrenza necessaria a saturare un NVMe è
`QD = BW · latenza / dimensione_IO`. Con blocchi da 32 KiB, 7 GB/s e ~100 µs di latenza,
`QD ≈ 21` richieste in volo: senza un'interfaccia asincrona a coda profonda il disco resta
inutilizzato.

Per noi il conto è diverso: con IO da **256 MiB**, `QD = 7 GB/s · 38 ms / 256 MiB ≈ 1`.
**Una sola lettura in volo satura già il disco**; un thread pool di `aiofiles` non è il
collo di bottiglia della banda. Dove `io_uring` ci darebbe qualcosa è altrove:

- **`O_DIRECT`**: evita la copia attraverso il page cache — 256 MiB a 15–25 GB/s di banda
  memoria single-thread sono **10–17 ms** per shard, sullo stesso ordine del trasferimento
  PCIe che segue; e rende EMH-3 indipendente dal page cache (§8.3).
- **Buffer registrati e pinned**: leggere direttamente nel buffer che poi va in H2D elimina
  uno staging; il passo successivo sarebbe GPUDirect Storage (NVMe → VRAM senza host).

Quindi la risposta quantitativa a D6 di #69: `io_uring` per noi **non è una questione di
queue depth**; ha senso solo come `O_DIRECT` + buffer pinned registrati, e resta subordinato
a #24 (se il collo è il forward CPU, il tier NVMe non è sul path critico).

### 8.5 Il control plane: costo di un lookup fuori processo

`get_num_new_matched_tokens` (`scheduler.py:210`) chiama `_tp_shard_client.query()` e, se
il backend sta ancora caricando, restituisce `None` e memorizza una *probe*: il lookup è
**non bloccante**, spalmato su più step dello scheduler, con una state machine
IDLE → Loading → Ready (documentata in `docs/vllm-request-state-machine.md`) e un path
esplicito per la "deriva di identità" — la richiesta che cambia mentre la risposta è in
volo (`scheduler.py:274-292`). È una buona progettazione ed è anche **il vero costo di un
daemon esterno**: ogni domanda fatta a un altro processo può tornare quando lo stato è già
cambiato, e va gestita. Per D4 di #69 (`osxd`) è esattamente la complessità da mettere in
conto: una predizione PT-PEP che arriva dopo che il gating ha già deciso è lo stesso
problema con un altro nome.

### 8.6 RDMA e indice distribuito

Il MetaServer registra gli hash dei blocchi per namespace (`internode/metaserver_client.rs`,
con batching e cap sulla profondità di coda); un nodo che ha evictato un blocco dopo la
registrazione produce un fetch fallito — e #401 mostra che oggi un fallimento locale abbatte
la connessione RDMA sana. È la staleness classica di un indice separato dai dati: servono
lease o versioni cross-nodo, che localmente esistono (`lease.rs`) ma la roadmap marca
"MetaServer HA" come lavoro futuro. Efficienza di rete: 194 GB/s su 8 × 400 Gbps
(= 400 GB/s di line rate) è il **48 %** — nella norma per RDMA con registrazione di memoria
e messaggi medi, non un numero anomalo. Per noi (§5) resta non applicabile.

### 8.7 Domini di guasto e versioning

Il daemon è il dominio di guasto di tutti gli engine dell'host: se muore, tutti perdono la
cache e devono degradare a miss senza cadere. Va detto a loro merito che
`python/tests/test_connector_fault_tolerance.py` e `session_crash_helper.py` testano
esattamente questo scenario. Il versioning connector ↔ server passa per `pegaflow-proto`;
un mismatch oggi fallisce in modo oscuro (#338). Sono i due costi ricorrenti di
qualunque architettura a daemon, e sono i due che un eventuale `osxd` (#69/D4) erediterebbe
per intero.

### 8.8 Sicurezza, vista dal sistema

Un handle CUDA IPC è una capability senza revoca: una volta aperto, il daemon mappa la
memoria KV dell'engine per la vita della mappatura. L'esposizione non è però "memoria
arbitraria" — l'handle mappa solo l'allocazione che il client ha esportato, e `layout.rs`
valida i range. L'esposizione reale è quella di §4.2 e §7.7: chiunque raggiunga il socket
gRPC può interrogare e leggere il KV di **altri client nello stesso namespace**, e il
namespace è una chiave di layout, non di identità. In un container con un solo tenant è
accettabile; su un host condiviso non lo è, e nel codice non c'è nulla che distingua i due
casi.

### 8.9 Ricadute concrete per OSX

| Tema | Cosa cambia per noi | Aggancio |
|---|---|---|
| `I(E; X)` come claim misurabile | Il gap LRU vs SEE+PT-PEP a pari capacità **è** il risultato del paper | #19, #21 |
| Rilascio completion-driven dei buffer pinned | Requisito di progetto per la pipeline a stream, prima di scriverla | #5 |
| Pool pinned per nodo NUMA, first-touch dal thread giusto | Poche righe, fino a 2× sul tier caldo del Z8 dual-socket | #49, #28 |
| `io_uring` = `O_DIRECT` + buffer registrati, non queue depth | Ridefinisce D6: subordinato a #24, scope più piccolo | #69/D6, #24 |
| EMH-3 dipende dal page cache | Liability in co-locazione; `O_DIRECT` la rimuove | §6.2 |
| Contesa PCIe con un daemon KV | Un partizionamento esplicito di banda è un vincolo di progetto, non un dettaglio | #33, #57 |

---

## 9. Raccomandazioni operative

| # | Azione | Priorità | Aggancio |
|---|---|---|---|
| 1 | Aggiungere PegaFlow come related work nel paper (Filone B) con l'inquadramento di §6.3 | **Alta** | Sprint 5 / Berg |
| 2 | Riformulare le rivendicazioni di novelty sui 4 punti di §6.3 prima della prossima stesura | **Alta** | Filone B |
| 3 | Non adottare PegaFlow: prerequisiti assenti (§5) e benefici non applicabili al nostro pattern | **Alta** | — decisione, non lavoro |
| 4 | Recuperare i numeri dal blog originale da rete non filtrata prima di citarli | Media | §2 |
| 5 | Formulare il claim del paper come misura di `I(E; X)`: gap LRU vs SEE+PT-PEP a pari capacità (§7.6) | **Alta** | issue #19, #21 |
| 6 | Progettare la pipeline a stream con rilascio completion-driven dei buffer pinned, non a timer (§7.8) | Media | issue #5 |
| 7 | Pool pinned per nodo NUMA con first-touch dal thread bound al nodo della GPU (§8.3) | Media | issue #49, #28 |
| 8 | `io_uring` per `AsyncNVMeIO` **solo** come `O_DIRECT` + buffer pinned registrati; subordinato a #24 (§8.4) | Bassa | gap noto README, #24 |
| 9 | Quando #8 si sblocca (`torch>=2.7`/`cu128`), rivalutare *solo* se nel frattempo arriva hardware RDMA | Bassa | issue #8 |
| 10 | Se un giorno si co-installa: partizionamento esplicito di DDR pinned e banda PCIe, e EMH-3 senza page cache (§6.2, §8.3) | Bassa | issue #33, #57 |

---

## 10. Giudizio sintetico

Ingegneria seria e tesi architetturale corretta — disaccoppiare la vita della cache dal
processo dell'engine è la mossa giusta e sarà copiata. Ma "production-grade" è, ad oggi,
un'etichetta di marketing che il tracker del progetto smentisce su tre fronti misurabili:
memory safety sul data path (#403), hardening del control plane non fatto (#353) e **zero
autenticazione nel codice del server**. I numeri pubblicati sono veri e insieme poco
trasferibili: misurano scenari da cloud provider (restart frequenti, 8 istanze per host, 8×400
Gbps per nodo, MLA+TP8) e non includono alcun confronto diretto con LMCache, Mooncake o
l'offload nativo di vLLM.

Per noi: **non adottabile e non necessario**, ma **rilevante per il paper**. Il rischio da
gestire non è tecnico, è di posizionamento — e si gestisce citandolo, non evitandolo.

---

## 11. Fonti

- [novitalabs/pegaflow](https://github.com/novitalabs/pegaflow) — codice letto a `b5d4acf` (v0.24.0, 2026-08-29), clone shallow
- [pegaflow/README.md](https://github.com/novitalabs/pegaflow/blob/master/README.md)
- [pegaflow/docs/vllm-patch.md](https://github.com/novitalabs/pegaflow/blob/master/docs/vllm-patch.md), [docs/deployment.md](https://github.com/novitalabs/pegaflow/blob/master/docs/deployment.md)
- [pegaflow issues aperte](https://github.com/novitalabs/pegaflow/issues) — #403, #402, #401, #408, #353, #339, #338, roadmap #314
- [vllm-project/vllm#31371](https://github.com/vllm-project/vllm/issues/31371) — RFC della patch, chiusa `not planned`
- vLLM blog, *vLLM x Novita AI: PegaFlow for Production-Grade External KV Cache*, 2026-05-18 — `https://vllm.ai/blog/2026-05-18-pegaflow` (**non raggiungibile da questa sessione**)
- [LMCache](https://blog.lmcache.ai/) e [Mooncake](https://kvcache-ai.github.io/Mooncake/) — alternative citate per confronto, non analizzate qui
