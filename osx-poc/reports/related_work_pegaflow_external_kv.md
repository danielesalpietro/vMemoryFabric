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
memorizzare il KV logico una volta invece di 8 (una per rank TP). Ma MLA ha già un KV latente
compresso e condiviso tra le teste: la combinazione "MLA + TP8" è il caso di massima
ridondanza possibile nel baseline. È un numero misurato nel punto in cui la vecchia soluzione
sprecava di più. Su un modello MHA/GQA con TP2 il fattore di dedup è 2, non 8.

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

## 7. Raccomandazioni operative

| # | Azione | Priorità | Aggancio |
|---|---|---|---|
| 1 | Aggiungere PegaFlow come related work nel paper (Filone B) con l'inquadramento di §6.3 | **Alta** | Sprint 5 / Berg |
| 2 | Riformulare le rivendicazioni di novelty sui 4 punti di §6.3 prima della prossima stesura | **Alta** | Filone B |
| 3 | Non adottare PegaFlow: prerequisiti assenti (§5) e benefici non applicabili al nostro pattern | **Alta** | — decisione, non lavoro |
| 4 | Recuperare i numeri dal blog originale da rete non filtrata prima di citarli | Media | §2 |
| 5 | Valutare `io_uring` per `AsyncNVMeIO` guardando `backing/uring.rs` come riferimento | Media | gap noto README |
| 6 | Quando #8 si sblocca (`torch>=2.7`/`cu128`), rivalutare *solo* se nel frattempo arriva hardware RDMA | Bassa | issue #8 |
| 7 | Se un giorno si co-installa: definire partizionamento esplicito di DDR pinned e banda PCIe (§6.2) | Bassa | issue #33, #57 |

---

## 8. Giudizio sintetico

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

## 9. Fonti

- [novitalabs/pegaflow](https://github.com/novitalabs/pegaflow) — codice letto a `b5d4acf` (v0.24.0, 2026-08-29), clone shallow
- [pegaflow/README.md](https://github.com/novitalabs/pegaflow/blob/master/README.md)
- [pegaflow/docs/vllm-patch.md](https://github.com/novitalabs/pegaflow/blob/master/docs/vllm-patch.md), [docs/deployment.md](https://github.com/novitalabs/pegaflow/blob/master/docs/deployment.md)
- [pegaflow issues aperte](https://github.com/novitalabs/pegaflow/issues) — #403, #402, #401, #408, #353, #339, #338, roadmap #314
- [vllm-project/vllm#31371](https://github.com/vllm-project/vllm/issues/31371) — RFC della patch, chiusa `not planned`
- vLLM blog, *vLLM x Novita AI: PegaFlow for Production-Grade External KV Cache*, 2026-05-18 — `https://vllm.ai/blog/2026-05-18-pegaflow` (**non raggiungibile da questa sessione**)
- [LMCache](https://blog.lmcache.ai/) e [Mooncake](https://kvcache-ai.github.io/Mooncake/) — alternative citate per confronto, non analizzate qui
