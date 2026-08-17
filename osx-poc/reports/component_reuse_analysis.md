# Component Reuse Analysis — Petals / ExLlamaV2 / ExLlamaV3 / tabbyAPI → vMemoryFabric

**Ruolo:** analisi da solutions architect — cosa, di quanto già scritto in Petals,
ExLlamaV2, ExLlamaV3 e tabbyAPI, è concretamente riusabile in `vMemoryFabric`, e perché
sì o perché no. Non è un esercizio teorico: ogni voce è confrontata contro il codice
*attuale* di `vMemoryFabric` (`osx-poc/src/eat/`, `osx-poc/src/tier/`,
`osx-poc/src/scheduler/`), non contro la roadmap.

**Data:** 2026-08-16
**Prerequisito:** [`related_work_petals_exllama.md`](related_work_petals_exllama.md) (v2,
verificato su codice) — questa nota assume note quelle scoperte.

---

## 0. Metodo e stato reale di vMemoryFabric (non la roadmap — il codice oggi)

Prima di valutare cosa riusare, va detto cosa `vMemoryFabric` **non ha ancora**, perché
altrimenti il confronto è fuorviante:

- **Nessun networking, nessun multinodo.** `osx-poc/src/` non contiene socket, RPC, DHT
  o codice distribuito di alcun tipo — è tutto single-process, single-host, single-GPU
  (`osx-poc/src/tier/gpu.py:52-56`, un solo `torch.device`). La combinazione
  "vMemoryFabric + grastorp multinodo" descritta nella nota di related-work è
  un'architettura target, non lo stato di questo repo.
- **Il tier DDR4 è puramente passivo.** `TierManager`/`GPUTransfer`
  (`osx-poc/src/tier/manager.py`, `gpu.py`) trattano la DDR4 solo come un buffer di
  passaggio: uno shard vi arriva da NVMe, vi resta come blob di byte opachi, e ne riparte
  verso la VRAM con una `torch.Tensor.to(device)` bloccante o `non_blocking` — **nessun
  calcolo avviene mai nel tier DDR4**.
- **Granularità fissa, non paginata.** `SlabAllocator` (`osx-poc/src/eat/slab.py`) è un
  pool piatto di `n_slots` da `SHARD_SIZE_BYTES` (256 MB) con una free-list lineare —
  nessun page-hash, nessuna deduplicazione, nessun reference counting tra shard che
  condividono contenuto.
- **AER (replica multi-GPU) è uno stub puro.** `AERManager.replication_factor()` ritorna
  sempre `1` (`osx-poc/src/scheduler/aer.py:46-53`) — il trigger logic esiste ed è
  testato, ma non c'è alcun meccanismo di trasferimento dati tra GPU perché non c'è una
  seconda GPU nel dev setup.

Questo è il metro di paragone corretto: non "cosa fa vMemoryFabric rispetto a un sistema
multinodo maturo", ma "cosa, in questi quattro progetti, risolve un problema che
vMemoryFabric ha *oggi*, nel suo scope PoC attuale".

---

## 1. Licenze — vincolo che filtra tutto il resto

| Progetto | Licenza | Compatibile con riuso di codice in vMemoryFabric (MIT)? |
|---|---|---|
| vMemoryFabric | MIT | — |
| Petals | MIT | ✅ Sì |
| ExLlamaV2 | MIT | ✅ Sì |
| ExLlamaV3 | MIT | ✅ Sì |
| **tabbyAPI** | **AGPL-3.0** | ❌ **No** — copyleft virale: incorporare codice tabbyAPI (anche parziale) obbligherebbe la porzione derivata, se non l'intero progetto, sotto AGPL-3.0 |

Questo esclude a priori qualunque riuso di codice letterale da tabbyAPI (che comunque,
per contenuto, è solo un layer HTTP/OpenAI-compatible — vedi §4). Petals, ExLlamaV2 ed
ExLlamaV3 sono tutti MIT: nessun vincolo di licenza al riuso, letterale o come pattern.

---

## 2. SÌ — riusabile (codice o pattern), con motivazione puntuale

### 2.1 exllamav3 `moe_cpu_host.py` — pattern di compute offload sul tier RAM

**Cosa:** un processo separato possiede pesi in memoria host pinned/huge-page, comunica
col processo GPU via un job ring in shared memory con flag di sincronizzazione (nessun
round-trip Python), stream i dati "caldi" verso la GPU mentre calcola i dati "freddi" in
loco con kernel AVX-512.

**Perché riusarlo — è la lacuna più diretta di vMemoryFabric oggi:** `GPUTransfer.to_vram()`
(`osx-poc/src/tier/gpu.py:60-105`) è l'unico modo in cui la DDR4 interagisce col resto del
sistema, ed è un trasferimento bloccante/passivo — la DDR4 non calcola mai nulla. Il
pattern di `MoeCpuHost` è esattamente il tassello mancante per far diventare il tier
EMH-1c (DDR4) un partecipante attivo invece di un semplice buffer di passaggio, cosa che
il nome stesso "Tier Manager" promette ma il codice attuale non fa. Non è un'idea
astratta: la struttura (processo figlio + shared memory + arena pinned + job ring) è
riusabile come design diretto per un nuovo componente `tier/cpu_compute.py`.

**Cosa NON portare 1:1:** il codice è specifico ai kernel MoE di exllamav3 (trellis EXL3,
routing top-k) — va riscritto per il tipo di calcolo che vMemoryFabric vuole eseguire
lato DDR4 (se mai), non copiato. Il valore riusabile è l'architettura di
sincronizzazione (shared-memory job ring, huge-page arena, GPU-streaming per il
sottoinsieme "caldo"), non le funzioni CUDA/AVX-512 stesse.

### 2.2 exllamav3 `cpu_cache.py` (`CPUPageCache`) — pagine content-hash, eviction chain-aware

**Cosa:** cache a pagine fisse indicizzate per hash di catena, con eviction che
distingue catene orfane, alberi interi, e LRU sulla radice.

**Perché riusarlo:** è un upgrade diretto e ben specificato di due componenti esistenti
di vMemoryFabric messi insieme:
- `SlabAllocator` (`osx-poc/src/eat/slab.py`) oggi non ha alcun concetto di hash/dedup —
  due shard con contenuto identico occupano due slot distinti. Il design a pagine
  hash-addressed di `CPUPageCache` è un riferimento diretto per estendere lo slab
  allocator con deduplicazione (rilevante per esperti condivisi/broadcast tra layer).
- `SEEPolicy`/`LRUPolicy` (`osx-poc/src/tier/policies.py`) oggi rankano gli shard uno
  per uno, senza nessuna nozione di dipendenza tra shard (es. shard che fanno parte
  della stessa catena logica). L'eviction "catena-aware" di `CPUPageCache` è un modello
  concreto per estendere `SEEPolicy.rank()` quando (se) vMemoryFabric introduce shard con
  dipendenze tra loro.

**Cosa NON portare 1:1:** l'implementazione è accoppiata alle `Cache`/`PageTable` di
exllamav3 (tensori CUDA per layer, non lo shard-per-file di vMemoryFabric) — va
riscritta contro `EATEntry`/`SlabAllocator`, non importata.

### 2.3 Petals — pattern di fault tolerance (retry + blacklist + re-routing)

**Cosa:** `InferenceSession.step()` e `sequence_manager.on_request_failure()`
(`src/petals/client/inference_session.py:284-362`) implementano retry con backoff,
blacklist temporanea del peer fallito, e re-derivazione del path via DHT.

**Perché riusarlo (in prospettiva, non oggi):** è l'unico pezzo dei quattro progetti che
risolve un problema — nodo remoto che sparisce a metà operazione — che esiste solo nel
target futuro "vMemoryFabric + grastorp multinodo", non nel PoC attuale (single-host).
Vale la pena registrarlo come riferimento di design per quando quel lavoro comincerà,
perché è un pattern maturo e testato in produzione (Petals gestisce esattamente
l'eterogeneità di rete che il fabric multinodo dovrà gestire), non perché ci sia oggi
qualcosa da modificare in `osx-poc/src/`.

**Perché NON è "riuso di codice" in senso stretto:** è profondamente accoppiato a
`hivemind.DHT` e al modello dati di Petals (sequenza di blocchi transformer, non pagine
di memoria) — quello che si porta è la strategia (retry-then-blacklist-then-reroute), non
le classi.

---

## 3. NO — non riusabile, con motivazione puntuale

### 3.1 hivemind (DHT/P2P di Petals) come libreria intera

Introdurrebbe una dipendenza pesante (hivemind è pensato per training distribuito P2P su
WAN, con autograd-aware RPC che vMemoryFabric non userà mai) per risolvere un problema —
service discovery su LAN/fabric — che ha soluzioni molto più leggere (gRPC/Consul/etcd,
o anche solo una libreria DHT minimale) quando il multinodo arriverà davvero. Importare
hivemind oggi significherebbe importare macchinari (compressione tensori
per-autograd-step, DHT record TTL pensati per churn di peer su Internet) che non
corrispondono a un fabric LAN affidabile e controllato dall'operatore.

### 3.2 Il modello a "blocchi di layer consecutivi" di Petals

L'unità di distribuzione di Petals è un blocco di transformer layer contigui — un
concetto di grana grossa e specifico all'architettura del modello. `vMemoryFabric` lavora
a livello di pagina/shard di memoria, un'astrazione più fine e agnostica rispetto al tipo
di oggetto AI (l'obiettivo dichiarato nel README è proprio superare "layer da assegnare a
un chip"). Riusare il codice di block-assignment di Petals (`block_selection.py`)
significherebbe importare l'esatta astrazione che vMemoryFabric è pensato per superare.

### 3.3 Il core ExLlamaV2/V3 (`model.py`, `cache.py`, kernel CUDA)

Sono estensioni C++/CUDA strettamente accoppiate ai propri kernel di attenzione e al
proprio formato di quantizzazione (EXL2/EXL3) — non librerie generiche importabili. Non
esiste un modo di "riusare" `set_device_map()` o `CachePage` senza portare con sé l'intero
motore di inferenza. Quello che si riusa è il *design* (già coperto in §2), non il
codice.

### 3.4 tabbyAPI

Oltre al blocco di licenza (§1), il contenuto stesso non è rilevante: è un layer
HTTP/OpenAI-compatible sopra un motore di inferenza (`endpoints/OAI/`,
`endpoints/Kobold/`) — vMemoryFabric non ha (e nello scope PoC attuale non ha bisogno di)
un server API pubblico; il suo confine è l'hook verso vLLM (`GCSGWorker`,
`osx-poc/src/scheduler/gcsg.py`), non un endpoint REST.

### 3.5 EXL2/EXL3 quantizzazione a bit-budget variabile (`conversion/optimize.py`)

Tecnicamente riusabile (MIT, nessun vincolo architetturale forte), ma **fuori scope**:
richiederebbe a vMemoryFabric di possedere un proprio pipeline di quantizzazione, mentre
oggi lo fa deliberatamente vLLM/AWQ/Marlin a monte (`GCSGWorker._quantize_int4`,
`_AWQShadowExpert`, `osx-poc/src/scheduler/gcsg.py:406-480`). Aggiungere un secondo
schema di quantizzazione proprietario duplicherebbe una responsabilità già delegata,
senza un problema concreto che lo giustifichi oggi.

---

## 4. Raccomandazione, in ordine di priorità

1. **Adottare il pattern `moe_cpu_host.py` per rendere DDR4 un tier attivo** (§2.1) — è
   la lacuna più diretta tra "quello che il nome Tier Manager promette" e quello che il
   codice fa oggi. Impatto: nuovo modulo `tier/`, nessuna modifica di licenza.
2. **Estendere `SlabAllocator` con pagine content-hash** ispirandosi a `CPUPageCache`
   (§2.2) — dedup tra shard identici, propedeutico anche a un futuro AER (replica) che
   oggi non ha alcun meccanismo di trasferimento dati.
3. **Registrare (non implementare ora) il pattern retry/blacklist/reroute di Petals**
   (§2.3) come riferimento per quando il lavoro multinodo comincia — nessuna azione di
   codice nel PoC attuale.
4. **Non importare hivemind, il modello a blocchi di Petals, i core CUDA di ExLlamaV2/V3,
   o alcun codice tabbyAPI** (§3) — per i motivi sopra: astrazione sbagliata, accoppiamento
   troppo forte, o licenza incompatibile.

---

## 5. Fonti

Stesso set di repository e commit di `related_work_petals_exllama.md` §5, più il codice
attuale di `vMemoryFabric`: `osx-poc/src/eat/slab.py`, `osx-poc/src/tier/manager.py`,
`osx-poc/src/tier/gpu.py`, `osx-poc/src/tier/policies.py`, `osx-poc/src/scheduler/aer.py`,
`osx-poc/src/scheduler/gcsg.py`.
