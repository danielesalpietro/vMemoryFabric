# Related Work — Petals vs ExLlamaV2/tabbyAPI: gestione dell'eterogeneità e confronto con vMemoryFabric + grastorp

**Status:** Nota di ricerca / posizionamento competitivo. Non è un report sperimentale — non
contiene benchmark eseguiti in questo repository, ma un'analisi qualitativa della letteratura
di progetto e dei repository pubblici citati.

**Data:** 2026-08-16
**Progetto:** OSX — Operating System for Experts (repo: `vMemoryFabric`)
**Scope:** confronto architetturale tra due approcci esistenti alla distribuzione di modelli
su hardware eterogeneo (Petals, ExLlamaV2/tabbyAPI) e l'approccio di `vMemoryFabric` +
`grastorp` (progetto correlato, non incluso in questo repository).

*(Nota di verifica: sia **Petals** che **ExLlamaV2** e la sua API **tabbyAPI** sono
regolarmente presenti e attivi su GitHub).*

---

## 1. Petals (P2P Swarm & Pipeline Parallelism)

**Petals** (sviluppato da BigScience / Yandex e ospitato su GitHub sotto l'organizzazione
`petals-infra`) è progettato con una filosofia in stile **BitTorrent / Peer-to-Peer**. Nasce
per consentire la collaborazione tra utenti con hardware completamente disomogeneo e
distribuito geograficamente tramite Internet.

### Come gestisce l'eterogeneità

- **Assegnazione dei layer (pipeline parallelism):** Petals divide un grande modello (es.
  Llama-3-70B, Mixtral) in blocchi di layer consecutivi (transformer blocks). Un server con
  una RTX 3090 (24 GB) può ospitare 12 layer, mentre un nodo con una scheda da 8 GB ne ospita
  solo 4.
- **Routing e P2P:** quando invii un prompt, il client contatta uno "swarm" (uno sciame di
  nodi). Il primo token attraversa i layer dal Nodo A → Nodo B → Nodo C.
- **Speculative decoding distribuito:** per superare la latenza di rete, Petals usa forme di
  generazione speculativa: i nodi con VRAM locale generano più token futuri in parallelo per
  ridurre il numero di round-trip di rete.

### Limiti strutturali

1. **Latenza di rete elevata:** ogni passaggio tra layer su nodi diversi richiede il
   trasferimento degli *attivatori* (hidden states) tramite la rete. Su una normale
   connessione Internet o LAN non dedicata, la latenza di generazione (token/s) crolla per le
   comunicazioni sequenziali.
2. **Nessun memory fabric sottostante:** Petals gestisce il passaggio di testimone a livello
   applicativo (tensor flow), non a livello di memoria o di pagina. Non c'è condivisione di
   VRAM, ma un semplice passaggio sequenziale dei dati.

---

## 2. ExLlamaV2 + tabbyAPI (Split-GPU & Tensor/Pipeline su singolo host)

**ExLlamaV2** (sviluppato da `turboderp` su GitHub) è una libreria di inferenza C++/CUDA ad
altissime prestazioni focalizzata su modelli quantizzati tramite il formato EXL2. **tabbyAPI**
(sviluppato da `theroyallab`) ne è il server API ufficiale e moderno.

### Come gestisce l'eterogeneità

ExLlamaV2 non è nato come un sistema distribuito su rete multinodo, ma è eccezionale nel
gestire **GPU eterogenee all'interno dello stesso host** (es. una RTX 3090 collegata via PCIe
4.0 affiancata a una RTX 5080 o una serie 2000).

- **Splitting flessibile (memory allocation ratio):** invece di pretendere schede identiche
  (come il classico tensor parallelism di PyTorch/NCCL), ExLlamaV2 permette di specificare la
  VRAM esatta da allocare su ogni scheda (es. `[18, 10, 8]` GB).
- **Layer allocation manuale/dinamico:** distribuisce i layer del modello in modo asimmetrico
  sui bus PCIe locali per evitare saturazioni su schede con meno VRAM o con ampiezza di banda
  inferiore.
- **Paged attention & continuous batching:** tramite integrazioni recenti e l'uso di
  **tabbyAPI**, gestisce la KV cache e il prompt caching usando la paged attention
  (allocazione non contigua della memoria di contesto).

### Limiti strutturali

1. **Vincolato al singolo server (host bus):** ExLlamaV2 lavora tramite bus PCIe locali; non
   possiede un layer RDMA nativo per estendersi su rete Ethernet multinodo in modo
   trasparente.
2. **Allocazione statica dei layer:** la VRAM viene allocata per layer completi. Se un blocco
   non entra nello spazio rimanente di una GPU, deve essere forzato sulla GPU successiva,
   creando colli di bottiglia nei bus di trasferimento.

---

## 3. Confronto con l'architettura vMemoryFabric + grastorp

| Caratteristica              | Petals                              | ExLlamaV2 / tabbyAPI                          | vMemoryFabric + grastorp                                   |
|------------------------------|--------------------------------------|------------------------------------------------|--------------------------------------------------------------|
| **Topologia**                | Multinodo P2P su Internet/WAN.       | Multi-GPU eterogeneo **solo locale** (PCIe).    | Multinodo eterogeneo su LAN/fabric (RDMA, PCIe).             |
| **Livello di operazione**    | Applicativo / pipeline.              | Tensor engine / driver C++ CUDA.                | **Hypervisor / layer di memoria (paged fabric)**.             |
| **Frammentazione VRAM**      | A livello di blocchi di layer.       | A livello di blocchi di layer e VRAM locale.    | **A livello di singola pagina di memoria logica**.             |
| **Paging e swapping**        | Assente.                             | Paged attention per KV cache locale.            | **Paging distribuito continuo (RAM/NVMe via grastorp)**.       |

Il punto di rottura di `vMemoryFabric` rispetto a Petals o ExLlamaV2 sta nel fatto che
entrambi pensano a "layer del modello da assegnare a un chip". `vMemoryFabric` ribalta il
problema: la memoria VRAM/RAM/NVMe di tutti i nodi viene aggregata prima in un **memory pool
distribuito**, mentre `grastorp` (progetto correlato, esterno a questo repo) coordina la
gerarchia di persistenza e paging. Il motore di inferenza vede uno spazio di indirizzamento
unificato, e la rete non è più un collo di bottiglia per il passaggio di layer, ma un bus di
paging della memoria.

---

## 4. Fonti

- Petals — organizzazione `petals-infra` su GitHub (BigScience / Yandex).
- ExLlamaV2 — `turboderp/exllamav2` su GitHub.
- tabbyAPI — `theroyallab/tabbyAPI` su GitHub.
