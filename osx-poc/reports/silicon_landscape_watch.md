# Silicon Landscape Watch — tassonomia del nuovo silicio AI per problema risolto

**Data:** 2026-08-16
**Natura del documento:** osservatorio strategico, non roadmap. Materiale di
contesto/letteratura per il paper Sprint 5 (Berg) — non genera item di
lavoro su `osx-poc/src/`, non tocca la issue #8 (quella resta scoped su
hardware execution reale: RTX 5080, AMD, Tenstorrent — vedi quell'issue per
lo stato operativo). Origine: critica esterna ricevuta il 2026-08-16 sulla
mappatura vendor/hardware discussa in issue #8, valutata e integrata qui.
**Revisione 2026-08-16 (round 2):** integrato CXL come categoria mancante,
corretta l'incoerenza tassonomica di "Open Silicon" (era una categoria
alla pari, ora è una nota trasversale), rimossa la digressione fuori
scope su un progetto esterno non correlato — tutti e tre i punti da una
seconda critica esterna, verificati prima di accettarli (vedi §3).

---

## 1. Perché questo documento esiste

La issue #8 classifica i vendor per "tipo di silicio" (GPU vs non-GPU). Una
critica esterna ha proposto un framework diverso e più utile per capire
*dove sta andando* il mercato, non solo *cosa comprare adesso*: classificare
per **problema attaccato**, non per architettura fisica. Tre colli di
bottiglia ricorrenti: memory bandwidth, data movement, costo energetico per
token inferito. La tesi centrale — "il mercato non converge verso più
potenza di calcolo, ma verso meno movimento dei dati" — è direttamente
rilevante per un progetto che tratta il memory tiering come primo
cittadino architetturale (EMH).

## 2. La tassonomia proposta

| Categoria | Obiettivo | Esempi citati |
|---|---|---|
| Scale-Up Compute | Più FLOPS, architettura general-purpose | NVIDIA, AMD |
| Memory-Centric Compute | Meno spostamento dati, pesi vicino al calcolo | Tenstorrent, d-Matrix, Untether AI, EnCharge AI, EdgeCortix |
| Inference ASIC | Token/s/Watt, non training | Groq, Etched, FuriosaAI, Rebellions, Hailo, Mythic |
| Dataflow Architectures | Memoria distribuita nel dataflow | SambaNova |
| Wafer-Scale Compute | Tutta la memoria on-chip | Cerebras |
| Photonic Fabrics / Network-Centric AI | Connessione compute↔memoria via fotonica, non il chip in sé | Celestial AI, Lightmatter, Ayar Labs |
| Analog / In-Memory Compute | Moltiplicazione matriciale come fenomeno analogico, non digitale | Mythic, EnCharge, Rain AI |
| Memory Pooling / Disaggregation (standard aperto, non vendor) | Pooling ed espansione di memoria system-level tra host, indipendente dal chip di calcolo | CXL Consortium — Intel Xeon 6, AMD EPYC (adozione CPU); Samsung/SK Hynix/Micron (moduli CXL-attached fino a 256GB); Astera Labs/Microchip/IntelliProp (switch) |
| Neuromorphic Computing | Paradigma computazionale alternativo | Loihi, BrainScaleS |

**Nota tassonomica:** "Open Silicon" (RISC-V, stack aperto, no CUDA
lock-in) è stato rimosso come riga a sé — non risponde alla stessa domanda
delle altre ("che collo di bottiglia attacca?"), risponde a "che modello
industriale/di licensing ha?". È un asse trasversale, non una categoria
alla pari: Tenstorrent, ad esempio, è *sia* Memory-Centric Compute
(architettura Tensix/NoC) *sia* Open Silicon (modello industriale
RISC-V/no lock-in) — le due etichette descrivono livelli diversi dello
stesso vendor, non vendor diversi.

## 3. Verifica — cosa regge, cosa è illustrativo

Controllato puntualmente prima di archiviare questo documento, non preso
per buono dal testo originale:

- **Le aziende citate sono reali e attive**, non un elenco gonfiato.
  Verificato in particolare **Mythic** (categoria Analog AI): tutt'altro
  che marginale nel 2026 — round da $125M, partnership Honda su chip
  analogici automotive, acquisizione di Videantis. Smentisce l'ipotesi
  iniziale (mia) che fosse un esempio superato.
- **La tabella a stelle Training vs Inference** (NVIDIA ⭐⭐⭐⭐⭐/⭐⭐⭐⭐⭐,
  Groq ⭐/⭐⭐⭐⭐⭐, ecc.) è **illustrativa, non verificata puntualmente** —
  utile come intuizione qualitativa (pochi competono con NVIDIA su
  training general-purpose, la maggior parte punta sull'inferenza), da non
  citare come dato nel paper senza fonte primaria.
- **CXL 3.0/3.1** (categoria Memory Pooling/Disaggregation, aggiunta in una
  revisione successiva del documento): verificato, non solo citato. Intel
  Xeon 6 (Granite Rapids+) e AMD EPYC (Turin+) hanno adozione CXL a livello
  CPU; moduli CXL-attached DRAM fino a 256GB di Samsung/SK Hynix/Micron
  sono già in produzione; il settore parla di "Phase 3: memory pooling"
  calendarizzata 2026-2027. Era l'assenza più seria della prima versione di
  questo documento — è l'analogo hardware più diretto dell'EMH tra tutto
  ciò che è citato qui.
- `vllm-ascend` (plugin ufficiale sotto l'org GitHub `vllm-project`, non un
  fork terzo) dichiara esplicitamente supporto **Mixture-of-Experts** per
  hardware Ascend NPU — verificato. Rilevante in relazione al caveat già
  registrato in issue #8: `tt-forge` di Tenstorrent non copre Mixtral/MoE
  nella sua lista pubblica di 800+ varianti testate, mentre il plugin
  Ascend lo dichiara come caso d'uso supportato. Non un'indicazione
  d'acquisto (nessun path di importazione/community pratico per un lab
  occidentale), solo un dato tecnico più preciso di quanto emerso nella
  prima revisione di questo documento.

## 4. Rilevanza specifica per vMemoryFabric

Tre categorie, non tutte e otto, hanno un collegamento diretto con questo
progetto — le altre restano contesto di mercato generico:

- **Memory-Centric Compute** è l'inquadramento corretto per Tenstorrent
  (già in issue #8), ma allarga la lente: d-Matrix, Untether AI, EnCharge
  AI, EdgeCortix attaccano lo stesso problema (pesi vicino al calcolo,
  meno traffico DRAM) con silicio diverso. Il parallelo utile per il paper:
  OSX fa a livello **software** (`EAT`/`TierManager`, promozione
  hot-expert a VRAM) quello che questa categoria fa a livello **silicio**
  — stessa filosofia, layer diverso dello stack. Vale come riferimento di
  letteratura/motivazione, non come target di porting.
- **Photonic Fabrics / Network-Centric AI** (Celestial AI, Lightmatter,
  Ayar Labs) è l'unico buco reale rispetto a quanto discusso finora in
  issue #8 — non ne avevamo parlato. L'idea di un fabric fotonico che
  disaggrega compute e memory pool è l'equivalente hardware dell'EMH: la
  stessa frase del README ("hierarchical memory placement... network
  topology awareness") descrive tanto l'obiettivo di OSX quanto il
  problema che questi vendor risolvono in silicio. Da citare nella
  literature review del paper come corrente di ricerca affine, non da
  valutare come hardware d'acquisto.
- **Memory Pooling / Disaggregation (CXL)** è l'analogia hardware più
  diretta dell'EMH tra tutte quelle citate in questo documento: CXL fa a
  livello di **fabric fisico** (pooling/espansione di memoria tra host,
  standard aperto multi-vendor) esattamente quello che EAT/TierManager
  fanno a livello **software** (promozione/eviction hot-cold tra
  VRAM/DDR/NVMe). A differenza di Memory-Centric Compute e Photonic
  Fabrics, CXL non è un'analogia di filosofia progettuale ma uno standard
  che sta arrivando su hardware server mainstream (Xeon 6, EPYC Turin+)
  nella stessa finestra temporale del PoC — la citazione più diretta per
  la sezione related-work del paper.

Le altre categorie (Scale-Up, Wafer-Scale, Dataflow, Inference ASIC,
Neuromorphic) restano utili come mappa di mercato generale ma non toccano
direttamente le scelte architetturali di questo progetto.
