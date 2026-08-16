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
**Revisione 2026-08-16 (round 3):** aggiunto §5 sull'ecosistema silicio
cinese (Ascend, Cambricon, Biren, Moore Threads, MetaX), fact-checked
punto per punto — scartato senza verifica in round 1, reintrodotto dopo
verifica invece di restare nell'oblio solo perché non è hardware
acquistabile per il PoC.

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

## 5. Hardware upstream: l'ecosistema cinese

Scartato nella prima revisione di questo documento con l'argomento "non è
lab-actionable per un PoC single-developer" — vero, ma è un criterio
parziale per un documento che si chiama *landscape watch*: la scala non
va in oblio solo perché non è acquistabile qui. Fact-check fatto ora,
non preso dal testo della critica originale (che conteneva almeno un
errore, vedi sotto):

- **Huawei Ascend 910C**: singolo chip a ~60% delle performance di
  inferenza di un H100 (2.4 vs 4 PFLOPS FP16/FP8, dato da ricerca
  DeepSeek riportata da Tom's Hardware/TrendForce) — ma il sistema
  **CloudMatrix 384** (package di più 910B) supera il rack NVIDIA GB200
  NVL72 su alcune metriche chiave a livello sistema. Il gap è reale al
  livello del singolo chip, si restringe o si inverte al livello sistema.
- **Cambricon**: il nome corretto è **Siyuan 590** (7nm, 2024, modellato
  su A100) e **Siyuan 690** (target classe-H100, produzione di massa
  2026) — non "MLU370/590" come nella critica originale ricevuta, che
  usava il branding di prodotto precedente. Target dichiarato 2026: 500K
  acceleratori totali, di cui 300K Siyuan 590/690.
- **Biren BR100**: GPGPU rivendicato alla pari di H100 al lancio (2022);
  IPO a Hong Kong gennaio 2026 — non più solo un design da monitorare,
  un'azienda quotata.
- **Moore Threads**: IPO sullo STAR Market di Shanghai, debutto 5
  dicembre 2025.
- **MetaX C600**: HBM3e, supporto FP8, produzione di massa Q1 2026.
- Segnale aggregato più significativo dei singoli chip: l'intero
  ecosistema cinese ha completato l'adattamento "Day-0" di DeepSeek V4 nel
  2026 — passaggio da deployment in ritardo rispetto a NVIDIA a deployment
  simultaneo. È un indicatore di maturità dello stack, non un aneddoto
  isolato su un singolo vendor.

**Perché resta fuori dalla roadmap operativa (issue #8) ma non da questo
documento**: nessun canale di acquisto diretto per un lab occidentale,
nessuna community/documentazione in inglese comparabile a CUDA/ROCm,
restrizioni all'export in entrambe le direzioni. Questo continua a
escluderlo come opzione hardware per il PoC — ma la scala (500K
acceleratori/anno dichiarati, IPO multiple, parità a livello sistema su
alcune metriche) è un fatto di mercato che un osservatorio strategico non
può omettere solo perché non è comprabile da qui. La critica originale
aveva individuato il buco giusto nel punto sbagliato del documento (voleva
sezioni su CANN/MindSpore dentro un file di scoping interno); qui, come
fatto di scala verificato, il buco era reale.
