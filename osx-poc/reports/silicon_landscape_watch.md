# Silicon Landscape Watch — tassonomia del nuovo silicio AI per problema risolto

**Data:** 2026-08-16
**Natura del documento:** osservatorio strategico, non roadmap. Materiale di
contesto/letteratura per il paper Sprint 5 (Berg) — non genera item di
lavoro su `osx-poc/src/`, non tocca la issue #8 (quella resta scoped su
hardware execution reale: RTX 5080, AMD, Tenstorrent — vedi quell'issue per
lo stato operativo). Origine: critica esterna ricevuta il 2026-08-16 sulla
mappatura vendor/hardware discussa in issue #8, valutata e integrata qui.

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
| Open Silicon (modello industriale, non architettura) | RISC-V, stack aperto, no CUDA lock-in | Tenstorrent (leader ideologico) |
| Neuromorphic Computing | Paradigma computazionale alternativo | Loihi, BrainScaleS |

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
- Un riferimento nel testo originale a un progetto "NORTHSTREAM" non
  corrisponde a nulla in questo repository — chiarito dall'autore: è un
  progetto separato dell'autore, non correlato a vMemoryFabric/OSX, fuori
  scope qui.

## 4. Rilevanza specifica per vMemoryFabric

Due categorie, non tutte e otto, hanno un collegamento diretto con questo
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

Le altre categorie (Scale-Up, Wafer-Scale, Dataflow, Inference ASIC,
Neuromorphic) restano utili come mappa di mercato generale ma non toccano
direttamente le scelte architetturali di questo progetto.

## 5. Fuori scope qui, annotato per il futuro

L'autore ha segnalato un tema collegato ma distinto, da un altro suo
progetto (NORTHSTREAM): la qualità e velocità dei **dati in ingresso** alla
memoria del modello — non solo dati statici tiered (come fa EMH), ma
flussi informativi continui/real-time. Argomento esplicitamente rimandato
("lo vediamo più avanti") — non sviluppato in questo documento, annotato
solo perché riemerga nel punto giusto quando verrà ripreso.
