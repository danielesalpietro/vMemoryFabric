# Related Work — Matrice di confronto per dominio

**Data:** 2026-08-13
**Scopo:** griglia di confronto sistematica tra GCSG (questo progetto) e i
sette sistemi letti per la related-work survey del paper (`logbook_paper.md`),
raggruppata per dominio funzionale invece che per paper — per isolare
similitudini/differenze asse per asse. Alimenta direttamente §8 (Related
Work) del paper, `sprint5_berg_plan.md` B2.

**Livello di confidenza per sistema**, dichiarato una volta qui invece che
ripetuto in ogni riga:

| Sistema | Fonte | Confidenza |
|---|---|---|
| GCSG | codice proprio, letto per intero | Alta |
| FineMoE | paper completo + codice reale ispezionato | Alta |
| AdapMoE | paper completo + codice reale ispezionato (formula-matched) | Alta |
| ReMoE | paper completo | Alta sul metodo; **repo verificato privo di codice** |
| FloE | paper completo incluse appendici | Alta |
| PagedWeight | paper completo | Alta |
| MoE-Lightning | **solo fonti secondarie incrociate** (arXiv bloccato da questa sessione) | Media — metadata cross-verificati da 3 fonti indipendenti, dettagli di design da ricerca web |
| MoEpic | **solo fonti secondarie** (arXiv bloccato) | **Bassa** — nessun paper primario letto, autori non verificati |

---

## 1. Problema targettizzato

| Sistema | Problema primario |
|---|---|
| GCSG | Garantire (in teoria — vedi §6) che l'esecuzione tramite expert degradati non perda qualità, quando il router è sicuro della propria decisione |
| FineMoE | Trade-off latenza-memoria nel serving MoE via offloading fine-grained |
| AdapMoE | Overhead di on-demand loading su edge device a memoria limitata |
| ReMoE | Bassa località di riuso degli expert tra token adiacenti (pressione su cache/offload) |
| FloE | Inferenza "on-the-fly" (percettibilmente istantanea) su GPU consumer memory-constrained |
| PagedWeight | Tensione tra peso del modello MoE e crescita della KV-cache sulla stessa VRAM |
| MoE-Lightning | Throughput batch massimo su GPU low-cost/memory-constrained |
| MoEpic *(bassa confidenza)* | Hit-rate della cache expert sotto budget VRAM fisso |

## 2. Granularità della decisione

| Sistema | Unità su cui si decide |
|---|---|
| GCSG | Per-token, per-layer (intero forward di un expert) |
| FineMoE | Per-iterazione di decode (expert map completa, non solo top-1) |
| AdapMoE | Per-token/per-layer (numero di expert K) + per-layer (dimensione cache, via DP) |
| ReMoE | Parametri del router/gate soltanto — nessuna granularità a runtime, è training-time |
| FloE | Intra-expert: colonna/canale (sparsity) + matrice up-proj intera (quantizzazione) |
| PagedWeight | Per weight-page (gate_up/down di un singolo expert, per layer) |
| MoE-Lightning | Whole-expert, con "paged weights" per segmentare il transfer |
| MoEpic *(bassa confidenza)* | Sub-expert: split verticale top/bottom di un singolo expert |

## 3. Segnale usato per decidere

| Sistema | Segnale | Endogeno al modello a runtime? |
|---|---|---|
| GCSG | Confidenza gating top-1 + entropia routing + contamination-rate | **Sì** — unico basato su incertezza del router stesso |
| FineMoE | Expert map (distribuzione completa) + hint semantici dal prompt | Parzialmente (expert map sì, hint prompt no) |
| AdapMoE | Gap score top1-top2 + sensitività Fisher/Hessiana (offline) | Parzialmente (score sì, sensitività è offline) |
| ReMoE | Nessuno a runtime — loss di training (KL-anchor + località temporale) | No |
| FloE | Soglie di sparsità su attivazioni (magnitude-based, offline-calibrate) | No |
| PagedWeight | Sensitività Hessiana offline + routing mass online + prompt residual (regressione) | Parzialmente |
| MoE-Lightning | Hierarchical Roofline Model — derivato dall'hardware, non dai dati | No |
| MoEpic *(bassa confidenza)* | Hotness (implicito) + iterazione a punto fisso | Probabile, non confermato |

## 4. Momento della decisione

| Sistema | Offline | Online/runtime | Ibrido |
|---|:---:|:---:|:---:|
| GCSG | | ✅ | |
| FineMoE | | ✅ | |
| AdapMoE | ✅ (soglia/DP) | ✅ (applicazione) | ✅ |
| ReMoE | ✅ (fine-tuning) | | |
| FloE | ✅ (soglie sparsità) | ✅ (predittori prefetch) | ✅ |
| PagedWeight | ✅ (sensitività globale) | ✅ (routing, residual) | ✅ |
| MoE-Lightning | ✅ (HRM, setup-time) | | |
| MoEpic *(bassa confidenza)* | ✅ (fixed-point) | | |

## 5. Quantizzazione

| Sistema | Presente? | Tipo |
|---|---|---|
| GCSG | Sì, ma ereditata dal checkpoint (AWQ/Marlin) o simulata (path 1, INT4) | Non è il contributo del sistema |
| FineMoE | No | — |
| AdapMoE | Sì (nell'eval), ma ortogonale al metodo | HQQ 4bit / 4+2bit |
| ReMoE | No nel metodo; il rilascio GGUF è una scelta di distribuzione | — |
| FloE | **Sì, centrale** | INT2 (HQQ) su up-proj + sparsità contestuale su gate/down |
| PagedWeight | **Sì, centrale** | Any-Precision bit-plane, bitwidth dinamico per pagina |
| MoE-Lightning | No | — |
| MoEpic *(bassa confidenza)* | No (lossless per design, split non compressione) | — |

## 6. Garanzia di qualità a runtime — l'asse dove GCSG si isola davvero

| Sistema | Meccanismo | Verifica empirica (esecuzione + confronto reale)? |
|---|---|---|
| **GCSG** | Shadow execution progettata per confrontare — **ma il confronto non è implementato** (issue #19/#22, `logbook_paper.md`) | **No, oggi** — unico sistema che almeno tenta il pattern, nessun altro lo fa nemmeno |
| FineMoE | N/A — nessuna approssimazione introdotta | N/A |
| AdapMoE | Stima teorica (Taylor/Fisher), soglia calibrata offline | No |
| ReMoE | Eval standard post-fine-tuning (non runtime) | No |
| FloE | Soglie di sparsità/quantizzazione calibrate offline, degradazione riportata come costo atteso statico | No |
| PagedWeight | Stima analitica del danno (formula g·η), mai un'esecuzione di verifica | No |
| MoE-Lightning | N/A — nessuna approssimazione (paper di scheduling puro) | N/A |
| MoEpic *(bassa confidenza)* | N/A — lossless | N/A |

**Conclusione della riga più importante della matrice**: nessuno degli otto
sistemi esegue oggi un confronto empirico reale shadow-vs-riferimento a
runtime. GCSG è l'unico che ha *progettato* il meccanismo per farlo — il
gap #19/#22 è quindi un gap di completamento, non un gap di originalità
rispetto alla letteratura.

## 7. Overlap I/O-compute (pipelining)

| Sistema | Presente? | Meccanismo |
|---|---|---|
| GCSG | No (issue #5/#20, aperto) | — |
| FineMoE | Sì | Scheduler C++ multi-mutex (`ArcherTaskPool`), non un lock singolo |
| AdapMoE | Sì | CUDA stream controller, fine-grained |
| ReMoE | N/A | — |
| FloE | Sì | Transfer asincrono compatto: pinned memory + SIMD (AVX-512) + multi-stream |
| PagedWeight | Sì | Page movement asincrono a "safe boundaries" |
| MoE-Lightning | **Sì, è il contributo centrale** | CGOPipe — overlap a tre vie GPU-compute/CPU-compute/I-O |
| MoEpic *(bassa confidenza)* | Menzionato ("efficient pipeline overlap") | Dettagli non confermati |

## 8. Hardware di validazione e disponibilità codice

(già stabilito nelle sessioni precedenti, riportato qui per completezza della griglia)

| Sistema | Hardware | Codice pubblico |
|---|---|---|
| GCSG | RTX 3090 (WSL2 + Linux reale), RTX A5000, A100 (path 1, solo meccanica) | Questo repo |
| FineMoE | Testbed 6-GPU | ✅ `github.com/IntelliSys-Lab/FineMoE-EuroSys26` — verificato reale |
| AdapMoE | RTX 4090, A6000 | ✅ `github.com/PKU-SEC-Lab/AdapMoE` — verificato reale |
| ReMoE | Non specificato nel paper letto | ⚠️ repo esiste ma **senza codice del metodo** (solo GGUF model-card) |
| FloE | H100, A100, A6000, RTX 3090 | ❌ non trovato |
| PagedWeight | RTX 6000 Ada, GH200 Grace Hopper | ❌ non trovato (preprint) |
| MoE-Lightning | T4 (singola e multipla) | ❌ non trovato |
| MoEpic | Non confermato | ❌ non trovato |

---

## Parte 2 — Numeri auto-riportati ("speed-race / quality-race")

**Avvertenza esplicita, da non omettere se questa tabella finisce nel
paper**: questi sono i numeri dichiarati da ciascun paper, su modelli,
hardware e metriche **diversi tra loro**. Non è una gara equa — è un
riepilogo di cosa ciascun sistema rivendica, non una misura indipendente
comparabile 1:1. Trattarla come tale sarebbe l'errore opposto a quello
già corretto per il claim di GCSG (`logbook_paper.md`, entry
"CORREZIONE").

### Speed (throughput / latenza)

| Sistema | Claim | Modello | Hardware | Baseline di confronto |
|---|---|---|---|---|
| FineMoE | −47% latenza, +39% hit rate | Qwen1.5-MoE-A2.7B | testbed 6-GPU | SOTA offloading esistenti |
| AdapMoE | 1.35× speedup, −25% attivazioni expert | Mixtral 8x7B/8x22B | RTX 4090, A6000 | Mixtral-offloading, Pre-gated MoE |
| FloE | 48.7× vs DeepSpeed-MII, 2.6× vs Mixtral-GPU | Mixtral 8x7B | RTX 3090 | DeepSpeed-MII, Mixtral-offloading, Fiddler |
| PagedWeight | 1.94× throughput, fino a −72% memoria GPU | Qwen1.5-MoE-A2.7B, Mixtral 8x7B, Gemma-4-26B-A4B | RTX 6000 Ada, GH200 | APL uniforme, DP-LLM, MxMoE |
| MoE-Lightning | 10.3× throughput | Mixtral 8x7B | T4 singola | Sistemi SOTA con offloading |
| GCSG | Non applicabile — nessun claim di throughput oggi (overhead shadow misurato come costo, r=0.993 con latenza, non come speedup) | Mixtral 8x7B-AWQ | RTX 3090, RTX A5000 | Baseline hook-only (stesso progetto) |

### Quality (accuratezza / degradazione)

| Sistema | Claim | Metrica | Modello |
|---|---|---|---|
| FineMoE | Non centrale al paper (nessuna approssimazione) | — | — |
| AdapMoE | Zero perdita misurabile | MMLU, ARC-Challenge | Mixtral 8x7B/8x22B |
| FloE | −4.4% ~ −7.6% (media) | 7 task downstream (BoolQ, SciQ, OBQA, Winogrande, MMLU-5shot, ecc.) | Mixtral 8x7B |
| PagedWeight | "FP16-equivalent" a memoria ridotta | Perplexity (Wikitext2/C4), GSM8K, MATH-500 | 3 modelli MoE |
| MoE-Lightning | Non centrale (nessuna approssimazione) | — | — |
| GCSG | −0.19pp (path 2/3) — **ma vedi §6 sopra: non misura ciò che l'abstract dichiara** | MMLU-5shot | Mixtral 8x7B-AWQ |

---

## Riferimenti

- `osx-poc/logbook_paper.md` — letture integrali e bibliografia di lavoro
- `osx-poc/reports/development_roadmap.md` §1 — rischio #19/#22 collegato alla riga 6 sopra
- Issue GitHub: #19, #20, #21, #22, #23, #24
