# Logbook — Paper Track (Filone B, Sprint 5 / Berg)

Dev diary scoped to the paper track only — mirrors `LOGBOOK.md`'s format
(one dated section per working session, nothing overwritten, corrections
recorded as new entries) but kept separate because the paper track has a
different audience and a different failure mode: `LOGBOOK.md` records
"did the code work," this file records "is the claim actually true and
is it actually new." Tracks against `osx-poc/reports/sprint5_berg_plan.md`
§3 (Filone B — Paper).

---

## Piano di lavoro — sotto-obiettivi del Filone B

| # | Sotto-obiettivo | Corrisponde a | Stato |
|---|---|---|---|
| 1 | Decisioni preliminari: venue (arXiv), formato (ACM `sigconf`, stile EuroSys), deadline (~fine agosto 2026), autore (Daniele Carmelo Salpietro) | §B0 del piano | ✅ fatto, 2026-08-12/13 |
| 2 | Related-work survey | §B3 (prima metà) | 🟡 in corso — prima ricerca fatta oggi, vedi entry sotto |
| 3 | Riuso materiale esistente (`gcsg_shadow_execution_report.md`, `poc_final_report.md`) + espansione outline in prosa | §B1/§B2 | 🔲 outline in bozza in `sprint5_berg_plan.md` §B2, non ancora scritto come testo |
| 4 | Figure (architettura vettoriale, grafici da dati grezzi) | §B3 (seconda metà) | 🔲 non iniziato |
| 5 | Setup scrittura LaTeX (`osx-poc/paper/`, template ACM `sigconf`, `.bib`) | §B4 | 🔲 non iniziato — bibliografia di lavoro di questa entry ne è il seme |
| 6 | Draft completo + revisione interna + checklist arXiv (categoria, licenza, `.bbl`) | §B5/§B4 | 🔲 non iniziato |

Il sotto-obiettivo 2 (related work) è stato esplicitamente identificato nel
piano come il rischio più alto sulla finestra stretta fino a fine agosto —
motivo per cui parte per primo, non per ultimo, nonostante il numero
d'ordine qui sopra segua l'outline del piano e non la priorità reale.

---

## 2026-08-13 — Prima ricerca di related work: originalità, e una sovrapposizione seria da affrontare

**Contesto:** richiesta diretta del project owner ("originalità?") dopo la
sintesi dei risultati del PoC. Non si poteva rispondere con sicurezza senza
controllare la letteratura — la risposta immediata a quella domanda,
in chat, era esplicitamente costruita su un'assunzione non verificata.
Questa entry è la prima verifica reale, non la survey completa.

### Metodologia

Due query mirate via web search, 2026-08-13, non esaustive — primo passo,
non conclusione:

1. `MoE inference expert offloading memory hierarchy VRAM DRAM SSD caching system 2025 2026`
2. `quantized MoE serving quality safety shadow verification gating confidence expert`

### Risultato 1 — la gerarchia di memoria (M1/M2) è in uno spazio affollato, non vuoto

L'idea centrale di EAT+TierManager — livelli hot/warm/cold su
VRAM/DDR/NVMe/PMEM, promozione/eviction guidata da hotness — **non è uno
spazio vuoto nel 2025-2026**. Trovati almeno 5-6 sistemi che fanno
sostanzialmente la stessa cosa (dettaglio in bibliografia sotto, Cluster A).

**Sovrapposizione più seria: FineMoE, EuroSys 2026.** Stessa venue/anno
target di questo progetto, expert hit/miss caching per serving
memory-constrained — il tipo di coincidenza che un lettore nota per primo.
**Priorità reale del prossimo giro di lavoro: leggerlo per intero e
differenziare esplicitamente**, non limitarsi a citarlo insieme agli altri.

### Risultato 2 — GCSG (shadow-verification della qualità sotto quantizzazione) non ha trovato un corrispettivo diretto

Nessuno dei risultati della seconda query fa esattamente quello che fa
GCSG: eseguire un pool "shadow" a runtime per verificare se la
quantizzazione aggressiva sta degradando la confidenza del gating, prima
che il degrado arrivi all'output. I risultati più vicini lavorano su assi
diversi — PagedWeight sceglie il bitwidth dinamicamente per pagina/expert
(non verifica in parallelo), AdapMoE adatta il gating in base a una
sensitivity stimata (non esegue una verifica shadow), MoEQuant è
calibrazione a tempo di training, non a runtime.

**Non trovare un match in due query non è prova di originalità** — è solo
la premessa per una survey più profonda mirata specificamente al
meccanismo di GCSG, non alla categoria generale "quantizzazione MoE".

### Non fatto ancora (prossimi passi, in ordine di urgenza)

1. Lettura integrale di FineMoE (EuroSys'26) — stessa venue/anno, massima
   priorità — e differenziazione esplicita per la sezione Related Work.
2. Lettura di MoEpic e MoE-Lightning (trovati solo via fonti secondarie,
   non ancora con URL diretto al paper — da recuperare prima di poterli
   leggere).
3. Query aggiuntive mirate specificamente al meccanismo di GCSG, non alla
   categoria generale: "shadow expert verification", "speculative
   quantization safety inference", "contamination detection quantized
   LLM", "runtime quality guard mixture of experts".
4. Metadata completi (autori, venue, anno esatto, abstract) per ogni voce
   della bibliografia sotto — oggi solo titolo+URL dai risultati di
   ricerca, non verificati alla fonte. Nessuna voce qui sotto è pronta per
   un vero `.bib` senza questo passaggio.

---

## Bibliografia di lavoro (prima raccolta, 2026-08-13)

**Avvertenza esplicita:** titoli e URL sotto sono reali (dai risultati di
ricerca), ma nessuno di questi paper è stato ancora aperto/letto per
intero in questa sessione — autori, venue esatta (tranne FineMoE, dichiarata
esplicitamente EuroSys'26 dalla fonte), anno preciso e abstract non sono
verificati alla fonte primaria. Trattare come lista di candidati da
verificare, non come bibliografia citabile as-is.

### Cluster A — Expert offloading / gerarchia di memoria per MoE serving (il più vicino a M1/M2)

| Paper | Riferimento | Nota |
|---|---|---|
| **FineMoE** — "Taming Latency-Memory Trade-Off in MoE-Based LLM Serving" | EuroSys 2026 — [PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf) | **Priorità massima** — stessa venue/anno target |
| FloE — "On-the-Fly MoE Inference on Memory-constrained GPU" | arXiv:2505.05950 — [PDF](https://arxiv.org/pdf/2505.05950) | |
| MoEpic — split GPU-top/CPU-bottom per expert, adaptive cache config via fixed-point iteration | citato solo via aggregatore, URL diretto al paper da recuperare | non ancora verificato alla fonte |
| MoE-Lightning — pipelining CPU-GPU-I/O | citato solo via aggregatore, URL diretto al paper da recuperare | non ancora verificato alla fonte |
| ReMoE — "Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference" | arXiv:2605.27081 — [PDF](https://arxiv.org/pdf/2605.27081) | |
| AdapMoE — "Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference" | arXiv:2408.10284 — [PDF](https://arxiv.org/pdf/2408.10284) | asse gating adattivo, non caching puro — rilevante anche per Cluster B |
| Harvest — "Opportunistic Peer-to-Peer GPU Caching for LLM Inference" | arXiv:2602.00328 — [PDF](https://arxiv.org/pdf/2602.00328) | caching P2P generico, non expert-specifico — rilevanza da verificare |
| Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts | arXiv:2404.05019 — [PDF](https://arxiv.org/pdf/2404.05019) | parallelismo a tempo di training — probabilmente meno rilevante, da verificare |

### Cluster B — Quantizzazione/qualità per MoE (contesto per GCSG)

| Paper | Riferimento | Nota |
|---|---|---|
| PagedWeight — "Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization" | arXiv:2607.16184 — [PDF](https://arxiv.org/html/2607.16184v1) | bitwidth dinamico per pagina/expert, non shadow-verification |
| MoEQuant — "Enhancing Quantization for MoE LLMs via Expert-Balanced Sampling and Affinity Guidance" | arXiv:2505.03804 — [PDF](https://arxiv.org/pdf/2505.03804) | calibrazione a training-time |
| MoQE — "Mixture of Quantized Experts: Complementary Effect of Low-bit Quantization and Robustness" | arXiv:2310.02410 — [PDF](https://arxiv.org/pdf/2310.02410) | |
| "Training Quantization for MoE Models" (Expert-Aware Scale) | OpenReview — [PDF](https://openreview.net/pdf/7e4fd1adf7076c8b23539447ab7137b911d58643.pdf) | nessun arXiv ID catturato |

### Adiacenti ma su asse diverso (probabilmente non core related work — da confermare)

| Paper | Riferimento | Perché adiacente ma diverso |
|---|---|---|
| Zero-Knowledge Proof Based Verifiable Inference of Models | arXiv:2511.19902 — [PDF](https://arxiv.org/pdf/2511.19902) | verificabilità, ma crittografica — meccanismo del tutto diverso da GCSG |
| Expert-Sample for Test-Time Scaling in Fine-Grained MoE | arXiv:2602.02443 — [PDF](https://www.arxiv.org/pdf/2602.02443) | |
| Federated Fine-Tuning of Sparsely-Activated LLMs on Resource-Constrained Devices | arXiv:2508.19078 — [PDF](https://arxiv.org/pdf/2508.19078) | |

### Fonti secondarie (solo orientamento, non citabili come related work)

| Fonte | URL | Perché non citabile direttamente |
|---|---|---|
| "Expert Offloading for Scalable AI" | [emergentmind.com](https://www.emergentmind.com/topics/expert-offloading) | aggregatore/wiki di topic, non fonte primaria |
| "Do Quantized MoE Models Lose Their Experts?" | [Medium](https://medium.com/data-science-in-your-pocket/do-quantized-moe-models-lose-their-experts-1e0c3a3421c0) | blog post, non peer-reviewed |
| "NVMe KV Cache Offloading for LLM Inference" | [Spheron blog](https://www.spheron.network/blog/nvme-kv-cache-offloading-llm-inference/) | blog, e riguarda KV cache non pesi degli expert — tangenziale |

---

## Riferimenti interni

- Piano completo del Filone B: `osx-poc/reports/sprint5_berg_plan.md` §3
- Materiale da riusare per Design/Evaluation: `osx-poc/reports/gcsg_shadow_execution_report.md`, `osx-poc/reports/poc_final_report.md`
- Dev diary generale (non paper-specific): `osx-poc/LOGBOOK.md`
