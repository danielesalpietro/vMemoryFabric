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

---

## 2026-09-02 — Il vicino ingegneristico: l'offloader nativo di vLLM, e il paragrafo che il paper deve contenere

**Contesto:** review della sessione su Elastic Expert Parallelism (issue #68,
discussion #70). Cercando se vLLM avesse un offload degli esperti nativo — per
rispondere a "quanto di vMemoryFabric c'è già in vLLM?" — l'ho trovato:
`vllm/model_executor/offloader/`, presente da v0.25.0 (2026-07-08), assente in
v0.13.0. Due backend, `uva` (zero-copy da pinned CPU, selezione per nome di
parametro) e `prefetch` (gruppi di layer, prefetch H2D asincrono un passo
avanti, buffer statici, custom op nel grafo compilato). Verifica sul codice
in `reports/related_work_vllm_offloader.md`; piano di misura in
`reports/test_plan_emh_vs_vllm_offloader.md`.

**Cosa cambia per il Related Work.** L'entry 2026-08-13 ha identificato
FineMoE come vicino *accademico*. Questo è il vicino *ingegneristico*, ed è
dentro l'engine da cui dipendiamo: un reviewer che conosce vLLM lo nominerà
per primo. "Nessuno fa offload di esperti in vLLM" non è più vero. Il claim
difendibile è più stretto: *nessuno lo fa in modo routing-aware, a
granularità di esperto, su più di due tier, con eviction dinamica* — quattro
assi, ognuno verificato con una riga di codice citata nel report. Il fatto
decisivo è di layout, non di policy: nel `FusedMoE` di vLLM gli esperti di un
layer sono un solo tensore `w13_weight (num_experts, …)`, e l'offloader
lavora per parametro. Non può muovere un esperto singolo senza un `FusedMoE`
diverso — che è ciò che l'EAT `(expert_id, shard_idx)` presuppone.

**Cosa cambia per la Evaluation.** Il confronto giusto è contro `prefetch`,
non contro `cpu_offload_gb` nudo: è ciò che un utente vLLM competente userebbe
oggi per far entrare Mixtral in 24 GB. Il piano dei test è pre-registrato
(ipotesi H1–H7 con soglie) e costruito per essere confrontabile nonostante il
confondimento di versione (V0 per GCSG, V1 per l'offloader): modello di costo
comune, normalizzazione per versione, livello L0 senza vLLM.

**Onestà da mettere nel testo, non da scoprire in review.** Gli offloader
servono gli stessi pesi ($\Delta Q = 0$); EMH con shadow INT4 no. Il
confronto è a due obiettivi. E il vantaggio di EMH è una funzione dello skew
del routing: a routing uniforme il modello predice *nessun guadagno* oltre la
capacità (H5). Se il paper non lo dice, lo dirà il reviewer.

### Bozza — §8 Related Work, paragrafo "Production engines" (inglese, per il draft)

> **Production engines.** vLLM recently added native weight offloading with
> two backends: a UVA zero-copy path that reads parameters from pinned host
> memory on demand, and a prefetch path that groups decoder layers, offloads
> a fixed subset chosen by a static arithmetic pattern, and hides host-to-
> device copies behind the compute of the preceding layers using static GPU
> buffers and compiler-visible custom ops. Both operate at parameter
> granularity; since vLLM's fused MoE layout stores all experts of a layer in
> a single tensor, neither can place or move an individual expert, and
> neither consults the router. vLLM's Elastic Expert Parallelism addresses an
> orthogonal capacity problem — horizontal scaling of the expert-parallel
> group under traffic variation — by adding data-parallel ranks and filling
> them with replicas of hot experts; it assumes homogeneous ranks and
> renegotiates the KV-cache budget to the global minimum. EMH differs from
> the offloader on three axes: residency decisions are routing-aware and taken
> at expert-shard granularity, the hierarchy spans more than two tiers, and
> eviction is dynamic. We do not claim general superiority: the offloader
> preserves model quality by construction, integrates with CUDA graphs, and
> — as our cost model predicts and §6 confirms — matches EMH under uniform
> routing. EMH's advantage is a function of routing skew, and we report it as
> such.

### Bozza — §7 Limitations, frase da aggiungere

> Our comparison against vLLM's native offloader spans two vLLM releases,
> because the offloader targets the V1 engine while our worker integrates
> with V0. We normalize every metric against a no-offload baseline on the same
> release, calibrate the cost model on engine-free microbenchmarks, and report
> the cross-release gap explicitly; the confound is reduced, not removed.

**Aggiornamento alla bibliografia di lavoro (Cluster A, nuova riga):**

| Paper / sistema | Riferimento | Nota |
|---|---|---|
| vLLM weight offloader (`uva`, `prefetch`) | `vllm-project/vllm`, `vllm/model_executor/offloader/`, ≥ v0.25.0 | codice, non paper — si cita come software con versione e commit; verificato 2026-09-02 |
| vLLM Elastic Expert Parallelism | RFC vllm#20323; `vllm/distributed/elastic_ep/` | asse ortogonale, non concorrente; `reports/related_work_elastic_ep.md` |

**Sotto-obiettivo 2 (related-work survey): resta 🟡.** FineMoE va ancora
letto per intero; questa entry aggiunge il vicino ingegneristico, non
sostituisce quella lettura.
