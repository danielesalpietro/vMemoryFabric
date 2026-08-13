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

## 2026-08-13 (continua) — Letture integrali FineMoE/AdapMoE/ReMoE/FloE/PagedWeight + MoEpic/MoE-Lightning via web, isolamento del meccanismo GCSG

**Contesto:** richiesta esplicita del project owner di approfondire — leggere per
intero FineMoE/MoEpic/MoE-Lightning e isolare con più precisione cosa
distingue GCSG, seguito diretto dei "Non fatto ancora" dell'entry precedente.

### Metodologia

Letti per intero (PDF diretti, non solo abstract/intro come nel primo giro):
FineMoE (EuroSys'26, 14 pagine), AdapMoE (ICCAD'24, corpo completo + inizio
references), ReMoE (ICML'26, 16 pagine incluse le appendici con le
dimostrazioni), FloE (ICML'25, 15 pagine incluse appendici teoriche),
PagedWeight (13 pagine, completo). MoEpic (arXiv:2509.08342) e MoE-Lightning
(arXiv:2411.11217, ASPLOS'25) **non recuperabili come PDF**: `arxiv.org` e i
mirror provati (`ar5iv.labs.arxiv.org`, `pschafhalter.com`,
`semanticscholar.org`, `themoonlight.io`) sono tutti bloccati dalla egress
policy di rete di questa sessione (403 dal proxy, non un errore transitorio —
verificato via `/__agentproxy/status`). Ricostruiti via più query di ricerca
web incrociate (abstract, dblp, ACM DOI per MoE-Lightning). Metadata di
MoE-Lightning cross-verificati anche indirettamente: sia PagedWeight che
ReMoE lo citano per intero nelle rispettive bibliografie (Cao et al., ASPLOS
2025, arXiv:2411.11217) — stesso paper, tre fonti indipendenti concordi.
MoEpic resta senza autori verificati (nessuna fonte secondaria li riportava);
non citabile ancora come bibliografia formale.

In parallelo, riletto `src/scheduler/gcsg.py` per intero (non solo il
docstring di modulo, letto nel primo giro) per capire esattamente cosa fa
`run_shadow()` a runtime.

### Risultato 1 — asse di differenziazione più preciso

Tutti i sei sistemi di offloading/caching letti (MoE-Infinity, Mixtral-
Offloading, FineMoE, MoEpic, MoE-Lightning, e in parte AdapMoE/PagedWeight)
rispondono alla stessa domanda strutturale: **dove/cosa posizionare nella
gerarchia di memoria o quanto comprimere**, guidati da statistiche di
traffico (hotness, similarità semantica/di traiettoria, sensitività
Hessiana offline). Nessuno di essi decide "questo token, adesso, in base a
quanto il router stesso è sicuro della propria decisione, tollera
un'esecuzione degradata?" — un trust-gate per-token basato su segnali
*endogeni al modello* (confidenza del gating + entropia della distribuzione
di routing) a runtime, non un problema di posizionamento in memoria.

Il parente più stretto resta AdapMoE (unico altro sistema che usa un segnale
per-token/per-layer — gap top1/top2 — per una decisione di qualità), ma con
tre differenze strutturali: (1) AdapMoE decide *riduzione di computo* (droppa
il secondo expert), GCSG decide *sostituzione di precisione* di un secondo
percorso eseguito in parallelo; (2) AdapMoE non ha un termine di entropia
né un contamination-rate auto-limitante; (3) AdapMoE è open-loop — soglia
derivata da un'espansione di Taylor/informazione di Fisher, mai verificata
a runtime contro un'esecuzione di riferimento. PagedWeight è il parente più
stretto sull'asse "quality-aware, runtime, per-request", ma la sua unità di
adattamento è la weight-page sotto pressione KV-cache, guidata da un
regressore offline+online che *stima* il danno — mai un'esecuzione shadow
reale che lo misuri empiricamente.

Nessuno dei sette paper esegue un secondo forward pass ridondante e lo
confronta con quello reale come primitiva di verifica runtime. Questo
pattern (draft economico eseguito in parallelo, poi verificato) è
strutturalmente più vicino a *speculative decoding* o a *shadow/dark-launch
deployment* (pratica SRE) che a qualunque lavoro di serving MoE trovato.

### Risultato 2 — scoperta critica: `run_shadow()` scarta il proprio output

Seguita la catena completa in `gcsg.py`: `_evaluate_gcsg_for_rows()` (riga
1657 circa) chiama `self.guard.run_shadow(...)`, il cui valore di ritorno
non è mai assegnato; dentro `run_shadow()`, la chiamata
`shadow_pool[shadow_expert_id](hidden_states, layer_id)` (riga 318 circa)
esegue il forward INT4 reale ma **anche qui il risultato non è catturato in
nessuna variabile**. In tutto il file non esiste un confronto tra l'output
dello shadow expert e l'output dell'expert reale — nessuna KL-divergence,
nessuna cosine similarity, nessuna soglia di scostamento.
`contamination_flag`/`contamination_rate` contano *quante volte* lo shadow
si è attivato, non se il suo output divergeva da quello reale.

**Conseguenza diretta per `gcsg_shadow_execution_report.md`:** il numero
72.11% vs 72.3% (baseline hook-only, −0.19pp) misura se eseguire il calcolo
shadow ridondante — il cui risultato viene scartato — danneggia comunque la
generazione reale (un test di **non-interferenza/sicurezza del sistema**),
non "se servire attraverso l'expert degradato preservi la qualità", che è la
premessa dichiarata nell'abstract del report stesso e nel docstring del
modulo. Con l'output scartato, la differenza di 1 domanda su 570 è
verosimilmente rumore (coerente con la caveat già presente in report §7
sull'assenza di un intervallo di confidenza formale), non segnale di un vero
costo di qualità.

**Non invalida la premessa di GCSG** — anzi rafforza l'argomento di
originalità, perché nessuno dei sette paper fa questo confronto neppure loro
— ma è un gap reale tra la claim del report/abstract e ciò che il codice
misura oggi. Due strade, da decidere col project owner prima di scrivere la
sezione Design/Evaluation del paper:
1. Implementare davvero il confronto shadow-vs-reale (catturare l'output
   reale dell'expert nello stesso hook `.gate`/forward della decoder layer,
   calcolare una divergenza — es. KL sui logit finali o cosine sull'hidden
   state — e riportare *quel* numero come "costo di qualità dello shadow").
2. Restringere esplicitamente la claim del paper a "sicurezza/non-
   interferenza dell'eseguire un canary computation nell'ombra", che è
   ciò che il PoC misura davvero oggi — riformulando abstract/§9 di
   conseguenza.

### CORREZIONE (2026-08-13, dal project owner) — il gap è più a monte: per i path 2/3 non esiste nemmeno un riferimento a precisione piena da confrontare

Il Risultato 2 sopra si fermava a "manca il codice che confronta shadow vs
reale". Il project owner ha verificato meglio: per i path effettivamente
usati sul checkpoint reale (`_AWQShadowExpert`, path 3; `_MarlinFusedShadowExpert`,
path 2 — gli unici due esercitati nel run MMLU del report), lo "shadow" non
è affatto una copia a precisione degradata rispetto a un originale più
preciso — **è un secondo forward attraverso lo stesso modulo già
quantizzato**, dichiarato esplicitamente nel docstring di `_AWQShadowExpert`:
"Lo 'shadow' qui è un secondo forward attraverso lo stesso expert già
caricato... non richiede una replica fisica del peso." Il checkpoint reale
(`casperhansen/mixtral-instruct-awq`) è AWQ/Marlin 4-bit end-to-end fin dal
caricamento — non esiste, in nessun punto del sistema in quella
configurazione, una versione a precisione piena con cui confrontare, quindi
anche implementando il confronto (Risultato 2, opzione 1) non ci sarebbe
alcuna divergenza di precisione da misurare: shadow e reale userebbero
letteralmente gli stessi pesi.

**Solo il path 1 (`_ShadowExpertINT4`) ha una vera divergenza di precisione
misurabile** — pesi fp16 grezzi (`w13_weight`) vs versione quantizzata a
INT4 simulato (`_quantize_int4`, int8 non packed) calcolata al volo — perché
lì il modello base non è pre-quantizzato e la "degradazione" è introdotta
davvero dallo shadow path stesso, non già presente ovunque.

**Conseguenza, verificata incrociando col report (`gcsg_shadow_execution_report.md`
§6-7-9) e col LOGBOOK generale:** "validato in qualità" (72.11/72.28/72.3%
MMLU) e "ha una divergenza di precisione reale" non si sono MAI verificati
insieme in nessun run di questo progetto fino ad oggi.
- I run MMLU (§6) sono tutti sul checkpoint AWQ reale, path 2/3 — zero
  divergenza di precisione per costruzione (paragrafo sopra).
- Path 1 è stato esercitato sotto offload reale solo come check di
  meccanica/correttezza (Sprint 4 sotto-obiettivo 6, A100, checkpoint
  Mixtral-8x7B-Instruct-v0.1 *non quantizzato* — un checkpoint diverso da
  quello usato per l'MMLU) — nessun numero di qualità esiste per path 1,
  né con né senza l'ipotetico confronto per-token.

Quindi il numero -0.19pp del report non misura "quanto costa in qualità
fidarsi di un expert degradato quando il router è sicuro" — non può, perché
nella configurazione in cui è stato misurato la nozione stessa di "degradato
vs reale" non si applica. Misura, nella lettura più caritatevole, se
raddoppiare il forward di un modulo AWQ/Marlin già caricato introduce
instabilità/rumore nella generazione — un risultato di robustezza
ingegneristica reale e non banale (i due root-cause di crash/stallo in
§4-5 restano scoperte genuine), ma ortogonale alla premessa "quality-safe
shadow execution under aggressive quantization" del titolo del report.

Opzioni concrete per procedere (sostituiscono le due sopra, ora più precise):
1. **Esperimento vero**: portare path 1 a scala MMLU sul checkpoint reale
   non quantizzato (o su un checkpoint fp16/bf16 caricato apposta), con un
   confronto per-token shadow-vs-reale implementato — l'unico modo di
   misurare davvero l'ipotesi del titolo. Costo: hardware più grande
   (visto il precedente A100 80GB con cpu_offload_gb=28 solo per farci
   stare il modello fp16), tempo di sviluppo per il confronto per-token,
   nuovo run MMLU completo — rischio concreto sulla scadenza di fine agosto.
2. **Ridisegnare i path 2/3** perché abbiano una divergenza di precisione
   reale anche dentro il regime già quantizzato — es. lo shadow chiama una
   versione ulteriormente degradata (2-bit, o un sottoinsieme di canali
   prunato) rispetto al 4-bit "reale" del checkpoint — mantiene il
   deployment realistico (AWQ/Marlin) ma richiede lavoro di ingegneria
   nuovo (un vero path di quantizzazione aggiuntiva dentro shadow, non
   presente oggi).
3. **Riformulare la claim** del paper su ciò che è stato davvero misurato:
   non "quality-safe under degradation" ma "safe to insert a redundant
   verification-shaped computation into a real serving pipeline without
   destabilizing it" — zero nuovi esperimenti, ma restringe sensibilmente
   cosa GCSG può rivendicare nel paper rispetto alla premessa originale del
   docstring di modulo.

Nessuna opzione scelta in questa entry — decisione del project owner,
probabilmente vincolata dalla finestra stretta fino a fine agosto 2026
(§B0).

### Bibliografia — metadata aggiornati (Cluster A)

| Paper | Riferimento verificato | Nota aggiornata |
|---|---|---|
| MoE-Lightning | Shiyi Cao, Shu Liu, Tyler Griggs, Peter Schafhalter, Xiaoxuan Liu, Ying Sheng, Joseph E. Gonzalez, Matei Zaharia, Ion Stoica. ASPLOS 2025 (Vol. 1), DOI 10.1145/3669940.3707267, arXiv:2411.11217 — metadata cross-verificati (dblp + ACM DOI + citazioni concordi in PagedWeight e ReMoE) | CGOPipe (pipelining CPU-GPU-I/O) + Hierarchical Roofline Model + paged weights; sistema di *scheduling per throughput batch*, nessun meccanismo di verifica qualità/shadow — asse ortogonale a GCSG |
| MoEpic | arXiv:2509.08342, "Accelerating Mixture-of-Expert Inference with Adaptive Expert Split Mechanism" — **autori non ancora verificati, non citabile as-is** | Split verticale top/bottom per expert (top in GPU, bottom in CPU) + configurazione cache via iterazione a punto fisso; nessuna quantizzazione, nessuna verifica di qualità — lossless per design, asse ortogonale a GCSG |

### Non fatto ancora

- Autori/venue di MoEpic non verificati alla fonte (arXiv irraggiungibile da
  questa sessione) — da recuperare in un ambiente con accesso, o chiedendo
  al project owner di incollare l'abstract/PDF direttamente.
- MoEQuant, MoQE, Harvest (Cluster B/adiacenti nella bibliografia sopra) non
  ancora letti per intero.
- Decisione da prendere col project owner sui due percorsi del Risultato 2
  prima di scrivere Design/Evaluation nel paper vero e proprio.

## 2026-08-13 (continua) — Verifica diretta del codice: fork di FineMoE/AdapMoE/ReMoE clonati e confrontati con gli originali

**Contesto:** il project owner ha forkato sul proprio profilo GitHub i tre
repo con codice confermato (FineMoE-EuroSys26, AdapMoE, ReMoE — vedi tabella
nell'entry precedente) per una verifica diretta invece di fidarsi solo del
testo dei paper. Cloni fatti in questa sessione (shallow, `--depth 1`),
codice letto direttamente.

### FineMoE e AdapMoE — codice reale, confermato corrispondente al paper

- **FineMoE-EuroSys26**: 45 file Python+C++, ~6.100 righe. `core/prefetch/`
  contiene uno scheduler C++ reale (`task_scheduler.cpp/h`,
  `archer_prefetch_handle.cpp/h`); `finemoe/memory/` contiene
  `expert_prefetcher.py`/`expert_tracer.py` — corrisponde all'architettura
  di prefetching fine-grained descritta nel paper. Costruito esplicitamente
  sopra MoE-Infinity (dichiarato nel README).
- **AdapMoE**: 22 file Python, ~5.450 righe. Controllato `src/dp.py` riga
  per riga contro §4.4.2 del paper: la funzione `f(index, size)` implementa
  esattamente i quattro termini di costo f1-f4 (Eq. 11-14), e
  `get_cache_size()` è la stessa ricorsione DP-knapsack di Eq. 19
  (`dp[i][j] = min(dp[i-1][j-k] + f(i,k))`). Match diretto codice-formula,
  non solo somiglianza di intento.

### ReMoE — il repo "ufficiale" non contiene il codice del metodo

`git ls-remote` su `danielesalpietro/ReMoE` (fork) e `BUAA-OSCAR/ReMoE`
(originale, quello citato nei risultati di ricerca come "[ICML'26] ReMoE")
restituisce **lo stesso identico commit** (`701420b...`), singolo branch
`main`, **un solo file: `README.md`**. Il contenuto non è il codice di
router fine-tuning descritto nel paper — è una model card che rimanda a un
checkpoint GGUF già fine-tuned e quantizzato
(`Zhu149248/DeepSeek-V2-Lite-Chat-ReMoE-GGUF` su HuggingFace, pensato per
Ollama). Zero training loop, zero implementazione del router fine-tuning,
zero harness di valutazione. Non è un fork "sbagliato" — è il repo
realmente linkato dal paper/dai risultati di ricerca, verificato identico
in entrambe le direzioni — ma il tag "[ICML'26] ReMoE" nei risultati di
ricerca descrive un repo che esiste ed è taggato correttamente, senza però
contenere codice sorgente riproducibile. Aggiornamento diretto alla tabella
data al project owner in chat: ReMoE passa da "repo ufficiale, alta
affidabilità" a "repo esiste, zero codice del metodo — non verificabile
staticamente, solo il checkpoint derivato è ispezionabile".

### Implicazione

Se serve verificare/confrontare numeri contro un baseline reale in fase di
Evaluation del paper, FineMoE e AdapMoE sono staticamente ispezionabili ed
eseguibili (repo completi); ReMoE no — l'unico modo di verificare qualcosa
di quel paper sarebbe scaricare il checkpoint GGUF e testarlo a runtime,
non leggerne l'implementazione.

---

## Riferimenti interni

- Piano completo del Filone B: `osx-poc/reports/sprint5_berg_plan.md` §3
- Materiale da riusare per Design/Evaluation: `osx-poc/reports/gcsg_shadow_execution_report.md`, `osx-poc/reports/poc_final_report.md`
- Dev diary generale (non paper-specific): `osx-poc/LOGBOOK.md`
