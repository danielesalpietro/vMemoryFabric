# Sprint 5 (Berg) — Piano di sviluppo: PoC delivery + paper

**Data:** 2026-08-12
**Stato di partenza:** Sprint 4 (Tekniska) completo al 100%, tutti e 4 i target
non funzionali del PoC già raggiunti e misurati su hardware reale (vedi
README, sezione "Development roadmap"). Sprint 5 parte da ~5% → planning.
**Durata prevista:** ~2.5 settimane (2026-08-12 → target 2026-08-31, non una
deadline formale ma il target dichiarato — vedi §3/B0), più stretta delle 4
settimane originariamente stimate per "Weeks 13–16" del roadmap.
**Audience della delivery:** stakeholder/reviewer esterni — non solo chiusura
interna. Questo alza l'asticella su riproducibilità e pacchettizzazione
rispetto a uno sprint puramente interno.

---

## 0. Correzione fatta in apertura di sprint (igiene issue tracker)

Prima di pianificare qualunque lavoro nuovo, un controllo di coerenza tra
README/CHANGELOG e GitHub ha trovato un disallineamento reale: gli issue
[#1](https://github.com/danielesalpietro/vMemoryFabric/issues/1),
[#4](https://github.com/danielesalpietro/vMemoryFabric/issues/4) e
[#17](https://github.com/danielesalpietro/vMemoryFabric/issues/17) erano
documentati come "closed" in README/CHANGELOG (2026-08-12/13) ma risultavano
ancora **OPEN** su GitHub — verificato interrogando l'API, non assunto dal
testo. Chiusi ora, con commento che rimanda ai commit reali che li hanno
risolti (5cd88eb, 64f6bdc, b9871cf/6727a04/3e6c751/256a293). Questo genere di
disallineamento è esattamente il tipo di dettaglio che un reviewer esterno
nota per primo — da qui la regola operativa sotto (§2, A1): **nessun issue si
considera chiuso finché non lo è su GitHub**, non nel LOGBOOK.

---

## 1. Obiettivo e principio guida

Due filoni paralleli ma dipendenti:

- **Filone A — PoC delivery**: portare il PoC in uno stato consegnabile a
  terzi (stakeholder/reviewer), non solo "funzionante per chi ha scritto il
  codice".
- **Filone B — Paper**: trasformare i report tecnici già esistenti
  (`gcsg_shadow_execution_report.md`, `mmlu_final_report.md`, LOGBOOK.md) in
  un paper sottomettibile a una venue di sistemi reale.

Esplicitamente **fuori scope** di Sprint 5 (restano "future work" nel paper,
non lavoro da fare ora): Sprint 6/Stockholm (telemetria), M4 (RecursiveMAS),
PMEM (#7) e dual-GPU/AER (#8) — tutti bloccati da hardware non disponibile,
non da decisioni di design.

---

## 2. Filone A — PoC delivery

### A1. Triage dei restanti issue aperti

Rimangono aperti dopo la chiusura di §0: #2, #3, #5, #6, #7, #8, #12, #18.
Per ciascuno, decisione esplicita fix-vs-documenta, non lasciata implicita:

**Da sistemare prima della delivery** (piccoli, ma pesano sulla prima
impressione di un reviewer esterno che clona il repo e prova a farlo
girare):

| # | Issue | Perché ora |
|---|-------|-----------|
| [#12](https://github.com/danielesalpietro/vMemoryFabric/issues/12) | `make lint/test/bench` falliscono per WORKDIR/path relativi | È il primo comando che chiunque esterno lancia — oggi richiede di già sapere il workaround (`cd osx-poc && ...`), non documentato nel Quickstart in modo ovvio |
| [#3](https://github.com/danielesalpietro/vMemoryFabric/issues/3) | `bench_tier.py` p95/p99 sporcati da CUDA cold-start | I numeri di questo benchmark finiscono nel paper (§B2/B3) — vanno puliti prima di essere citati in una sede accademica |
| [#6](https://github.com/danielesalpietro/vMemoryFabric/issues/6) | manca `pyproject.toml`/`ruff.toml` | Costo basso, "good first issue"; un repo senza config di lint è un segnale negativo per un reviewer che guarda il codice |
| [#18](https://github.com/danielesalpietro/vMemoryFabric/issues/18) | nessun fingerprint ambiente, assunzioni silenti (`OMP_NUM_THREADS`, `shm_size`, GPU model) | Già mordente per davvero su RunPod (Sprint 4); un reviewer esterno che riproduce su hardware diverso dal nostro ci sbatte contro allo stesso modo — fix minimo: fail-loud invece di assumere |

**Da documentare come limitazione nota, non da fixare in questo sprint**
(rientra nel §7 "Limitations" del paper così com'è):

- [#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2) — RLock
  contention: già deciso 2026-08-12 di lasciarlo aperto deliberatamente,
  nessuna traffic reale lo esercita ancora. Nel paper diventa "known
  limitation, not yet triggered by production traffic", non un bug nascosto.
- [#5](https://github.com/danielesalpietro/vMemoryFabric/issues/5) — CUDA
  stream pipelining: future work esplicito.
- [#7](https://github.com/danielesalpietro/vMemoryFabric/issues/7),
  [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) — bloccati
  da hardware (PMEM, RTX 5080), già trattati come tali nel roadmap.

### A2. Report finale consolidato del PoC

Oggi i risultati sono sparsi tra README (sezione roadmap, molto lunga),
`gcsg_shadow_execution_report.md`, `mmlu_final_report.md`, e centinaia di KB
di `LOGBOOK.md`. Serve un unico **PoC Final Report** (`osx-poc/reports/`) che:

- raccoglie M1/M2/M3 con i numeri finali per i 4 target non funzionali
  (PT-PEP <3ms p99, PT-PEP hit rate >70%, GCSG degradation <2%, promotion
  latency entro 1.5× bandwidth teorica), tutti già misurati — qui si tratta
  di consolidare, non ri-misurare;
- diventa la base fattuale unica da cui il paper (Filone B) attinge, così i
  due documenti non divergono nel tempo;
- include esplicitamente la deviazione hardware Sprint 4 (RTX A5000 su
  RunPod invece dell'RTX 3090 di riferimento dichiarato nel README) in un
  unico punto ben visibile, non solo in nota sparsa nel LOGBOOK.

### A3. Riproducibilità (necessaria per audience esterna)

- Script "repro end-to-end" unico (`build → smoke → test → bench → MMLU`)
  con tempi attesi dichiarati, così un reviewer sa se è ancora "in corso" o
  "bloccato";
- fissare esplicitamente quale hardware/ambiente è quello "di riferimento"
  per i numeri citati nel paper (3090 Docker-on-Windows per gran parte del
  lavoro, A5000 RunPod bare-metal Linux per Sprint 4) — evitare che un
  reviewer confronti numeri tra ambienti diversi senza saperlo;
- verificare `make smoke && make test` verde da zero su un container pulito,
  non solo "ha funzionato l'ultima volta che l'ho girato".

### A4. Release

- Tag `v0.5.0` (o `v1.0.0-poc`, da decidere: dipende se si considera questo
  il primo rilascio "completo" del PoC agli occhi di terzi);
- aggiornare banner README ("Current release") e chiudere `CHANGELOG.MD`
  per Sprint 5;
- aggiornare la tabella roadmap (Sprint 5 da 🔲 pending a ✅/🟡 con link a
  questo piano e al PoC Final Report).

### A5. Gate di uscita del Filone A

Sprint 5 non si considera "delivered" finché non sono vere **tutte** queste
condizioni insieme, verificate non assunte:

1. issue tracker coerente con README (nessun altro caso come §0);
2. `make smoke && make test` verde da zero su ambiente pulito;
3. PoC Final Report pubblicato con i 4 target non funzionali confermati in
   un'unica lettura coerente;
4. release taggata.

---

## 3. Filone B — Paper (target: arXiv, taglio/formato in stile OSDI/EuroSys)

### B0. Decisione presa (2026-08-12)

- **Venue:** arXiv (self-archive) — non una submission a OSDI/EuroSys/MLSys
  in senso stretto. Questo cambia alcune cose concretamente rispetto a una
  submission a program committee:
  - nessun page limit imposto da un CFP, nessun processo di peer review,
    nessuna deadline hard di sottomissione a un sistema esterno (HotCRP e
    simili);
  - **niente anonimizzazione double-blind** — si scrive con nomi/affiliazioni
    reali fin da subito, a differenza di una submission OSDI/EuroSys vera
    (entrambe double-blind review);
  - resta comunque una pubblicazione pubblica e citabile (DOI arXiv), quindi
    va trattata con lo stesso rigore sulle claim quantitative di una vera
    submission (§B5) — "non c'è un reviewer che la respinge" non vuol dire
    "meno accurata".
- **Formato/taglio:** modellato su OSDI/EuroSys comunque, per scelta — cioè
  lunghezza e struttura da systems paper "vero" (~12–14 pagine + bibliografia,
  due colonne), non un tech report libero. Template scelto (2026-08-12):
  **ACM `sigconf`** (stile EuroSys) — le sue dipendenze `.sty` sono standard
  su TeX Live e compatibili con la build LaTeX di arXiv, nessun problema di
  pacchettizzazione atteso.
- **Deadline:** nessuna deadline formale esterna, ma target dichiarato
  **entro fine agosto 2026** (~19 giorni da oggi, 2026-08-12). Trattata come
  vincolo reale nel piano sotto, non come "quando capita".

Conseguenza pratica sul timeline (§4): niente ciclo di revisione da program
committee da aspettare, ma il tempo resta comunque stretto — la voce a
rischio più alto rimane la related-work survey (B3), unica parte non ancora
iniziata e non comprimibile sotto una soglia minima di credibilità.

### B1. Materiale già pronto da riusare (non da riscrivere da zero)

`gcsg_shadow_execution_report.md` ha già una struttura paper-shaped:
Abstract, Motivation, Experimental Setup, GCSG Design, due Root-Cause
analysis reali (issue #10/#16 e lo stall non-deadlock), MMLU Evaluation,
Limitations, **Related Work** (riferimenti upstream vLLM già citati),
Conclusions and Future Work. Questo è il nucleo delle sezioni Design/
Implementation/Evaluation/Limitations del paper — lavoro di adattamento ed
espansione, non stesura da zero.

`mmlu_final_report.md` fornisce la tabella per-subject e i numeri di
accuratezza già pronti per la sezione Evaluation.

### B2. Outline paper (bozza di lavoro)

1. Abstract
2. Introduction — motivazione: MoE serving non ha un livello di sistema
   dedicato al lifecycle degli expert (placement statico, nessuna
   topology-awareness, replica uniforme o assente)
3. Background & Motivation
4. Design — Expert Memory Hierarchy (EMH): M1 EAT, M2 Tier Manager, M3
   Scheduler (PT-PEP, GCSG, AER)
5. Implementation — vincoli reali dell'ambiente di sviluppo (Docker-on-
   Windows, WSL2, single-GPU) trattati come parte onesta della storia, non
   nascosti
6. Evaluation — MMLU quality degradation, promotion latency, PT-PEP
   latency/hit-rate, EAT lookup latency **incluso il risultato negativo del
   Bloom filter** (rimosso perché più lento di un dict semplice — un buon
   negative result per un systems paper, va raccontato non nascosto)
7. Discussion / Limitations — RLock contention non ancora esercitata da
   traffico reale (#2), PMEM/dual-GPU deferred, M4 out of scope
8. Related Work — sistemi di MoE serving esistenti (DeepSpeed-MoE,
   FasterMoE/Tutel, SwapAdvisor e simili)
9. Future Work — Sprint 6 (telemetria), M4, PMEM/dual-GPU quando disponibili
10. Conclusion

### B3. Materiale nuovo da produrre (non esiste ancora nel repo)

- **Related work reale con citazioni verificate** — quanto già citato nei
  report è solo la conferma upstream dei bug vLLM (#10/#16), non una survey
  della letteratura MoE-serving. Questa è ricerca bibliografica vera da
  fare, non un riordino di materiale esistente.
- **Figure pulite** — il diagramma di architettura oggi è ASCII nel README,
  serve una figura vettoriale vera; grafici latenza/accuratezza a partire
  dai dati grezzi già presenti (`gpu_telemetry_20260812.csv`, i vari
  `mmlu_*.jsonl`, `bench_tier_pod_*.log`) invece delle sole tabelle markdown.
- **Bibliografia** (`.bib`).

### B4. Infrastruttura di scrittura

Creare `osx-poc/paper/` con template ACM `sigconf` (deciso in B0) e file
`.bib`. Non esiste ancora nulla di questo tipo nel repo — verificato
(`find -iname "*.tex"` vuoto).

**Autore (deciso 2026-08-12):** Daniele Carmelo Salpietro — Principal
Solutions Architect (indipendente) — [salpietro.it](https://www.salpietro.it) —
daniele@salpietro.it. Riportato anche nel blocco Citation del README
(sostituisce il precedente placeholder "OSX Research Team").

Checklist specifica per arXiv (diversa da una submission a program
committee, va verificata esplicitamente prima di considerare B4 chiusa):

- sorgente LaTeX autosufficiente (niente riferimenti a file esterni non
  inclusi nel pacchetto caricato);
- `.bbl` compilato incluso nel pacchetto, non solo il `.bib` — arXiv non
  garantisce di rieseguire bibtex nell'ambiente esatto atteso;
- categoria primaria arXiv da scegliere (candidate: `cs.DC` — Distributed,
  Parallel, and Cluster Computing, dato il taglio memory-hierarchy/serving
  system; eventuale cross-list `cs.LG`);
- licenza di distribuzione (es. CC-BY 4.0) da dichiarare esplicitamente.

### B5. Revisione interna

Giro di review sul draft completo prima di considerare il paper pronto,
in particolare sulle claim quantitative (ogni numero deve poter essere
ricondotto a un file JSONL/CSV/log reale in `osx-poc/`, non a un valore
trascritto a mano).

---

## 4. Timeline rivista (target 2026-08-31, ~2.5 settimane)

Compressa rispetto alla stima originale a 4 settimane, perché B0 ha fissato
un target reale a fine agosto. Priorità: tutto ciò che alimenta il paper
(B3 in particolare) parte súbito, non in settimana 3 come nella prima
bozza; gli item del Filone A che non bloccano il testo del paper (A4/A5)
possono chiudere a ridosso o appena dopo il 31/8 senza mettere a rischio la
data del paper.

| Settimana | Date | Filone A | Filone B |
|---|---|---|---|
| 1 | 12–18 ago | A1 (fix #12/#3/#6/#18 — priorità a #3, i suoi numeri finiscono nel paper) | B1 (riuso materiale) + B2 (outline) + **avvio B3** (related-work survey — la voce a rischio più alto, parte per prima non per ultima) |
| 2 | 19–25 ago | A2 (PoC Final Report — sblocca i numeri definitivi per la sezione Evaluation) | B3 continua (figure/grafici dai dati grezzi) + assemblaggio draft completo |
| 3 | 26–31 ago | A3–A4–A5 (repro, release, gate — possono scivolare di qualche giorno oltre il 31/8 senza toccare la data del paper) | B5 (revisione interna) + checklist arXiv (B4) + submission |

Rischio principale resta lo stesso di prima ma ora è sul percorso critico:
la related-work survey (B3) è l'unica voce non ancora iniziata e non
comprimibile sotto una soglia minima senza abbassare la credibilità del
paper — per questo è spostata in settimana 1 invece che in settimana 3.

---

## 5. Prossimi passi immediati

1. ~~Decidere la venue target reale (B0)~~ — fatto 2026-08-12: arXiv, stile
   OSDI/EuroSys, target fine agosto 2026.
2. Partire in parallelo su A1 (i 4 fix piccoli, priorità #3) e sull'avvio di
   B3 (related-work survey) — sono i due percorsi a maggior rischio di
   ritardo se lasciati per ultimi.
3. Aprire issue GitHub per ciascun item di A1 non ancora tracciato in modo
   da avere lo stesso livello di tracciabilità degli sprint precedenti
   (pattern già in uso: issue → commit con `(closes #N)` → riga di
   CHANGELOG).
4. ~~Decidere autori/affiliazioni reali per il frontespizio del paper
   (B4)~~ — fatto 2026-08-12: Daniele Carmelo Salpietro, Principal Solutions
   Architect (indipendente), salpietro.it, daniele@salpietro.it. Restano da
   scegliere solo categoria arXiv e licenza (B4), non bloccanti per iniziare
   a scrivere.
