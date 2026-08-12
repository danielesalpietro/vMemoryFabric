# Sprint 5 (Berg) — Piano di sviluppo: PoC delivery + paper

**Data:** 2026-08-12
**Stato di partenza:** Sprint 4 (Tekniska) completo al 100%, tutti e 4 i target
non funzionali del PoC già raggiunti e misurati su hardware reale (vedi
README, sezione "Development roadmap"). Sprint 5 parte da ~5% → planning.
**Durata prevista:** 4 settimane (Weeks 13–16 del roadmap originale).
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

## 3. Filone B — Paper (systems paper, venue reale — OSDI/EuroSys/MLSys 2027)

### B0. Decisione da prendere subito (bloccante per B4)

La citation nel README indica genericamente "OSDI 2027 / EuroSys 2027 /
MLSys 2027". Prima di investire in related work/figure serve **una** venue
target con la sua deadline reale, perché cambia formato (template LaTeX),
lunghezza (page limit), e taglio del paper (experience/systems paper vs.
full research contribution). Non blocco il piano su questo, ma è il primo
input mancante — lo marco come azione immediata in §5.

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

Creare `osx-poc/paper/` con template LaTeX della venue scelta in B0 e file
`.bib`. Non esiste ancora nulla di questo tipo nel repo — verificato
(`find -iname "*.tex"` vuoto).

### B5. Revisione interna

Giro di review sul draft completo prima di considerare il paper pronto,
in particolare sulle claim quantitative (ogni numero deve poter essere
ricondotto a un file JSONL/CSV/log reale in `osx-poc/`, non a un valore
trascritto a mano).

---

## 4. Timeline proposta

| Settimana | Filone A | Filone B |
|---|---|---|
| 1 | A1 (triage + fix #12/#3/#6/#18) | B0 (decisione venue) |
| 2 | A2 (PoC Final Report) | B1–B2 (riuso materiale + outline) |
| 3 | A3 (riproducibilità) | B3 (related work + figure + dati) |
| 4 | A4–A5 (release + gate) | B4–B5 (draft completo + revisione) |

Le due settimane finali hanno del buffer implicito: A3/A4 sono più leggeri
di A1/A2, e possono assorbire slittamenti di B3 (la related-work survey è
la voce a rischio maggiore di stima, essendo ricerca non ancora iniziata).

---

## 5. Prossimi passi immediati

1. Decidere la venue target reale (B0) — input necessario per non sprecare
   lavoro su un template sbagliato.
2. Partire da A1 (i 4 fix piccoli) in parallelo, indipendenti da B0.
3. Aprire issue GitHub per ciascun item di A1 non ancora tracciato in modo
   da avere lo stesso livello di tracciabilità degli sprint precedenti
   (pattern già in uso: issue → commit con `(closes #N)` → riga di
   CHANGELOG).
