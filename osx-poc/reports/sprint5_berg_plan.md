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

**Addendum 2026-08-14 — la stessa regola non bastava.** Un secondo controllo,
partito da una domanda indipendente ("il progetto su GitHub è aggiornato?"),
ha trovato una versione più grave dello stesso problema: Sprint 3
(Oskarshamn) e Sprint 4 (Tekniska) erano entrambi documentati come completi
da giorni — issue chiuse su GitHub, come da regola sopra — ma il codice che
li implementa **non era mai arrivato su `develop`**. Viveva su
`Sprint-3-Oskarshamn` e `claude/sprint-5-berg-plan-e4dsc3`, entrambi mai
mersati; `develop` eseguiva ancora gli stub Sprint 0/1 di M2/M3
(`raise NotImplementedError`) mentre README/CHANGELOG descrivevano quel
lavoro come "✅ complete". La regola qui sopra verificava lo stato
dell'issue, non se il commit che la chiudeva fosse mai arrivato dove conta.
Corretto mergendo in sequenza `Sprint-3-Oskarshamn` → `claude/sprint-5-berg-
plan-e4dsc3` → `claude/log-folder-organization-uh5f1a` (PR #25) →
`claude/rlock-data-quality-9pcxzq` (PR #26) su `develop` (PR #9, #29, #25,
#26) — vedi CHANGELOG.MD [Berg] §Tracking per il dettaglio dei conflitti
risolti. **Regola estesa: nessun issue/sprint si considera chiuso finché il
codice che lo risolve non è in `develop`, non solo su un branch.**

---

## 1. Obiettivo e principio guida

Due filoni paralleli ma dipendenti:

- **Filone A — PoC delivery**: portare il PoC in uno stato consegnabile a
  terzi (stakeholder/reviewer), non solo "funzionante per chi ha scritto il
  codice".
- **Filone B — Paper**: trasformare i report tecnici già esistenti
  (`gcsg_shadow_execution_report.md`, `poc_final_report.md`, LOGBOOK.md) in
  un paper sottomettibile a una venue di sistemi reale.

Esplicitamente **fuori scope** di Sprint 5 (restano "future work" nel paper,
non lavoro da fare ora): Sprint 6/Stockholm (telemetria), M4 (RecursiveMAS),
PMEM (#7) e dual-GPU/AER (#8) — tutti bloccati da hardware non disponibile,
non da decisioni di design.

---

## 2. Filone A — PoC delivery

### A1. Triage dei restanti issue aperti

Rimangono aperti dopo la chiusura di §0: #3, #5, #6, #7, #8, #12, #18.
Per ciascuno, decisione esplicita fix-vs-documenta, non lasciata implicita.

**Aggiornamento 2026-08-13 — #2 non è più "documenta come limitazione
nota":** la decisione sotto (presa 2026-08-12, un giorno prima) è stata
superata da un fix reale, non solo da una nuova misura. Issue
[#23](https://github.com/danielesalpietro/vMemoryFabric/issues/23) ha
prodotto quattro strategie di locking selezionabili su
`ExpertAccessTable` (`locking_strategy="single"|"striped"|"lockfree_read"`,
Opzioni A/B/C, più un seqlock su `EATEntry.version` per l'Opzione D) —
`lockfree_read` è ora il **default** di produzione (nessun chiamante
esistente lo overridava). Misurato su `bench_eat.py` §contention (chiavi
disgiunte) e nuovo §churn (stesse chiavi, pattern realistico M2/M3): p99
reader passa da ~1000µs (single, comportamento pre-fix) a ~1-2µs
(lockfree_read); il torn-read che l'Opzione C accetta deliberatamente è
stato misurato, non solo teorizzato: 10/119038 letture (~0.008%) sotto
churn concorrente, zero su single/striped nello stesso run. Dettagli e
numeri completi nel branch `claude/rlock-data-quality-9pcxzq`.
**Rimisurato su hardware reale 2026-08-13** (Z8/`Z8-G4-RTX3090` via
`full-gpu-tests`, [run #150](https://github.com/danielesalpietro/vMemoryFabric/actions/runs/31726685030)):
a differenza dei numeri di contention pre-esistenti su #2 (più versioni
incoerenti tra loro a seconda dell'host, vedi nota su deviazione hardware
Sprint 4 sotto), qui sandbox CI e Z8 raccontano la stessa storia — p99
reader `lockfree_read` ~700-900x più veloce di `single` su entrambi gli
host, torn-read confermati solo su `lockfree_read` (rari, ~0.002-0.008%,
zero su single/striped). Nessuna riconciliazione necessaria. **#2 e #23
sono chiuse su GitHub** (state_reason: completed), coerente con la
regola §0 ("nessun issue si considera chiuso finché non lo è su GitHub").

**Da sistemare prima della delivery** (piccoli, ma pesano sulla prima
impressione di un reviewer esterno che clona il repo e prova a farlo
girare):

| # | Issue | Perché ora | Stato |
|---|-------|-----------|-------|
| [#12](https://github.com/danielesalpietro/vMemoryFabric/issues/12) | `make lint/test/bench` falliscono per WORKDIR/path relativi | È il primo comando che chiunque esterno lancia — oggi richiede di già sapere il workaround (`cd osx-poc && ...`), non documentato nel Quickstart in modo ovvio | 🔲 aperto |
| ~~[#3](https://github.com/danielesalpietro/vMemoryFabric/issues/3)~~ | `bench_tier.py` p95/p99 sporcati da CUDA cold-start | I numeri di questo benchmark finiscono nel paper (§B2/B3) — vanno puliti prima di essere citati in una sede accademica | ✅ **chiuso 2026-08-14** — warm-up shard dedicato in `bench_ddr4_to_vram()`, vedi CHANGELOG [Berg] |
| ~~[#6](https://github.com/danielesalpietro/vMemoryFabric/issues/6)~~ | manca `pyproject.toml`/`ruff.toml` | Costo basso, "good first issue"; un repo senza config di lint è un segnale negativo per un reviewer che guarda il codice | ✅ **chiuso** — PR #11 |
| [#18](https://github.com/danielesalpietro/vMemoryFabric/issues/18) | nessun fingerprint ambiente, assunzioni silenti (`OMP_NUM_THREADS`, `shm_size`, GPU model) | Già mordente per davvero su RunPod (Sprint 4); un reviewer esterno che riproduce su hardware diverso dal nostro ci sbatte contro allo stesso modo — fix minimo: fail-loud invece di assumere | 🔲 aperto |

**Da documentare come limitazione nota, non da fixare in questo sprint**
(rientra nel §7 "Limitations" del paper così com'è):

- ~~[#2](https://github.com/danielesalpietro/vMemoryFabric/issues/2) — RLock
  contention: già deciso 2026-08-12 di lasciarlo aperto deliberatamente,
  nessuna traffic reale lo esercita ancora.~~ **Superato 2026-08-13** — vedi
  addendum in cima a §2/A1: fix reale implementato (issue #23), non più solo
  documentazione di una limitazione. Nel paper diventa "measured and fixed
  during Sprint 5" con i numeri del nuovo benchmark §churn, non "known
  limitation".
- [#5](https://github.com/danielesalpietro/vMemoryFabric/issues/5) — CUDA
  stream pipelining: future work esplicito.
- [#7](https://github.com/danielesalpietro/vMemoryFabric/issues/7),
  [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) — bloccati
  da hardware (PMEM, RTX 5080), già trattati come tali nel roadmap.

### A2. Report finale consolidato del PoC — fatto (2026-08-13)

`osx-poc/mmlu_final_report.md` (M3/MMLU-only, fermo allo snapshot di
Sprint 3) è stato prima aggiornato con tutti i run di Sprint 4, poi
rinominato/esteso in **`osx-poc/reports/poc_final_report.md`** — ora copre
M1 (EAT), M2 (Tier Manager) e M3 (GCSG/MMLU) in un unico documento, con in
apertura (§0) i 4 target non funzionali valutati insieme invece che sparsi:

- **PT-PEP <3ms p99**: 🟡 nominalmente rispettato, ma l'unico numero reale
  mai registrato è un singolo fallimento isolato (3.71ms), mai
  ri-misurato con un valore pulito — segnalato esplicitamente come
  evidenza debole rispetto agli altri tre target, non presentato come
  "rispettato" allo stesso livello.
- **PT-PEP hit rate >70%**: ✅ 87.2%, con la riserva già nota (held-out
  same-distribution, non OOD).
- **GCSG degradation <2%**: ✅ margine ampio, corroborato da 5 run
  indipendenti (piattaforma/hardware/quantizzazione/data-path tutti
  diversi, tutti entro 0.7pp l'uno dall'altro).
- **Promotion latency entro 1.5× bandwidth teorica**: ✅ rispettato a P50
  su entrambi i round di benchmark disponibili (non solo uno, come nella
  versione precedente del report).

Aggiunta anche una scoperta non richiesta ma reale: i numeri di
contention dell'issue #2 (RLock) esistevano già in tre versioni
incoerenti tra loro (~1360× originale, ~61-91× ri-misurato in sandbox);
ricalcolando i rapporti dagli stessi log di regressione già usati per
M1/M2 emerge una **quarta cifra, mai calcolata prima** (~348×/~413×, dai
due round sul pod RunPod) — che non conferma nessuna delle altre due,
rinforzando (non risolvendo) la decisione già presa di lasciare l'issue
aperta.

A2 è quindi completo: un solo documento, `osx-poc/reports/poc_final_report.md`,
è ora la base fattuale unica da cui il paper (Filone B) attinge — la
deviazione hardware Sprint 4 (RTX A5000 su RunPod invece dell'RTX 3090 di
riferimento) è dichiarata esplicitamente per ogni singolo dataset citato,
non più in nota sparsa nel LOGBOOK.

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

### A4. Release — ✅ fatto (2026-08-26), con una scelta diversa da quella qui sotto

- ~~Tag `v0.5.0` (o `v1.0.0-poc`, da decidere: dipende se si considera questo
  il primo rilascio "completo" del PoC agli occhi di terzi)~~ — deciso
  diversamente: **`v0.6.0-beta.1`**, non `v0.5.0`/`v1.0.0-poc`. Motivo: la
  numerazione interna del CHANGELOG (Karlshamn `v0.1.0-dev` → Berg
  `v0.6.0-dev`) era già arrivata a v0.6.0-dev quando si è posta la domanda
  per davvero — taggare `v0.5.0` sarebbe stato un passo indietro rispetto
  allo stato di sviluppo reale, e `v1.0.0-poc` avrebbe dichiarato una
  maturità (produzione-ready) che questo stesso rilascio esplicitamente
  non ha (PMEM non ancora nel path di produzione — #57; tier CPU-offload
  #33 non production-ready; Filone B/paper indietro). `-beta.1` comunica
  correttamente lo stato senza sovra- né sotto-dichiarare. Pubblicata come
  GitHub Release marcata `prerelease: true`:
  <https://github.com/danielesalpietro/vMemoryFabric/releases/tag/v0.6.0-beta.1>;
- ✅ banner README ("Current release") aggiornato con link alla release e
  chiarimento che le versioni precedenti (Tekniska, Oskarshamn, ...) erano
  solo etichette CHANGELOG, mai tag git reali fino ad ora;
- ✅ tabella roadmap aggiornata (Sprint 5: Filone A 🟢 delivered con link
  alla release, Filone B 🟡 ancora indietro — non un singolo stato
  aggregato, per non nascondere che i due filoni sono a punti diversi).

### A5. Gate di uscita del Filone A

Sprint 5 non si considera "delivered" finché non sono vere **tutte** queste
condizioni insieme, verificate non assunte:

1. ✅ **issue tracker coerente con README** (nessun altro caso come §0) —
   verificato 2026-08-25/26 dalla prima Project Plan Review
   ([`ppr_20260825_sprint5_review.html`](ppr_20260825_sprint5_review.html)),
   che ha trovato e corretto in PR
   [#59](https://github.com/danielesalpietro/vMemoryFabric/pull/59) proprio
   un caso come §0 (PR #42/#48/#50/#52/#54 mersate il 24 agosto ma non
   ancora riflesse in README/CHANGELOG) — chiuso lo stesso giorno in cui è
   stato trovato, non lasciato aperto;
2. ✅ **`make smoke && make test` verde da zero su ambiente pulito** —
   verificato 2026-08-26: clone nuovo, build da zero, 13/13 smoke,
   224 passed / 3 skipped;
3. ✅ **PoC Final Report pubblicato con i 4 target non funzionali
   confermati in un'unica lettura coerente** — fatto dal 2026-08-13,
   vedi `reports/poc_final_report.md` §0;
4. ✅ **release taggata** — `v0.6.0-beta.1` pubblicata su GitHub Releases
   2026-08-26 alle 15:58 UTC (`prerelease: true`), tag sul commit
   `31f526a` (HEAD di `develop` a quel momento, cioè include già la
   chiusura delle condizioni 1–3 sopra) — verificato via API
   (`get_release_by_tag`/`get_tag`), non solo assunto dal "fatto" riportato
   in chat: <https://github.com/danielesalpietro/vMemoryFabric/releases/tag/v0.6.0-beta.1>.
   *(Nota per la cronaca, non più attuale: la sessione che aveva preparato
   il tag annotato su `6a6cdfe` non è riuscita a pusharlo — `git push
   origin v0.6.0-beta.1` rifiutato con HTTP 403 dal proprio proxy git,
   scoped al solo branch di lavoro — e ha lasciato questa condizione
   esplicitamente aperta invece di marcarla chiusa sulla fiducia. Il
   push/pubblicazione è stato completato subito dopo con accesso pieno al
   repo, sul tag e sulle note preparati in quella sessione.)*

**Gate A5 completo — 4/4 chiuso al 2026-08-26.** Sprint 5 (Berg) si
considera "delivered" per il Filone A secondo il criterio di questo §.
Il Filone B (paper arXiv) resta un percorso separato e più indietro — vedi
la PPR linkata sopra.

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

`poc_final_report.md` (§0, §3) fornisce ora sia i 4 target non funzionali
sia la tabella per-subject/i numeri di accuratezza, già pronti per la
sezione Evaluation.

### B2. Outline paper — v2, approvata dal project owner il 2026-09-02

**Cosa è cambiato e perché.** La v1 (2026-08-12, riportata in fondo a questa
sezione per storia) metteva l'Expert Memory Hierarchy al centro come
contributo. La verifica sul codice dei tre sistemi vicini — l'offloader
nativo di vLLM, MoE-Infinity a HEAD, FineMoE (EuroSys '26) — ha mostrato che
granularità di esperto, prefetch predittivo e hotness persistente sono già
lì (`reports/related_work_vllm_offloader.md`, `reports/related_work_finemoe.md`,
`logbook_paper.md` entry 2026-09-02 e 2026-09-02 bis). L'unico contributo
che nessuno dei tre può contestare è **GCSG**: la verifica shadow della
qualità sotto quantizzazione aggressiva, a runtime. Il paper si riorienta
di conseguenza: **GCSG è il contributo, EMH è il substrato che lo rende
possibile**, e si presenta come *diversa* dai vicini, mai come *in più*.
Proposta e motivazione per esteso: issue #68 (commento del 2026-09-02),
discussion #70 (D10). Approvata.

**Titolo di lavoro** (da confermare in B5): *Runtime Quality Guarding for
Quantized MoE Serving: Shadow Execution over a Heterogeneous Expert Memory
Hierarchy*.

1. **Abstract** — il problema è la qualità sotto quantizzazione aggressiva
   nel serving MoE, non la memoria; GCSG la guarda a runtime con shadow
   execution; EMH è ciò che rende la shadow execution sostenibile in VRAM;
   evaluation contro l'offloader di vLLM e MoE-Infinity su hardware
   identico; nessuna claim sul prefetch predittivo prima del numero.
2. **Introduction** — motivazione riscritta: i sistemi di expert offloading
   esistenti preservano la qualità per costruzione perché servono gli stessi
   pesi; quando i pesi sono quantizzati aggressivamente per entrare in
   memoria, **nessuno di loro sa se il gating sta degradando**. GCSG chiude
   quel buco. Contributi elencati in quest'ordine: (i) GCSG; (ii) EMH come
   substrato eterogeneo (DDR4/PMEM/NVMe come spazio unico) su cui la shadow
   execution gira senza esplodere la VRAM; (iii) un piano di misura
   pre-registrato e confrontabile (modello di costo in byte/token,
   normalizzazione per versione). Il lead time di PT-PEP entra fra i
   contributi **solo se H8 passa** (§6).
3. **Background & Motivation** — MoE routing e sparsità (k ≪ E); perché la
   quantizzazione INT4 degli esperti è il modo in cui si entra in 24 GB;
   cosa fanno oggi offloader vLLM, MoE-Infinity, FineMoE e cosa **non**
   guardano (la qualità). Qui entra la caratterizzazione dello skew del
   routing con l'entropia coarse/fine (M9, definizione di FineMoE, con
   attribuzione).
4. **Design** — **4.1 GCSG** (contributo principale): shadow pool, soglie
   θ_gate/θ_entropy/θ_contamination, contamination rate per richiesta, memory
   math del pool (da `gcsg_shadow_execution_report.md`, #19). **4.2 EMH come
   substrato**: EAT (M1), Tier Manager (M2), tier PMEM (#7/#52/#57), SEE
   policy — presentati per ciò che devono garantire a GCSG (residenza degli
   shadow expert, budget VRAM), non come novità. **4.3 Predizione e
   scheduling**: PT-PEP descritto come *un* predittore, con la sua proprietà
   architetturale distintiva (gira sul testo, prima della tokenizzazione,
   fuori dal critical path) e il suo limite dichiarato (nessuna correzione
   in-flight, a differenza di FineMoE). AER **non compare** nel design:
   citato in Future Work solo come replica *fra tier* su hardware
   asimmetrico, se D5 lo tiene in vita.
5. **Implementation** — worker vLLM V0 (0.10.1), `--enforce-eager`, finestra
   di pin e sue conseguenze (`vllm_torch27_compat_analysis.md`); vincoli
   dell'ambiente (Docker-on-Windows, WSL2, single-GPU, poi Z8 bare-metal)
   come parte onesta della storia; i due bug upstream (#10/#16) come
   root-cause reali.
6. **Evaluation** — segue `reports/test_plan_emh_vs_vllm_offloader.md`,
   pre-registrato (H1–H9): **6.1 L0b** bake-off dei predittori sulle stesse
   tracce (decide come si racconta PT-PEP — è il primo esperimento in
   ordine di esecuzione); **6.2 L0** primitive di trasferimento, calibra il
   modello di costo; **6.3 L1** serving su singola GPU contro A0 (nessun
   offload, due versioni), A1 `uva`, A2 `prefetch`, A3 EMH, A5 MoE-Infinity,
   a parità di VRAM, con W1 (MMLU), W2 (decode sintetico), W3 (sweep di
   skew); **6.4 qualità**: MMLU 5-shot con e senza GCSG, contamination rate
   — è il numero che gli altri sistemi non hanno; **6.5 negative results**
   raccontati: Bloom filter (rimosso), P3 se EMH "vince" a routing uniforme
   (misura sbagliata), H8 se PT-PEP non batte il kNN. **6.6 L2** solo se #8
   si sblocca in tempo (D7: KV cache al minimo globale sull'eterogeneo).
7. **Discussion / Limitations** — confondimento di versione V0/V1 (ridotto,
   non eliminato); shard 256 MB vs ≈ 90 MB per esperto INT4; qualità
   asimmetrica (gli offloader non toccano i pesi); leakage PT-PEP su MMLU;
   eager vs CUDA graph; RLock (#2); single-GPU; M4 out of scope.
8. **Related Work** — quattro paragrafi, tre già in bozza in
   `logbook_paper.md`: *Expert offloading systems* (MoE-Infinity, FineMoE —
   entry 2026-09-02 bis), *Production engines* (offloader vLLM ed EEP —
   entry 2026-09-02), *Quantization for MoE* (PagedWeight, AdapMoE, MoEQuant
   — Cluster B, da verificare alla fonte), *Classic MoE serving*
   (DeepSpeed-MoE, FasterMoE/Tutel, SwapAdvisor). FineMoE va citata dal PDF
   appena raggiungibile: oggi i suoi meccanismi sono verificati sul codice
   ufficiale, i numeri no.
9. **Future Work** — correzione in-flight della predizione (#21, da FineMoE);
   PMEM nel path reale (#57); dual-GPU (#8) e con esso D7 e AER fra tier;
   Sprint 6 telemetria (#47); M4.
10. **Conclusion**.

**Ordine di esecuzione che ne consegue** (sostituisce la sequenza B1→B5 della
timeline §4, ormai scaduta): L0b → L0 → immagine V1 → L1-W2 → L1-W1 → L1-W3 →
draft. Il draft dell'abstract e dell'introduzione si scrive **dopo L0b**, non
prima, perché il modo in cui si racconta PT-PEP dipende da H8.

<details>
<summary>B2 v1 (2026-08-12) — superata, conservata per storia</summary>

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

La motivazione della v1 ("MoE serving non ha un livello di sistema dedicato
al lifecycle degli expert") è stata smentita dalla verifica sul codice:
MoE-Infinity e FineMoE sono esattamente quel livello. Vedi
`reports/related_work_finemoe.md` §3.
</details>

### B3. Materiale nuovo da produrre (non esiste ancora nel repo)

- **Related work reale con citazioni verificate** — quanto già citato nei
  report è solo la conferma upstream dei bug vLLM (#10/#16), non una survey
  della letteratura MoE-serving. Questa è ricerca bibliografica vera da
  fare, non un riordino di materiale esistente. **Prima ricerca fatta
  2026-08-13**, tracciata in `osx-poc/logbook_paper.md` (piano a
  sotto-obiettivi + bibliografia di lavoro): la gerarchia di memoria M1/M2
  è in uno spazio affollato (5-6 sistemi simili trovati, inclusa **FineMoE,
  EuroSys 2026 — stessa venue/anno target**), mentre GCSG (shadow-verification
  della qualità sotto quantizzazione) non ha trovato un corrispettivo
  diretto. **Aggiornamento 2026-09-02:** FineMoE e MoE-Infinity verificati
  sul **codice ufficiale** (`reports/related_work_finemoe.md`) — il PDF è
  irraggiungibile dall'ambiente di lavoro, i meccanismi sono verificati, i
  numeri no; l'offloader nativo di vLLM e EEP verificati sul sorgente
  (`reports/related_work_vllm_offloader.md`, `reports/related_work_elastic_ep.md`).
  Esito: GCSG resta senza corrispettivo; EMH no. Da qui la v2 di B2.
  B3 non si chiude senza la lettura del PDF di FineMoE per i numeri e le
  baseline.
- **Figure pulite** — il diagramma di architettura oggi è ASCII nel README,
  serve una figura vettoriale vera; grafici latenza/accuratezza a partire
  dai dati grezzi già presenti (`logs/sprint4_tekniska/misc/gpu_telemetry_20260812.csv`, i vari
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
