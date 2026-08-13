# Development Roadmap — vMemoryFabric / OSX

**Data:** 2026-08-13
**Scopo:** vista *forward-looking* per rilascio — cosa contiene ciascuna
release pianificata, con impatto e rischio espliciti per ogni voce. Non
sostituisce due documenti già esistenti, li completa:

- `README.md` §"Development roadmap" — narrativa **retrospettiva**,
  sprint per sprint, con lo storico di cosa è stato fatto e come.
- `osx-poc/reports/sprint5_berg_plan.md` — piano **dettagliato dello
  sprint corrente** (Berg), task-level.

Questo documento risponde a una domanda diversa da entrambi: **cosa va in
quale release**, per tenere la rotta rilascio dopo rilascio e valutare
impatto/rischio prima di iniziare a lavorarci, non a posteriori. Le 24
issue GitHub aperte/chiuse ad oggi sono la fonte di verità sottostante —
questo documento le raggruppa e le pesa, non le sostituisce.

**Non tutte le assegnazioni qui sotto sono decise.** Dove non lo sono, è
scritto esplicitamente "proposta, da validare" — coerente con come questo
progetto tratta le decisioni (vedi B0 in `sprint5_berg_plan.md`: nessuna
scelta di piano è "presa" finché non è scritta come tale).

---

## 1. Release corrente — v0.5.0-dev "Berg" (Sprint 5, in corso)

**Target:** fine agosto 2026 (~2.5 settimane da oggi). **Audience:**
esterna (stakeholder/reviewer) — prima immagine pubblica del progetto oltre
al codice nudo.

**Contenuto pianificato** (dettaglio completo in `sprint5_berg_plan.md`,
qui solo il riepilogo per-release):

| # | Issue | Contenuto | Impatto | Rischio |
|---|---|---|---|---|
| [#3](https://github.com/danielesalpietro/vMemoryFabric/issues/3) | `bench_tier.py` p95/p99 sporcati da CUDA cold-start | Fix — warm-up prima della misura | **Alto**: i suoi numeri finiscono nel paper (Evaluation) | Basso — fix meccanico, isolato |
| [#6](https://github.com/danielesalpietro/vMemoryFabric/issues/6) | manca `pyproject.toml`/`ruff.toml` | Fix — config lint | Medio — prima impressione per un reviewer che apre il codice | Basso |
| [#12](https://github.com/danielesalpietro/vMemoryFabric/issues/12) | `make lint/test/bench` falliscono su path relativi | Fix — primo comando che un esterno lancia | **Alto**: blocca chiunque provi il Quickstart alla lettera | Basso |
| [#18](https://github.com/danielesalpietro/vMemoryFabric/issues/18) | nessun fingerprint ambiente, assunzioni silenti | Fix — fail-loud invece di assumere | Medio — già mordente su RunPod | Basso-medio (tocca più script) |
| #2, #5, #7, #8 | RLock contention, CUDA stream, PMEM, dual-GPU | **Non fixate** — documentate come limitazione nota nel paper (§7 Discussion) | — | Rischio di reputazione se presentate come bug nascosti invece che limitazioni dichiarate — già mitigato scrivendole esplicitamente |
| Filone B (paper) | Related work (B3), draft, LaTeX, arXiv checklist | Vedi sotto | **Alto** — è il deliverable con più visibilità esterna | **Alto** — vedi rischio nuovo sotto |

### Rischio nuovo, non presente in `sprint5_berg_plan.md` quando è stato scritto (2026-08-12)

Emerso il 2026-08-13 durante la related-work survey: la premessa centrale
di GCSG ("quality-safe shadow execution under aggressive quantization")
non è supportata dai dati per i path effettivamente usati nel run MMLU
pubblicato — per i checkpoint AWQ/Marlin non esiste alcun riferimento a
precisione piena da cui lo shadow possa divergere (`logbook_paper.md`,
entry "CORREZIONE" del 2026-08-13; issue [#19](https://github.com/danielesalpietro/vMemoryFabric/issues/19),
opzione alternativa in [#22](https://github.com/danielesalpietro/vMemoryFabric/issues/22)).

**Questo è un rischio diretto sul percorso critico di Berg**, non un item
separabile: la sezione Design/Evaluation del paper (B2, punto 4/6) non può
essere scritta onestamente finché non è deciso quale delle opzioni in #19/#22
si persegue (riformulare la claim, misura "lite", o il pieno esperimento su
path 1). Questa decisione andrebbe presa **in settimana 1** di Berg, insieme
a B3 (related-work), non rimandata a settimana 3 — competere per lo stesso
tempo limitato con B3 è il rischio reale, non menzionato nel piano originale.

**Non ancora deciso — richiede una scelta del project owner prima di
proseguire con B2/B4.**

### Gate di uscita (invariato da `sprint5_berg_plan.md` §2.A5)

1. issue tracker coerente con README/CHANGELOG;
2. `make smoke && make test` verde da zero;
3. PoC Final Report pubblicato con i 4 target confermati;
4. release taggata (`v0.5.0` o `v1.0.0-poc`, ancora da decidere).

---

## 2. Prossima release — v0.6.0-dev "Stockholm" (Sprint 6, telemetria)

**Non ancora iniziato.** Scope già descritto in README ma **senza issue
GitHub numerate** — unica release pianificata che non ha ancora tracciabilità
allo stesso livello delle altre (gap da colmare aprendo issue reali prima di
iniziare, non lavorarci "a memoria" dal README).

| Fase | Contenuto | Impatto | Rischio | Blocco |
|---|---|---|---|---|
| 1 — telemetria single-worker | Adapter Prometheus su `.stats()` già esistenti (GCSGGuard, AERManager, PTPEPClassifier, TierManager, EAT) — zero nuova strumentazione | Medio — osservabilità, non tocca validità scientifica del PoC | Basso — dati già raccolti, solo esposizione | Nessuno — può iniziare subito dopo Berg |
| 2 — aggregazione multi-worker | Prometheus multi-target vs. pushgateway (decisione di design non ancora presa) | Basso nel breve, alto se/quando serve scalare | Medio — richiede una scelta architetturale non banale | **#8** (dual-GPU/AER, arrivo RTX 5080) |

**Azione consigliata prima di iniziare Stockholm:** aprire issue GitHub
per la Fase 1 (adapter Prometheus, wiring `make metrics-up`/`metrics-down`)
— oggi esiste solo come paragrafo README, non come lavoro tracciabile.

---

## 3. Backlog "Systems Hardening" — proposta, non ancora assegnato a una release

Le quattro issue aperte oggi (2026-08-13) dalla lettura diretta di FineMoE/
AdapMoE/PagedWeight/FloE/MoE-Lightning. Nessuna bloccata da hardware, nessuna
nello scope di Berg — per coerenza con la disposizione già presa in
`sprint5_berg_plan.md` §A1 sui loro issue "genitore" (#2 e #5, entrambi
esplicitamente rimandati a "future work"), la stessa logica si applica ai
loro derivati.

| # | Issue | Impatto se implementata | Rischio | Dipendenze |
|---|---|---|---|---|
| [#24](https://github.com/danielesalpietro/vMemoryFabric/issues/24) | Misura diretta transfer-bound vs compute-bound | **Alto** — precede qualunque decisione su #20/worker_cpu, evita di costruire la soluzione sbagliata | Basso tecnico, costo di provisioning (budget, non infrastruttura) | **#3** (deve chiudere prima — stesso artefatto di misura, e #3 è già dentro Berg: si sblocca prima del previsto) |
| [#20](https://github.com/danielesalpietro/vMemoryFabric/issues/20) | Compact async transfer (FloE) + CGOPipe/HRM (MoE-Lightning) per #5 | Alto se #24 conferma transfer-bound; nullo se conferma compute-bound | Medio — layout trick di FloE tocca granularità EAT/slab, non solo `GPUTransfer` | #24 (esito ne determina la priorità reale) |
| [#21](https://github.com/danielesalpietro/vMemoryFabric/issues/21) | Collegare σ (SEEPolicy) a PT-PEP | Medio — oggi γ=0.4 è sprecato su un contributo nullo | Medio — rischio di introdurre un segnale rumoroso se l'oggetto giusto (prompt-level vs per-shard) non è quello corretto | Nessuna tecnica — solo verifica empirica prima di collegare |
| [#23](https://github.com/danielesalpietro/vMemoryFabric/issues/23) | EAT RLock — opzioni A (RLock→Lock) → E (estensione nativa) | Basso oggi (traffico reale è single-threaded, come già deciso su #2 il 2026-08-12), alto se/quando M3 genera concorrenza reale | Basso per l'opzione A (gratis, un pomeriggio), crescente per B→E | Stessa condizione di #2: **non prioritario finché non esiste traffico concorrente reale** |

**Osservazione sul pattern**: #5 è deferred dal Sprint 0 (Karlshamn) senza
mai essere assegnato a una release con una data — cinque sprint dopo, è
ancora "future work". Il rischio concreto per questo backlog è lo stesso:
se non viene assegnato esplicitamente a una release (anche solo come
"candidato per lo sprint dopo Stockholm"), rischia la stessa sorte.
**Decisione richiesta**: assegnarlo a un nome di sprint reale (Svezia-themed,
da scegliere) o lasciarlo esplicitamente come backlog senza data, ma
dichiarato come scelta, non per omissione.

---

## 4. Bloccate da hardware — nessuna release fissa, opportunistiche

| # | Issue | Blocco | Quando rivalutare |
|---|---|---|---|
| [#7](https://github.com/danielesalpietro/vMemoryFabric/issues/7) | PMEM (EMH-2) | Bare-metal Z8 G4 non ancora disponibile | All'arrivo dell'hardware |
| [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) | Dual-GPU / AER | Arrivo RTX 5080 | All'arrivo dell'hardware — sblocca anche Stockholm Fase 2 |

Non richiedono pianificazione di sprint — solo un trigger (arrivo hardware)
già chiaro e non ambiguo.

---

## 5. Vista compatta — tutte le issue aperte, per release

```
v0.5.0-dev "Berg"     (in corso, target 31/8)   #3 #6 #12 #18 (fix)
                                                  #2 #5 #7 #8 (documentate, non fix)
                                                  #19 #22 (decisione paper — RISCHIO NUOVO)

v0.6.0-dev "Stockholm" (non iniziato)             Fase 1: nessuna issue ancora aperta (gap)
                                                  Fase 2: bloccata da #8

Backlog "Systems       (proposta, non assegnato)  #24 → #20 → #21, #23
 Hardening"

Bloccate da HW          (opportunistiche)          #7, #8
```

---

## 6. Decisioni che servono dal project owner per rendere questo piano definitivo

Elencate qui perché sono le uniche cose che bloccano questo documento dal
diventare "il" piano invece di una proposta:

1. **#19/#22** (rischio nuovo su Berg): quale opzione perseguire, e quando
   — vedi §1. Blocca B2/B4 del paper.
2. **Naming/collocazione del backlog "Systems Hardening"** (§3): sprint
   dedicato con nome reale, o backlog esplicitamente senza data?
3. **v0.5.0 vs v1.0.0-poc** per il tag di release Berg (già una domanda
   aperta in `sprint5_berg_plan.md` §A4, riportata qui per visibilità).
4. **Issue GitHub per Stockholm Fase 1** — aprirle ora (anche se Stockholm
   non è iniziato) per avere lo stesso livello di tracciabilità del resto,
   o esplicitamente rimandarle a quando Berg chiude?

---

## Riferimenti

- `README.md` §"Development roadmap" — storico sprint-per-sprint
- `osx-poc/reports/sprint5_berg_plan.md` — piano dettagliato Sprint 5
- `osx-poc/logbook_paper.md` — dev diary del filone paper, incluso il
  rischio #19/#22
- Issue GitHub #1-#24 (repo `danielesalpietro/vMemoryFabric`)
