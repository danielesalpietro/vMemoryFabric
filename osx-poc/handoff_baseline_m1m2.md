# Handoff — congelare la baseline M1/M2 pre-upgrade (issue #8)

**Per:** la sessione Claude Code con accesso diretto alla Z8 G4 bare-metal
(quella che ha già fatto il recupero dati, la verifica torch/sm_120 e
l'analisi di compatibilità vLLM del 31/08).
**Da:** sessione remota che ha coordinato Sprint 5 (Berg) — nessun accesso di
rete alla Z8, lavora solo via GitHub.
**Data:** 2026-08-31.

---

## 0. Prima di tutto: onboarding — non saltarlo

Stesso principio delle volte precedenti: prima di eseguire qualunque comando,
leggi in quest'ordine — la fonte di verità è sempre il documento linkato, non
il riassunto qui sotto.

1. **`README.md`** (root del repo) — stato attuale, release `v0.6.0-beta.1`,
   roadmap, known limitations (voce dual-GPU/#8 già aggiornata).
2. **`osx-poc/CHANGELOG.MD`**, sezione `[Berg]` in cima.
3. **`osx-poc/LOGBOOK.md`**, entry più recenti in cima — in particolare
   25-31 agosto (PPR, chiusura Gate A5, dual-GPU discovery, analisi vLLM).
4. **[Issue #8](https://github.com/danielesalpietro/vMemoryFabric/issues/8)
   su GitHub, per intero** — contiene tutta la storia: il precedente RTX PRO
   6000, la verifica reale del 30/08 (nvidia-smi, `get_arch_list()`, test
   CUDA fallito su `cuda:1`), e l'analisi di compatibilità del 31/08
   (`vllm==0.9.0` + `torch==2.7.0+cu128` minimo, finestra utilizzabile
   `0.9.0`–`0.10.1`, chiusa da `mixtral_quant.py`/`vllm/worker/worker.py`
   rimossi, non da torch). **Leggila lì**, non fidarti del riassunto sotto.
5. **`osx-poc/reports/vllm_torch27_compat_analysis.md`** — il documento
   completo dell'analisi del 31/08 (PR #66, già su `develop`), con la
   sezione finale "Cosa NON è stato verificato" che elenca esplicitamente
   la baseline M1/M2 come prerequisito ancora aperto — è il compito di
   questa sessione.
6. **`osx-poc/handoff_rtx5060ti16gb.md`** — l'handoff precedente su questo
   stesso filone, per capire le convenzioni già stabilite (dati grezzi mai
   alterati, branch da `develop` aggiornato, nessuna decisione unilaterale).

---

## 1. Perché questo task, e cosa NON è

L'analisi di compatibilità vLLM/torch (31/08, PR #66) ha concluso che un
upgrade a `torch==2.7.0+cu128` + `vllm==0.9.0` è tecnicamente fattibile, ma
elenca esplicitamente cosa manca prima di un vero upgrade — tra cui, dalla
sezione "Cosa NON è stato verificato":

> Baseline M1/M2 non ancora congelata sul toolchain attuale (prerequisito
> esplicito prima di qualsiasi upgrade reale, non di questa sola analisi).

Questo era già scritto come condizione nell'issue #8 fin dall'aggiornamento
del 16/08: *"congelare un set di metriche baseline M1/M2 sull'attuale
toolchain prima di qualsiasi upgrade"* — per rendere un eventuale delta
post-upgrade **misurabile**, non solo assunto.

**Questo task NON è l'upgrade.** È: eseguire e archiviare la suite di
benchmark/test/eval **sul toolchain attuale, invariato**
(`torch==2.5.1+cu124`, `vllm==0.6.6.post1`, RTX 3090 attiva,
`NVIDIA_VISIBLE_DEVICES=0` invariato) — un checkpoint "prima", non un "dopo".

---

## 2. Il compito

Tutta la strumentazione esiste già nel Makefile — nessun codice nuovo da
scrivere. Eseguire, nell'ordine, **senza toccare nessun pin di versione**:

1. `make smoke` (`scripts/smoke_test.py`) — sanity hardware/env
2. `make test` (pytest completo con `--cov`) — regressione funzionale
3. `make bench-eat` (`benchmarks/bench_eat.py`) — M1: latenza/throughput EAT
   (hit/miss, baseline dict, contention, contention_churn,
   contention_by_strategy)
4. `make bench-tier` (`benchmarks/bench_tier.py`) — M2: latenza promozione
   (`nvme_to_ddr4`, `ddr4_to_vram`, `promote_live_tensor` pin=False/True)
5. **MMLU 5-shot** via `scripts/run_mmlu_in_slices.sh` →
   `scripts/eval_mmlu_gcsg.py` — stesso protocollo già usato finora (570
   domande, 57 subject, 5-shot standard Hendrycks et al.) — è il numero di
   *qualità* (GCSG quality degradation) più esposto a un eventuale bump
   torch/vllm, quindi il più importante da avere come riferimento "prima".
   Timeout lunghi attesi (vedi commenti nello script, 850s+ per fetta è
   normale sotto WSL2/cpu_offload_gb — se questa sessione gira su bare-metal
   Linux reale invece di WSL2, annotalo: potrebbe essere più veloce, è un
   dato in sé).

**Dichiara esplicitamente l'ambiente** prima di girare i comandi: commit
hash (`git rev-parse HEAD`), se l'immagine `osx-poc:dev` esiste già o va
ricostruita (come per la verifica del 30/08), versione driver/`nvidia-smi`.
Se l'immagine va ricostruita, dichiara l'equivalenza esplicitamente come
già fatto per il check sm_120, non darla per scontata.

---

## 3. Dove salvare — dati grezzi, non riassunti a mente

Segui la convenzione già in uso in `logs/new_z8_bare_metal/README.md`:
nuova cartella `logs/new_z8_bare_metal/baseline_pretorch27_20260831/` (o la
data reale in cui gira) con:

- output grezzo di `make smoke` e `make test`
- JSON/log grezzi di `bench_eat.py` e `bench_tier.py`
- il `.jsonl` prodotto da `run_mmlu_in_slices.sh`

Se un file ha lo stesso problema di banner-container-davanti-al-JSON già
visto altrove: **originale intatto byte-per-byte**, copia pulita accanto
come `*.clean.json` — mai sovrascrivere il grezzo.

Poi un documento breve, `osx-poc/reports/baseline_pretorch27_20260831.md`
(nome/data da adattare a quando gira davvero), con le cifre di riferimento
(p50/p95/p99 EAT, latenze promote M1/M2, accuracy MMLU aggregata e per
sezione) e link ai raw — è il termine di paragone che, dopo un eventuale
futuro upgrade, permetterà di scrivere "delta reale: X%" invece di "sembra
uguale".

---

## 4. Regole del gioco (stesse di sempre, non opzionali)

- **Verifica prima di scrivere.** Ogni cifra ha una fonte tracciabile (log,
  JSON, output di comando) nella cartella `logs/`.
- **Non sovrascrivere dati/misure discordanti.** Se un numero non coincide
  con uno già pubblicato altrove (es. i numeri MMLU di Sprint 3/4 nel
  README), riportali entrambi e segnala la discrepanza — non scegliere
  quello che "sembra giusto".
- **Nessun pin toccato.** Questo task gira sul toolchain attuale invariato —
  `requirements-vllm.txt` non si tocca in questa sessione.
- **Dati grezzi mai alterati** — vedi §3 sopra.
- **Workflow git**: branch nuovo da `develop` aggiornato (non riusare branch
  vecchi), es. `claude/baseline-m1m2-pretorch27`, push, lascia
  aprire/mergiare la PR alla sessione con accesso a GitHub — stesso pattern
  di `claude/z8-raw-data-recovery(-2)` e `claude/z8-toolchain-evidence`.
- **Nessuna decisione unilaterale su scope aggiuntivo.** Se emerge qualcosa
  non richiesto qui (es. vuoi anche rigirare `bench_cpu_kernel.py` o
  `bench_pmem_tier.py`), segnalalo e aspetta conferma invece di includerlo
  o ometterlo di tua iniziativa — non è nella lista del §2 sopra
  intenzionalmente, per tenere questo giro comparabile 1:1 con la sezione
  "Cosa NON è stato verificato" dell'analisi del 31/08, non più ampio.

---

## 5. Dove riportare

Aggiorna l'issue #8 con una nuova sezione datata (stesso pattern delle
sezioni "Verifica reale"/"Analisi compatibilità" già presenti) che linka il
documento `baseline_pretorch27_*.md` e i raw in `logs/`, oppure scrivimi qui
se preferisci che sia la sessione remota a scriverlo sull'issue dopo aver
letto il tuo output — indifferente, basta che l'informazione arrivi lì, non
solo in questa conversazione.
