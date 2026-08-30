# Handoff — sblocco toolchain per RTX 5060 Ti 16GB (issue #8)

**Per:** la sessione Claude Code con accesso diretto alla Z8 G4 bare-metal
(quella che ha già fatto il recupero dati e la verifica torch/sm_120).
**Da:** sessione remota che ha coordinato Sprint 5 (Berg), la Project Plan
Review e il triage dell'issue #8 — nessun accesso di rete alla Z8, lavora
solo via GitHub.
**Data:** 2026-08-30.

---

## 0. Prima di tutto: onboarding — non saltarlo

Questo non è un task isolato. Il progetto ha una storia di ~3 settimane con
convenzioni precise, nate da errori reali già fatti e corretti. Prima di
toccare qualunque pin di versione, leggi in quest'ordine:

1. **`README.md`** (root del repo) — stato attuale, roadmap, target del PoC,
   known limitations. Il banner in cima è tenuto sincronizzato con la
   release reale: **`v0.6.0-beta.1`** è la prima release taggata del
   progetto (pubblicata 2026-08-26), Gate A5 del Filone A è chiuso 4/4.
2. **`osx-poc/CHANGELOG.MD`**, sezione `[Berg]` in cima — cosa è cambiato
   di recente, in ordine cronologico inverso.
3. **`osx-poc/LOGBOOK.md`**, entry più recenti in cima — il dev diary
   completo, prosa non compressa. In particolare le entry del 25-28 agosto
   (Project Plan Review, chiusura Gate A5, annuncio della beta).
4. **`osx-poc/LOGBOOK_NEW_Z8.md`** — la migrazione a bare-metal del 24
   agosto, che tu stesso conosci già ma che chi lavora da remoto ha dovuto
   ricostruire leggendolo.
5. **`osx-poc/LOGBOOK_ISSUE33.MD`** — la storia lunga (22 sessioni) del
   tier CPU-offload, per capire il livello di rigore che questo progetto
   si aspetta su ogni claim quantitativa.
6. **[Issue #8](https://github.com/danielesalpietro/vMemoryFabric/issues/8)
   su GitHub, per intero, non solo l'ultima sezione** — la contengono tutte
   le decisioni pregresse su AER/dual-GPU, inclusa la sezione
   "Verifica reale 2026-08-30" che tu stesso hai prodotto i dati per
   scrivere (nvidia-smi, `get_arch_list()`, il test CUDA fallito sulla
   5060 Ti). Leggila lì, non fidarti del riassunto qui sotto.
7. **`osx-poc/reports/ppr_20260825_sprint5_review.html`** (o la versione
   `.docx` accanto) — la Project Plan Review che ha trovato che il Filone B
   (paper) era fermo da giorni mentre il backlog cresceva altrove. Serve
   per capire perché il project owner è ora attento a non ripetere lo
   stesso pattern con questo task.

**Perché insisto sull'onboarding**: nelle ultime settimane si sono
accumulati due incidenti dello stesso tipo — codice/dati reali che
esistevano solo su una macchina o un branch, mai riflessi nei documenti
pubblici (README, CHANGELOG, issue tracker). Sono stati trovati e corretti
solo perché qualcuno ha *verificato* invece di assumere. Le regole al §2
sotto esistono per non ripetere l'errore una terza volta — con te.

---

## 1. Cosa è successo finora su questo filone (issue #8)

Riassunto compresso — la fonte di verità resta l'issue GitHub, non questo
file:

- Issue #8 (dual-GPU/AER) è bloccata da Sprint 0 (Karlshamn), in attesa di
  una seconda GPU. Prevista una RTX 5080; arrivata invece una **RTX 5060 Ti
  16GB**, affiancata alla RTX 3090 24GB esistente sulla Z8.
- L'issue aveva già un precedente reale e documentato: una RTX PRO 6000
  Blackwell (sm_120) provata ad agosto e scartata perché
  `torch==2.5.1+cu124` (pin in `requirements-vllm.txt`) non supporta
  sm_120 — `RuntimeError: CUDA error: no kernel image is available`.
- **2026-08-30, tu stesso l'hai verificato sull'host reale**: `nvidia-smi`
  vede entrambe le GPU correttamente (indici 0/1, 24576/16311 MiB, bus
  PCIe distinti); `torch.cuda.get_arch_list()` conferma **sm_120 assente**;
  un'operazione CUDA minima su `cuda:1` fallisce con lo stesso identico
  errore della RTX PRO 6000. La 3090 funziona normalmente.
- Due caveat che hai già segnalato tu stesso, e che restano validi:
  1. l'immagine `osx-poc:dev` non esiste più su questa Z8 (verosimile
     conseguenza della riprovisionatura del 24 agosto) — ricostruita
     un'equivalente per il test, non lo stack completo con vllm incluso;
  2. `docker-compose.yml` ha `NVIDIA_VISIBLE_DEVICES=0`/
     `CUDA_VISIBLE_DEVICES=0` — blocco *indipendente* dal toolchain, la
     seconda GPU resta invisibile ai container finché questo pin non
     viene cambiato, a prescindere da torch.
- Conclusione già scritta nell'issue: sbloccare AER richiede
  `torch>=2.7` su `cu128` (supporto Blackwell consumer), che trascina il
  pin `vllm==0.6.6.post1` — dichiarato in `requirements-vllm.txt` come
  gruppo coerente, non separabile a cuor leggero.

---

## 2. Il compito per questa sessione

**Non è ancora "fai l'upgrade".** È: stabilire *se* e *come* si può fare
in sicurezza, prima che chiunque tocchi un pin.

1. **Identifica quale versione di vLLM è compatibile con `torch>=2.7` /
   `cu128`.** Parti da `requirements-vllm.txt` e dalla changelog/release
   notes di vLLM per capire quale versione minima richiede quel torch, e
   se introduce breaking change rilevanti per `GCSGWorker`
   (`osx-poc/src/scheduler/gcsg.py`) — in particolare i tre path di
   quantizzazione (AWQ ModuleList, Marlin, INT4 shadow) che oggi dipendono
   da dettagli interni di vLLM 0.6.6.post1 (vedi i commenti/docstring in
   `gcsg.py` e la storia in `LOGBOOK.md`, entry Oskarshamn/Tekniska).
2. **Non installare nulla di definitivo, non toccare i pin ancora.**
   Prima produci solo l'analisi di compatibilità (documento o commento
   sull'issue), poi aspetta conferma esplicita prima di procedere con un
   vero bump — esattamente come hai già fatto oggi per la verifica
   torch/sm_120: leggi, misuri, riporti, non decidi da solo di applicare
   un fix strutturale.
3. **Prerequisito esplicito già scritto nell'issue, da rispettare**:
   *"congelare un set di metriche baseline M1/M2 sull'attuale toolchain
   prima di qualsiasi upgrade"* — se questa sessione o una successiva
   procede davvero con l'upgrade, va prima eseguito e salvato un giro di
   `make smoke && make test` + i benchmark M1/M2 (`bench_eat.py`,
   `bench_tier.py`) sul toolchain attuale, così un delta post-upgrade è
   misurabile e non solo assunto.
4. **Il pin `NVIDIA_VISIBLE_DEVICES=0` in `docker-compose.yml` è un task
   separato**, non bloccante per l'analisi di compatibilità vLLM ma
   necessario prima che la seconda GPU sia utilizzabile per davvero.
   Puoi proporlo come PR indipendente quando pronta.

---

## 3. Regole del gioco (osservate finora in questo progetto, non opzionali)

- **Verifica prima di scrivere, sempre.** Ogni claim quantitativa in
  questo repo ha una fonte tracciabile (log, JSON, output di comando). Se
  non l'hai misurato tu stesso in questa sessione, dillo esplicitamente
  invece di darlo per assunto.
- **Non sovrascrivere dati/misure discordanti.** Se un numero non
  coincide con uno già scritto altrove, riportali entrambi e segnala la
  discrepanza — non scegliere quello che "sembra giusto".
- **Un issue/gate non è chiuso finché non è verificabile per davvero**,
  non finché il lavoro intorno sembra finito. Vale anche al contrario: se
  qualcosa è già mersato su `develop` ma un documento dice ancora "non
  fatto", quel documento è la cosa da correggere, non lasciare come sta.
- **Dati grezzi mai alterati.** Se un file di output ha un problema di
  formato (es. banner di un container davanti al JSON), il file grezzo
  resta intatto byte-per-byte — si aggiunge una copia pulita accanto
  (`*.clean.json`), mai si sovrascrive l'originale. Vedi
  `logs/new_z8_bare_metal/README.md` per la convenzione già in uso.
- **Workflow git**: crea un branch da `develop` aggiornato (non riusare
  branch vecchi non aggiornati — è già successo che un branch locale
  restasse indietro rispetto a `develop` dopo un merge), pusha, e lascia
  aprire/mergiare la PR alla sessione con accesso a GitHub — è il pattern
  già usato per i due recuperi dati precedenti
  (`claude/z8-raw-data-recovery`, `claude/z8-raw-data-recovery-2`).
- **Nessuna decisione unilaterale su cosa scartare o ampliare in
  silenzio.** Se emerge uno scope aggiuntivo non richiesto (è già successo
  con le cartelle extra di telemetria), segnalalo e aspetta conferma
  invece di includerlo o ometterlo di tua iniziativa.

---

## 4. Dove riportare

Aggiorna l'issue #8 con una nuova sezione datata (stesso pattern delle
sezioni "Aggiornamento"/"Verifica reale" già presenti), oppure scrivimi
qui se preferisci che sia la sessione remota a scriverlo sull'issue dopo
aver letto il tuo output — indifferente, basta che l'informazione arrivi
lì, non solo in questa conversazione.
