# Bootstrap anti-Alzheimer — checklist di sessione

Questo file esiste per un solo motivo: ogni nuova sessione lunga su
questo progetto tende a ri-scoprire da zero problemi già diagnosticati,
a ipotizzare cause nuove per sintomi già visti, o a saltare passi di
verifica già stabiliti come necessari. Leggilo PRIMA di iniziare
qualunque test/benchmark/run su questo repo, non dopo aver già
inseguito tre ipotesi sbagliate.

Non sostituisce `LOGBOOK.md`/`LOGBOOK_ISSUE33.MD` (la storia completa,
prosa, cronologica) — è l'indice rapido che dice COSA controllare e
DOVE, prima di rimettersi a indagare da capo. Se una voce qui sotto
sembra sbagliata o superata, controlla prima nel logbook corrispondente
(potrebbe essere cambiata) e correggi questo file di conseguenza — non
lasciarlo silenziosamente stale.

## 0. Prima di ipotizzare QUALUNQUE causa per un sintomo nuovo

**Controlla la sezione 5 (known issues) di questo file PRIMA di
inventare una spiegazione nuova.** Il costo di rileggere 15 righe è
minuscolo comparato a un'ora spesa a inseguire thread-tuning/PCIe/altro
quando la causa reale era già scritta da mesi in `LOGBOOK.md` (vedi
2026-08-17, entry "continued 15" per un esempio concreto di questo
esatto errore).

Se il sintomo non è in sezione 5: cerca prima con
`grep -rn "<parola chiave>" LOGBOOK.md LOGBOOK_ISSUE33.MD` prima di
teorizzare. Solo se non c'è nulla, procedi a un'indagine nuova — e
quando la risolvi, AGGIUNGI la causa a sezione 5 di questo file, non
solo al logbook.

## 1. Pre-flight, prima di lanciare qualunque test

- [ ] `git status` + `git fetch origin <branch>` — un'altra sessione
  Claude Code potrebbe lavorare in PARALLELO sullo stesso branch
  (successo più volte in questo progetto). Se ci sono commit nuovi,
  `git log HEAD..origin/<branch> --oneline` prima di procedere.
- [ ] Rileggi le ultime 1-2 entry di `LOGBOOK_ISSUE33.MD` (o del
  logbook pertinente) — non fidarti del riassunto in testa alla
  conversazione, potrebbe essere stato compattato/perso dettaglio.
- [ ] Se il test misura TEMPI/PERFORMANCE (non solo correttezza):
  vai a sezione 2 PRIMA di scegliere dove girare il test — l'ambiente
  sbagliato produce numeri inutilizzabili, non solo lenti.
- [ ] Se il test tocca allocazione RAM/CPU host: misura l'uso REALE
  attuale prima di decidere se serve liberare risorse o allocare di
  più (`Win32_PerfFormattedData_PerfOS_Processor` per CPU,
  `Win32_OperatingSystem` per RAM) — non assumere.

## 2. Dove girare un test — locale (Z8/WSL2) vs RunPod

**Regola pratica**: se il test misura TEMPI di generate()/forward
reali su un carico non banale, gira su RunPod (Linux reale), non in
locale. WSL2 ha una limitazione strutturale nota (vedi 5.1) che rende
qualunque numero di tempo raccolto in locale sospetto finché non
confermato altrove.

Il locale (Z8, 28 core/436GB RAM/RTX3090) resta valido per:
- correttezza/parità numerica (non dipende da `pin_memory`)
- unit test, lint, iterazione rapida sul codice
- smoke test brevi (secondi, non minuti) dove uno stall improbabile
  è accettabile da tollerare/riprovare

## 3. Come eseguire un test (pattern incrementale)

- [ ] Fetta piccola prima (16), poi raddoppia (32→64→full) — non
  lanciare mai il carico pieno alla cieca. Ogni fetta è un checkpoint:
  se una fallisce/stalla, le precedenti restano valide.
- [ ] `--watchdog-timeout` esplicito e ragionevole per la dimensione
  del test — MA vedi 5.2: il watchdog interno di `eval_mmlu_gcsg.py`
  (self-SIGTERM/SIGKILL da dentro il container) non è affidabile sotto
  questo setup Docker/WSL2. Tieni una finestra di terminale/comando
  pronto per `docker kill <container>` da HOST come piano B, non
  fidarti che lo script si fermi da solo.
- [ ] Se il sospetto è sotto-parallelizzazione CPU: prova
  `OMP_NUM_THREADS=N MKL_NUM_THREADS=N` (N = core allocati a
  Docker/WSL2) sia come `-e` a `docker compose run` sia inline prima
  del comando Python — ma non aspettarti miracoli se la vera causa è
  altrove (vedi 5.1).
- [ ] Prima di runnare, avvia la telemetria (sezione 4) — MAI
  reattivamente a metà run. I primi minuti (caricamento modello,
  costruzione pool) sono spesso i più informativi e vanno persi se la
  raccolta parte tardi.

## 4. Telemetria — cosa raccogliere, sempre, con timestamp

Timestamp ISO 8601 con offset su OGNI riga, di OGNI sampler — è quello
che permette di allineare "cosa stava succedendo nel codice" con "cosa
diceva l'hardware" a posteriori. Un numero senza timestamp preciso è
quasi inutile per il debug post-mortem.

| Segnale | Comando | Note |
|---|---|---|
| Host CPU/RAM | PowerShell: `Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'"` + `Win32_OperatingSystem` | Girare come processo staccato (`Start-Process -WindowStyle Hidden`), loop con `Start-Sleep`, mai bloccante |
| GPU (util/mem/power/temp) | `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` | |
| **PCIe RX/TX** | `nvidia-smi dmon -s t -c 1` (colonne `rxpci`/`txpci`, MB/s) | Funziona anche su GeForce (RTX 3090 confermato) nonostante alcune metriche NVML siano ristrette sui consumer — MA vedi 5.4: un valore osservato NON prova un tetto, serve un benchmark dedicato per quello |
| Docker per-container | `docker stats --no-stream --format "..."` | Se il container sparisce, il sampler smette di scrivere righe — non un errore, controllare `docker ps` per conferma |
| PMEM/Optane | `Get-PmemPhysicalDevice` (PowerShell) | Vedi 5.5: su questa macchina la PMEM è in Memory Mode 100%, INVISIBILE come capacità separata — questo comando conferma la config (`Persistent memory size: 0 GB` su tutti i moduli) ma non dà telemetria di uso in tempo reale senza `ipmctl` (richiede admin, non sempre disponibile) |

Ferma i sampler esplicitamente a fine sessione di test
(`Stop-Process`/`docker exec ... pkill` o equivalente) invece di
lasciarli solo al self-timeout — più pulito per la prossima sessione.
Scrivi sempre un `SUMMARY.md` accanto ai CSV (vedi
`logs/z8_local/telemetry_fase6a/SUMMARY.md` per il formato).

## 5. Known issues — NON re-investigare da zero

### 5.1 — WSL2 `pin_memory=False`: run che dovrebbero durare minuti durano ore

vLLM stampa `WARNING ... Using 'pin_memory=False' as WSL is detected.
This may slow down the performance.` — quando lo vedi, FERMATI prima
di inseguire altre ipotesi (thread count, PCIe, casting, ecc.): questo
progetto ha già misurato, mesi fa, che questa singola limitazione può
trasformare un run da 570 domande da ~35-40 minuti (RunPod, Linux
reale, `pin_memory=True`) a UNA NOTTE INTERA (WSL2 locale) — vedi
`LOGBOOK.md`, sessioni 2026-08-11/12 ("the stall was never a
deadlock"). Non è specifico a nessun path di codice in particolare:
colpisce qualunque carico che si appoggia al meccanismo di swap/offload
di vLLM.

**Azione**: se un test locale sotto WSL2 è anormalmente lento,
`grep "pin_memory" <log>` PRIMA di ogni altra indagine. Se presente,
la spiegazione più probabile è questa, non un bug nel codice del
progetto — sposta la misura su RunPod invece di ottimizzare
sull'ambiente sbagliato.

### 5.2 — Self-SIGKILL da PID 1 non funziona in questo setup Docker/WSL2

Il watchdog di `eval_mmlu_gcsg.py` (`os.kill(os.getpid(), SIGKILL)`
dall'interno del processo stesso, che è PID 1 nel container) NON ha
terminato il processo (osservato 2026-08-17: il processo è rimasto
vivo, `State: R`, accumulando CPU time, per 10+ minuti dopo il SIGKILL
loggato). `docker kill <container>` da FUORI (host) ha funzionato
immediatamente sullo stesso processo. Causa esatta non confermata
(sospetto: interazione tra thread nativi OpenMP/CUDA e la
paravirtualizzazione GPU-PV di WSL2, non verificato).

**Azione**: non fidarsi del watchdog interno per fermare da solo un
run bloccato/lento sotto questo setup — tenere pronto `docker kill`
da host come piano B sempre.

### 5.3 — `_ShadowExpertINT4.__call__()`: cast+scale ora memoizzati

Fino al 2026-08-17, ricalcolava `w13_q.to(dtype) * scale` ad OGNI
chiamata (ogni token, ogni layer, ogni expert freddo) invece che una
volta sola — trascurabile sul path INT4 originale (sorgente int8,
piccolo), costoso sul path fp32-cache di Fase 6a (sorgente già fp32,
centinaia di MB). Fix: memoization per `(layer_id, dtype)` in
`__init__`/`__call__`. Se un futuro path aggiunge un NUOVO tipo di
cache pesi a questa classe, verificare che lo scale sia effettivamente
costante per la vita dell'oggetto (la memoization assume che sì).

### 5.4 — Traffico PCIe osservato ≠ tetto del canale

Un valore di banda osservato (`nvidia-smi dmon -s t`) durante un
workload che non sta esplicitamente cercando di saturare il bus NON
dimostra un limite strutturale (WSL2/GPU-PV o altro) — dimostra solo
quanto quel workload specifico sta usando. Per una risposta vera serve
un microbenchmark dedicato che ci provi a saturare il canale
(`scripts/bench_pcie_bandwidth_wsl2.py`, pinned vs pageable, varie
dimensioni) e un confronto con un ambiente Linux reale.

### 5.5 — PMEM (Intel Optane DC) in Memory Mode 100%

Lo Z8 ha 4× `NMA1XXD128GPS` (Optane DC PMM, ~505GB nominali) oltre a
15× DIMM DDR4 standard (~240GB). Confermato via
`Get-PmemPhysicalDevice`: tutti e 4 i moduli a "Persistent memory
size: 0 GB" → intera capacità allocata a Memory Mode (volatile), non
App Direct. In Memory Mode la PMEM DIVENTA la RAM visibile al sistema
operativo (i 503.7GB totali riportati da Windows combaciano con la
capacità PMEM, non con DDR4+PMEM sommati) mentre i 240GB di DDR4
fungono da cache invisibile gestita dalla CPU. Per indicazione del
project owner, la cache DDR4 "si attiva" (nel senso che PMEM comincia
a essere esercitata) sopra l'80% di uso della cache DDR4 stessa —
soglia non verificabile da software senza `ipmctl` con privilegi
admin (non sempre disponibile in sessione). Qualunque numero di
"RAM libera" misurato da software NON distingue le due — è un limite
noto, non un errore di misura.

### 5.6 — Dettagli ambiente locale minori ma ricorrenti

- Docker `WORKDIR` è `/workspace` (repo root), non `osx-poc` — i
  comandi vanno prefissati `osx-poc/...` o eseguiti dopo `cd osx-poc`
  dentro il container.
- `VLLM_ATTENTION_BACKEND=XFORMERS` necessario per costruire un
  `vllm.LLM()` semplice (senza `worker_cls=GCSGWorker`) in locale — il
  backend flash-attention di default crasha nel profiling interno
  (`cu_seqlens_q must have dtype int32`), causa non investigata a
  fondo (probabile drift di versione).
- `hf_overrides={"head_dim": 128}` necessario per costruire l'engine su
  questo checkpoint Mixtral, altrimenti `num_heads` risolve a `None`.
- Checkpoint locale reale: `/data/nvme/models/mixtral-instruct-awq`
  dentro il container (override via `OSX_MMLU_MODEL_PATH` se serve un
  path diverso, es. per il pattern "copia locale prima del load" su
  RunPod — vedi 5.7).

### 5.7 — RunPod: mmap su volume di rete può bloccarsi indefinitamente

Caricare un checkpoint via mmap direttamente da un Network Volume
RunPod (FUSE/MooseFS) può bloccarsi indefinitamente (osservato,
diagnosticato via `/proc/PID/status` — stato `D`, wchan
`folio_wait_bit_common` — NON specifico al flag `--enable-cpu-offload`,
un run di controllo senza quel flag si è bloccato allo stesso modo).
Il filesystem stesso non è lento (confermato con `dd`, 4-5GB/s) — il
problema è specifico a mmap su quel tipo di storage.

**Azione**: copiare il checkpoint su disco container locale PRIMA del
load, mai mmap-are direttamente da un Network Volume. Pattern già
gestito da `OSX_MMLU_MODEL_PATH` in `eval_mmlu_gcsg.py`.

## 6. Come registrare le change nel logbook

- Formato standard per entry narrative: **What we set out to do / What
  we did / Result / Why this matters / Current state / Next session**
  — un'entry per sessione di lavoro, non per singola azione.
- **Mai cancellare/riscrivere una correzione** — aggiungerla in fondo
  con una nota esplicita ("Correzione, stessa sessione: ..."), lasciando
  l'errore originale visibile. La cronologia degli errori è
  informazione, non rumore da ripulire.
- Per sessioni con MOLTI interventi live (debug interattivo, telemetria,
  run multipli, kill/restart) — oltre alla prosa, aggiungere una
  **tabella con timestamp reali**: intervento, timestamp, stato
  attuale, nota di rollback. Serve a valutare impatto e rollback anche
  quando la sessione stessa non ricorda più l'ordine esatto degli
  eventi. Timestamp reali = presi da `date -Iseconds`, `docker inspect
  --format '{{.State.StartedAt}}'`, prima riga di un CSV di telemetria
  — MAI stimati a memoria.
- Se una run/test FALLISCE senza produrre risultato: documentarlo lo
  stesso, con la stessa cura di un successo — un "non ha funzionato,
  ecco perché" è valore quanto un numero.
- Timestamp ovunque possibile: non solo nel logbook, anche nei file di
  telemetria (sezione 4) e nei nomi dei log di run falliti (es.
  `run_n16_KILLED_thread_starved_23min_no_result.log` — il nome stesso
  porta informazione, non serve aprire il file per sapere cos'è
  successo).

## 7. Dove guardare per il resto

- `LOGBOOK.md` — storia completa del progetto, tutte le issue.
- `LOGBOOK_ISSUE33.MD` — storia completa di issue #33 (tier DDR4/CPU
  offload), questo file ne è solo l'indice operativo.
- `reports/` — report finali per issue/sprint chiusi (non aggiornati
  live, solo a chiusura).
- Questo file va aggiornato ogni volta che emerge un nuovo "known
  issue" riutilizzabile (sezione 5) o una nuova pratica di processo che
  vale la pena rendere default (sezioni 1-4, 6) — non lasciarlo fermo
  all'ultima sessione che l'ha creato.
