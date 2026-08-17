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

### 5.8 — RunPod: accesso SSH diretto per Claude Code (egress di questa sessione è allowlisted)

L'egress di rete di una sessione Claude Code è allowlisted a domini
specifici — `runpod.io`/`api.runpod.io` danno `403` dal proxy
dell'ambiente, e il proxy `ssh.runpod.io` richiede comunque una PTY
interattiva vera (fallisce con `Error: Your SSH client doesn't support
PTY` da una shell non-interattiva). L'unico percorso che funziona è
l'**IP pubblico diretto del pod + porta mappata** per la 22 (RunPod
"Expose TCP Ports", eventualmente richiede un restart del pod se non
era già attivo) — non l'endpoint proxy, non `.internal`.

Keypair dedicata già esistente su questa macchina (riusata da Sprint 4
in poi, non ricrearla): `~/.ssh/id_ed25519_runpod_sprint4` (+ `.pub`).
Chiave pubblica:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFacxFKKVMWddof4o2tQ9OgZsHkX4TuhsxaS1T2G6vwX vmemoryfabric-sprint4-runpod-20260811
```

**Azione**: appena il project owner ha accesso alla console web del pod
(RunPod → pod → Connect → Start Web Terminal), fargli lanciare questo
comando sul pod stesso (idempotente, append-only, imposta anche i
permessi che `sshd` richiede):

```bash
mkdir -p ~/.ssh && echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFacxFKKVMWddof4o2tQ9OgZsHkX4TuhsxaS1T2G6vwX vmemoryfabric-sprint4-runpod-20260811" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

Aggiungere la chiave anche in Account → Settings → SSH Public Keys
lato RunPod è utile per i PROSSIMI pod creati dopo l'aggiunta, ma non
retroagisce su un pod già in esecuzione — per un pod già up, il comando
sopra è il percorso affidabile.

### 5.9 — `nproc`/`free`/`/proc/meminfo` dentro un pod RunPod mentono: mostrano l'HOST, non la quota cgroup

Confermato di nuovo 2026-08-17 (pod RTX3090, dopo Malmö/RTX A6000 —
pattern ricorrente, non un caso isolato): dentro il container `nproc`
ha riportato 256 e `free -h` 1.0Ti totali — la topologia/RAM dell'HOST
fisico condiviso, non quello che il cgroup di QUESTO pod può davvero
usare. Il numero vero:

```bash
# CPU (cgroup v1 — controllare anche v2, /sys/fs/cgroup/cpu.max, se v1 assente)
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us
# core reali = quota_us / period_us

# RAM (cgroup v1 — v2: /sys/fs/cgroup/memory.max e memory.current)
cat /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes
```

**Azione**: MAI usare `nproc`/`free` grezzi per calcolare
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` o per giudicare quanta RAM è
disponibile per un pool CPU-resident — sempre i due comandi sopra
prima di lanciare qualunque run che dipenda da questi numeri. Codice
di produzione: `_read_cgroup_available_gb()` in `src/scheduler/gcsg.py`
implementa già questa lettura corretta (v1→v2→psutil), riusarla invece
di richiamare `psutil`/`os.cpu_count()` grezzi in nuovo codice.

### 5.10 — `py-spy`/profiling live spesso bloccato su pod RunPod (manca SYS_PTRACE)

`py-spy dump --pid <PID>` (e qualunque altro tool che richieda
`ptrace`: `gdb attach`, `strace`, spesso anche `perf`) fallisce con
"Permission Denied... SYS_PTRACE capability" su un pod RunPod — il
container non è avviato con quella capability e non è modificabile
dall'interno via SSH (servirebbe ricreare il pod con
`--cap-add=SYS_PTRACE`, non un'opzione esposta dalla console RunPod
standard). Verificato 2026-08-17, non un caso isolato del singolo pod.

**Azione**: non perdere tempo a inseguire varianti di py-spy/gdb su un
pod RunPod. Passare subito a instrumentare il codice stesso con
`time.monotonic()`/`log.info()` nei punti sospetti (vedi il pattern
già applicato in `_build_cpu_shadow_pool_awq()`, timing per-layer e
per-expert) — più lento da iterare ma l'unico che funziona in questo
ambiente.

### 5.11 — CPU shadow pool (fp32 AWQ, Fase 6a): FALSIFICATO che fosse "solo WSL2" — causa reale isolata, non è la pool build

Attenzione a non ri-applicare ciecamente la lezione di 5.1: il primo
run reale con `--enable-cpu-offload` genuinamente attivo (path AWQ,
Fase 6a Passo 3) su un pod RunPod **Linux reale, senza alcun warning
`pin_memory=False`**, NON ha completato nemmeno 1 prompt su 16 in 600s.
La causa NON è WSL2 (Linux reale, nessun warning). **Non è nemmeno la
build del pool CPU** (prima ipotesi di questa stessa sezione, ora
corretta: la build è VELOCE, ~44s/expert × 2 = ~88s totali, completa
PRIMA che "LLM ready" venga stampato — misurato con timing dedicato,
`grep 'GCSG DIAG.*TOTALE' <log>`).

**Causa reale, isolata 2026-08-17**: il forward per-TOKEN di
`_ShadowExpertINT4.__call__()` (una riga alla volta,
`hidden_states.shape=(1, 4096)`, non batchato attraverso la sequenza)
costa **~0.10s/chiamata**, costante — con migliaia di chiamate
necessarie anche per n=16 prompt, questo da solo spiega minuti di
wall time. **Testato e falsificato che sia thread-oversubscription**:
`OMP_NUM_THREADS=27` vs `=2` allo stesso checkpoint (calls=200)
producono lo STESSO avg (~0.10s/call) — il parallelismo aiuta la
build del pool (tensori grandi) ma è irrilevante sul forward a
singola riga. Sospetto più probabile, non ancora confermato: costo
fisso per-chiamata (dispatch/allocazione PyTorch) o limite di
throughput per-core genuino su questo tipo di istanza RunPod
ottimizzata per GPU (vCPU condivise/deprioritizzate) — non un bug di
configurazione facilmente risolvibile.

**Risultato completo misurato** (pod RTX3090/AMD EPYC 7C13, 2026-08-17):
n=16, 1133.1s (~18.9 min), accuracy 8/16 (50.0%),
`shadow_activations=10081` — **identico byte-per-byte** al baseline
no-op (stesso n, stessa selezione expert) in accuracy E conteggio
attivazioni: correttezza numerica PROVATA su hardware reale, non solo
in unit test. Performance: **25.7x più lento** del baseline senza
offload sullo stesso n=16 (44.1s, Malmö) — non production-ready,
estrapolato al full 570 sarebbe dell'ordine delle ore.

**Azione**: se un run con `--enable-cpu-offload` sul path AWQ è lento,
NON teorizzare thread-tuning o pool-build — sono già esclusi con dati.
`grep 'GCSG DIAG: shadow forward calls' <log>` per vedere il tasso
reale di chiamate/secondo. Un microbenchmark PyTorch puro (matmul
1×4096 @ 4096×N su CPU, fuori da vLLM) è il prossimo passo se si vuole
isolare dispatch-overhead vs. limite hardware genuino — non ancora
fatto. Il cap RAM-aware (`_check_cpu_ram_budget()`) resta comunque
utile (previene OOM su pool grandi) ma non tocca questo costo
per-chiamata.

### 5.12 — `scp` diretto di un file locale (Windows) su un pod: il diff esplode per via di CRLF

Copiare un file modificato localmente (Windows, `core.autocrlf=true`,
quindi working tree in CRLF) direttamente via `scp` sul checkout git
del pod (Linux, blob committati in LF puro) produce un `git diff` che
segna OGNI riga come cambiata, anche se le modifiche reali sono poche
righe — il file trasferito ha `\r\n`, il blob tracciato ha `\n`. `git`
normalmente gestisce questa conversione in modo trasparente su
push/pull/checkout, ma `scp` la bypassa copiando i byte grezzi.

**Azione**: dopo un `scp` diretto Windows→pod di un file di codice,
SEMPRE `sed -i 's/\r$//' <file>` sul pod prima di fidarsi di
`git diff --stat` per giudicare la dimensione della modifica. Meglio
ancora: preferire `git push`/`git pull` quando possibile (gestisce la
conversione da solo) — usare `scp` diretto solo per iterare più in
fretta durante un debug attivo, come fatto qui.

### 5.13 — SSH+nohup verso un pod: serve `< /dev/null`, non solo redirect di stdout/stderr

Lanciare un processo in background su un pod via SSH con
`nohup cmd > out.log 2>&1 &` (senza chiudere anche lo STDIN) fa restare
appesa la sessione SSH stessa finché il processo backgrounded non
termina — anche se il comando "in foreground" è già tornato. Causa:
il processo figlio eredita comunque il file descriptor di stdin
collegato al canale SSH, e SSH aspetta che TUTTI i descriptor del
canale si chiudano, non solo che il comando visibile finisca.

**Azione**: sempre `nohup cmd > out.log 2>&1 < /dev/null & disown` (non
solo `> out.log 2>&1 &`) per lanciare un run lungo su un pod da uno
script/tool non interattivo — altrimenti ogni lancio va in timeout e
finisce nella coda dei task in background del tool, funzionalmente
innocuo ma fonte di confusione inutile.

### 5.14 — `/proc/cpuinfo`: alcuni flag AVX-512 hanno l'underscore, altri no — non assumere la convenzione

Il naming dei flag nel kernel Linux per `/proc/cpuinfo` è
inconsistente: `avx512f`/`avx512bw`/`avx512cd`/`avx512dq`/`avx512vbmi`/
`avx512vl`/`avx512ifma` NON hanno underscore, ma `avx512_vnni`/
`avx512_bf16`/`avx512_fp16`/`avx512_bitalg`/`avx512_vbmi2`/
`avx512_vpopcntdq` SÌ (verificato 2026-08-17 su un pod RunPod H200/
Xeon Platinum 8568Y+, Emerald Rapids — `benchmarks/perf_test_hardware.py`
cercava `"avx512vnni"` senza underscore, dando un falso negativo su un
host che lo supporta realmente). Non è un errore di battitura isolato —
è la convenzione reale del kernel, va verificata caso per caso con un
grep diretto su `/proc/cpuinfo`, non assunta per analogia con
`avx512f`.

**Azione**: prima di aggiungere un check per un nuovo flag CPU (AVX-512
o altro, es. AMX: `amx_bf16`/`amx_int8`/`amx_tile`, anche questi senza
underscore), `grep -o 'nomeflag' /proc/cpuinfo` su hardware reale per
confermare la stringa esatta — mai assumere la convenzione dagli altri
flag già nel codice.

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
- `benchmarks/perf_test_hardware.py` + `benchmarks/perf_tuning_report.py`
  (`make perf-test-hardware` / `make perf-tuning-report`, issue #33
  "continued 19") — framework riutilizzabile per caratterizzare
  CPU/RAM/GPU/PCIe di un host/pod e derivarne raccomandazioni (thread
  count, budget RAM CPU-pool, verdetto cpu-offload). Rieseguirlo su un
  pod/host NUOVO invece di ripetere a mano un test A/B thread-count o un
  microbenchmark GFLOPS isolato — è esattamente il lavoro che quei due
  script fanno già, con l'esito annotato contro i dati già noti (range
  24-26x pipeline / 44.6x isolato) invece di un numero isolato.
- `logs/runpod_perf_framework_validation_20260817/COMPARISON_ANALYSIS.md`
  (issue #33 "continued 21") — confronto a tre host (Z8, RunPod
  RTX3090/EPYC, RunPod H200/Xeon Platinum) coi JSON grezzi dello stesso
  framework — punto di partenza se serve aggiungere un quarto host al
  confronto invece di ripartire da zero.
- Questo file va aggiornato ogni volta che emerge un nuovo "known
  issue" riutilizzabile (sezione 5) o una nuova pratica di processo che
  vale la pena rendere default (sezioni 1-4, 6) — non lasciarlo fermo
  all'ultima sessione che l'ha creato.
