# Logbook — Nuova Z8 G4 bare-metal

Dev diary scoped alla transizione dal vecchio setup Docker-on-Windows/WSL2
alla nuova Z8 G4 (Ubuntu 24.04 bare-metal, RTX 3090, accessibile via SSH),
tenuto separato da `LOGBOOK.md`/`LOGBOOK_ISSUE33.MD` perché copre lavoro di
infrastruttura/provisioning che si estende su più sessioni ed è un
prerequisito trasversale (non uno specifico issue applicativo) — stessa
convenzione degli altri log: una sezione per sessione di lavoro, prosa,
correzioni registrate non cancellate.

---

## 2026-08-24 — Pre-check hardware, fix provisioning, primo clone e build

**Release:** none — setup e verifica ambiente, nessun codice applicativo
modificato.

### What we set out to do

Verificare se la nuova Z8 G4 (SSH `admin@151.64.182.141:2222`) è pronta per
riprendere i test del progetto, in vista della sostituzione del vecchio
ambiente di riferimento Docker-on-Windows/WSL2.

### What we did

Pre-check hardware via SSH, solo comandi in lettura:

- CPU: 2× Xeon Gold 6244 (Cascade Lake), 16C/32T totali, `avx512_vnni`
  presente, nessun AMX (atteso, pre-Sapphire Rapids).
- RAM: 235 GiB DDR4 utilizzabili su 2 nodi NUMA (nodo0 ~112GB, nodo1
  ~129GB); GPU (RTX 3090) sul nodo 0.
- GPU: RTX 3090, 24GB VRAM, driver 595.84.
- OS: Ubuntu 24.04.2 LTS, kernel 6.8.0-138-generic.
- `io_uring` abilitato (`io_uring_disabled=0`) — non disponibile nel vecchio
  setup WSL2/Docker.
- NVIDIA Container Toolkit 1.20.0 già installato.

Trovato e corretto/chiarito nel corso della sessione:

- L'installer automatico aveva messo l'intero OS (`/`, `/boot/efi`, un
  volume XFS applicativo) sulla region0 di Optane PMEM (`pmem0s`, modalità
  `sector`) invece che su un disco dedicato — riconosciuto come bug del
  processo di provisioning, da correggere nella prossima release della ISO
  (fuori scope di questa sessione, in carico ad altra sessione dedicata).
- Root filesystem inizialmente stretto (53G liberi su 98G) per un loop-file
  Docker orfano (`/var/lib/docker-loop.xfs`) residuo di provisioning
  precedente; risolto in parallelo spostando il vero Docker Root Dir su
  `/mnt/wdc-docker` (disco WD ~466G, XFS) — confermato via
  `docker info` → `Docker Root Dir: /mnt/wdc-docker/docker`. Spazio libero
  su `/` tornato a 85G.
- `/etc/docker/daemon.json` porta strumentazione Vast.ai
  (`kaalia_docker_shim` come runtime nvidia, registry-mirror `*.vast.ai`) —
  confermato intenzionale: questo nodo è anche un nodo Vast.ai in fase di
  sviluppo/test, non un residuo da rimuovere.
- Region1 PMEM (era `pmem1s`, modalità `sector`/BTT, 252GiB, inutilizzata)
  riconfigurata via `ndctl` in modalità `fsdax` → `/dev/pmem1`, 248.1GiB,
  byte-addressable. Non ancora formattata né montata — solo allocata.
- Installati `ndctl`, `ipmctl`, `python3-pip`; utente `admin` aggiunto al
  gruppo `docker`; verificato passthrough GPU end-to-end in un container
  (`docker run --gpus all ... nvidia-smi -L` vede correttamente la 3090).
- Runner GitHub Actions self-hosted (`berlin-3eie`, label
  `self-hosted,Linux,X64,z8,gpu`) era erroneamente agganciato al repo
  `kickstart-berlin` invece che a `vMemoryFabric` — deregistrato da
  `kickstart-berlin` e ri-registrato su `vMemoryFabric` (via API +
  `config.sh`/`svc.sh`), online e verificato. Rimosso anche il vecchio
  runner offline `Z8-G4-RTX3090` (era il setup Docker-on-Windows precedente,
  ora sostituito da questo nodo).
- Repo clonato sulla Z8 in `~/vMemoryFabric`, branch
  `claude/ddr4-ram-processing-unzseh` (allineato a `develop`, contiene il
  lavoro issue #33 DDR4/PMEM tier — scelto perché è il branch più rilevante
  per l'hardware appena sbloccato).
- Avviata `make build` (immagine Docker CUDA 12.1.1 + Python 3.12) —
  in corso a fine sessione, esito da registrare nel prossimo aggiornamento.

### Stato aperto

- `nvcc`/CUDA toolkit ancora assente sull'host — non bloccante, il workflow
  del progetto gira interamente dentro l'immagine Docker.
- `make smoke` non ancora lanciato — prossimo passo appena la build finisce.
- Label del runner non verificate contro `ci.yml` prima della migrazione
  (fatto dopo, via API: confermate `self-hosted`+`gpu` presenti).

### Sequenza di test concordata (da eseguire nelle prossime sessioni)

1. **Re-check**: rilanciare `make perf-test-hardware`/`make perf-tuning-report`
   (framework di
   [`COMPARISON_ANALYSIS.md`](../logs/runpod_perf_framework_validation_20260817/COMPARISON_ANALYSIS.md),
   issue #33) su questa Z8 bare-metal — finora quell'hardware (Xeon Gold
   6244 + 3090) era stato misurato solo sotto Docker-on-Windows/WSL2. Stesso
   discorso per `bench_tier.py`/`bench_eat.py`: i numeri RLock/#2 erano
   stati rimisurati solo sul vecchio runner Windows (`full-gpu-tests` run
   #150).
2. `pinned_memory: true` e `use_io_uring: true` in
   `osx-poc/configs/osx_default.yaml` (oggi entrambi `false` per limite del
   vecchio ambiente dev) — flag semplici, nessun codice nuovo, poi
   ri-benchmark (`bench_tier.py`).
3. Benchmark CPU-offload pinnato NUMA (`numactl --cpunodebind=0
   --membind=0`, coerente col nodo 0 della GPU) — alimenta l'ipotesi aperta
   in issue #45 (banda RAM per-core limitata su vCPU condivise vs. core
   fisici dedicati); su questa macchina abbiamo core dedicati, campione più
   pulito dei tre host già misurati.
4. **PMEM/EMH-2** — non un semplice re-bench: non esiste ancora un modulo
   tier PMEM in `src/tier/` (`pmem.enabled: false` in config, solo
   placeholder). Serve prima: `mkfs`+`mount -o dax` su `/dev/pmem1`, un
   nuovo modulo tier analogo a `gpu.py`/`io.py`, aggancio a `TierManager` —
   poi, solo dopo, benchmark.

### Why this matters

Questa Z8 bare-metal sblocca sperimentalmente tre dei vincoli elencati nella
tabella "Dev environment constraints" del README principale (pinned CUDA
memory, io_uring, Optane PMEM) finora solo documentati come
"deferred"/non disponibili. È il prerequisito diretto per la prossima fase
di lavoro su issue #33/#45, non uno specifico fix applicativo — da qui la
scelta di tenerlo come log separato.

### Aggiornamento (stessa sessione) — sampler di telemetria host attivati

`make build` è rimasta in esecuzione per tutta la sessione (immagine CUDA
12.1.1 + Python 3.12 dal `Dockerfile` del progetto, prima build su questa
macchina, cache vuota).

Nel frattempo, su richiesta esplicita, riscritti in bash i sampler usati
sulla vecchia Z8 Windows (`logs/z8_local/telemetry_fase6a/host_sampler.ps1`,
`pcie_sampler.ps1` — PowerShell, non eseguibili su Linux) come
`osx-poc/scripts/telemetry/{host_sampler,pcie_sampler,docker_stats_sampler}.sh`,
stesso schema CSV per confrontabilità con le vecchie run (una differenza
nota: `ram_free_gb` qui è `MemAvailable` da `/proc/meminfo`, non l'analogo
esatto di `FreePhysicalMemory` di Windows). A differenza della vecchia
sessione, questi tre script ora sono tooling versionato nel repo, non solo
artefatti one-off dentro `logs/`.

Avviati in background (`setsid`+`disown`, sopravvivono alla chiusura della
sessione SSH) su `~/vMemoryFabric/logs/new_z8_bare_metal/telemetry_20260824_1253/`:
verificati con dati reali dopo ~20s — CPU/RAM host, GPU (RTX 3090 idle:
~29W, 33°C, 9MiB/24576MiB VRAM), PCIe RX/TX (0/0, atteso: nessun
trasferimento in corso). `docker_stats.csv` per ora ha solo l'header — si
popola quando parte il primo container (`make smoke`/benchmark), `make
build` non ne avvia uno.

Nessun `DURATION_S` esplicito passato → default 7200s (2h) per host/pcie
sampler, come nella vecchia versione PowerShell; da rilanciare se una
sessione di test supera quella finestra.

### Aggiornamento (stessa sessione) — build completata, smoke test verde dopo fix issue #12

`make build` completata con successo (~13 min, cache vuota, prima build su
questa macchina): immagine `osx-poc:dev`, CUDA 12.1.1 + Python 3.12 +
torch 2.5.1+cu124 + vLLM 0.6.6.post1.

Primo `make smoke` fallito subito:
`python: can't open file '/workspace/scripts/smoke_test.py': [Errno 2] No
such file or directory`. Non un problema di questa macchina — è
esattamente il bug già tracciato in
[issue #12](https://github.com/danielesalpietro/vMemoryFabric/issues/12)
(aperto dal 2026-08-14, mai risolto): `docker-compose.yml` monta `.` su
`/workspace` (repo root), ma `scripts/`/`src/`/`tests/` vivono sotto
`osx-poc/`, e `docker compose run` usa sempre il `WORKDIR` dell'immagine
(`/workspace`), non la cwd host da cui gira `make` — anche lanciandolo da
dentro `osx-poc/`.

Applicata la fix già raccomandata nella issue (opzione 1, non l'opzione 2
che avrebbe richiesto toccare ogni target del Makefile): aggiunto
`working_dir: /workspace/osx-poc` al servizio `osx-dev` in
`docker-compose.yml` — nessun rebuild necessario, è un parametro runtime di
compose, non dell'immagine. Applicato sia sul repo locale (Windows) sia sul
clone della Z8. **Non ancora committato/pushato** — fix presente solo nei
working tree, in attesa di conferma prima di aprire una PR.

Con il fix, `make smoke` è verde: **13/13 PASS**, nessun FAIL:

- Python 3.12, PyTorch 2.5.1+cu124, CUDA: RTX 3090 23.6GB VRAM
- CUDA tensor roundtrip OK ma 141.8ms (⚠ warning atteso: nessun pinned
  memory finché non si flippa `osx_default.yaml` — vedi passo 2 della
  sequenza di test concordata)
- vLLM 0.6.6.post1, Transformers 4.57.6, ONNXRuntime 1.18.0,
  prometheus_client 0.20.0, `src/` (eat/tier/scheduler) importabile, volume
  NVMe scrivibile

Nota: l'output dello smoke test dice ancora "Setup: Docker/Windows · RTX
3090" ed etichetta il warning sulla latenza come "atteso su Docker/Windows"
— è testo statico nello script, non riflette ancora che stiamo girando su
Linux bare-metal; non blocca nulla, ma da tenere a mente leggendo l'output
finché `smoke_test.py` non viene aggiornato.

### Stato aperto (aggiornato)

- Gate `make smoke` ora verde — sbloccato il passo 1 della sequenza di test
  concordata (re-check `perf-test-hardware`/`perf-tuning-report` +
  `bench_tier.py`/`bench_eat.py`).
- Fix issue #12 da committare/pushare (o lasciare solo locale?) — decisione
  da prendere con l'utente.
- `nvcc`/CUDA toolkit host, repo su Z8, Vast.ai: come sopra, invariato.

### Aggiornamento (stessa sessione) — re-check `perf-test-hardware`/`perf-tuning-report` (passo 1 della sequenza)

Primo run con dimensioni default (`hidden=512/intermediate=1536`, frazione
di Mixtral) — non comparabile al confronto a 3 host esistente. Rilanciato
con `OSX_BENCH_HIDDEN=4096 OSX_BENCH_INTERMEDIATE=14336` (dimensioni
Mixtral reali, stesse di
[`COMPARISON_ANALYSIS.md`](../logs/runpod_perf_framework_validation_20260817/COMPARISON_ANALYSIS.md)).
Output salvato su Z8 in
`logs/new_z8_bare_metal/perf_test_hardware_20260824/{hw_z8_baremetal,tuning_report_z8_baremetal}.json`
(non ancora copiato sul repo locale/Windows).

Confronto diretto con la vecchia Z8 (Docker-on-Windows/WSL2) e i due pod
RunPod già misurati (issue #45):

| Host | Banda RAM (GB/s) | CPU GFLOPS batch=1 | Slowdown CPU/GPU previsto |
|---|---:|---:|---:|
| Z8 vecchia (Docker-on-Windows/WSL2) | 19.84 | 22.97 | 17.1× |
| **Z8 nuova (bare-metal Linux)** | **35.83** | **31.60** | **14.0×** |
| RunPod EPYC/3090 | 10.98 | 11.66 | 36.9× |
| RunPod H200/Xeon Platinum | 5.96 | 3.32 | 569.7× |

Bare-metal migliora sia banda RAM (~1.8×) sia GFLOPS CPU (~1.38×) rispetto
al vecchio ambiente — supporta l'ipotesi che parte dell'overhead misurato
finora fosse WSL2/Docker-on-Windows, non solo il silicio. **La conclusione
di issue #33 non cambia però nella sostanza**: il forward CPU-offload resta
~14× più lento della GPU a batch=1 anche sul miglior hardware misurato
finora — ancora utile solo per risparmio VRAM, non per velocità
(`perf_tuning_report.py` lo segnala esplicitamente come coerente col range
già osservato in produzione, 24-26x pipeline reale).

Altri dati dal tuning report: PCIe pinned 7.8-8.9 GB/s (vs pageable
~3.1 GB/s), nessun tetto evidente su questo host (a differenza del sospetto
tetto WSL2/GPU-PV ipotizzato in passato su `bench_pcie_bandwidth_wsl2.py`).
Segnale batching: fino a ~10x throughput recuperabile GEMV→GEMM batched
(non implementato). Budget RAM: fino a 68 shadow-expert INT4 in cache CPU
col margine di sicurezza attuale (24GB riservati).

### Aggiornamento (stessa sessione) — `make test`, `bench_eat.py`, `bench_tier.py` (completa il passo 1)

`make test`: **210 passed, 3 skipped**, 0 fail, coverage 88%.

`bench_eat.py` (RLock/issue #2 re-check, contention disgiunta): `single`
p99=179.6µs vs `lockfree_read` p99=0.975µs → **~184×**. È una quinta cifra
per lo stesso benchmark, di nuovo diversa dalle precedenti (~1360×
originale, ~61-91× sandbox, ~348-413× RunPod, ~700-900× vecchia Z8
Windows) — coerente col pattern di varianza host-dipendente già
documentato e deliberatamente non "risolto" nel README/issue #2, non un
problema nuovo. Sotto churn (stesse chiavi): `lockfree_read` p50=0.869µs
vs `single` p50=110.1µs, torn-read rate 0.0134% (7/52289) — stesso ordine
di grandezza già noto.

`bench_tier.py`: **crash immediato** al primo tentativo —
`torch.OutOfMemoryError` dentro `bench_ddr4_to_vram()`. Investigato prima
di toccare nulla (l'utente ha chiesto esplicitamente perché si stesse
modificando codice già usato per altri test): causa reale trovata,
**aperta [issue #48](https://github.com/danielesalpietro/vMemoryFabric/issues/48)**.

Root cause: `SlabAllocator.get_buffer()` restituisce sempre lo slot fisso
da 256MB (`SHARD_SIZE_BYTES`), non i 4MB sintetici dichiarati dal
benchmark; `_ddr4_to_vram()` non evicta mai il tensore promosso
(`self._vram[key]` resta permanentemente popolato). Con `N_SHARDS=100`
(alzato da 20 in un commit **successivo** ai due run pod del 12/8 che
hanno prodotto i numeri "✅ rispettato" già citati in
`poc_final_report.md` — quei numeri vengono dalla sezione
`promote_live_tensor`, non da `ddr4_to_vram`): 100×256MB = 25.6GB,
oltre i 24GB di una singola 3090. Per quanto risulta dai log esistenti,
la sezione `ddr4_to_vram` a N=100 non era mai stata eseguita fino in
fondo su GPU reale da quando `N_SHARDS` fu alzato — bug dormiente, non
introdotto da questa sessione, non specifico di questo host.

Fix applicata in `bench_ddr4_to_vram()`: `await mgr.evict(...)` dopo ogni
`promote()` misurata (fuori dalla finestra `t0`/`latencies_us`, non
altera la latenza), footprint VRAM steady-state ~1 shard invece di
accumulare tutti gli N_SHARDS. Non tocca `promote_live_tensor`, quindi
non retroattiva sui numeri già pubblicati. Applicata sia sul repo locale
sia sul clone Z8 (non ancora committata, come il fix di #12).

Ri-eseguito con successo dopo la fix:

| Sezione | P50 | Note |
|---|---:|---|
| `promote_live_tensor` pin=True | 196.99µs | within 1.5× target: **true** — coerente coi 194.1/207.8µs già citati nel report |
| `promote_live_tensor` pin=False | 1552.83µs | within 1.5× target: false (atteso, path pageable storico) |
| `ddr4_to_vram` (N=100, con evict) | 84049µs | prima misura completa mai riuscita per questa sezione a N=100 |
| `nvme_to_ddr4` (N=100) | 4128µs | — |

### Stato aperto (aggiornato)

- **Passo 1 completo**: re-check `perf-test-hardware`/`perf-tuning-report`,
  `make test`, `bench_eat.py`, `bench_tier.py` tutti fatti.
- Due fix non ancora committate/pushate, entrambe solo nei working tree
  (locale + clone Z8): issue #12 (`docker-compose.yml` working_dir) e
  issue #48 (`bench_tier.py` eviction). Decisione da prendere su PR.
- JSON/log dei risultati di oggi non ancora copiati dal Z8 al repo
  locale/CI — da decidere se vanno committati come nuovo dataset di
  confronto (4° host) o restano solo artefatti locali sulla Z8.

### Aggiornamento (stessa sessione) — passo 2 saltato (non era un flag), passo 3 fatto (NUMA pinning)

Prima di toccare `osx_default.yaml`, controllato come `pinned_memory` e
`use_io_uring` sono davvero usati nel codice — nessuno dei due è un flag
letto da qualcosa:

- **Pinning**: deciso a runtime da `GCSGWorker._should_pin_transfers()`
  (`scheduler/gcsg.py:1901`) via `vllm.platforms.interface.in_wsl()` —
  automatico su Linux reale, già attivo su questa Z8 senza toccare
  config (confermato dal `pin_True` di `bench_tier.py` sopra).
- **io_uring**: non implementato per niente — `AsyncNVMeIO` (`tier/io.py`)
  usa sempre `aiofiles`/asyncio, il flag in config non è letto da nessuna
  parte. Implementarlo per davvero è sviluppo vero (binding liburing/
  aiouring, nuovo backend), non chiudibile in questa sessione.

Deciso con l'utente: saltare il passo 2 (nulla da flippare, pinning già
confermato attivo), passare al passo 3.

**Passo 3 — benchmark CPU-offload pinnato NUMA** (issue #45): `numactl`
non presente nell'immagine, installato al volo nel container
(`apt-get install numactl`, non persistente — da reinstallare ad ogni
`docker compose run` finché non entra nel `Dockerfile`, se si deciderà
di tenerlo). `numactl --membind` inizialmente bloccato
(`set_mempolicy: Operation not permitted`) — servita la capability
`--cap-add SYS_NICE` su `docker compose run` per sbloccarlo.

Rilanciato `perf-test-hardware`/`perf-tuning-report` con
`numactl --cpunodebind=0 --membind=0` (nodo 0, stesso nodo della GPU),
dimensioni Mixtral reali, confrontato con il run non pinnato di oggi:

| Config | Banda copy RAM | CPU GFLOPS batch=1 | Slowdown CPU/GPU previsto |
|---|---:|---:|---:|
| Non pinnato (32 core, cross-NUMA) | 35.83 GB/s | 31.60 | 14.0× |
| NUMA-pinned nodo0 (16 core, mem locale) | 21.33 GB/s | 33.23 | 12.9× |

Risultato controintuitivo ma spiegabile: pinnare al nodo della GPU
**riduce** la banda copy aggregata (metà core/canali disponibili) ma
**migliora leggermente** il GFLOPS reale del forward (+5%) e abbassa lo
slowdown (14.0×→12.9×) — per il workload GEMV reale (memory-bandwidth-
bound ma anche latency-sensitive) conta evitare l'hop cross-socket UPI
più che avere banda aggregata massima. Miglioramento reale ma piccolo
(~8%), non cambia la conclusione di fondo di issue #33 (cpu-offload utile
solo per risparmio VRAM, non per velocità).

Segnale parziale per issue #45 (non risolutivo): qui forzare un singolo
nodo NUMA riduce la banda copy ma non peggiora — anzi migliora
leggermente — il throughput GEMV reale. Indizio che il problema
osservato sul pod H200 (banda 5.96 GB/s, GFLOPS 3.32) non fosse
necessariamente "banda aggregata bassa" di per sé, ma qualcos'altro
nell'allocazione vCPU condivisa di quel pod specifico — non verificabile
da qui, servirebbe rimisurare direttamente su un pod H200 nuovo con lo
stesso controllo NUMA.

Output salvati su Z8:
`logs/new_z8_bare_metal/perf_test_hardware_20260824/{hw_z8_baremetal_numa0,tuning_report_z8_baremetal_numa0}.json`.

Topologia confermata dall'utente: 8C/16T per socket × 2 socket (16 core
fisici/32 thread totali), GPU fisicamente su CPU0 — coerente con
`nvidia-smi topo -m` (GPU0 NUMA affinity nodo 0) già usato per il
pinning sopra, nessuna correzione necessaria al pinning stesso.

**Caveat trovato controllando questo dettaglio**: `cgroup_cores_available`
nel report NUMA-pinned mostra ancora **32**, non 16. `_read_cgroup_cpu_count()`
(`perf_test_hardware.py`) legge la quota cgroup (`/sys/fs/cgroup/cpu.max`,
qui "max", nessun limite impostato sul container) con fallback a
`os.cpu_count()` — nessuno dei due vede l'affinità impostata da
`numactl --cpunodebind`, che è scheduling affinity, non quota cgroup.
Conseguenza: `perf_tuning_report.py` raccomanda comunque
`omp_mkl_num_threads_for_pool_build: 32` anche sotto pinning a un solo
socket (dove sono schedulabili solo 16 thread) — oversubscription 2:1 se
qualcuno impostasse davvero `OMP_NUM_THREADS=32` in quelle condizioni. I
GFLOPS misurati restano validi (throughput realmente osservato sotto la
restrizione), ma la raccomandazione di thread count del tuning report è
cieca al NUMA pinning — non corretto in questa sessione (richiede
decidere se leggere `os.sched_getaffinity(0)` oltre alla quota cgroup,
una scelta di design, non un one-liner come #12/#48). Aperta
[issue #49](https://github.com/danielesalpietro/vMemoryFabric/issues/49)
per tracciare la revisione di design (non un fix immediato).

### Stato aperto (aggiornato) — fine sessione

- Passi 1 e 3 della sequenza concordata completi. Passo 2 chiuso come
  "non applicabile" (non era un flag reale). Passo 4 (PMEM/EMH-2) resta
  da fare — richiede sviluppo (mkfs+mount dax, nuovo modulo tier),
  non ancora iniziato.
- Fix committate su branch dedicati da `develop` e PR aperte:
  [PR #50](https://github.com/danielesalpietro/vMemoryFabric/pull/50)
  (issue #12, `fix/issue-12-docker-workdir`),
  [PR #51](https://github.com/danielesalpietro/vMemoryFabric/pull/51)
  (issue #48, `fix/issue-48-bench-tier-oom`) — non ancora mergiate.
- `numactl` installato solo ad-hoc nel container corrente, non nel
  `Dockerfile` — da decidere se aggiungerlo in modo permanente se il
  NUMA pinning diventa parte del workflow standard.
- JSON/log della sessione odierna (perf-test-hardware, tuning-report,
  bench_eat, bench_tier, NUMA) tutti su Z8 in `logs/new_z8_bare_metal/`,
  non ancora copiati/committati nel repo locale.

### Aggiornamento (stessa sessione) — passo 4: implementazione tier PMEM (EMH-2)

Sviluppo vero, non un flag: mount DAX + nuovo modulo tier + aggancio a
TierManager + nuovi test + nuovo benchmark.

**Mount**: `/dev/pmem1` formattato XFS e montato con `mount -o dax`
(`dax=always` confermato) su `/mnt/pmem_emh2` (248G, 244G liberi).
Persistito in `/etc/fstab` (sopravvive a reboot). Bind-mount nel
container via `docker-compose.override.yml` (gitignored, host-specific —
`.example` committato nel repo per documentare il pattern) su
`/data/pmem`.

**`src/tier/pmem.py`** (nuovo): `PMEMTransfer`, pool allocator a slot
fissi come `SlabAllocator` (DDR4) ma backed da un file mmap-ato
(`numpy.memmap`) sul mount DAX invece che DRAM anonima. Pool file
pre-allocato con `posix_fallocate` (evita file sparse: un buco allocato
lazy al primo write avrebbe confuso qualunque benchmark di throughput).
Limitazione dichiarata nel docstring: mmap normale, non `MAP_SYNC` — dà
accesso byte-addressable reale ma non la garanzia di durabilità
fsync-free che `MAP_SYNC` darebbe (non esposto da Python `mmap` oggi).

**`eat/types.py`**: `Tier.PMEM = 3` aggiunto (non rinumerato tra DDR4/NVME
per non toccare un valore già in uso altrove).

**`tier/manager.py`**: `TierManager.__init__` accetta `pmem_path`/
`pmem_n_slots` opzionali (`None` = tier disabilitato, comportamento
invariato per tutti gli altri ambienti dev). Nuovi hop `promote()`:
NVME→PMEM, PMEM→DDR4 (single-hop ciascuno, non incatenati in un percorso
NVME→VRAM a più hop — scelta di scope, vedi docstring). Nuovo hop
`evict()`: PMEM→NVME. **NVME→DDR4 diretto resta invariato e disponibile**
a prescindere da `pmem_path` — PMEM è una rotta aggiuntiva, non una
sostituzione (verificato con test dedicato).

**Bug trovato e fixato nel proprio codice nuovo, stesso pattern di issue
#48**: `PMEMTransfer.read()` inizialmente ritornava l'intero slot fisso
(256MB) invece del payload reale scritto (4MB nel benchmark) — identica
causa radice di #48 (`SlabAllocator.get_buffer()` fa la stessa cosa),
questa volta scoperta *prima* di essere segnalata come issue, controllando
un numero sospetto (`pmem_to_ddr4` P50=116ms per uno shard sintetico da
4MB, matematicamente vicino a 256MB/banda-letta-misurata anziché
4MB/banda). Fix: `read()` ora taglia alla `size_bytes` reale tracciata in
`alloc()`, non all'intero slot — evita di ripetere il bug invece di
scoprirlo dopo. Nessuna issue aperta per questo: fixato prima del commit,
mai arrivato in uno stato "pubblicato".

**Test** (`tests/test_tier.py`): `TestPMEMTransfer` (8 test, portabili —
mmap su file normale si comporta identicamente a DAX per la
correttezza, nessun `@pytest.mark.gpu`) + `TestTierManagerPMEM` (4 test:
NVME→PMEM, PMEM→DDR4 con verifica byte-a-byte del payload, NVME→DDR4
diretto invariato, evict PMEM→NVME) + 2 test in `TestTierManager` per il
caso `pmem_path=None`. Suite completa: **224 passed, 3 skipped** (invariati).

**`benchmarks/bench_pmem_tier.py`** (nuovo, `make bench-pmem-tier`): tre
sezioni — `raw_bandwidth` (buffer 512MB, stessa dimensione di
`perf_test_hardware.py bench_ram()` per confrontabilità), `nvme_to_pmem`,
`pmem_to_ddr4` (shard sintetici 4MB, stessa convenzione di
`bench_tier.py`). A differenza di `bench_ddr4_to_vram()` non serve evict
nel loop: PMEM (252GB) e DDR4 (~236GB) hanno ampio margine per tenere
tutti gli N_SHARDS=100 residenti insieme (25.6GB ciascuno), a differenza
della VRAM 24GB che aveva causato #48.

Risultati (dopo il fix di `read()`):

| Sezione | Valore |
|---|---:|
| raw_bandwidth write | 0.61 GB/s |
| raw_bandwidth read | 1.85 GB/s |
| nvme_to_pmem P50 (shard 4MB) | 5.41 ms |
| pmem_to_ddr4 P50 (shard 4MB) | 2.73 ms |

**Caveat onesto sui numeri raw_bandwidth**: singola misura senza warm-up
né ripetizioni (a differenza di `perf_test_hardware.py bench_ram()` che
comunque è anch'esso single-shot) — 0.61GB/s in scrittura è più basso
delle specifiche tipiche Optane DC (ordine 2-3GB/s per DIMM). Possibili
fattori non isolati: conversione unwritten-extent XFS al primo write
anche dopo `posix_fallocate`, nessun controllo NUMA (non verificato se
region1/PMEM è sullo stesso socket della CPU che esegue il benchmark),
nessuna ripetizione per scartare l'effetto first-touch. Non
approfondito oltre in questa sessione — numero riportato come prima
caratterizzazione, non come dato definitivo. Buon candidato per un
prossimo giro se questi numeri diventano rilevanti per una decisione.

File coinvolti: `src/tier/pmem.py` (nuovo), `src/eat/types.py`,
`src/tier/manager.py`, `src/tier/__init__.py`, `tests/test_tier.py`,
`benchmarks/bench_pmem_tier.py` (nuovo), `osx-poc/Makefile`,
`docker-compose.override.yml.example` (nuovo, committato),
`docker-compose.override.yml` (nuovo, gitignored), `.gitignore`.
Non ancora committato su nessun branch — tutto nei working tree
(locale + Z8).

Output JSON: `logs/new_z8_bare_metal/pmem_tier_20260824/bench_pmem_tier.json`
sulla Z8, non ancora copiato/committato nel repo locale.
