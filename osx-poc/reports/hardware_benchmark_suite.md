# Hardware Benchmark Suite — validazione bottom-up dell'infrastruttura sotto OSX/vMemoryFabric

**Status:** Nota di metodologia / riferimento, **non un report sperimentale** — nessuno dei
tool elencati qui è stato eseguito in questo repository. Scopo: fissare quali benchmark
"standard de facto" del settore vanno usati per validare l'infrastruttura fisica (memoria,
CPU, GPU, storage) su cui girano i moduli applicativi di OSX (M1 EAT, M2 Tier Manager, M3
Scheduler), e come si relazionano ai benchmark applicativi già presenti nel repo
(`benchmarks/bench_eat.py`, `benchmarks/bench_tier.py`).

**Data:** 2026-08-18
**Progetto:** OSX — Operating System for Experts (repo: `vMemoryFabric`)
**Vedi anche:** [`README.md`](../../README.md) §"Dev environment constraints" per lo stato
reale (dev vs. target) dell'hardware citato qui sotto.

---

## 1. Perché un layer di benchmark separato da `bench_eat.py`/`bench_tier.py`

I benchmark applicativi del repo misurano la **logica di OSX** — latenza di lookup EAT,
latenza di promozione tier, contesa di lock — assumendo che il bus di memoria, il link
PCIe e il volume NVMe sotto di essi si comportino secondo le specifiche teoriche del
vendor. Quell'assunzione non è mai stata verificata direttamente in questo repo: se
`bench_ddr4_to_vram` misura 194µs di p50, non sappiamo se quel numero è vicino al limite
fisico del link PCIe di questa macchina o se c'è un 2-3× di margine perso a una
configurazione NUMA sbagliata, a RDIMM non alla frequenza dichiarata, o a un volume NVMe
che non tiene la coda IOPS attesa.

La prassi standard del settore per questo tipo di verifica è **bottom-up**: isolare ogni
sottosistema fisico con un tool dedicato, verificarlo contro le specifiche teoriche, e solo
*dopo* fidarsi dei numeri prodotti da un benchmark applicativo che vive sopra quello stack.
Le sezioni 2-4 elencano i tool de facto per ciascun sottosistema; la §5 li mette in ordine
di esecuzione; la §6 li mappa contro lo stato reale hardware/software di questo progetto
(vedi README).

---

## 2. Sottosistema Memoria e CPU

| Tool | Cosa misura | Perché conta per OSX |
|---|---|---|
| **[STREAM Benchmark](https://www.cs.virginia.edu/stream/)** | Sustained memory bandwidth (MB/s) — Copy/Scale/Add/Triad su array grandi quanto la cache | È il limite fisico su cui sbatte ogni transfer DDR4-bound, incluso l'hop NVMe→DDR4 di `TierManager.promote()` e il caricamento degli shard "warm" in EMH-1c |
| **[Intel MLC (Memory Latency Checker)](https://www.intel.com/content/www/us/en/download/736633/intel-memory-latency-checker-intel-mlc.html)** | Latenza e banda incrociata, per-NUMA-node | Rileva se i thread di `TierManager`/`GCSGWorker` leggono da un banco RAM remoto rispetto al socket che li esegue — una misconfigurazione NUMA silenziosa che nessun benchmark applicativo del repo può distinguere da "overhead intrinseco del codice" |
| **[HPL (High-Performance Linpack)](https://www.netlib.org/benchmark/hpl/)** | TFLOPS su sistemi lineari densi | Meno direttamente rilevante per OSX (che è memory-bound, non compute-bound), ma è il baseline storico per certificare che la CPU non sia in throttling termico prima di attribuire una regressione a un cambiamento software |
| **[HPCG (High Performance Conjugate Gradients)](https://www.hpcg-benchmark.org/software/)** | Pattern di accesso memoria irregolari, più vicini a carichi reali (reti neurali, gather/scatter) | Più rappresentativo di HPL per il pattern di accesso di un `SlabAllocator`/EAT reale — accessi sparsi per `expert_id`/`shard_idx`, non sequenziali |

---

## 3. Sottosistema GPU e AI

| Tool | Cosa misura | Perché conta per OSX |
|---|---|---|
| **[NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests)** (`all_reduce_perf`, `sendrecv_perf`) | Banda passante reale GPU↔GPU e GPU↔host, oltre i dati di targa PCIe/NVLink | È esattamente il tipo di misura che manca oggi al path DDR4→VRAM di `GPUTransfer` — la differenza pin=True/pin=False misurata in `bench_tier.py::bench_promote_live_tensor` (194µs vs 684µs p50) è plausibile ma non è mai stata confrontata contro il tetto fisico del link PCIe di questa macchina, solo contro un numero teorico hardcoded (`_PCIE_GEN3_BANDWIDTH_BYTES_PER_SEC` in `bench_tier.py`) |
| **[MLPerf (MLCommons)](https://github.com/mlcommons/inference)** ([Training](https://github.com/mlcommons/training)) | Metriche applicative standardizzate — Tokens/second, Time-to-First-Token, su modelli reali pre-configurati | Standard di settore per confrontare OSX con altri sistemi di inference (vedi anche `related_work_petals_exllama.md`), ma richiede una suite Training/Inference completa — fuori scope per il PoC attuale, rilevante per Sprint 5/Berg (paper) se si vuole un numero comparabile pubblicabile |
| **[DCGM](https://github.com/NVIDIA/DCGM)** (`dcgmproftester`) | Stress al 100% TDP per ore — certifica che alimentazione/raffreddamento reggano senza thermal throttling | Diagnostico, non un benchmark applicativo: rileva se una regressione di latenza osservata da `bench_tier.py` è in realtà throttling termico della GPU, non un problema di `TierManager` |

---

## 4. Storage e I/O

| Tool | Cosa misura | Perché conta per OSX |
|---|---|---|
| **[FIO (Flexible I/O Tester)](https://github.com/axboe/fio)** | IOPS/latenza per pattern di I/O configurabile (letture 4K random, high queue depth, ecc.) | Direttamente rilevante per `AsyncNVMeIO` (M2) e per `bench_nvme_to_ddr4`: senza un numero FIO di riferimento per il volume NVMe usato in dev, non c'è modo di sapere se la latenza NVMe→DDR4 misurata riflette il volume fisico o l'overhead di `asyncio`/`aiofiles` (vedi README, "Perché `asyncio + aiofiles` invece di `io_uring`") |

---

## 5. Ordine di esecuzione raccomandato

Prassi bottom-up standard, applicata a questo progetto:

1. **STREAM** — esclude problemi ai banchi DDR4 prima di fidarsi di qualunque numero DDR4-bound
2. **Intel MLC** — esclude una misconfigurazione NUMA, se/quando il deployment target è multi-socket (non il caso della Z8 G4 dev, socket singolo)
3. **FIO** — stabilisce il tetto fisico IOPS/latenza del volume NVMe usato da `AsyncNVMeIO`
4. **NCCL Tests** — stabilisce il tetto fisico PCIe/NVLink per il path DDR4→VRAM, quando c'è più di una GPU da cui misurare un `sendrecv_perf` reale (oggi non c'è, vedi §6)
5. Solo a questo punto, i numeri applicativi di `bench_eat.py`/`bench_tier.py` sono interpretabili contro un tetto fisico noto, non contro un numero teorico hardcoded
6. **HPL/HPCG/MLPerf/DCGM** — test applicativi/di certificazione aggregati, da lanciare per un final sign-off (es. prima di un tagged release o del paper Sprint 5/Berg), non ad ogni iterazione

---

## 6. Applicabilità allo stato reale del progetto (dev vs. target)

Incrociando la tabella "Dev environment constraints" del README con i tool sopra:

| Tool | Eseguibile oggi (Z8 G4, RTX 3090 singola, Docker-on-Windows) | Note |
|---|---|---|
| STREAM | ✅ sì | Nessuna dipendenza da GPU/NUMA multi-socket |
| Intel MLC | ⚠️ parzialmente | Socket singolo in dev — utile solo come baseline pre-deployment multi-socket, non rileva nulla di NUMA-specifico qui |
| HPL / HPCG | ✅ sì (CPU-only) | Non memory-bound come il resto della suite di OSX — utile solo come sign-off generico, non come diagnostica mirata |
| FIO | ✅ sì | Direttamente applicabile al volume NVMe usato da `AsyncNVMeIO` — il candidato con il ritorno più immediato per issue #3 (`bench_tier.py` p95/p99 skewed) |
| NCCL Tests | ❌ bloccato | Richiede ≥2 GPU — stesso blocco hardware di issue [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) (AER/dual-GPU, RTX 5080 in arrivo) |
| MLPerf | ❌ fuori scope PoC | Richiede una suite Training/Inference completa su modelli standardizzati — non pianificato prima di Sprint 5/Berg (paper) |
| DCGM | ⚠️ parzialmente | `dcgmproftester` richiede il DCGM host engine, non incluso nell'immagine Docker-on-Windows attuale — utile solo su bare-metal Linux target |

Nessuno di questi tool è oggi wired in `osx-poc/Makefile` o in CI: restano una checklist di
validazione manuale da eseguire quando l'infrastruttura fisica cambia (nuova macchina,
nuovo volume NVMe, arrivo della seconda GPU) o prima di un sign-off importante (release
taggata, numeri pubblicati nel paper), non un sostituto dei benchmark applicativi esistenti.

---

## 7. Prossimi passi (non ancora eseguiti)

- FIO contro il volume NVMe di dev — il candidato più diretto, collegabile a issue #3
- STREAM sulla Z8 G4 — stabilisce il tetto teorico per interpretare `bench_nvme_to_ddr4`/`bench_ddr4_to_vram`
- NCCL Tests — bloccato fino all'arrivo della seconda GPU (issue #8)
- MLPerf/DCGM — da valutare in sede di pianificazione Sprint 5/Berg, non prima

---

## 8. Licenze

Tutti i tool sopra sono gratuiti nell'uso; la distinzione è tra open source e
proprietario-ma-free:

| Tool | Licenza | Costo |
|---|---|---|
| STREAM | Open source (custom academic license, John McCalpin) | ✅ Gratuito |
| Intel MLC | Proprietario, gratuito — EULA Intel con restrizioni su redistribuzione/pubblicazione di risultati comparativi | ✅ Gratuito |
| HPL | Open source (Netlib) | ✅ Gratuito |
| HPCG | Open source (Sandia National Labs) | ✅ Gratuito |
| NCCL Tests | Open source (NVIDIA, BSD-style) | ✅ Gratuito |
| MLPerf (MLCommons) | Open source — solo la sottomissione di risultati ufficiali richiede membership MLCommons a pagamento | ✅ Gratuito per l'uso interno (submission ufficiale = a pagamento) |
| DCGM | Proprietario, gratuito — architettura open-core: librerie/binding open source su GitHub, alcuni tool di diagnostica restano proprietari | ✅ Gratuito |
| FIO | Open source (GPLv2) | ✅ Gratuito |

Nessuno dei tool elencati in questo report richiede una licenza commerciale per essere
scaricato ed eseguito localmente — anche quelli proprietari (Intel MLC, DCGM) sono
distribuiti a costo zero dal rispettivo vendor. L'unico costo reale nell'intera suite è
opzionale e riguarda MLPerf: la *sottomissione* di un risultato alla classifica ufficiale
MLCommons richiede una membership a pagamento, ma eseguire il benchmark per misurazioni
interne (come previsto qui) resta gratuito.

Nota: l'URL di download di Intel MLC contiene un ID interno alla pagina Intel che può
cambiare nei redesign del sito — se il link smette di funzionare, cercare "Intel Memory
Latency Checker" dal developer portal Intel lo ritrova comunque.
