# Z8 locale — telemetria Fase 6a cpu-offload, prima run reale su path AWQ (2026-08-17)

Raccolta durante il primo tentativo di eseguire `--enable-cpu-offload`
genuinamente attivo sul path AWQ reale (issue #33 Fase 6a Passo 3,
commit `652b31a`), in preparazione del pattern di validazione
16→32→64→full. Nessuno dei due tentativi di run (n=16) è arrivato a un
risultato — vedi `LOGBOOK_ISSUE33.MD`, entry "continued 15" e la sua
correzione, per il racconto completo e la causa identificata
(`pin_memory=False` sotto WSL2 — stessa patologia già diagnosticata e
risolta per questo progetto nelle sessioni 2026-08-11/12, non un bug
nuovo di Fase 6a).

## File

| File | Cadenza | Righe finali | Contenuto |
|---|---|---|---|
| `host_perf.csv` | 10s | 331 | CPU% totale host, RAM used/free/total (Windows, PowerShell `Win32_PerfFormattedData_PerfOS_Processor`/`Win32_OperatingSystem`) |
| `gpu_telemetry.csv` | 10s | 331 | `nvidia-smi --query-gpu` — util GPU/mem, VRAM used/total, power, temp |
| `pcie_throughput.csv` | 5s | 363 | `nvidia-smi dmon -s t` — RX/TX PCIe in MB/s (funziona su questa RTX 3090 nonostante alcune restrizioni NVML su GeForce) |
| `docker_stats.csv` | 10s | 267 | `docker stats --no-stream` per container attivo |
| `host_sampler.ps1` / `pcie_sampler.ps1` | — | — | script sorgente dei due sampler PowerShell |

Timestamp ISO 8601 con offset (`+02:00`) su tutte le righe — pensati per
essere incrociati con i timestamp `[T+Ns]` nei log applicativi
(`logs/z8_local/mmlu_fase6a_cpu_offload/`).

## Finestra temporale coperta

- Sampler host/GPU: 2026-08-17 13:54:52 → 14:53:31 (fermato manualmente)
- Sampler PCIe: 2026-08-17 14:22:25 → 14:53:34 (fermato manualmente,
  avviato più tardi degli altri due — aggiunto a metà sessione su
  richiesta esplicita)
- Sampler docker stats: 2026-08-17 13:55:03 → 14:47:00 (ultima riga
  utile; il processo è rimasto attivo più a lungo ma senza container da
  riportare dopo quel punto, nessuna riga nuova prodotta)

Copre entrambi i tentativi di run n=16 (il primo non-thread-tuned,
killato manualmente dopo 23.5 min; il secondo con
`OMP_NUM_THREADS=28`/`MKL_NUM_THREADS=28`, arrivato al timeout interno
dello script a 30 min e poi killato da fuori con `docker kill` dopo che
il self-SIGKILL dall'interno del container non ha funzionato).

## Letture chiave (dettaglio nel LOGBOOK)

- **Host CPU totale**: quasi sempre sotto il 26%, spesso a singola
  cifra — il carico non si è mai distribuito sull'host nonostante i 28
  core allocati a Docker/WSL2.
- **Container CPU**: ~270% senza thread tuning, picchi 350-440% (con
  cali periodici a ~100%) dopo `OMP_NUM_THREADS=28`/`MKL_NUM_THREADS=28`
  — miglioramento parziale, mai vicino a un uso pieno dei 28 core.
- **GPU**: VRAM stabile ~23.6/24GB per tutta la finestra attiva (nessuna
  nuova allocazione dopo il caricamento iniziale), utilizzo compute
  perlopiù basso (2-29%) con rari picchi al 100% (singoli step di
  decode), potenza quasi sempre ~41W (idle) salvo quei picchi.
- **PCIe**: traffico bursty, non uno stream continuo — punte fino a
  ~2.3 GB/s TX / ~1.1 GB/s RX, ma intervallate da campioni quasi a zero.
  Non ancora attribuito con certezza a una causa singola (vLLM
  `cpu_offload_gb=4` nativo vs. `route_forward()` di GCSG vs. altro) —
  serve `osx-poc/scripts/bench_pcie_bandwidth_wsl2.py` (creato, non
  ancora eseguito) per isolare la banda reale disponibile da quella
  effettivamente usata dal workload attuale.
- **RAM**: mai sotto pressione — 364-418GB liberi su 503.7GB per tutta
  la finestra, ben lontano dalla soglia dell'80% di uso DDR4 che, per
  indicazione del project owner, fa scattare il fallback su PMEM
  (Intel Optane DC, confermato installato e configurato al 100% in
  Memory Mode — `Get-PmemPhysicalDevice` mostra tutti e 4 i moduli
  NMA1XXD128GPS a "Persistent memory size: 0 GB").
