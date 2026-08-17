# Malmö RunPod — issue #33, real-hardware validation (2026-08-17)

Sessione di validazione su hardware reale (non WSL2) per issue #33 (DDR4
compute-offload) — dettagli completi in `LOGBOOK_ISSUE33.MD`. Questa
cartella raccoglie tutti gli artefatti recuperati dal pod prima dello
shutdown, stesso principio già seguito per `logs/sprint4_tekniska/`.

## Pod

```
Nome:            vMemoryFabric - Malmö - CPU Offload DDR
GPU:             RTX A6000 x1 (46068 MiB VRAM, CC 8.6)
CPU:             Intel Xeon Gold 6342 (Ice Lake-SP) — AVX-512F/BW/VL/VNNI
                 confermati via /proc/cpuinfo, non solo dalla scheda
                 prodotto
vCPU allocate:   9 nominali — quota cgroup reale 7.65 core
                 (/sys/fs/cgroup/cpu.max = "765000 100000")
RAM:             503 GiB (host, visibile al container)
Container disk:  50 GB (overlay)
Pod volume:      128 GB, mount /data/nvme (MooseFS, eu-se-1)
Prezzo:          $0.55/hr totale
```

Dettagli completi: `pod_config/` (lscpu, free, nvidia-smi, os-release,
uname, df, pip freeze, git state, quota cgroup).

## Cosa è stato eseguito

**Piano concordato**: Punto 2 (regressione baseline, `--wire-tier-manager`
senza `--enable-cpu-offload`, fette 16→32→64→570, ripetuto ×2) + Punto 3
(conferma esplicita del no-op di `--enable-cpu-offload` sul path 2/3
reale, fette 16 e 32).

**Bug ambientale trovato e aggirato**: caricamento checkpoint via mmap
(safetensors/vLLM) si blocca indefinitamente sul volume di rete
MooseFS/FUSE (`/data/nvme`) — confermato NON causato dal nostro codice
(un run di controllo identico, senza alcun flag di issue #33, si blocca
allo stesso modo). `dd` sullo stesso file legge a 4-5 GB/s: il problema è
specifico all'mmap su FUSE, non al filesystem in generale. Workaround:
checkpoint copiato su disco locale del container (`/root/models/...`),
nuovo override `OSX_MMLU_MODEL_PATH` aggiunto a `eval_mmlu_gcsg.py`
(commit `b719111`).

## Risultati

### Punto 2 — regressione baseline (`--wire-tier-manager`, cpu-offload OFF)

| Fetta | Round 1 | Round 2 | Baseline storico |
|---|---|---|---|
| 16 | 50.0% (8/16) | 50.0% (8/16) — identico | — (campione troppo piccolo per confronto) |
| 32 | 65.6% (21/32) | 65.6% (21/32) — identico | — |
| 64 | 70.3% (45/64) | 70.3% (45/64) — identico | — |
| 570 | **72.1% (411/570)** | non rilanciato (esito atteso, round 1-3 già concordanti) | 72.3% (412/570), 2026-08-09 |

Determinismo perfetto tra round 1 e round 2 a tutte le fette testate —
non solo "vicino", identico byte-per-byte (stesso `shadow_activations`,
stessa `activation_rate`). Il run completo conferma: nessuna delle
modifiche di Fase 0-5 (issue #33) ha degradato il path di produzione già
validato.

### Punto 3 — conferma no-op di `--enable-cpu-offload`

| Fetta | Senza flag | Con `--enable-cpu-offload` | Diff |
|---|---|---|---|
| 16 | 50.0% (8/16), shadow_activations=10081 | 50.0% (8/16), shadow_activations=10081 | **nessuna** |
| 32 | 65.6% (21/32), shadow_activations=23665 | 65.6% (21/32), shadow_activations=23665 | **nessuna** |

Confermato empiricamente (non solo per lettura del codice): il flag è
un no-op sicuro sul checkpoint reale (path 3, AWQ ModuleList) — nessuna
differenza misurabile in nessuna metrica.

## Telemetria GPU

`misc/gpu_telemetry.csv` — 547 campioni, ogni 5s, dalle 00:32 alle 01:18
UTC (intera sessione).

- Picco utilizzo GPU: **99%** (durante i `generate()` reali)
- Picco VRAM: **43729 MiB** / 46068 MiB (combacia col budget 0.95 di
  `gpu_memory_utilization` di vLLM: 18.95GB pesi + 21GB KV-cache + overhead)
- Potenza media: 124.9W (media su TUTTA la sessione, inclusi i lunghi
  periodi di inattività tra un run e l'altro — non rappresentativa del
  consumo sotto carico attivo, che tocca ~290W nei picchi)

## File in questa cartella

- `pod_config/` — snapshot HW/env completo (stesso schema di
  `logs/sprint4_tekniska/pod_config/`)
- `runs/` — log completi di ogni run (`run_r1_*.log`, `run_r2_*.log`,
  `run_offload_*.log`, `run_control_*.log`)
- `results/` — output JSONL strutturato di ogni run
- `misc/gpu_telemetry.csv` — telemetria GPU continua per tutta la sessione

## Cosa manca per attivare davvero `--enable-cpu-offload`

Vedi `LOGBOOK_ISSUE33.MD`, sessione 2026-08-17 (continued) — "Fase 6":
il checkpoint reale usa path 3 (AWQ ModuleList) o path 2 (Marlin), mai
path 1 (l'unico coperto da `_build_cpu_shadow_pool()`, Fase 2). Serve un
dequantizzatore AWQ lato CPU per path 3 (fattibile, scope chiaro); path 2
(Marlin) è più difficile (formato bit-interleaved specifico del kernel
GPU) e a priorità più bassa.
