# RunPod (RTX3090) — issue #33, prima run cpu-offload reale completata (2026-08-17)

Sessione di validazione su hardware reale (non WSL2) per issue #33 (DDR4
compute-offload, Fase 6a) — dettagli completi in `LOGBOOK_ISSUE33.MD`,
entry "continued 17". Questa cartella raccoglie tutti gli artefatti
recuperati dal pod, stesso principio già seguito per
`logs/malmo_runpod/` e `logs/sprint4_tekniska/`.

## Pod

```
GPU:             RTX 3090 (24576 MiB VRAM)
CPU:             AMD EPYC 7C13 (Zen3) — NESSUN avx512* in /proc/cpuinfo
                 (diverso dallo Xeon Ice Lake di Malmö)
vCPU:            quota cgroup reale 27.2 core (cfs_quota_us=2720000,
                 cfs_period_us=100000) — nproc riporta 256 (topologia
                 host, non la quota — vedi BOOTSTRAP §5.9)
RAM:             limite cgroup reale ~116.4GiB (free -h riporta 1.0Ti,
                 stesso motivo del punto sopra)
Checkpoint:      casperhansen/mixtral-instruct-awq, scaricato da HF
                 direttamente su disco locale container in 28.96s
```

Dettagli completi: `pod_config/` (lscpu, free, nvidia-smi, os-release).

## Cosa è stato eseguito

Cinque tentativi di run n=16 con `--wire-tier-manager --enable-cpu-offload
--quantization awq`, iterando sul codice tra un tentativo e l'altro (non
solo ripetendo lo stesso comando):

| Run | Esito | Causa/scoperta |
|---|---|---|
| v1 | Fallito, watchdog 600s, 0/16 | Nessun `pin_memory` warning — non è WSL2. GPU 0%, RAM cgroup all'84% |
| v2 | Fallito, watchdog 900s, 0/16 | Tentato `py-spy dump` — bloccato (manca SYS_PTRACE, container RunPod) |
| — | — | Aggiunto timing diagnostico + cap RAM-aware nel codice |
| v4 | Fallito, watchdog 1800s, 0/16 | **Scoperta chiave**: `logging.basicConfig()` mancava nello script — tutte le righe `log.info()` di `scheduler.gcsg` (mai viste in nessun run precedente) andavano perse. Fix applicato. Con logging funzionante: pool CPU si costruisce in ~88s, VELOCE, completa prima di "LLM ready" — non era mai il collo di bottiglia |
| — | — | Aggiunto contatore/timing al forward per-token (`_ShadowExpertINT4.__call__`) |
| v5 | **COMPLETATO**, 1133.1s | Vero collo di bottiglia isolato: forward a singola riga (1×4096), ~0.10s/chiamata, migliaia di chiamate |
| thread2 | Test A/B, fermato dopo 200 campioni | `OMP_NUM_THREADS=2` vs `=27`: stesso avg (~0.10s/call) — **thread-oversubscription falsificato come causa** |

## Risultati numerici (run v5, l'unico completato)

```json
{"total": 16, "correct": 8, "accuracy": 0.5, "shadow_activations_cumulative": 10081,
 "elapsed_s": 1133.098283892963, "cpu_offload_enabled": true, "quantization": "awq"}
```

- **Correttezza**: `shadow_activations=10081` e accuracy 50.0% — **identici
  byte-per-byte** al baseline Malmö dove `--enable-cpu-offload` era un
  no-op (stesso n=16, stessa selezione expert `[0, 1]`). Prova diretta su
  hardware reale che il dequant AWQ CPU di Fase 6a Passo 3 non degrada la
  qualità.
- **Performance**: 1133.1s contro 44.1s del baseline senza offload
  (Malmö) — **25.7x più lento**. Non production-ready; estrapolato al
  full 570 sarebbe dell'ordine delle ore.
- **Causa della lentezza**: costo fisso di ~0.10s per ogni forward CPU a
  singola riga (non batchato), indipendente dal numero di thread —
  verosimilmente un limite di throughput per-core di questo tipo di
  istanza RunPod (GPU-ottimizzata, vCPU condivise), non un bug di
  configurazione. Non ancora isolato con un microbenchmark PyTorch puro
  (prossimo passo se si vuole inseguire la velocità).

## File in questa cartella

- `pod_config/` — snapshot HW/env (lscpu, free, nvidia-smi, os-release)
- `runs/` — log completi di ogni tentativo (v1 fallito, v2 py-spy
  bloccato, v4 pool-veloce-ma-forward-lento, v5 completato, thread2 A/B)
- `results/run_n16_v5.jsonl` — unico risultato strutturato completo
- `misc/gpu_telemetry.csv`, `misc/host_telemetry.csv` — telemetria
  continua per la sessione

## Aggiornamento — microbenchmark isolato + confronto controllato a 3 fasi (stesso giorno, entry "continued 17/18")

Il microbenchmark PyTorch isolato indicato sopra come "prossimo passo"
è stato eseguito nella stessa sessione:

```
CPU batch=1: p50 37.04ms (9.51 GFLOPS)   GPU batch=1: p50 0.83ms (424.4 GFLOPS)
```

**Risposta**: limite di throughput per-core genuino (9.5 GFLOPS è
basso ma reale per un GEMV su questa CPU senza AVX-512), non overhead
di dispatch — ma resta un gap 2.7x non spiegato tra questo benchmark
isolato (37ms) e la pipeline reale (~100ms/call). PCIe escluso come
causa (`bench_pcie_bandwidth_wsl2.py`, mai eseguito prima: pinned D2H
~20-24 GB/s, ben oltre il teorico Gen3x16).

**Spiegazione fisica del GEMV lento** (osservazione del project owner,
confermata da `top -H` durante un run reale: un solo core al 99.9%,
gli altri 26 a zero): un matmul a singola riga (1×4096) è
memory-bandwidth-bound, non compute-bound — bassa intensità
aritmetica, ogni peso letto dalla RAM produce un solo output. oneDNN
correttamente non lo distribuisce su più core (aggiungerne non
aiuterebbe, tutti competerebbero per la stessa banda). Batchare 2+
token (GEMV→GEMM) alzerebbe l'intensità aritmetica e sbloccherebbe il
multi-core — idea registrata per una sessione futura, non implementata.

**Confronto controllato a 3 fasi** (stesso n=16, `OSX_GCSG_FORCE_CPU_ROUTE`
nuovo per forzare il routing CPU):

| Fase | Tempo | Accuracy | shadow_activations |
|---|---|---|---|
| 1 — solo GPU | 43.9s | 50.0% | 10081 |
| 2 — solo CPU (forzata) | 1072.3s | 50.0% | 10081 |
| 3 — "misto" (= v5) | 1133.1s | 50.0% | 10081 |

**Scoperta più importante di questa tabella**: la Fase 3 non era mai
stata un vero "misto" — a cold start (nessun traffico EAT reale) la
classificazione hot/cold usa la sola recency di inserimento, e il
nostro pool `[0, 1]` cade sistematicamente nella metà "fredda".
Confermato empiricamente: Fase 2 (1072.3s) e Fase 3/v5 (1133.1s)
sono statisticamente equivalenti (5.7% di differenza, rumore). Causa
architetturale: `ExpertAccessTable` non persiste tra processi —
issue #27 (EPM) propone esattamente questa persistenza ma non è mai
stata implementata. Un vero "misto" richiede o quella persistenza, o
un pattern due-pass (`generate()` → `refresh_shadow_pool_selection()`
→ `generate()`) nello stesso processo — nessuno dei due fatto oggi,
decisione esplicita di chiudere qui la sessione.

Zero errori matematici su tutte e tre le fasi (stessa accuracy, stesso
conteggio attivazioni) — la correttezza resta indipendente da dove
gira il forward.

## Cosa resta aperto

Nessuna azione bloccante per issue #33: correttezza provata (ora tre
volte, non solo una), feature resta opt-in/default-off (verificato con
test dedicati). Prossimi passi possibili, nessuno bloccante:
1. Issue #27 (EPM) o pattern due-pass, per un vero confronto misto.
2. Batching (GEMV→GEMM) sul forward CPU, per sbloccare i core idle.
3. Test finale su hardware bare-metal reale (richiesto dal project
   owner) invece di un'istanza RunPod condivisa/virtualizzata.
