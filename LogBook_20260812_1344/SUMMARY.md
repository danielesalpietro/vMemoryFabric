# Sprint 4 (Tekniska) — RunPod session, 2026-08-12

Pod: RTX A5000, EU-RO-1, Network Volume "Sprint-4-Tekniska" (72GB, id `lxg28gfsja`).
Immagine: fix sshd/CMD (`56ad3e6`) applicato manualmente sul pod (repo + PYTHONPATH,
non ancora rebuildata su GHCR — vedi `osx-poc/LOGBOOK.md`).

## 1. Soak test pinning (§9, issue #17) — `soak_test_pinning/soak_test.log`

- 1000/1000 iterazioni, shard 256MB (stessa unità TierManager/EAT), 0 mismatch byte-esatti.
- Nessun degrado: drift cycle totale -1%, transfer H2D+D2H -3%.
- Pin-alloc drift -31% (allocatore più veloce dopo warm-up, non più lento).
- **Conclusione: pinned memory reale è sicura e stabile sotto carico sostenuto su Linux reale (fuori WSL2).**

## 2. MMLU 5-shot, run a fette (18x32) — `mmlu_sliced_run/`

- `orchestrator.log` (log completo), `mmlu_results.jsonl` (risultato per fetta).
- 570/570 domande, 57/57 soggetti, **412/570 corrette (72.28%)**.
- Tempo totale: ~38-40 minuti (17 fette in 37m06s + ultima fetta).
- Breakdown per fetta: model load medio ~96s, generate() reale ~20-25s (l'80% del tempo è overhead di ricaricamento checkpoint, non lavoro vero).
- Correlazione `shadow_activations` vs tempo per fetta: **r=0.95** (forte) — costo di latenza reale, non nel roadmap del README.
- Correlazione `shadow_activations` vs accuratezza per fetta: r=0.04 (nulla) — contaminazione non degrada selettivamente la qualità.

## 3. MMLU 5-shot, burn-test single-shot (570 in una chiamata) — `mmlu_burn_singleshot/burn_singleshot.log`

- Un solo processo, un solo `GCSGWorker`, un solo `generate()` su tutti i 570 prompt.
- **Nessuno stallo** — lo storico blocco WSL2 a richiesta ~27-31 non si riproduce su Linux reale.
- Tempo totale: **570.3s (~9.5 min)** = 106.3s model load + 464.0s generate() reale.
- **412/570 corrette (72.3%) — identico al run a fette**, stesso numero esatto di risposte giuste.
- **~4.2x più veloce del run a fette**, stessa qualità.

## 4. Confronto con baseline WSL2 (LOGBOOK 2026-08-10, storico)

| | WSL2 (storico) | Fette (oggi) | Single-shot (oggi) |
|---|---|---|---|
| Corrette | 411/570 (72.11%) | 412/570 (72.28%) | 412/570 (72.3%) |
| Tempo | ore ("overnight") | ~38-40 min | ~9.5 min |
| Stallo a ~27-31? | sì (mai risolto) | n/a (per design) | **no** |

## 5. Ambiente — `environment/environment_snapshot.txt`

GPU: RTX A5000 (non 3090 — deviazione dal piano originale, accettata). Driver 580.159.04,
CUDA 13.0, torch 2.5.1+cu124. Checkpoint `casperhansen/mixtral-instruct-awq` (~24.6GB,
3 shard safetensors) su `/data/nvme/models/`.

## Punti aperti per il ragionamento successivo

- Il pattern "un processo per fetta" era un aggiramento per uno stallo mai diagnosticato
  con certezza su WSL2. Oggi risulta non necessario su Linux reale — vale la pena
  aggiornare `run_mmlu_in_slices.sh`/la metodologia di default del progetto?
- Il costo di latenza dello shadow execution (r=0.95 con `shadow_activations`) non è
  documentato da nessuna parte finora — merita una nota nel report tecnico.
- `TierManager`/EAT restano fuori dal path dati reale (issue #17, non affrontato oggi —
  questo run usa `cpu_offload_gb` nativo di vLLM, non `TierManager`).
- GPU usata è A5000, non 3090 — ogni confronto quantitativo assoluto con benchmark
  precedenti va preso con questa riserva.
