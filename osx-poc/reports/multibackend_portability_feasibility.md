# Studio di fattibilità: vMemoryFabric multi-backend

*Preparato come base per issue evolutiva. Ambito: portare M1/M2/M3 da CUDA-only a un'architettura multi-backend sul modello di `vllm.platforms`.*

## Executive summary

**Verdetto: fattibile, non banale, da fare a fasi.** Il progetto non è "CUDA-only" per un singolo motivo isolato — è CUDA-only su tre livelli indipendenti che vanno risolti separatamente:

1. **Livello infrastruttura** (Docker, CI, requirements) — solo configurazione, nessun codice.
2. **Livello device/memoria** (M2 `TierManager`/`GPUTransfer`) — soft coupling, refactorabile dietro un'astrazione tipo `Platform`.
3. **Livello kernel di calcolo** (Marlin/AWQ in M3 `GCSGWorker`) — hard blocker reale: questi kernel non hanno equivalente su altri backend e vanno sostituiti o resi opzionali per backend.

Nessun blocco è insormontabile, ma il livello 3 impone che "multi-backend" per ora significhi realisticamente **CUDA + ROCm** (il backend più vicino), non un salto diretto a XPU/HPU/Neuron/TPU.

## Inventario accoppiamento — cosa blocca cosa

### Hard blocker (kernel/infra non portabili, richiedono sostituzione o feature-gating)

| # | Punto | Perché blocca |
|---|---|---|
| 1 | `gcsg.py:492,599` — `torch.ops.vllm.fused_marlin_moe`, `torch.ops._moe_C.marlin_gemm_moe`; classi `_PinnedMarlinExperts` (484-576), `_MarlinFusedShadowExpert` (579-676) | Kernel CUDA compilati per Ampere+. Nessun equivalente ROCm/XPU/HPU/Neuron/TPU. |
| 2 | `gcsg.py:103-106,211-217` — `pynvml` in `GCSGGuard._check_vram_budget` | NVML è NVIDIA-only. ROCm usa `amdsmi`/`rocm-smi`; altri backend non hanno un concetto equivalente diretto. |
| 3 | `Dockerfile:7` — `FROM nvidia/cuda:12.1.1-devel-ubuntu22.04`; `requirements-vllm.txt:24-26` — `torch==2.5.1+cu124` | Immagine base e wheel PyTorch legati a CUDA 12.4. |
| 4 | `docker-compose.yml:8-12` — `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES` | Runtime Docker NVIDIA-specifico. |
| 5 | `.github/workflows/ci.yml:50-79` — job `full-gpu-tests` su runner self-hosted taggato `gpu` | Nessuna distinzione di backend nella matrice CI; infrastruttura di test reale locked a una singola RTX 3090 fisica. |

### Soft coupling (astraibile senza riscrivere la logica)

| # | Punto | Fix |
|---|---|---|
| 1 | `tier/gpu.py:55,103,124,129,134,149` — `torch.device(f"cuda:{id}")`, `torch.cuda.stream`, `torch.cuda.mem_get_info`, `torch.cuda.Stream`, `torch.cuda.empty_cache` | Sostituire con dispatch tipo `current_platform.*` (vedi sotto). API PyTorch concettualmente equivalenti esistono su `torch.xpu.*`, `torch.hpu.*`. |
| 2 | `tier/manager.py:35,48,265` — `gpu_device: int` passato a `GPUTransfer` | Eredita il fix di (1), nessun lavoro aggiuntivo. |
| 3 | `gcsg.py:1329-1371,1276-1327` — `_pin_awq_expert_to_gpu`, `_promote_module_via_tier_manager`: check `.device.type != "cuda"`, `.to("cuda")` hardcoded | Sostituire con `current_platform.device_type`. |
| 4 | `gpu.py:97`, `gcsg.py:1326,1514` — `.pin_memory()` | Semantica di "pinned host memory" varia per backend — da validare per backend, non bloccante. |
| 5 | Naming/default hardcoded RTX 3090 / `device_id=0` sparsi in 6+ file (`gpu.py`, `manager.py`, `eat/types.py`, `scheduler/__init__.py`, `gcsg.py`) | Cosmetic + parametrizzazione. |

### Punto di leva già presente

`gcsg.py:1260-1274` (`_should_pin_transfers`) **già importa** `vllm.platforms.interface.in_wsl()`. Il progetto consuma già `vllm.platforms` in un punto (solo per rilevare WSL2) — è il gancio naturale da cui estendere l'uso di `current_platform` in modo sistematico, invece di introdurre un pattern nuovo da zero.

### GCSGWorker — l'hook su vLLM è già backend-agnostico

Punto positivo non ovvio: il meccanismo centrale (subclass di `Worker`, forward hook su `MixtralMoE.gate`, override di `execute_model()` che legge `ExecuteModelRequest.seq_group_metadata_list`) usa **solo strutture dati generiche di vLLM**, che vLLM stesso astrae per backend. Questo hook *non* è un hard blocker — segue automaticamente qualunque backend vLLM supporti, a patto che i punti CUDA-espliciti sopra (righe 1329-1371, 543-576) vengano risolti.

## Architettura di riferimento: `vllm.platforms`

*(vLLM non è installato in questa sessione — quanto segue è basato su conoscenza generale dell'architettura pubblica, da verificare contro la versione vLLM effettivamente pinnata dal progetto prima dell'implementazione.)*

Pattern: una classe astratta `Platform` (`vllm/platforms/interface.py`) con un enum `PlatformEnum` (`CUDA`, `ROCM`, `XPU`, `HPU`, `NEURON`, `TPU`, `CPU`) e metodi come `get_device_name()`, `get_device_total_memory()`, `is_pin_memory_available()`, `get_current_memory_usage()`. Ogni backend implementa una sottoclasse (`CudaPlatform`, `RocmPlatform`, `XPUPlatform`, ...) che incapsula le chiamate device-specifiche. Il resto della codebase vLLM chiama sempre `current_platform.<metodo>()`, mai `torch.cuda.*` direttamente. Il dispatch avviene per autodetect o via variabile d'ambiente.

**Questo è esattamente il pattern che `GPUTransfer` dovrebbe adottare**: non reinventare un'astrazione propria, ma dipendere da `current_platform` di vLLM (già una dipendenza del progetto) invece di `torch.cuda` diretto.

## Fattibilità per backend

| Backend | Complessità stimata (1–9) | Blocco principale |
|---|---|---|
| **ROCm** | **5** | Marlin/AWQ non hanno kernel ROCm equivalenti pronti all'uso → serve percorso di quantizzazione alternativo (es. GPTQ generico, o feature-gating che disabilita GCSG shadow-expert su ROCm). NVML → amdsmi. Resto (M2, hook vLLM) è refactor meccanico. |
| **Intel XPU** | **7** | Come ROCm ma ecosistema di quantizzazione ancora meno maturo per MoE; `torch.xpu` più giovane, meno testato in produzione su carichi MoE. |
| **Intel Gaudi (HPU)** | **8** | Modello di memoria e compilazione (graph mode via `habana_frameworks`) sufficientemente diverso da rendere anche M2 (stream/pin memory) da ripensare, non solo da astrarre. |
| **AWS Neuron** | **9** | Compilazione statica del grafo (`torch-neuronx`), nessun concetto diretto di "shadow expert" promosso a runtime — il modello di esecuzione di GCSG (intercettare gating in forward pass e promuovere tensori on-the-fly) è in tensione strutturale con il modello Neuron. |
| **Google TPU** | **9** | Stesso problema di Neuron, aggravato da XLA (compilazione lazy, no eager tensor manipulation stile GCSG). |

## Roadmap proposta

**Fase 0 — Astrazione senza nuovo backend (prerequisito, nessun rischio funzionale)**
Introdurre il dispatch `current_platform` in `GPUTransfer` e nei punti soft-coupling di `gcsg.py`, mantenendo CUDA come unico backend reale supportato. Nessun comportamento cambia, ma la base diventa estendibile. Stima: settimane, non mesi — è refactor meccanico su ~10 punti noti.

**Fase 1 — ROCm come primo secondo backend**
Il più vicino a CUDA (HIP re-usa gran parte della API `torch.cuda.*`). Richiede: sostituire NVML→amdsmi, validare `pin_memory()` su ROCm, e decidere la strategia per Marlin/AWQ (disabilitare shadow-expert path su ROCm inizialmente è un compromesso ragionevole, non uno stop-the-world).

**Fase 2+ — XPU/HPU/Neuron/TPU**
Da valutare singolarmente solo dopo Fase 1, perché ciascuno introduce un problema architetturale diverso (non solo "porta il kernel"), come evidenziato nella tabella sopra.

## Rischi e domande aperte

- **Infrastruttura di test**: tutta la validazione storica (Sprint 1-4, LOGBOOK.md) è su una singola RTX 3090 fisica. Un secondo backend richiede hardware reale per la validazione, non solo codice — non è simulabile in CI standard senza un runner dedicato.
- **Marlin/AWQ come feature, non come infrastruttura**: la decisione più importante non è tecnica ma di prodotto — se il shadow-expert pinning via Marlin è il cuore differenziante di GCSG, "multi-backend" potrebbe significare "core degradato" su non-CUDA finché non esiste un kernel equivalente altrove.
- **Versione vLLM pinnata**: lo studio assume l'architettura `vllm.platforms` nella sua forma pubblica nota; va riverificata contro la versione esatta (0.6.6.post1, da conversazione precedente) prima di aprire l'issue, perché l'API di `Platform` è cambiata nel tempo tra versioni vLLM.

## Raccomandazione

Aprire l'issue evolutiva su **Fase 0 soltanto** (astrazione `current_platform`, zero nuovi backend), con Fase 1 (ROCm) come issue di follow-up separata una volta che l'astrazione esiste e non ha regressioni sul path CUDA esistente. Trattare Fase 2+ come esplorativa, non pianificata.
