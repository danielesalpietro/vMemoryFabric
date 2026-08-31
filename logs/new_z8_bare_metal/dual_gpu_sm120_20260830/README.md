# Verifica toolchain dual-GPU / `sm_120` — dati grezzi

**Issue:** [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) (dual-GPU/AER)
**Host:** Z8 G4 bare-metal (`berlin-3eie`), RTX 3090 24GB + RTX 5060 Ti 16GB, driver 595.84

Esecuzione della checklist "Da fare prima/durante l'arrivo" dell'issue #8, sulla
seconda GPU realmente installata (**RTX 5060 Ti 16GB**, non la RTX 5080
originariamente attesa — stessa famiglia Blackwell, `sm_120`).

## Nota sulle date

La verifica è stata **eseguita** il 2026-08-30 e i risultati riportati
sull'issue #8 lo stesso giorno. Questi file sono la **ri-esecuzione del
2026-08-31**, fatta per archiviare gli output grezzi: la prima passata era stata
condotta in container poi rimossi, lasciando i numeri solo in prosa sull'issue —
esattamente il pattern che `handoff_rtx5060ti16gb.md` §0 dice di non ripetere.
I risultati coincidono; le uniche differenze sono i valori casuali di
`torch.randn().sum()` e la VRAM riportata da torch (`24117`/`15849` MiB, al netto
del context CUDA) contro quella di `nvidia-smi` (`24576`/`16311` MiB).
Il nome della directory conserva la data della misura originale.

## File

| File | Comando | Contenuto |
|---|---|---|
| `nvidia_smi.txt` | `nvidia-smi` | Output completo, entrambe le GPU |
| `nvidia_smi_query.txt` | `nvidia-smi --query-gpu=... --format=csv` + `nvidia-smi -L` | Indici, VRAM, bus PCIe, compute cap, UUID |
| `check_sm120.py` | — | Script dei 3 controlli, eseguito identico sui due stack |
| `check_sm120_torch251_cu124.txt` | `python3.12 check_sm120.py` | **Pin attuale** (`requirements-vllm.txt`) — atteso e osservato: FALLISCE |
| `check_sm120_torch270_cu128.txt` | idem | **Candidato** — atteso e osservato: PASSA |
| `pip_freeze_torch251_cu124.txt` | `pip freeze` | Ambiente esatto del run sopra |
| `pip_freeze_torch270_cu128.txt` | `pip freeze` | idem |

## Risultato

| stack | `sm_120` in `arch_list` | `cuda:0` (3090) | `cuda:1` (5060 Ti) |
|---|---|---|---|
| `torch==2.5.1+cu124` | **no** | OK | `RuntimeError: CUDA error: no kernel image is available` |
| `torch==2.7.0+cu128` | **sì** | OK | **OK** |

L'errore sulla riga in basso a sinistra è **identico** a quello già documentato
in `LOGBOOK.md` (entry 2026-08-12/13) per la RTX PRO 6000 Blackwell su RunPod:
stesso pin di torch, stessa famiglia architetturale, stesso fallimento.

## Ambiente dei container

Non è l'immagine del progetto: **`osx-poc:dev` non esiste più su questa Z8**
(verosimile conseguenza della riprovisionatura del 2026-08-24 verso
`/mnt/wdc-docker`). Ricostruito un ambiente equivalente per i soli tre controlli:
stesso `nvidia/cuda:12.1.1-*-ubuntu22.04`, stesso Python **3.12.13** (coincide col
log pytest della sessione del 24/08), stesso wheel torch.

Differenze rispetto all'immagine reale, da tenere presenti:
- variante `-base` invece di `-devel` — irrilevante qui, il wheel torch porta le
  proprie librerie CUDA (lo nota `requirements-vllm.txt` stesso);
- **`vllm` NON installato** — questi file non testano lo stack completo;
- `--gpus all` / `NVIDIA_VISIBLE_DEVICES=all` invece del `=0` di
  `docker-compose.yml`, override a runtime, nessun file modificato.

## Vincolo indipendente, ancora aperto

`docker-compose.yml` pinna `NVIDIA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`
(commento `# RTX 3090 only`): la seconda GPU resta invisibile ai container **a
prescindere da torch**. È un blocco separato da quello della toolchain e va
rimosso prima che la 5060 Ti sia utilizzabile per davvero (handoff §2.4).

## Seguito

Analisi di compatibilità vLLM ↔ `torch>=2.7`:
[`osx-poc/reports/vllm_torch27_compat_analysis.md`](../../../osx-poc/reports/vllm_torch27_compat_analysis.md),
evidenze in [`../vllm_torch27_compat_20260831/`](../vllm_torch27_compat_20260831/).
