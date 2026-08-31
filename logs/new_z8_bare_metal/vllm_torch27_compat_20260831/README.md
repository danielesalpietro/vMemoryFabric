# Analisi compatibilità vLLM ↔ torch≥2.7/cu128 — dati grezzi (2026-08-31)

Evidenze prodotte sulla Z8 G4 bare-metal (`berlin-3eie`) per l'analisi richiesta
da `osx-poc/handoff_rtx5060ti16gb.md` §2.1 (issue [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8)):
quale versione di vLLM è compatibile con `torch>=2.7`/`cu128`, e quali breaking
change toccano `GCSGWorker` (`osx-poc/src/scheduler/gcsg.py`).

Ogni file porta in testa il comando e la fonte che l'hanno prodotto, più
timestamp e host, così l'analisi in
[`osx-poc/reports/vllm_torch27_compat_analysis.md`](../../../osx-poc/reports/vllm_torch27_compat_analysis.md)
è ricontrollabile e ri-eseguibile senza fidarsi della prosa.

## File

| File | Fonte | Contenuto |
|---|---|---|
| `vllm_pypi_index.json` | `pypi.org/pypi/vllm/json` | Indice completo release vLLM (96 release al 2026-08-31), archiviato per riproducibilità |
| `collect_vllm_torch_pins.py` | — | Script che produce `vllm_torch_pins.txt` dall'indice sopra |
| `vllm_torch_pins.txt` | `pypi.org/pypi/vllm/{v}/json`, `info.requires_dist` | Pin `torch` dichiarato da ogni release vLLM da 0.6.6.post1 a 0.28.0 |
| `torch_cu128_wheels.txt` | `download.pytorch.org/whl/cu128/torch/` | Wheel cu128 esistenti per cp312/linux-x86_64 |
| `vllm_cuda_supported_archs.txt` | `CMakeLists.txt` ai tag | `CUDA_SUPPORTED_ARCHS` dei kernel propri di vLLM + bisezione della prima release con arch `12.0` |
| `vllm_internals_bisect.txt` | HTTP status su `raw.githubusercontent.com` | Presenza/rimozione dei file interni vLLM usati da `gcsg.py`, per 12 tag |
| `vllm_api_signatures_diff.txt` | sorgenti ai tag | `AWQMoEMethod.apply()` e `fused_marlin_moe()` a confronto 0.6.6.post1 ↔ 0.9.0 |
| `vllm_090_hook_points.txt` | sorgenti al tag v0.9.0 | I 4 hook del docstring di `gcsg.py` + i 3 path di quantizzazione verificati su 0.9.0 |

## Nota su una correzione fatta in corsa

Una prima passata aveva concluso che **0.9.0** fosse la prima release vLLM con
arch `12.0` nei kernel, avendo campionato solo `0.6.6.post1 / 0.9.0 / 0.10.1 / 0.11.0`.
La bisezione completa in coda a `vllm_cuda_supported_archs.txt` mostra che la
prima è in realtà **0.8.0**. La conclusione finale (minimo utilizzabile = 0.9.0)
non cambia, ma il motivo sì: non sono i kernel vLLM a vincolare, è `torch` —
0.8.x pinna `torch==2.6.0`, per cui **non esiste** una wheel cu128
(`torch_cu128_wheels.txt`), quindi niente `sm_120` lato torch.

Entrambe le versioni del fatto sono lasciate agli atti invece di sovrascrivere
la prima, per la convenzione già in uso in questo repo (vedi il paragrafo
analogo in `CHANGELOG.MD`, entry PPR 2026-08-25).

## Limite di questa analisi

È verifica **statica**: pin dichiarati, presenza di simboli e firme ai tag.
Nessun `pip install vllm` è stato eseguito, nessun modello caricato, nessun run
GCSG end-to-end. Che `sm_120` sia negli archi supportati e che i simboli
esistano **non dimostra** che Mixtral-8x7B-AWQ giri su vLLM 0.9.0 con GCSG
attivo. Serve un run reale prima di considerare chiusa la questione.
