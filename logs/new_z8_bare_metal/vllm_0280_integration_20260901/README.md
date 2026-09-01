# vLLM giugno–agosto 2026 (0.22.1 → 0.28.0) — dati grezzi (2026-09-01)

Evidenze prodotte per l'analisi
[`osx-poc/reports/vllm_2026Q3_integration_analysis.md`](../../../osx-poc/reports/vllm_2026Q3_integration_analysis.md):
quali novità vLLM della finestra giugno–agosto 2026 esistono davvero nel
sorgente, e dove `vMemoryFabric` può innestarsi senza forkare vLLM.

**Metodo.** Le due sdist ufficiali `vllm-0.22.1.tar.gz` (2026-06-05, prima
release della finestra) e `vllm-0.28.0.tar.gz` (2026-08-26, ultima release al
momento della verifica) sono state scaricate da
`files.pythonhosted.org/packages/source/v/vllm/`, estratte e confrontate con
`grep`/`diff`/`wc`. **Nessun `pip install vllm`, nessun run, nessun benchmark** —
stesso perimetro dichiarato in
[`vllm_torch27_compat_20260831/`](../vllm_torch27_compat_20260831/), di cui
questa sessione è il seguito.

Le sdist stesse non sono committate (36–40 MB l'una); sono ri-scaricabili
dagli URL sopra e i comandi che hanno prodotto ogni file sono in testa al file
stesso.

## File

| File | Fonte | Contenuto |
|---|---|---|
| `vllm_pypi_index_20260901.json` | `pypi.org/pypi/vllm/json` | Indice completo release vLLM (96 release, ultima 0.28.0), archiviato per riproducibilità |
| `vllm_release_window_jun_aug_2026.txt` | idem | Le 8 release pubblicate nella finestra giugno–agosto 2026, con data di upload |
| `vllm_torch_pins_window.txt` | `pypi.org/pypi/vllm/<v>/json` | Pin `torch`/`torchvision` di ognuna di quelle 8 release |
| `vllm_0280_feature_verification.txt` | sorgente 0.28.0 | Ogni nome citato nella richiesta, cercato nell'albero `.py` — con i due falsi positivi isolati (`P-EAGLE`, `EAGLE 3.1`) |
| `vllm_subsystem_delta_0221_0280.txt` | sorgenti 0.22.1 e 0.28.0 | LOC e file aggiunti/rimossi nei 5 sottosistemi rilevanti per vMemoryFabric, sull'esatta finestra |
| `vllm_0280_extension_points.txt` | sorgente 0.28.0 | I 6 punti di innesto (A–F) citati dall'analisi, riportati come sorgente letterale invece che parafrasati |
| `gcsg_hooks_vs_0280.txt` | sorgente 0.28.0 | Stato in 0.28.0 di ogni dipendenza interna di `gcsg.py` — estende a 0.28.0 la tabella di `vllm_torch27_compat_analysis.md` §2, che si fermava a 0.10.1 |

## Due correzioni alla lista di partenza, lasciate agli atti

La richiesta che ha originato questa sessione elencava fra le novità
**`P-EAGLE`** e **`EAGLE 3.1`**. Nessuno dei due esiste nel sorgente di 0.28.0:

- `grep -rli p_eagle` restituisce 5 file, ma sono **falsi positivi da
  sottostringa** (`_setup_eagle3_aux_hidden_state_outputs`, `drop_eagle_block`) —
  la riga di prova è in `vllm_0280_feature_verification.txt` §1;
- la stringa `eagle 3.1` / `eagle31` non compare mai: il metodo dichiarato in
  `config/speculative.py` si chiama `eagle3`, senza minor version.

`DFlash` e `DSpark` invece esistono e sono entrambi metodi speculativi di
prima classe (`SpeculativeMethod` Literal, stesso file). Il dettaglio è
registrato qui e non solo nell'analisi perché è esattamente il tipo di
dettaglio che, non verificato, si propaga per mesi — stessa ragione della nota
analoga in `vllm_torch27_compat_20260831/README.md`.

## Limite di questi dati

Dicono che un simbolo **esiste** al tag e con quale firma. Non dicono che
funzioni, né che regga il carico, né che l'innesto proposto nell'analisi
compili: nessuna delle due sdist è mai stata installata. Vale lo stesso
disclaimer di §4 dell'analisi precedente.
