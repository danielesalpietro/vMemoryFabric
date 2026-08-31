# Analisi di compatibilità vLLM ↔ `torch>=2.7`/`cu128` — issue #8

**Data:** 2026-08-31
**Host:** Z8 G4 bare-metal (`berlin-3eie`), RTX 3090 24GB + RTX 5060 Ti 16GB
**Richiesta da:** `osx-poc/handoff_rtx5060ti16gb.md` §2.1
**Stato:** analisi statica completa; **nessun pin modificato**, nessun `pip install vllm` eseguito

Dati grezzi e comandi che hanno prodotto ogni numero di questo documento:
[`logs/new_z8_bare_metal/vllm_torch27_compat_20260831/`](../../logs/new_z8_bare_metal/vllm_torch27_compat_20260831/)
e [`logs/new_z8_bare_metal/dual_gpu_sm120_20260830/`](../../logs/new_z8_bare_metal/dual_gpu_sm120_20260830/).

---

## Conclusione

**Versione minima utilizzabile: `vllm==0.9.0` con `torch==2.7.0+cu128`.
Finestra utilizzabile: `0.9.0` – `0.10.1`.**

La finestra si chiude per motivi che **non hanno nulla a che vedere con torch**:
sono due file interni di vLLM da cui `gcsg.py` dipende e che vengono rimossi.

---

## 1. Il vincolo torch

Pin `torch` dichiarato da ogni release vLLM
(fonte: `vllm_torch_pins.txt`, campo `info.requires_dist` di PyPI):

| vLLM | torch | data |
|---|---|---|
| 0.6.6.post1 → 0.7.3 | `2.5.1` | 2024-12 → 2025-02 |
| 0.8.0 → 0.8.5.post1 | `2.6.0` | 2025-03 → 2025-05 |
| **0.9.0 → 0.9.2** | **`2.7.0`** | **2025-05-28** |
| 0.10.0 → 0.10.1.1 | `2.7.1` | 2025-07 |
| 0.10.2 → 0.11.0 | `2.8.0` | 2025-09 |
| 0.28.0 (ultima) | `2.13.0` | 2026-08-26 |

`torch>=2.7` ⇒ **vLLM ≥ 0.9.0**.

### Verifica empirica su hardware reale

Stesso script (`check_sm120.py`) su due stack, per avere il controfattuale —
output grezzi in `dual_gpu_sm120_20260830/`:

| stack | `sm_120` in `arch_list` | `torch.randn(10, device='cuda:1').sum()` |
|---|---|---|
| `torch==2.5.1+cu124` (pin attuale) | **no** | `RuntimeError: CUDA error: no kernel image is available for execution on the device` |
| `torch==2.7.0+cu128` (candidato) | **sì** | **OK** |

La 3090 (`cuda:0`, sm_86) funziona in entrambi.

### Trappola: la wheel di default è quella sbagliata

`torch==2.7.0` **puro da PyPI è una build cu126** (dipende da
`nvidia-cuda-runtime-cu12==12.6.77`) e **non contiene `sm_120`**. Il pin di vLLM
dichiara `torch==2.7.0` senza local version: `pip install vllm==0.9.0` liscio
installa la variante sbagliata e riproduce esattamente l'errore del 2026-08-30.

Serve `--index-url https://download.pytorch.org/whl/cu128`. Le wheel cu128
esistono **da torch 2.7.0 in poi** — nessuna per 2.5.x/2.6.x
(fonte: `torch_cu128_wheels.txt`).

### I kernel propri di vLLM

`gcsg.py` non usa solo torch: passa per i kernel CUDA di vLLM via
`quant_method.apply()`. Da `CMakeLists.txt` (fonte: `vllm_cuda_supported_archs.txt`):

- `0.6.6.post1` → `"7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0"` — niente `12.0`
- **`0.8.0`** → prima release con `12.0` (elenco incondizionato)
- da `0.9.0` → `12.0` presente ma dietro `if(CMAKE_CUDA_COMPILER_VERSION >= 12.8)`

Quindi 0.8.x compila già kernel sm_120, ma pinna `torch==2.6.0` che non ha wheel
cu128: **il vincolo stringente è torch, non i kernel vLLM.** 0.9.0 resta il minimo.

---

## 2. Breaking change per `GCSGWorker`

### Cosa sopravvive intatto in 0.9.0

Verificato sul sorgente al tag, non sulle release notes
(fonte: `vllm_090_hook_points.txt`):

| Dipendenza di `gcsg.py` | Stato in 0.9.0 |
|---|---|
| `worker_cls` come stringa qualname in `EngineArgs` | presente (`arg_utils.py:402`) |
| `resolve_obj_by_qualname` come meccanismo di risoluzione | presente (`worker_base.py:147,558`) |
| `Worker.init_device()` / `load_model()` separati, `mem_get_info()` in `init_device` | presenti |
| `MixtralMoE.gate` = `ReplicatedLinear`, `router_logits` scartato come locale | presente (`mixtral.py:57,81,105`) |
| `execute_model(execute_model_req: ExecuteModelRequest)` + `.seq_group_metadata_list` | presenti |
| `MixtralMLP` con `w1`/`w2`/`w3` in `nn.ModuleList` (path AWQ) | presente |
| Gli 8 attributi `FusedMoE` passati a `apply()` | tutti presenti |
| `vllm.platforms.interface.in_wsl()` (per `_should_pin_transfers`) | presente |
| `_ShadowExpertINT4` | puro torch, zero dipendenze vLLM — immune |

`AWQMoEMethod.apply()` **guadagna** 4 parametri (`global_num_experts`,
`expert_map`, `apply_router_weight_on_input`, `activation`), tutti con default;
`gcsg.py` passa tutto per keyword ⇒ la chiamata a `gcsg.py:840` resta valida.
L'unico `assert` nuovo nel corpo è `activation == "silu"`, che è il default.

### Un allarme rivelatosi falso, registrato per non ripeterlo

`torch.ops.vllm.fused_marlin_moe` cambia firma tra 0.6.6.post1 e 0.9.0: il
parametro `num_bits: int` è sostituito da `quant_type_id: int`
(`ScalarType.from_id`) — fonte: `vllm_api_signatures_diff.txt`.

Sembrava rompere il path Marlin, ma **`gcsg.py` non chiama mai quell'op
direttamente**: la riga 668 è dentro un docstring che documenta cosa fa
`apply()` internamente. Il codice reale passa da `quant_method.apply()`.
La rottura è interna a vLLM e non propaga.

### Dove la finestra si chiude davvero

Bisezione su 12 tag (fonte: `vllm_internals_bisect.txt`):

| File usato da `gcsg.py` | Ultima versione in cui esiste | Cosa rompe |
|---|---|---|
| `vllm/model_executor/models/mixtral_quant.py` | **0.10.1** (404 in 0.10.2) | `_AWQShadowExpert` — la `ModuleList` di `MixtralMLP` |
| `vllm/worker/worker.py` | **0.10.1.1** (404 in 0.11.0) | `GCSGWorker(Worker)` — la classe base stessa |

Da 0.11.0 resta solo il motore V1 (`vllm/v1/worker/gpu_worker.py`), con API
worker completamente diversa (`execute_model(scheduler_output)`, non
`ExecuteModelRequest`). Non è un adattamento di firma: è riscrivere l'hook.

Inoltre **da 0.8.0 V1 è il default**: già in 0.9.0 servirebbe `VLLM_USE_V1=0`
esplicito perché `GCSGWorker` venga istanziato.

---

## 3. Documentazione da correggere in `gcsg.py`

Il docstring del modulo (righe 41-95) cita due cose non più vere:

1. `vllm.executor.gpu_executor.GPUExecutor` come fonte della verifica della
   sequenza `init_device()` → `load_model()`: **quel file non esiste più da
   0.7.0** (sostituito da `uniproc_executor.py`). Il comportamento descritto è
   invariato; la citazione no.
2. La firma `fused_marlin_moe(..., num_bits=...)` (riga 668), sostituita da
   `quant_type_id` in 0.9.0.

Non corretto in questa sessione: è scope aggiuntivo rispetto a quanto chiesto
dall'handoff, segnalato in attesa di conferma (handoff §3, ultima regola).

---

## 4. Cosa NON è stato verificato

- **Nessun run end-to-end.** Che `sm_120` sia negli archi supportati e che i
  simboli esistano ai tag **non dimostra** che Mixtral-8x7B-AWQ giri su vLLM
  0.9.0 con GCSG attivo. Serve un run reale.
- **Nessuna baseline M1/M2** (handoff §2.3): è prerequisito dell'*upgrade*, non
  dell'analisi.
- **`vllm` non è mai stato installato.** Solo `torch`, in container usa-e-getta.
- Il vincolo indipendente `NVIDIA_VISIBLE_DEVICES=0` in `docker-compose.yml`
  resta aperto (handoff §2.4) — tiene la seconda GPU invisibile ai container a
  prescindere da torch.
