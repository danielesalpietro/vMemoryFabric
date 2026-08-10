#!/usr/bin/env python3
"""Verifica numerica Fase 1, issue #10, direzione (a3): un MixtralMLP
standalone, popolato a mano dal checkpoint su disco via safetensors lazy
load, produce lo stesso output del modulo reale caricato dal loader vLLM
normale (quantization="awq")?

Contesto: la direzione (a3) evita sia il dequant manuale (a2) sia il
reverse del layout Marlin (a1) — costruisce solo un secondo MixtralMLP
(stessa classe che _AWQShadowExpert già usa per il path AWQ piatto) fuori
dal modello principale, per i soli expert_id/layer necessari, invece di
scrivere aritmetica di dequantizzazione a mano. Tre presupposti verificati
prima di scrivere questo script (non assunti):

  1. AWQLinearMethod.apply() richiede CUDA (nessun kernel CPU registrato
     per _C::awq_gemm) — quindi il modulo standalone va su .cuda(), non è
     un test offline puro.
  2. Il process group distribuito è già inizializzato dopo LLM(), nello
     stesso processo — costruire ReplicatedLinear (quant_config=AWQConfig)
     dopo funziona.
  3. MixtralMLP.__init__ NON passa params_dtype a ReplicatedLinear, che di
     default cade su torch.get_default_dtype() (vllm/model_executor/
     layers/linear.py:170-171) — senza wrapping esplicito sarebbe fp32 con
     tutta probabilità, non fp16. Uso vllm.model_executor.model_loader.
     utils.set_default_torch_dtype (la stessa utility che vLLM usa
     internamente per caricare in fp16), non un context manager scritto a
     mano — stesso principio "delega al codice reale" già applicato a
     _AWQShadowExpert.

Checkpoint: casperhansen/mixtral-instruct-awq (NON TheBloke — quello ha il
bug NaN root-causato in LOGBOOK 2026-08-09, vllm-project/vllm#2359).
cpu_offload_gb=4: unica configurazione in cui "awq" puro (pesi non
compattati, 22.97GB) carica per intero su una 3090 da 24GiB (vedi
scripts/smoke_test_fetta2_awq_with_offload.py).

Soglie dichiarate qui, prima di girare il confronto, non decise a
posteriori guardando il numero che esce: rtol=1e-2, atol=1e-3 — stesso
ordine di grandezza già usato in verify_marlin_shadow_expert.py (righe
110, 148) per un confronto dello stesso tipo (due percorsi kernel diversi
sullo stesso peso).

Usage:
    PYTHONPATH=src python scripts/verify_awq_manual_shadow_expert.py
"""
from __future__ import annotations

import json
import sys

import torch

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"
LAYER_ID = 0
EXPERT_IDS = (0, 1)   # confronto + discriminazione, stesso schema di verify_marlin_shadow_expert.py
RTOL = 1e-2
ATOL = 1e-3


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _build_standalone_expert(quant_config, hidden_size, intermediate_size,
                             num_experts, layer_id, expert_id, device):
    """Costruisce un MixtralMLP standalone per un solo (layer_id, expert_id),
    popolato via safetensors lazy load dal checkpoint su disco — non
    condivide memoria col modello principale caricato da LLM().
    """
    from safetensors import safe_open
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    from vllm.model_executor.models.mixtral_quant import MixtralMLP

    with set_default_torch_dtype(torch.float16):
        mlp = MixtralMLP(num_experts, hidden_size, intermediate_size,
                         quant_config=quant_config)

    with open(f"{MODEL_PATH}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}"
    open_shards: dict = {}
    try:
        for linear_name, linear_module in (("w1", mlp.w1), ("w2", mlp.w2), ("w3", mlp.w3)):
            for param_name in ("qweight", "qzeros", "scales"):
                key = f"{prefix}.{linear_name}.{param_name}"
                shard_file = weight_map[key]
                if shard_file not in open_shards:
                    open_shards[shard_file] = safe_open(
                        f"{MODEL_PATH}/{shard_file}", framework="pt", device="cpu",
                    )
                tensor_cpu = open_shards[shard_file].get_tensor(key)

                param = getattr(linear_module, param_name)
                expected_dtype = param.dtype   # int32 per qweight/qzeros, fp16 per scales
                if tuple(tensor_cpu.shape) != tuple(param.shape):
                    _fail(f"{key}: shape checkpoint {tuple(tensor_cpu.shape)} "
                          f"!= shape attesa dal modulo {tuple(param.shape)}")
                param.data.copy_(tensor_cpu.to(device=device, dtype=expected_dtype))
    finally:
        for handle in open_shards.values():
            del handle

    for linear_module in (mlp.w1, mlp.w2, mlp.w3):
        linear_module.quant_method.process_weights_after_loading(linear_module)

    return mlp.to(device)


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.quantization.awq import AWQConfig

    print(f"Loading {MODEL_PATH} (quantization=awq, plain — NON marlin, "
          f"cpu_offload_gb=4)...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq",
        cpu_offload_gb=4,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=2048,
        hf_overrides={"head_dim": 128},
    )
    print("LLM ready.")

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    hf_config = llm.llm_engine.model_config.hf_config
    layer0 = model.model.layers[LAYER_ID]
    experts = layer0.block_sparse_moe.experts
    print(f"experts type: {type(experts).__name__}, "
          f"hidden_size={hf_config.hidden_size}, "
          f"intermediate_size={hf_config.intermediate_size}, "
          f"num_local_experts={hf_config.num_local_experts}")
    if not hasattr(experts, "__getitem__") or hasattr(experts, "num_experts"):
        _fail("experts non è una ModuleList — non il path AWQ piatto (mixtral_quant.py) "
              "che questo script verifica")

    device = next(experts[0].parameters()).device
    print(f"device modello reale: {device}")

    with open(f"{MODEL_PATH}/quant_config.json") as f:
        quant_config_json = json.load(f)
    quant_config = AWQConfig.from_config(quant_config_json)
    print(f"AWQConfig da disco: {quant_config}")

    captured: dict = {}

    def _capture_hook(module, inputs, output):
        if "hidden_states" not in captured:
            captured["hidden_states"] = inputs[0].detach().clone()

    handle = layer0.block_sparse_moe.gate.register_forward_hook(_capture_hook)
    try:
        llm.generate(
            ["[INST] Explain quantum entanglement in one sentence. [/INST]"],
            SamplingParams(max_tokens=8, temperature=0.0),
        )
    finally:
        handle.remove()

    if "hidden_states" not in captured:
        _fail("hook never fired — no real hidden_states captured")
    hidden_states = captured["hidden_states"]
    print(f"Captured real hidden_states: shape={tuple(hidden_states.shape)}, "
          f"dtype={hidden_states.dtype}, device={hidden_states.device}")

    print(f"\nSoglie dichiarate PRIMA del confronto: rtol={RTOL}, atol={ATOL}")

    reference_outputs = {}
    shadow_outputs = {}

    for expert_id in EXPERT_IDS:
        print(f"\n--- expert_id={expert_id} ---")

        reference_module = experts[expert_id]
        with torch.no_grad():
            ref_out = reference_module(hidden_states)
        torch.cuda.synchronize()
        if not torch.isfinite(ref_out).all():
            _fail(f"expert_id={expert_id}: output di riferimento (loader reale) non finito")
        reference_outputs[expert_id] = ref_out
        print(f"riferimento (loader reale): mean={ref_out.mean().item():.6f}")

        shadow_module = _build_standalone_expert(
            quant_config, hf_config.hidden_size, hf_config.intermediate_size,
            hf_config.num_local_experts, LAYER_ID, expert_id, device,
        )
        with torch.no_grad():
            shadow_out = shadow_module(hidden_states)
        torch.cuda.synchronize()
        if not torch.isfinite(shadow_out).all():
            _fail(f"expert_id={expert_id}: output shadow standalone non finito")
        shadow_outputs[expert_id] = shadow_out
        print(f"shadow (standalone, safe_open): mean={shadow_out.mean().item():.6f}")

        if not torch.allclose(ref_out, shadow_out, rtol=RTOL, atol=ATOL):
            max_diff = (ref_out - shadow_out).abs().max().item()
            _fail(f"expert_id={expert_id}: shadow standalone diverge dal riferimento "
                  f"oltre rtol={RTOL}/atol={ATOL} (max abs diff={max_diff:.6f})")
        print(f"MATCH entro rtol={RTOL}/atol={ATOL}")

    diff_between_experts = (shadow_outputs[EXPERT_IDS[0]] - shadow_outputs[EXPERT_IDS[1]]).abs().max().item()
    print(f"\nmax abs diff tra shadow expert_id={EXPERT_IDS[0]} e {EXPERT_IDS[1]}: {diff_between_experts:.4f}")
    if diff_between_experts < 1e-2:
        _fail("i due expert shadow producono output quasi identico — "
              "l'estrazione per-expert sembra un no-op")

    print("\nPASS: MixtralMLP standalone (safe_open lazy load, "
          "set_default_torch_dtype reale, process_weights_after_loading "
          "esplicito) combacia col modulo caricato dal loader vLLM normale "
          "entro le soglie dichiarate, per due expert distinti.")


if __name__ == "__main__":
    main()
