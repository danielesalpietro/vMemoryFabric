#!/usr/bin/env python3
"""Mappa esatta layer-per-layer dello stato di offload, per issue #10/#16.

Due cose, non una: (1) il numero esatto di layer/expert offloaded sotto
cpu_offload_gb=4 (serve per calcolare il costo VRAM reale di (a), non la
stima ~5-6 layer), e (2) verifica empirica di QUALE oggetto viene wrappato
da maybe_offload_to_cpu() — il decoder layer intero (make_layers() lo
chiama su quell'oggetto, vllm/model_executor/models/utils.py:551) o anche
i singoli moduli expert. Se è solo il decoder layer, chiamare
experts[i](hidden_states) direttamente NON passa dal wrapper — userebbe il
forward originale sui parametri reali (che possono essere ancora su CPU).
Rilevante prima di scrivere qualunque fix: il fix corretto dipende da QUALE
meccanismo genera davvero il crash, non da quale sembra plausibile.

Verifica: forward.__qualname__ di un closure definito dentro
maybe_offload_to_cpu contiene "maybe_offload_to_cpu.<locals>.forward" —
un bound method normale (MixtralMLP.forward, ReplicatedLinear.forward)
non ce l'ha. Controlliamo sia il decoder layer sia i singoli expert.

Usage:
    PYTHONPATH=src python scripts/map_offload_state.py --quantization awq
    PYTHONPATH=src python scripts/map_offload_state.py --quantization awq_marlin
"""
from __future__ import annotations

import argparse

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"


def _is_wrapped(forward_fn) -> bool:
    return "maybe_offload_to_cpu" in getattr(forward_fn, "__qualname__", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", choices=["awq", "awq_marlin"], required=True)
    args = parser.parse_args()

    from vllm import LLM

    print(f"Loading {MODEL_PATH} (quantization={args.quantization}, cpu_offload_gb=4)...")
    llm = LLM(
        model=MODEL_PATH,
        quantization=args.quantization,
        cpu_offload_gb=4,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=2048,
        hf_overrides={"head_dim": 128},
    )
    print("LLM ready.\n")

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layers = model.model.layers
    is_marlin = args.quantization == "awq_marlin"

    print(f"Decoder layer .forward wrapping (maybe_offload_to_cpu su tutto il layer):")
    layer_wrapped = []
    for layer_id, layer in enumerate(layers):
        wrapped = _is_wrapped(layer.forward)
        layer_wrapped.append(wrapped)
        marker = "WRAPPED" if wrapped else "unwrapped"
        print(f"  layer {layer_id:2d}: layer.forward {marker}")

    print(f"\nPer-expert device + forward wrapping:")
    n_offloaded_layers_fully = 0
    expert_offload_map = []   # (layer_id, expert_id, device, expert_forward_wrapped)
    for layer_id, layer in enumerate(layers):
        experts = layer.block_sparse_moe.experts
        if is_marlin:
            device = experts.w13_qweight.data.device
            wrapped = _is_wrapped(experts.forward)
            print(f"  layer {layer_id:2d}: FusedMoE (tutti gli expert insieme) "
                  f"device={device}, experts.forward {'WRAPPED' if wrapped else 'unwrapped'}")
            expert_offload_map.append((layer_id, None, device, wrapped))
        else:
            for expert_id, expert in enumerate(experts):
                device = expert.w1.qweight.data.device
                wrapped = _is_wrapped(expert.forward)
                expert_offload_map.append((layer_id, expert_id, device, wrapped))
            devices = {e[2].type for e in expert_offload_map if e[0] == layer_id}
            wrapped_states = {e[3] for e in expert_offload_map if e[0] == layer_id}
            print(f"  layer {layer_id:2d}: expert devices={devices}, "
                  f"expert.forward wrapped states={wrapped_states}")

    print(f"\n--- Riepilogo ---")
    n_layers_offloaded = sum(1 for w in layer_wrapped if w)
    print(f"Decoder layer con .forward WRAPPED (maybe_offload_to_cpu attivo su quel layer): "
          f"{n_layers_offloaded}/{len(layers)}")

    if not is_marlin:
        offloaded_experts = [(l, e) for (l, e, d, w) in expert_offload_map if d.type == "cpu"]
        expert_forward_ever_wrapped = any(w for (_, _, _, w) in expert_offload_map)
        print(f"Expert individuali con parametri su CPU: {len(offloaded_experts)}/{len(layers)*8}")
        print(f"Almeno un expert.forward individuale risulta WRAPPED "
              f"(non solo il decoder layer)? {expert_forward_ever_wrapped}")
        if offloaded_experts:
            layers_with_offloaded_experts = sorted(set(l for l, e in offloaded_experts))
            print(f"Layer con almeno un expert offloaded: {layers_with_offloaded_experts}")
            per_layer_offloaded_count = {}
            for l, e in offloaded_experts:
                per_layer_offloaded_count[l] = per_layer_offloaded_count.get(l, 0) + 1
            print(f"Expert offloaded per layer (solo layer con >=1): {per_layer_offloaded_count}")
    else:
        offloaded_layers = [l for (l, e, d, w) in expert_offload_map if d.type == "cpu"]
        print(f"Layer Marlin (FusedMoE, tutto insieme) con w13_qweight su CPU: "
              f"{offloaded_layers}")


if __name__ == "__main__":
    main()
