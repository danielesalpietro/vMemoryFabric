#!/usr/bin/env python3
"""Verifica numerica indipendente di _MarlinFusedShadowExpert (issue #10).

Prima versione di questo script forzava top_k=1 per isolare "davvero" un
solo expert — crashava con CUDA illegal memory access dentro
fused_marlin_moe, sia dalla chiamata diretta sia da una chiamata di
riferimento indipendente (vero FusedMoE.forward() con top_k monkey-
patchato). Vedi gcsg.py::_MarlinFusedShadowExpert per la cronologia
completa. Il fix adottato non tocca più top_k (resta quello reale del
modello) — questo script verifica quel fix, non l'approccio precedente.

Senza poter forzare top_k=1 per un confronto "puro", la verifica procede
per proprietà osservabili invece che per uguaglianza esatta con una
seconda chiamata:

  1. Output finito (nessun NaN/Inf) — il caso base.
  2. Invarianza al secondo expert "arbitrario": con top_k reale (2) e
     tutti gli expert non-target ugualmente sfavoriti, quale dei 7 diventi
     il secondo selezionato non dovrebbe cambiare il risultato in modo
     percepibile (il suo peso dopo renormalize è numericamente nullo).
     Verificato rendendo un secondo expert specifico marginalmente meno
     sfavorito (logit -10 invece di -30) e controllando che l'output non
     cambi in modo apprezzabile — se cambiasse, il "secondo slot" non
     sarebbe davvero trascurabile e l'isolamento non funzionerebbe.
  3. Discriminazione reale: output per expert_id=0 e per expert_id=1 DEVONO
     differire in modo sostanziale — prova che il target selezionato conta
     davvero (non è un no-op che ignora router_logits).

Usage:
    PYTHONPATH=src python scripts/verify_marlin_shadow_expert.py
"""
from __future__ import annotations

import sys

import torch

from scheduler.gcsg import _MarlinFusedShadowExpert

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    from vllm import LLM, SamplingParams

    print(f"Loading {MODEL_PATH} (vanilla, quantization=awq_marlin)...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq_marlin",
        cpu_offload_gb=4,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=2048,
        hf_overrides={"head_dim": 128},
    )
    print("LLM ready.")

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer0 = model.model.layers[0]
    experts = layer0.block_sparse_moe.experts
    print(f"experts type: {type(experts).__name__}, num_experts={experts.num_experts}, "
          f"top_k={experts.top_k}, renormalize={experts.renormalize}, "
          f"has w13_qweight={hasattr(experts, 'w13_qweight')}")
    if not hasattr(experts, "w13_qweight"):
        _fail("experts has no w13_qweight — not the Marlin-packed path this script verifies")

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

    num_experts = experts.num_experts
    outputs = {}

    # 1. Output finito per due expert_id diversi.
    # Nota (2026-08-10): entry per-layer (fused, expert_id, num_experts),
    # non (fused_list, expert_id, num_experts) uniforme — vedi gcsg.py.
    for expert_id in (0, 1):
        shadow = _MarlinFusedShadowExpert([(experts, expert_id, num_experts)])
        output = shadow(hidden_states, layer_id=0)
        torch.cuda.synchronize()
        if not torch.isfinite(output).all():
            _fail(f"expert_id={expert_id}: output has non-finite values")
        print(f"expert_id={expert_id}: output finito, shape={tuple(output.shape)}, "
              f"mean={output.mean().item():.6f}")
        outputs[expert_id] = output

    # 2. Discriminazione reale tra due expert diversi.
    diff_between_experts = (outputs[0] - outputs[1]).abs().max().item()
    print(f"\nmax abs diff tra expert_id=0 e expert_id=1: {diff_between_experts:.4f}")
    if diff_between_experts < 1e-2:
        _fail(
            "expert 0 e expert 1 producono output quasi identico — il "
            "routing forzato sembra un no-op, non una vera selezione per expert"
        )

    # 3. Invarianza al secondo expert arbitrario (prova che il suo peso e'
    #    davvero trascurabile, non solo "piccolo").
    fused = experts
    target_expert_id = 0
    baseline_shadow = _MarlinFusedShadowExpert([(fused, target_expert_id, num_experts)])
    baseline_output = baseline_shadow(hidden_states, layer_id=0)
    torch.cuda.synchronize()

    router_logits_perturbed = torch.full(
        (hidden_states.shape[0], num_experts), -30.0,
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    router_logits_perturbed[:, target_expert_id] = 30.0
    router_logits_perturbed[:, (target_expert_id + 1) % num_experts] = -10.0  # meno sfavorito
    perturbed_output = fused.quant_method.apply(
        layer=fused,
        x=hidden_states,
        router_logits=router_logits_perturbed,
        top_k=fused.top_k,
        renormalize=fused.renormalize,
        use_grouped_topk=fused.use_grouped_topk,
        topk_group=fused.topk_group,
        num_expert_group=fused.num_expert_group,
        custom_routing_function=fused.custom_routing_function,
        scoring_func=fused.scoring_func,
        e_score_correction_bias=fused.e_score_correction_bias,
    )
    torch.cuda.synchronize()

    perturbation_diff = (baseline_output - perturbed_output).abs().max().item()
    print(f"max abs diff tra second-slot -30 vs -10 (stesso target): "
          f"{perturbation_diff:.6f}")
    if perturbation_diff > 1e-2:
        _fail(
            f"cambiare il logit del secondo expert (-30 -> -10) cambia "
            f"l'output di {perturbation_diff:.4f} — il secondo slot NON e' "
            f"trascurabile, l'isolamento non e' affidabile"
        )

    print("\nPASS: _MarlinFusedShadowExpert isola un expert reale e "
          "distinto via il kernel Marlin reale, senza toccare top_k, con "
          "contributo del secondo slot verificato trascurabile.")


if __name__ == "__main__":
    main()
