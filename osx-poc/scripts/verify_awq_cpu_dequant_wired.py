#!/usr/bin/env python3
"""Issue #33 Fase 6a — Passo 3: parità numerica della pipeline PRODUZIONE
(non più uno script standalone) contro il kernel CUDA AWQ reale.

Passo 2 (verify_awq_cpu_dequant_parity.py) verificava una REIMPLEMENTAZIONE
locale del dequant, mai collegata a GCSGWorker. Da allora
_dequantize_awq_linear_to_fp32()/_build_cpu_shadow_pool_awq() sono stati
scritti dentro src/scheduler/gcsg.py e agganciati al branch path-3 di
_load_shadow_pool() — questo script chiama quel codice REALE (stesso
import che userebbe un GCSGWorker vero), non una copia, per verificare che
il collegamento in produzione (bits/group_size derivati dalle shape reali,
non da quant_config.json come nel Passo 2) preservi la parità già
dimostrata.

Costruisce GCSGWorker via __new__ (stesso principio di tests/test_
scheduler.py: _build_cpu_shadow_pool_awq() non tocca self, un'istanza
vuota basta), chiama il metodo reale su un layer/expert del modello
caricato via vLLM plain LLM(), confronta contro lo stesso forward CUDA
reale del Passo 2. Stessa soglia stretta (1e-2 relativo): qui NON c'è
rumore di quantizzazione atteso, solo precisione fp16/fp32 residua.

Usage:
    PYTHONPATH=src VLLM_ATTENTION_BACKEND=XFORMERS python scripts/verify_awq_cpu_dequant_wired.py [--layer 0] [--expert 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from scheduler.gcsg import GCSGWorker

MODEL_PATH = Path("/data/nvme/models/mixtral-instruct-awq")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({"status": "skipped", "reason": "CUDA non disponibile"}))
        return

    print("Caricamento modello reale via vLLM (plain LLM(), quantization=awq)...")
    from vllm import LLM

    t0 = time.perf_counter()
    llm = LLM(
        model=str(MODEL_PATH),
        quantization="awq",
        cpu_offload_gb=4,
        gpu_memory_utilization=0.95,
        max_num_seqs=16,
        max_model_len=3328,
        enforce_eager=True,
        hf_overrides={"head_dim": 128},
    )
    print(f"Modello caricato in {time.perf_counter() - t0:.1f}s")

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer = model.model.layers[args.layer]
    module = layer.block_sparse_moe.experts[args.expert]
    print(f"Modulo estratto: layer={args.layer} expert={args.expert} type={type(module).__name__}")

    first_param_device = next(module.parameters()).device
    was_offloaded = first_param_device.type != "cuda"
    if was_offloaded:
        print(f"Modulo offloaded ({first_param_device}) — pinning esplicito su GPU...")
        module = module.to("cuda")
        layer.block_sparse_moe.experts[args.expert] = module
    else:
        print(f"Modulo già GPU-resident ({first_param_device}), nessun pinning necessario.")

    hidden = model.config.hidden_size
    torch.manual_seed(0)
    hidden_states_cpu = torch.randn(4, hidden, dtype=torch.float16)
    hidden_states_gpu = hidden_states_cpu.to("cuda")

    print("\nForward REALE (kernel CUDA AWQ, modulo MixtralMLP vero)...")
    with torch.no_grad():
        output_real = module(hidden_states_gpu)
    output_real_cpu = output_real.detach().to(torch.float32).cpu()
    print(f"output reale: shape={tuple(output_real_cpu.shape)} "
          f"range=[{output_real_cpu.min().item():.4f}, {output_real_cpu.max().item():.4f}]")

    print("\nForward CPU via GCSGWorker._build_cpu_shadow_pool_awq() REALE "
          "(bits/group_size derivati dalle shape, non da quant_config.json)...")
    worker = GCSGWorker.__new__(GCSGWorker)
    t0 = time.perf_counter()
    cpu_pool = worker._build_cpu_shadow_pool_awq([layer], expert_ids=[args.expert])
    build_s = time.perf_counter() - t0
    print(f"_build_cpu_shadow_pool_awq() completato in {build_s:.2f}s")

    shadow = cpu_pool[args.expert]
    with torch.no_grad():
        output_cpu = shadow(hidden_states_cpu.to(torch.float32), layer_id=0)
    print(f"output CPU: shape={tuple(output_cpu.shape)} "
          f"range=[{output_cpu.min().item():.4f}, {output_cpu.max().item():.4f}]")

    diff = output_real_cpu - output_cpu
    rel_l2_error = (diff.norm() / output_real_cpu.norm()).item()
    max_abs_diff = diff.abs().max().item()
    print(f"\nErrore relativo L2: {rel_l2_error:.6f}")
    print(f"Differenza assoluta massima: {max_abs_diff:.6f}")

    threshold = 1e-2
    if rel_l2_error < threshold:
        print(f"\nPARITÀ CONFERMATA (errore {rel_l2_error:.6f} < soglia {threshold}) — "
              "il codice REALE agganciato a GCSGWorker._load_shadow_pool() produce lo "
              "stesso risultato del kernel CUDA reale, non solo la reimplementazione "
              "standalone del Passo 2.")
    else:
        print(f"\nPARITÀ FALLITA (errore {rel_l2_error:.6f} >= soglia {threshold}) — "
              "probabile bug reale nel wiring di produzione, da investigare prima di "
              "fidarsi della pipeline.")


if __name__ == "__main__":
    main()
