#!/usr/bin/env python3
"""Testa se il crash del kernel Marlin (issue #10, commit 4ff2026) e' lo
STESSO meccanismo isolato per il path AWQ ModuleList (issue nuova,
5729dc7): offload CPU + pin_memory=False sotto WSL2.

La sessione originale su Marlin (4ff2026) ha isolato via router_logits/
top_k, concludendo "il bug e' dentro il kernel compilato, non ispezionabile
da Python" — MA non ha mai testato la variabile offload/pin_memory in
isolamento, perche' cpu_offload_gb=4 era gia' attivo per necessita' di VRAM
in OGNI run di quella sessione, mai confrontato contro un layer
non-offloaded o contro pin_memory forzato. Se la causa e' la stessa
individuata per il path AWQ ModuleList (swap-in offload + memoria non
pinned sotto WSL2), l'intera indagine Marlin precedente potrebbe aver
attribuito al kernel Marlin un sintomo che in realta' viene da un livello
sotto (il meccanismo di offload stesso, condiviso da entrambi i path).

A differenza del path ModuleList (un modulo per expert, offload
selezionabile per expert), FusedMoE (sia fp16 grezzo che Marlin-packed)
tiene TUTTI gli expert di un layer in un unico tensore batched
(w13_qweight shape (num_experts, ...)) — quindi l'offload, se avviene, e'
per LAYER intero, non per singolo expert. Cerchiamo un layer con
w13_qweight su CPU vs uno su CUDA, non un singolo expert.

Varianti, un processo ciascuna (stessa cautela sul contesto CUDA sporco
dopo un crash):
  --variant non-offloaded       : layer con w13_qweight su cuda
  --variant offloaded-unpinned  : layer con w13_qweight su cpu, pin_memory default (False sotto WSL)
  --variant offloaded-pinned    : stesso layer offloaded, pin_memory forzato True

Usage:
    PYTHONPATH=src python scripts/isolate_marlin_offload_variables.py --variant offloaded-unpinned
"""
from __future__ import annotations

import argparse
import sys

import torch

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"


def _find_layer(model, want_device: str):
    for layer_id, layer in enumerate(model.model.layers):
        experts = layer.block_sparse_moe.experts
        device = experts.w13_qweight.data.device
        if device.type == want_device:
            return layer_id, experts
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["non-offloaded", "offloaded-unpinned", "offloaded-pinned"],
        required=True,
    )
    args = parser.parse_args()

    if args.variant == "offloaded-pinned":
        import vllm.platforms.interface as iface
        iface.in_wsl = lambda: False
        print("Monkey-patched vllm.platforms.interface.in_wsl -> lambda: False "
              "(forza is_pin_memory_available() a True).")

    from vllm import LLM, SamplingParams
    from scheduler.gcsg import _MarlinFusedShadowExpert

    print(f"Loading {MODEL_PATH} (quantization=awq_marlin, cpu_offload_gb=4, "
          f"variant={args.variant})...")
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
    hf_config = llm.llm_engine.model_config.hf_config

    want_device = "cuda" if args.variant == "non-offloaded" else "cpu"
    found = _find_layer(model, want_device)
    if found is None:
        print(f"FAIL: nessun layer trovato con w13_qweight.device={want_device} — "
              f"il test non puo' esercitare la condizione che vuole isolare.")
        sys.exit(1)
    layer_id, experts = found
    actual_device = experts.w13_qweight.data.device
    print(f"Layer selezionato: {layer_id}, w13_qweight.device={actual_device} "
          f"(atteso {want_device})")
    if actual_device.type != want_device:
        print(f"FAIL: device trovato ({actual_device.type}) non combacia con quello "
              f"atteso ({want_device}).")
        sys.exit(1)

    hs = torch.randn(4, hf_config.hidden_size, dtype=torch.float16, device="cuda")
    # Nota (2026-08-10): _MarlinFusedShadowExpert prende ora entry per-layer
    # (fused, expert_id, num_experts) invece di (fused_list, expert_id,
    # num_experts) uniforme — vedi gcsg.py. Qui chiamiamo direttamente sul
    # tensore ORIGINALE (non pinnato) apposta, e' quello che questo script
    # isola.
    shadow = _MarlinFusedShadowExpert(
        [(experts, 0, hf_config.num_local_experts)],
    )
    print(f"Chiamo _MarlinFusedShadowExpert(expert_id=0).__call__() direttamente — "
          f"variant={args.variant}, device layer={actual_device}...")
    try:
        with torch.no_grad():
            out = shadow(hs, layer_id=0)
        torch.cuda.synchronize()
        print(f"NESSUN ERRORE — output finite={torch.isfinite(out).all().item()}, "
              f"shape={tuple(out.shape)}, mean={out.mean().item():.6f}")
    except Exception as e:
        print(f"ECCEZIONE: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
