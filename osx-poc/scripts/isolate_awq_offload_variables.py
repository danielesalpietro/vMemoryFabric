#!/usr/bin/env python3
"""Isola le variabili offload/pin_memory nel crash trovato in
isolate_awq_shadow_call_crash.py (illegal memory access chiamando
block_sparse_moe.experts[e](hidden_states) direttamente, fuori dal forward
sequenziale, su quantization="awq" con cpu_offload_gb=4).

Due varianti, un processo per variante (contesto CUDA "sporco" dopo un
illegal memory access — niente run incatenati nello stesso processo dopo un
crash atteso):

  --variant non-offloaded: stesso checkpoint/config (cpu_offload_gb=4), ma
      cerca un expert che con questo budget e' rimasto su GPU (non tutto il
      modello viene offloaded, solo ~4GB dei 23 totali) e chiama QUELLO
      direttamente. Se non crasha -> offload e' la variabile, non la
      chiamata diretta in se'. Se crasha comunque -> il problema e' piu'
      generale della sola interazione con l'offload.

  --variant pin-memory-forced: stesso expert offloaded gia' noto crashare
      (layer 0, primo expert con device=cpu), ma con
      vllm.platforms.interface.in_wsl monkey-patchato per forzare
      pin_memory=True prima di LLM(). Trovato il punto esatto in
      vllm/platforms/interface.py:230-238 (Platform.is_pin_memory_available
      chiama in_wsl() a runtime, non bindato a import-time — patchare il
      nome nel modulo funziona indipendentemente dall'ordine di import,
      purche' avvenga prima di LLM()).

Usage:
    PYTHONPATH=src python scripts/isolate_awq_offload_variables.py --variant non-offloaded
    PYTHONPATH=src python scripts/isolate_awq_offload_variables.py --variant pin-memory-forced
"""
from __future__ import annotations

import argparse
import sys

import torch

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"


def _find_expert(model, want_device: str):
    """Scandisce layer/expert reali finche' non trova un device che combacia.
    Non assume che layer 0/expert 0 sia rappresentativo per il device voluto.
    """
    for layer_id, layer in enumerate(model.model.layers):
        experts = layer.block_sparse_moe.experts
        for expert_id, expert in enumerate(experts):
            device = expert.w1.qweight.data.device
            if device.type == want_device:
                return layer_id, expert_id, expert
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["non-offloaded", "pin-memory-forced"], required=True)
    args = parser.parse_args()

    if args.variant == "pin-memory-forced":
        import vllm.platforms.interface as iface
        iface.in_wsl = lambda: False
        print("Monkey-patched vllm.platforms.interface.in_wsl -> lambda: False "
              "(forza is_pin_memory_available() a True).")

    from vllm import LLM, SamplingParams

    print(f"Loading {MODEL_PATH} (quantization=awq, cpu_offload_gb=4, variant={args.variant})...")
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

    want_device = "cuda" if args.variant == "non-offloaded" else "cpu"
    found = _find_expert(model, want_device)
    if found is None:
        print(f"FAIL: nessun expert trovato con device={want_device} — "
              f"il test non puo' esercitare la condizione che vuole isolare.")
        sys.exit(1)
    layer_id, expert_id, reference_module = found
    actual_device = reference_module.w1.qweight.data.device
    print(f"Expert selezionato: layer={layer_id}, expert={expert_id}, "
          f"w1.qweight.device={actual_device} (atteso {want_device})")
    if actual_device.type != want_device:
        print(f"FAIL: device trovato ({actual_device.type}) non combacia con quello atteso "
              f"({want_device}) — bug nella logica di ricerca dello script, non nel modello.")
        sys.exit(1)

    hs = torch.randn(4, hf_config.hidden_size, dtype=torch.float16, device="cuda")
    print(f"Chiamo reference_module(hidden_states sintetico) direttamente — "
          f"variant={args.variant}, device parametro={actual_device}...")
    try:
        with torch.no_grad():
            out = reference_module(hs)
        torch.cuda.synchronize()
        print(f"NESSUN ERRORE — output finite={torch.isfinite(out).all().item()}, "
              f"shape={tuple(out.shape)}, mean={out.mean().item():.6f}")
    except Exception as e:
        print(f"ECCEZIONE: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
