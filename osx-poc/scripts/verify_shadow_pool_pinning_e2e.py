#!/usr/bin/env python3
"""Verifica end-to-end (issue #10/#16, task "verifica stato reale, non il
codice scritto"): dopo il fix a _load_shadow_pool(), gli expert nello
shadow pool sono DAVVERO su GPU, e chiamarli DAVVERO non crasha più —
tramite GCSGWorker reale (worker_cls, non un test isolato sulla classe
shadow), stesso pattern di scripts/smoke_test_gcsg_mixtral8x7b.py.

Controlli, in ordine:
  1. shadow_pool contiene expert_ids 0 e 1 (shadow_pool_size=2) — non vuoto,
     non degradato a hook-only.
  2. Per il path AWQ ModuleList: ogni modulo pinnato ha i parametri
     effettivamente su cuda (non solo "il codice ha chiamato .to('cuda')
     senza sollevare" — controllo diretto del device).
  3. Chiamare lo shadow callable direttamente, fuori dal forward
     sequenziale — esattamente il pattern che crashava prima del fix — su
     un layer che PRIMA del pinning era offloaded (layer 0, confermato in
     scripts/map_offload_state.py): nessuna eccezione, output finito.
  4. generate() reale funziona ancora (nessuna regressione sul path
     normale del modello).

Usage:
    PYTHONPATH=src python scripts/verify_shadow_pool_pinning_e2e.py --quantization awq
    PYTHONPATH=src python scripts/verify_shadow_pool_pinning_e2e.py --quantization awq_marlin
"""
from __future__ import annotations

import argparse
import sys

import torch

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"
LAYER_ID_WAS_OFFLOADED = 0   # confermato in scripts/map_offload_state.py


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", choices=["awq", "awq_marlin"], required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from scheduler.gcsg import GCSGWorker

    print(f"Loading {MODEL_PATH} via GCSGWorker, quantization={args.quantization}, "
          f"cpu_offload_gb=4...")
    llm = LLM(
        model=MODEL_PATH,
        worker_cls="scheduler.gcsg.GCSGWorker",
        quantization=args.quantization,
        cpu_offload_gb=4,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=2048,
        hf_overrides={"head_dim": 128},
    )
    print("LLM ready.\n")

    worker = llm.llm_engine.model_executor.driver_worker
    if not isinstance(worker, GCSGWorker):
        _fail(f"driver_worker is {type(worker)}, not GCSGWorker")

    print(f"--- 1. shadow_pool contents ---")
    pool_ids = sorted(worker._shadow_pool.keys())
    print(f"shadow_pool expert_ids: {pool_ids}")
    if pool_ids != [0, 1]:
        _fail(f"shadow_pool={pool_ids}, atteso [0, 1] (shadow_pool_size=2) — "
              f"il fix non ha ripopolato il pool come atteso")

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer0_experts = model.model.layers[LAYER_ID_WAS_OFFLOADED].block_sparse_moe.experts
    is_marlin = args.quantization == "awq_marlin"

    print(f"\n--- 2. device dei parametri pinnati (layer {LAYER_ID_WAS_OFFLOADED}, "
          f"offloaded PRIMA del fix) ---")
    if is_marlin:
        # Path Marlin: il tensore ORIGINALE del layer resta offloaded (il
        # proxy e' una copia separata, non una mutazione in-place — vedi
        # _PinnedMarlinExperts) — verifichiamo invece che il proxy dentro
        # ai _MarlinFusedShadowExpert del pool sia su cuda.
        print(f"layer {LAYER_ID_WAS_OFFLOADED} tensore ORIGINALE w13_qweight.device="
              f"{layer0_experts.w13_qweight.data.device} (atteso invariato, ancora cpu — "
              f"il proxy e' una copia separata, non una mutazione in-place)")
        shadow0 = worker._shadow_pool[0]
        proxy0 = shadow0._fused[LAYER_ID_WAS_OFFLOADED]
        print(f"proxy._fused[{LAYER_ID_WAS_OFFLOADED}].w13_qweight.device={proxy0.w13_qweight.device} "
              f"(atteso cuda)")
        if proxy0.w13_qweight.device.type != "cuda":
            _fail("proxy del pool Marlin non e' su cuda")
    else:
        for expert_id in (0, 1):
            expert = layer0_experts[expert_id]
            device = next(expert.parameters()).device
            print(f"layer {LAYER_ID_WAS_OFFLOADED}, expert {expert_id}: device={device} (atteso cuda)")
            if device.type != "cuda":
                _fail(f"expert {expert_id} non e' stato pinnato in GPU")

    print(f"\n--- 3. chiamata diretta fuori sequenza sul layer prima offloaded "
          f"(pattern che crashava) ---")
    hf_config = llm.llm_engine.model_config.hf_config
    hs = torch.randn(4, hf_config.hidden_size, dtype=torch.float16, device="cuda")
    for expert_id in (0, 1):
        shadow = worker._shadow_pool[expert_id]
        try:
            with torch.no_grad():
                out = shadow(hs, layer_id=LAYER_ID_WAS_OFFLOADED)
            torch.cuda.synchronize()
        except Exception as e:
            _fail(f"expert_id={expert_id}: ECCEZIONE chiamando lo shadow callable "
                  f"direttamente — {type(e).__name__}: {e}")
        if not torch.isfinite(out).all():
            _fail(f"expert_id={expert_id}: output non finito")
        print(f"expert_id={expert_id}: nessuna eccezione, output finito, "
              f"mean={out.mean().item():.6f}")

    print(f"\n--- 4. generate() reale, nessuna regressione ---")
    outputs = llm.generate(
        ["[INST] What is 2+2? [/INST]"],
        SamplingParams(max_tokens=16, temperature=0.0),
    )
    text = outputs[0].outputs[0].text
    print(f"output: {text!r}")
    if not text.strip():
        _fail("generate() ha prodotto output vuoto")

    print(f"\nPASS ({args.quantization}): shadow pool popolato, expert pinnati "
          f"confermati su GPU, chiamata diretta fuori sequenza su un layer "
          f"prima offloaded non crasha piu', generate() normale invariato.")


if __name__ == "__main__":
    main()
