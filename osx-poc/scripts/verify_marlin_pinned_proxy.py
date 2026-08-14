#!/usr/bin/env python3
"""Verifica isolata del proxy _PinnedMarlinExperts (issue #10/#16, direzione
(a) per il path Marlin) PRIMA di integrarlo in _load_shadow_pool() — stesso
principio "verifica isolata prima di integrare" usato per _AWQShadowExpert
e il resto di questa indagine.

Costruisce il proxy su un layer OFFLOADED (layer 0, confermato in
scripts/map_offload_state.py), lo confronta contro _MarlinFusedShadowExpert
sullo STESSO layer, STESSO tensore originale (non un layer diverso — pesi
diversi per definizione, confronto privo di senso) — per poter chiamare
l'originale offloaded senza crashare (issue #10/#16: chiamata diretta a un
layer offloaded senza pin_memory crasha) pin_memory e' forzato via
monkeypatch, SOLO come strumento diagnostico interno a questo script di
verifica, non una config di produzione (vedi #16). Se il proxy e' corretto,
i due output devono combaciare entro una soglia dichiarata: stesso peso,
stesso expert, letto da due tensori fisicamente diversi (slice pinnata
sincrona vs tensore originale intero, via il meccanismo di swap pinnato).

rtol/atol dichiarati PRIMA di girare, stesso principio di
verify_awq_manual_shadow_expert.py.

Usage:
    PYTHONPATH=src python scripts/verify_marlin_pinned_proxy.py
"""
from __future__ import annotations

import sys

import torch

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"
POOL_EXPERT_IDS = [0, 1]   # shadow_pool_size=2, round-robin — stesso di osx_default.yaml
RTOL = 1e-2
ATOL = 1e-3


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    # Monkeypatch diagnostico SOLO per rendere chiamabile senza crash il
    # tensore originale offloaded del layer 0, per ottenere un riferimento
    # numerico da confrontare col proxy — non una config di produzione
    # (issue #16: pin_memory=True sotto WSL2 bypassa un guard deliberato di
    # vLLM, resta diagnostico finche' non e' capito perche' vLLM lo disabilita).
    import vllm.platforms.interface as iface
    iface.in_wsl = lambda: False
    print("Monkey-patched vllm.platforms.interface.in_wsl -> lambda: False "
          "(SOLO per rendere il riferimento layer-0 originale chiamabile "
          "senza crash in questo script — non una config di produzione).")

    from vllm import LLM, SamplingParams
    from scheduler.gcsg import _MarlinFusedShadowExpert, _PinnedMarlinExperts

    print(f"Loading {MODEL_PATH} (quantization=awq_marlin, cpu_offload_gb=4)...")
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
    layers = model.model.layers

    offloaded_layer_id = 0    # confermato offloaded in map_offload_state.py

    offloaded_experts = layers[offloaded_layer_id].block_sparse_moe.experts
    print(f"layer {offloaded_layer_id} w13_qweight.device={offloaded_experts.w13_qweight.data.device} "
          f"(atteso cpu — pin_memory forzato lo rende chiamabile senza crash "
          f"per il riferimento, ma il tensore resta fisicamente offloaded)")
    if offloaded_experts.w13_qweight.data.device.type != "cpu":
        _fail("layer 0 non e' offloaded come atteso — la config e' cambiata, verifica non valida")

    print(f"\nCostruisco _PinnedMarlinExperts su layer {offloaded_layer_id} "
          f"(offloaded), expert_ids={POOL_EXPERT_IDS}...")
    proxy = _PinnedMarlinExperts(offloaded_experts, POOL_EXPERT_IDS)
    for name in ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales", "w13_qzeros", "w2_qzeros"):
        t = getattr(proxy, name)
        print(f"  proxy.{name}: shape={tuple(t.shape)}, device={t.device} (atteso cuda, dim0={len(POOL_EXPERT_IDS)})")
        if t.device.type != "cuda":
            _fail(f"proxy.{name} non e' su cuda")
        if t.shape[0] != len(POOL_EXPERT_IDS):
            _fail(f"proxy.{name} dim0={t.shape[0]}, atteso {len(POOL_EXPERT_IDS)}")

    hs = torch.randn(4, hf_config.hidden_size, dtype=torch.float16, device="cuda")

    print(f"\nSoglie dichiarate PRIMA del confronto: rtol={RTOL}, atol={ATOL}")

    for local_index, expert_id in enumerate(POOL_EXPERT_IDS):
        print(f"\n--- expert_id={expert_id} (local_index={local_index}) ---")

        shadow_from_proxy = _MarlinFusedShadowExpert(
            [(proxy, local_index, len(POOL_EXPERT_IDS))],
        )
        with torch.no_grad():
            out_proxy = shadow_from_proxy(hs, layer_id=0)
        torch.cuda.synchronize()
        if not torch.isfinite(out_proxy).all():
            _fail(f"expert_id={expert_id}: output dal proxy non finito")
        print(f"proxy (layer offloaded, slice pinnata): mean={out_proxy.mean().item():.6f}")

        shadow_from_original = _MarlinFusedShadowExpert(
            [(offloaded_experts, expert_id, hf_config.num_local_experts)],
        )
        with torch.no_grad():
            out_original = shadow_from_original(hs, layer_id=0)
        torch.cuda.synchronize()
        if not torch.isfinite(out_original).all():
            _fail(f"expert_id={expert_id}: output dal tensore originale non finito")
        print(f"originale (STESSO layer 0, offloaded+pin_memory forzato, "
              f"tensore intero a 8 expert): mean={out_original.mean().item():.6f}")

        if not torch.allclose(out_proxy, out_original, rtol=RTOL, atol=ATOL):
            max_diff = (out_proxy - out_original).abs().max().item()
            _fail(f"expert_id={expert_id}: proxy diverge dall'originale oltre "
                  f"rtol={RTOL}/atol={ATOL} (max abs diff={max_diff:.6f})")
        print(f"MATCH entro rtol={RTOL}/atol={ATOL}")

    print("\nPASS: _PinnedMarlinExperts produce lo stesso risultato del tensore "
          "Marlin originale, per entrambi gli expert del pool, con lo slicing "
          "sull'asse expert confermato corretto.")


if __name__ == "__main__":
    main()
