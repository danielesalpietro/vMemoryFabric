#!/usr/bin/env python3
"""Issue #33 Fase 6a — misura completa della pipeline AWQ-dequant→cache→
forward su un expert reale del checkpoint di produzione. Chiude i pezzi
lasciati stimati (non misurati) dopo Passo 1/Passo 2: costo di
ri-quantizzazione INT4, impronta di memoria reale, costo del forward a
regime, E — non testato prima d'ora — se la doppia quantizzazione
(AWQ reale → `_quantize_int4()` simulato) degrada l'accuratezza oltre il
kernel CUDA reale, non solo la velocità.

Confronta DUE strategie di caching per lo stesso expert reale:
    fp32   — cache i pesi dequantizzati AWQ così come sono (Passo 1/2):
             nessun rumore di quantizzazione aggiuntivo, ma 21GB/expert
             stimati su tutti i 32 layer (Passo 1).
    int4   — ri-quantizza con `_quantize_int4()` (Fase 1, stesso schema
             già validato) subito dopo il dequant AWQ, cache l'INT8:
             ~4x più compatto, ma introduce un SECONDO round di rumore
             di quantizzazione sopra quello AWQ già presente — mai
             misurato prima quanto costi in accuratezza.

Ground truth: lo stesso modulo MixtralMLP reale e lo stesso kernel CUDA
AWQ di verify_awq_cpu_dequant_parity.py (Passo 2) — non un nuovo
riferimento.

Richiede gli stessi workaround locali di Passo 2 (VLLM_ATTENTION_BACKEND,
hf_overrides) — vedi quel file per il perché.

Usage:
    PYTHONPATH=src VLLM_ATTENTION_BACKEND=XFORMERS python scripts/bench_awq_cpu_pipeline.py [--layer 0] [--expert 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from scheduler.gcsg import _quantize_int4, _ShadowExpertINT4

MODEL_PATH = Path("/data/nvme/models/mixtral-instruct-awq")

# ── vendorizzato da casper-hansen/AutoAWQ, awq/utils/packing_utils.py ────────
# (MIT license) — identico a verify_awq_cpu_dequant*.py, duplicato per
# tenere questo script eseguibile in isolamento (nessun cross-import tra
# scripts/, stessa convenzione di bench_*.py in questo progetto).

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros
    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(iweights.shape[-1], dtype=torch.int32, device=izeros.device)
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)
    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]
    return iweights, izeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    scales_e = scales.repeat_interleave(group_size, dim=0)
    izeros_e = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros_e) * scales_e
    return iweight

# ── fine codice vendorizzato ──────────────────────────────────────────────


def _awq_dequant_expert(module) -> tuple[dict[str, torch.Tensor], float]:
    quant_config = json.loads((MODEL_PATH / "quant_config.json").read_text())
    bits, group_size = quant_config["w_bit"], quant_config["q_group_size"]

    t0 = time.perf_counter()
    dequantized = {}
    for proj in ("w1", "w2", "w3"):
        linear = getattr(module, proj)
        qweight = linear.qweight.detach().cpu()
        qzeros = linear.qzeros.detach().cpu()
        scales = linear.scales.detach().cpu()
        dequantized[proj] = dequantize_gemm(qweight, qzeros, scales, bits, group_size)
    dequant_s = time.perf_counter() - t0
    return dequantized, dequant_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({"status": "skipped", "reason": "CUDA non disponibile"}))
        return

    print("Caricamento modello reale via vLLM...")
    from vllm import LLM

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
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    module = model.model.layers[args.layer].block_sparse_moe.experts[args.expert]
    hidden = model.config.hidden_size
    if next(module.parameters()).device.type != "cuda":
        module = module.to("cuda")

    torch.manual_seed(0)
    hidden_states_cpu = torch.randn(4, hidden, dtype=torch.float32)
    hidden_states_gpu = hidden_states_cpu.to(torch.float16).to("cuda")

    print("\n[Ground truth] Forward reale via kernel CUDA AWQ...")
    with torch.no_grad():
        output_real = module(hidden_states_gpu).detach().to(torch.float32).cpu()

    # ── Componente 1: dequant AWQ (comune a entrambe le strategie) ──────────
    dequantized, dequant_s = _awq_dequant_expert(module)
    print(f"\n[Componente 1] Dequant AWQ (w1+w2+w3, un layer): {dequant_s * 1e3:.1f}ms")

    w13_fp32 = torch.cat([dequantized["w1"].T, dequantized["w3"].T], dim=0).to(torch.float32)
    w2_fp32 = dequantized["w2"].T.to(torch.float32)

    results = {}

    # ── Strategia "fp32": cache diretta, nessun rumore aggiuntivo ───────────
    shadow_fp32 = _ShadowExpertINT4([(w13_fp32, 1.0)], [(w2_fp32, 1.0)])
    shadow_fp32(hidden_states_cpu, layer_id=0)   # warm-up
    t0 = time.perf_counter()
    for _ in range(20):
        output_fp32 = shadow_fp32(hidden_states_cpu, layer_id=0)
    forward_fp32_s = (time.perf_counter() - t0) / 20

    mem_fp32_mb = (w13_fp32.numel() + w2_fp32.numel()) * 4 / (1024 ** 2)
    err_fp32 = ((output_real - output_fp32).norm() / output_real.norm()).item()

    results["fp32_cache"] = {
        "dequant_ms": dequant_s * 1e3,
        "requant_ms": 0.0,
        "forward_ms_per_call": forward_fp32_s * 1e3,
        "cached_memory_mb_per_layer": mem_fp32_mb,
        "cached_memory_gb_32_layers": mem_fp32_mb * 32 / 1024,
        "rel_l2_error_vs_real_kernel": err_fp32,
    }

    # ── Strategia "int4": ri-quantizza con _quantize_int4 (Fase 1), cache
    # l'INT8 — round-trip dequant addizionale ad ogni forward call, stesso
    # meccanismo già in _ShadowExpertINT4 per il path GPU. ─────────────────
    t0 = time.perf_counter()
    w13_int4 = _quantize_int4(w13_fp32)
    w2_int4 = _quantize_int4(w2_fp32)
    requant_s = time.perf_counter() - t0

    shadow_int4 = _ShadowExpertINT4([w13_int4], [w2_int4])
    shadow_int4(hidden_states_cpu, layer_id=0)   # warm-up
    t0 = time.perf_counter()
    for _ in range(20):
        output_int4 = shadow_int4(hidden_states_cpu, layer_id=0)
    forward_int4_s = (time.perf_counter() - t0) / 20

    mem_int4_mb = (w13_int4[0].numel() + w2_int4[0].numel()) * 1 / (1024 ** 2)  # int8 = 1 byte
    err_int4 = ((output_real - output_int4).norm() / output_real.norm()).item()

    results["int4_cache"] = {
        "dequant_ms": dequant_s * 1e3,
        "requant_ms": requant_s * 1e3,
        "forward_ms_per_call": forward_int4_s * 1e3,
        "cached_memory_mb_per_layer": mem_int4_mb,
        "cached_memory_gb_32_layers": mem_int4_mb * 32 / 1024,
        "rel_l2_error_vs_real_kernel": err_int4,
    }

    print(f"\n[Componente 2] Requant INT4 (_quantize_int4, w13+w2): {requant_s * 1e3:.2f}ms")
    print("\n[Componente 3] Memoria cachata per layer/expert:")
    print(f"  fp32: {mem_fp32_mb:.1f} MB/layer -> {mem_fp32_mb * 32 / 1024:.2f} GB/expert (32 layer)")
    print(f"  int4: {mem_int4_mb:.1f} MB/layer -> {mem_int4_mb * 32 / 1024:.2f} GB/expert (32 layer)")
    print("\n[Componente 4] Forward a regime (media 20 call, batch=4, fp32 compute):")
    print(f"  fp32_cache: {forward_fp32_s * 1e3:.2f}ms/call")
    print(f"  int4_cache: {forward_int4_s * 1e3:.2f}ms/call "
          f"(+{(forward_int4_s - forward_fp32_s) * 1e3:.2f}ms per il dequant INT4->fp32 inline)")

    print("\n[Accuratezza vs. kernel CUDA reale — NON testato prima d'ora]")
    print(f"  fp32_cache: errore relativo L2 = {err_fp32:.6f} "
          f"({'entro soglia stretta 1e-2' if err_fp32 < 1e-2 else 'FUORI soglia stretta'})")
    print(f"  int4_cache: errore relativo L2 = {err_int4:.6f} "
          f"({'entro soglia stretta 1e-2' if err_int4 < 1e-2 else 'FUORI soglia — doppia quantizzazione degrada oltre la sola precisione fp'})")

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
