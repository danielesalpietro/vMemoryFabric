#!/usr/bin/env python3
"""Issue #33 Fase 6a — Passo 2: parità numerica tra il dequant CPU
(Passo 1, `verify_awq_cpu_dequant.py`) e il kernel CUDA AWQ REALE, sullo
stesso expert dello stesso checkpoint, stesso input.

A differenza della parità Fase 1 (INT4 simulato vs. riferimento fp32, dove
uno scarto relativo ~0.24-0.32 era ATTESO — rumore di quantizzazione
reale), qui i pesi sono GIÀ quantizzati AWQ da monte: sia il path CPU sia
quello GPU calcolano sulla STESSA informazione quantizzata, solo con
codice diverso (dequant separato + GEMM fp32 vs. kernel CUDA fuso
dequant+matmul). Ci si aspetta quindi uno scarto vicino alla sola
precisione fp16/fp32, non rumore di quantizzazione — soglia qui molto più
stretta di Fase 1 (1e-2 relativo, non 0.5): uno scarto grande
indicherebbe un bug reale nel nostro dequant/layout, non un limite
atteso della tecnica.

Carica il modello reale via vLLM (plain LLM(), non GCSGWorker — non
servono gli hook GCSG per questo confronto), estrae il vero modulo
MixtralMLP di un expert, lo pinna in GPU se offloaded (stesso fix
issue #10/#16 — chiamare un modulo offloaded fuori sequenza crasha
altrimenti), ne esegue il forward reale, e lo confronta contro il nostro
path CPU sullo stesso input.

Richiede due workaround locali specifici a questa macchina (Z8, WSL2),
non necessari sul pod Malmö (Linux reale) dove il checkpoint era già
stato caricato con successo più volte in questa stessa issue:
  - `VLLM_ATTENTION_BACKEND=XFORMERS`: il backend flash-attention di
    default crasha nel profiling interno di vLLM
    ("cu_seqlens_q must have dtype int32") con `LLM()` semplice (senza
    `worker_cls="scheduler.gcsg.GCSGWorker"`) — non indagato a fondo
    (probabile drift di versione flash-attn/vLLM su questa build
    locale), aggirato passando a XFormers invece di investigare la
    causa esatta (fuori scope per Fase 6a).
  - `hf_overrides={"head_dim": 128}`: senza, `num_heads` risolve a None
    e il calcolo della dimensione del blocco KV-cache crasha — lo stesso
    override già usato in `eval_mmlu_gcsg.py`, mancante qui perché
    questo script non passa da `GCSGWorker`.

Usage:
    PYTHONPATH=src VLLM_ATTENTION_BACKEND=XFORMERS python scripts/verify_awq_cpu_dequant_parity.py [--layer 0] [--expert 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from scheduler.gcsg import _ShadowExpertINT4

MODEL_PATH = Path("/data/nvme/models/mixtral-instruct-awq")

# ── vendorizzato da casper-hansen/AutoAWQ, awq/utils/packing_utils.py ────────
# (MIT license) — identico a verify_awq_cpu_dequant.py (Passo 1), duplicato
# qui per tenere questo script eseguibile in isolamento, stessa convenzione
# di bench_*.py in questo progetto (nessun cross-import tra scripts/).

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


def _cpu_shadow_from_module(module, hidden: int) -> _ShadowExpertINT4:
    """Estrae qweight/qzeros/scales REALI dal modulo MixtralMLP caricato da
    vLLM (non da un file separato come nel Passo 1 — stessi tensori che il
    kernel CUDA userà, garantito essere lo STESSO oggetto, non solo lo
    stesso file)."""
    quant_config = json.loads((MODEL_PATH / "quant_config.json").read_text())
    bits, group_size = quant_config["w_bit"], quant_config["q_group_size"]

    dequantized = {}
    for proj_name, layer_attr in (("w1", "w1"), ("w2", "w2"), ("w3", "w3")):
        linear = getattr(module, layer_attr)
        qweight = linear.qweight.detach().cpu()
        qzeros = linear.qzeros.detach().cpu()
        scales = linear.scales.detach().cpu()
        dequantized[proj_name] = dequantize_gemm(qweight, qzeros, scales, bits, group_size)

    w13 = torch.cat([dequantized["w1"].T, dequantized["w3"].T], dim=0).to(torch.float32)
    w2 = dequantized["w2"].T.to(torch.float32)
    return _ShadowExpertINT4([(w13, 1.0)], [(w2, 1.0)])


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
    module = model.model.layers[args.layer].block_sparse_moe.experts[args.expert]
    print(f"Modulo estratto: layer={args.layer} expert={args.expert} type={type(module).__name__}")

    # Fix issue #10/#16: un modulo offloaded chiamato fuori sequenza crasha
    # (device mismatch) — pinna esplicitamente su GPU se serve, stesso
    # principio già validato in _pin_awq_expert_to_gpu().
    first_param_device = next(module.parameters()).device
    was_offloaded = first_param_device.type != "cuda"
    if was_offloaded:
        print(f"Modulo offloaded ({first_param_device}) — pinning esplicito su GPU...")
        module = module.to("cuda")
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

    print("\nForward CPU (dequant + _ShadowExpertINT4, stessi tensori qweight/qzeros/scales)...")
    shadow = _cpu_shadow_from_module(module, hidden)
    with torch.no_grad():
        output_cpu = shadow(hidden_states_cpu.to(torch.float32), layer_id=0)
    print(f"output CPU: shape={tuple(output_cpu.shape)} "
          f"range=[{output_cpu.min().item():.4f}, {output_cpu.max().item():.4f}]")

    diff = output_real_cpu - output_cpu
    rel_l2_error = (diff.norm() / output_real_cpu.norm()).item()
    max_abs_diff = diff.abs().max().item()
    print(f"\nErrore relativo L2: {rel_l2_error:.6f}")
    print(f"Differenza assoluta massima: {max_abs_diff:.6f}")

    # Soglia stretta (non 0.5 come Fase 1 — qui NON c'è rumore di
    # quantizzazione atteso, entrambi i path leggono la STESSA informazione
    # quantizzata): 1e-2 relativo, margine sopra la sola precisione
    # fp16/fp32, non sopra il rumore AWQ.
    threshold = 1e-2
    if rel_l2_error < threshold:
        print(f"\nPARITÀ CONFERMATA (errore {rel_l2_error:.6f} < soglia {threshold}) — "
              "il dequant CPU produce lo stesso risultato del kernel CUDA reale, "
              "non solo un output plausibile.")
    else:
        print(f"\nPARITÀ FALLITA (errore {rel_l2_error:.6f} >= soglia {threshold}) — "
              "probabile bug reale nel dequant/layout CPU, da investigare prima di "
              "fidarsi della pipeline.")


if __name__ == "__main__":
    main()
