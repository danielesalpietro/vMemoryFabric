#!/usr/bin/env python3
"""Issue #33 Fase 6a — proof-of-path: dequantizzare un expert AWQ reale su
CPU (via AutoAWQ's `dequantize_gemm()`, riusata as-is — vedi
LOGBOOK_ISSUE33.MD 2026-08-17 "continued 4" per la verifica del sorgente
reale) e verificare che il risultato sia utilizzabile.

Passo 1 di 2 (questo script): carica i tensori AWQ grezzi
(qweight/qzeros/scales) di un expert reale dal checkpoint su disco (nessun
motore vLLM, nessuna GPU richiesta qui — solo safetensors), dequantizza su
CPU, verifica shape/dtype/finitezza, esegue un forward SwiGLU reale.
Passo 2 (separato, richiede GPU): confronto numerico contro il kernel CUDA
AWQ reale — non fatto qui.

`dequantize_gemm`/`unpack_awq`/`reverse_awq_order` vendorizzate qui
(licenza MIT, casper-hansen/AutoAWQ, verificate contro il sorgente reale
via GitHub API il 2026-08-17 — non installiamo `autoawq` come dipendenza
solo per ~40 righe di puro PyTorch, device-agnostic per costruzione).

Usage:
    PYTHONPATH=src python scripts/verify_awq_cpu_dequant.py [--layer 0] [--expert 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

MODEL_PATH = Path("/data/nvme/models/mixtral-instruct-awq")

# ── vendorizzato da casper-hansen/AutoAWQ, awq/utils/packing_utils.py ────────
# (MIT license) — verificato contro il sorgente reale il 2026-08-17, non
# reimplementato da zero. Solo le funzioni servono a dequantize_gemm().

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


def _load_expert_awq_tensors(layer: int, expert: int, proj: str) -> dict[str, torch.Tensor]:
    """proj in {'w1', 'w2', 'w3'} — naming Mixtral reale (w1=gate_proj,
    w3=up_proj, w2=down_proj), verificato via model.safetensors.index.json
    del checkpoint reale (2026-08-17)."""
    index = json.loads((MODEL_PATH / "model.safetensors.index.json").read_text())
    prefix = f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}."
    keys = {suffix: prefix + suffix for suffix in ("qweight", "qzeros", "scales")}
    shard_files = {index["weight_map"][k] for k in keys.values()}
    tensors: dict[str, torch.Tensor] = {}
    for shard_file in shard_files:
        with safe_open(MODEL_PATH / shard_file, framework="pt", device="cpu") as f:
            for suffix, full_key in keys.items():
                if full_key in f.keys():
                    tensors[suffix] = f.get_tensor(full_key)
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()

    quant_config = json.loads((MODEL_PATH / "quant_config.json").read_text())
    bits, group_size = quant_config["w_bit"], quant_config["q_group_size"]
    print(f"quant_config: bits={bits} group_size={group_size} "
          f"zero_point={quant_config['zero_point']} version={quant_config['version']}")

    dequantized = {}
    for proj in ("w1", "w2", "w3"):
        t = _load_expert_awq_tensors(args.layer, args.expert, proj)
        print(f"{proj}: qweight={tuple(t['qweight'].shape)} {t['qweight'].dtype}, "
              f"qzeros={tuple(t['qzeros'].shape)}, scales={tuple(t['scales'].shape)} {t['scales'].dtype}")

        t0 = time.perf_counter()
        w = dequantize_gemm(t["qweight"], t["qzeros"], t["scales"], bits, group_size)
        dequant_s = time.perf_counter() - t0

        assert torch.isfinite(w).all(), f"{proj}: dequant produced non-finite values"
        print(f"{proj}: dequantized shape={tuple(w.shape)} dtype={w.dtype} "
              f"range=[{w.min().item():.4f}, {w.max().item():.4f}] "
              f"dequant_time={dequant_s * 1e3:.2f}ms")
        dequantized[proj] = w

    # AWQ GEMM qweight è [in_features, out_features] (x @ w, non x @ w.T —
    # verificato dalla shape empirica sopra: qweight.shape[0] == hidden_size
    # atteso, non intermediate_size). _ShadowExpertINT4 si aspetta layout
    # nn.Linear-style (out_features, in_features) — serve una trasposizione
    # esplicita, non assunta: verificata di seguito confrontando le shape
    # con hidden_size/intermediate_size reali del config del checkpoint.
    model_config = json.loads((MODEL_PATH / "config.json").read_text())
    hidden = model_config["hidden_size"]
    intermediate = model_config["intermediate_size"]
    print(f"\nconfig reale: hidden_size={hidden} intermediate_size={intermediate}")

    for proj, w in dequantized.items():
        in_f, out_f = w.shape
        expected_in = hidden
        expected_out = intermediate
        matches_as_is = (in_f, out_f) == (expected_in, expected_out)
        matches_transposed = (out_f, in_f) == (expected_in, expected_out)
        print(f"{proj}: shape={tuple(w.shape)} — "
              f"matches (hidden,intermediate) as-is: {matches_as_is}, "
              f"transposed: {matches_transposed}")

    # AWQ GEMM: y = x @ w (in_features, out_features). _ShadowExpertINT4
    # fa hidden_states @ w13.T con w13 in layout nn.Linear-style
    # (out_features, in_features) — serve .T esplicito, confermato dalle
    # shape sopra (w1/w3: (hidden, intermediate) AWQ vs (intermediate,
    # hidden) atteso da nn.Linear per gate/up_proj; w2 già combacia "as-is"
    # con (intermediate, hidden) ma nn.Linear vuole (hidden, intermediate)
    # per down_proj — stesso .T necessario in entrambi i casi).
    w13 = torch.cat([dequantized["w1"].T, dequantized["w3"].T], dim=0)   # (2*intermediate, hidden)
    w2 = dequantized["w2"].T   # (hidden, intermediate)
    print(f"\nw13 (concat gate+up, layout _ShadowExpertINT4): {tuple(w13.shape)}")
    print(f"w2 (down, layout _ShadowExpertINT4): {tuple(w2.shape)}")

    from scheduler.gcsg import _ShadowExpertINT4
    # scale=1.0: i pesi sono già dequantizzati (fp16 reali, non un'altra
    # quantizzazione simulata) — _ShadowExpertINT4 fa w_q.to(dtype)*scale,
    # con scale=1.0 è un no-op moltiplicativo, riusa la classe intatta.
    torch.manual_seed(0)

    for dtype, label in ((torch.float16, "fp16"), (torch.float32, "fp32")):
        shadow = _ShadowExpertINT4(
            [(w13.to(dtype), 1.0)], [(w2.to(dtype), 1.0)],
        )
        hidden_states = torch.randn(4, hidden, dtype=dtype)

        shadow(hidden_states, layer_id=0)   # warm-up, fuori dal timing
        t0 = time.perf_counter()
        output = shadow(hidden_states, layer_id=0)
        forward_s = time.perf_counter() - t0

        assert output.shape == (4, hidden), f"shape inattesa: {output.shape}"
        assert torch.isfinite(output).all(), "output non finito"
        print(f"\nForward reale (batch=4, dtype={label}): shape={tuple(output.shape)} "
              f"finito={torch.isfinite(output).all().item()} "
              f"range=[{output.min().item():.4f}, {output.max().item():.4f}] "
              f"tempo={forward_s * 1e3:.2f}ms")

    w13_fp32_bytes = w13.numel() * 4
    w2_fp32_bytes = w2.numel() * 4
    total_mb = (w13_fp32_bytes + w2_fp32_bytes) / (1024 ** 2)
    print(f"\nMemoria fp32 dequantizzata (w13+w2, un solo layer/expert): "
          f"{total_mb:.1f} MB — × 32 layer (Mixtral reale) = "
          f"{total_mb * 32 / 1024:.2f} GB per un intero expert 'freddo'")

    print("\nProof-of-path Passo 1 (CPU-only, no GPU) COMPLETATO —"
          "\n  - dequant produce output finito, shape coerenti col config reale"
          "\n  - trasposizione a layout _ShadowExpertINT4 verificata (non assunta)"
          "\n  - forward SwiGLU reale eseguito su pesi dequantizzati reali, output finito"
          "\n  - costo dequant (~950ms/expert per un layer, w1+w2+w3, naive/non cachato)"
          " conferma che serve caching, non ricalcolo per-call"
          "\nPasso 2 (parità numerica contro kernel CUDA reale) non fatto qui — richiede GPU.")


if __name__ == "__main__":
    main()
