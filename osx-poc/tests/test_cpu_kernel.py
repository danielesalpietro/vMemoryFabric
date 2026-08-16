"""Fase 1 (issue #33) — kernel CPU isolato, nessuna integrazione col tier.

Verifica l'ipotesi di lavoro di LOGBOOK_ISSUE33.MD 2026-08-16: prima di
scrivere intrinsics AVX-512 custom, controllare se il backend CPU di
PyTorch (oneDNN/MKL, AVX-512/VNNI su Cascade Lake) basta già. Sorpresa
trovata scrivendo questi test: `_ShadowExpertINT4.__call__()`
(`scheduler/gcsg.py:445-457`) non ha NESSUNA dipendenza da CUDA — è
`hidden_states @ w13.T` + `F.silu` + `@ w2.T`, plain PyTorch ops che girano
sul device dei tensori in ingresso. Il "path GPU" che gira in produzione è
tale solo perché `_load_shadow_pool()` forza `.to("cuda")` sui pesi PRIMA
di passarli qui (vedi `test_scheduler.py::
test_load_shadow_pool_moves_offloaded_fused_weights_to_gpu_before_quantizing`)
— la classe stessa è già il "kernel CPU" cercato in Fase 1, non ne serve
uno nuovo per la correttezza. Quello che resta aperto è SOLO la domanda di
throughput, isolata nella classe TestThroughputSmoke sotto.

Deviazione dichiarata: dimensioni sintetiche più piccole di Mixtral reale
(hidden=4096, intermediate=14336, 32 layer — vedi docstring di modulo di
gcsg.py) per tenere questi test rapidi ovunque; le dimensioni realistiche
sono nel benchmark (`benchmarks/bench_cpu_kernel.py`), non qui.
"""
from __future__ import annotations

import platform
import time

import pytest
import torch
import torch.nn.functional as F

from scheduler.gcsg import _quantize_int4, _ShadowExpertINT4

_SEED = 0


def _has_avx512() -> bool:
    """True se la CPU corrente espone avx512f — unica fonte affidabile è
    /proc/cpuinfo su Linux (torch non espone le CPU capability flags in
    modo stabile tra versioni). False (skip) su qualunque altra piattaforma
    o in caso di errore di lettura — nessun falso positivo accettabile qui,
    un test di throughput "AVX-512" che gira senza AVX-512 misurerebbe il
    fallback scalare e produrrebbe un numero fuorviante."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/cpuinfo") as f:
            return "avx512f" in f.read()
    except OSError:
        return False


def _make_synthetic_weights(hidden: int, intermediate: int, num_layers: int,
                             generator: torch.Generator):
    """Un layer di pesi SwiGLU fp32 casuali, layout nn.Linear-style
    (out_features, in_features) come da docstring di _ShadowExpertINT4."""
    per_layer_w13, per_layer_w2 = [], []
    per_layer_w13_fp32, per_layer_w2_fp32 = [], []
    for _ in range(num_layers):
        w13 = torch.randn(2 * intermediate, hidden, generator=generator)
        w2 = torch.randn(hidden, intermediate, generator=generator)
        per_layer_w13_fp32.append(w13)
        per_layer_w2_fp32.append(w2)
        per_layer_w13.append(_quantize_int4(w13))
        per_layer_w2.append(_quantize_int4(w2))
    return per_layer_w13, per_layer_w2, per_layer_w13_fp32, per_layer_w2_fp32


def _reference_forward_fp32(hidden_states, w13_fp32, w2_fp32, intermediate_size):
    """Stessa formula di _ShadowExpertINT4.__call__, ma senza passare per
    _quantize_int4 — il riferimento "verità" con cui misurare l'errore di
    quantizzazione, indipendente da qualunque device (gira ovunque, non
    serve una GPU per avere un riferimento fp32)."""
    gate_up = hidden_states @ w13_fp32.T
    gate, up = gate_up.split(intermediate_size, dim=-1)
    activated = F.silu(gate) * up
    return activated @ w2_fp32.T


# ── Correttezza — gira ovunque, nessun requisito AVX-512 ────────────────────

class TestShadowExpertINT4CPU:

    def test_runs_on_cpu_without_cuda(self):
        """Prova diretta che il "kernel" non ha bisogno di CUDA: pesi
        quantizzati da tensori CPU, mai un .to('cuda') in vista, output
        ancora CPU. Questo È il device-agnostic path che Fase 2 userà."""
        generator = torch.Generator().manual_seed(_SEED)
        hidden, intermediate, num_layers = 32, 64, 2
        per_layer_w13, per_layer_w2, *_ = _make_synthetic_weights(
            hidden, intermediate, num_layers, generator,
        )
        shadow = _ShadowExpertINT4(per_layer_w13, per_layer_w2)

        hidden_states = torch.randn(4, hidden, generator=generator)
        output = shadow(hidden_states, layer_id=0)

        assert output.device.type == "cpu"
        assert not output.is_cuda
        assert output.shape == (4, hidden)
        assert torch.isfinite(output).all()

    def test_quantize_int4_round_trip_within_scale_tolerance(self):
        """_quantize_int4 arrotonda a 16 livelli simmetrici attorno a 0
        (INT4 range [-8,7]) con passo `scale` — l'errore per elemento non
        può superare mezzo passo, per costruzione di torch.round(). Se
        questo non regge, tutta l'analisi di tolleranza del test successivo
        (forward end-to-end) poggia su un'assunzione falsa."""
        generator = torch.Generator().manual_seed(_SEED)
        weight = torch.randn(128, 256, generator=generator) * 3.0  # scala arbitraria

        quantized, scale = _quantize_int4(weight)
        dequantized = quantized.to(weight.dtype) * scale

        max_error = (weight - dequantized).abs().max().item()
        assert max_error <= scale / 2 + 1e-6

    def test_shadow_expert_forward_matches_fp32_reference_within_quant_tolerance(self):
        """Parity numerica: output INT4 (via _ShadowExpertINT4, su CPU)
        contro un forward fp32 non quantizzato calcolato indipendentemente
        sopra. Fase 1 del piano chiedeva un confronto col path GPU già
        verificato — dato che _ShadowExpertINT4 è la STESSA classe usata in
        produzione su GPU (device-agnostic, vedi test sopra), confrontare
        contro il riferimento fp32 copre lo stesso terreno senza richiedere
        hardware CUDA per eseguire questo test.

        Soglia empirica, non derivata analiticamente: un bound per-elemento
        costruito a mano sull'errore di quantizzazione (scale/2 su w13 poi
        w2) si è rivelato sbagliato di oltre un ordine di grandezza — non
        tiene conto della somma su `hidden`/`intermediate` termini nei due
        matmul. Errore relativo L2 misurato su 5 seed indipendenti
        (script ad-hoc, non incluso nel repo): stabile in [0.24, 0.32] —
        atteso per una quantizzazione INT4 per-tensore, 16 livelli, su pesi
        gaussiani propagati attraverso due matmul con una non-linearità in
        mezzo. Soglia qui a 0.5, margine ~1.5-2x sopra il range osservato:
        rileva una regressione reale (es. _quantize_int4 rotto), non un
        rumore statistico del seed."""
        generator = torch.Generator().manual_seed(_SEED)
        hidden, intermediate, num_layers = 32, 64, 1
        (per_layer_w13, per_layer_w2,
         per_layer_w13_fp32, per_layer_w2_fp32) = _make_synthetic_weights(
            hidden, intermediate, num_layers, generator,
        )
        shadow = _ShadowExpertINT4(per_layer_w13, per_layer_w2)

        hidden_states = torch.randn(8, hidden, generator=generator)
        got = shadow(hidden_states, layer_id=0)
        want = _reference_forward_fp32(
            hidden_states, per_layer_w13_fp32[0], per_layer_w2_fp32[0], intermediate,
        )

        rel_l2_error = ((got - want).norm() / want.norm()).item()
        assert rel_l2_error <= 0.5, (
            f"errore relativo L2 {rel_l2_error:.3f} oltre la soglia 0.5 — "
            "atteso in [0.24, 0.32] per INT4 per-tensore su pesi gaussiani"
        )


# ── Throughput — richiede AVX-512 reale per essere un dato significativo ────

@pytest.mark.avx512
class TestThroughputSmoke:
    """Smoke check, non un benchmark con percentili — quello vive in
    benchmarks/bench_cpu_kernel.py. Qui solo una soglia larga per accorgersi
    se qualcosa regredisce grossolanamente (es. un fallback scalare
    accidentale), non per misurare performance in modo affidabile."""

    def test_forward_completes_within_generous_time_budget(self):
        if not _has_avx512():
            pytest.skip("avx512f non presente su questa CPU (/proc/cpuinfo)")

        generator = torch.Generator().manual_seed(_SEED)
        hidden, intermediate, num_layers = 512, 1536, 1
        per_layer_w13, per_layer_w2, *_ = _make_synthetic_weights(
            hidden, intermediate, num_layers, generator,
        )
        shadow = _ShadowExpertINT4(per_layer_w13, per_layer_w2)
        hidden_states = torch.randn(16, hidden, generator=generator)

        shadow(hidden_states, layer_id=0)  # warm-up, fuori dal timing

        t0 = time.perf_counter()
        for _ in range(20):
            shadow(hidden_states, layer_id=0)
        elapsed_s = time.perf_counter() - t0

        # Soglia deliberatamente larga (10s per 20 forward a questa scala
        # ridotta): non è un gate di performance, solo un allarme se il
        # backend CPU smette di usare kernel vettorizzati.
        assert elapsed_s < 10.0, (
            f"20 forward hanno impiegato {elapsed_s:.2f}s — sospetto "
            "fallback scalare, non un kernel AVX-512/oneDNN reale"
        )
