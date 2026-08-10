#!/usr/bin/env python3
"""Isola la causa del crash trovato in verify_awq_manual_shadow_expert.py:
chiamare block_sparse_moe.experts[0](hidden_states) DIRETTAMENTE (fuori dal
forward sequenziale del modello, dopo che generate() e' gia' tornato) da'
`CUDA error: illegal memory access`, su quantization="awq" piano (NON
Marlin) con cpu_offload_gb=4, sul checkpoint reale casperhansen/
mixtral-instruct-awq. Rilevante perche' _AWQShadowExpert (gcsg.py:470) fa
esattamente questa chiamata da un hook, in produzione, e non e' mai stato
esercitato end-to-end su questo checkpoint/regime (solo sul tiny model,
senza cpu_offload_gb — vedi LOGBOOK 2026-08-09).

Due ipotesi indipendenti, con conseguenze diverse:
  H1 (lifetime del tensore): l'hidden_states catturato dall'hook durante
     generate() punta a memoria che l'allocator di vLLM ha gia' potuzialmente
     riassegnato una volta tornata la request — bug locale allo script di
     verifica, non un problema di _AWQShadowExpert.
  H2 (offload/chiamata fuori sequenza): il meccanismo di cpu_offload_gb (che
     sposta i pesi CPU->GPU via hook prima del forward normale di ogni
     layer) non si innesca quando il modulo viene chiamato direttamente,
     fuori dalla sequenza del modello — strutturale, riguarda
     _AWQShadowExpert in produzione.

Test, in ordine di costo (economico prima):
  1. hidden_states SINTETICO (torch.randn, stessa shape/dtype/device),
     stesso cpu_offload_gb=4. Se crasha comunque -> H1 esclusa, sospetto
     verso H2. Se non crasha -> il problema e' nel tensore catturato (H1),
     non nel meccanismo di chiamata diretta in se'.
  2. Solo se il test 1 NON crasha: richiama con l'hidden_states catturato
     davvero da hook, per confermare che quello (non il meccanismo in se')
     e' la causa.
  3. Solo se il test 1 crasha: richiama reference_module() una seconda
     volta (stesso input sintetico) nello stesso script, per distinguere
     deterministico (bug di stato piu' grossolano) da intermittente (piu'
     coerente con una race tra copia asincrona H2D e kernel launch).

Usage:
    PYTHONPATH=src python scripts/isolate_awq_shadow_call_crash.py
"""
from __future__ import annotations

import sys

import torch

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"
LAYER_ID = 0
EXPERT_ID = 0


def _try_forward(reference_module, hidden_states, label: str) -> bool:
    """Ritorna True se il forward completa senza eccezioni (dopo sync)."""
    print(f"\n--- {label} ---")
    print(f"input: shape={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}, "
          f"device={hidden_states.device}")
    try:
        with torch.no_grad():
            out = reference_module(hidden_states)
        torch.cuda.synchronize()
        finite = torch.isfinite(out).all().item()
        print(f"OK — output shape={tuple(out.shape)}, finite={finite}, "
              f"mean={out.mean().item():.6f}")
        return True
    except Exception as e:
        print(f"ECCEZIONE: {type(e).__name__}: {e}")
        return False


def main() -> None:
    from vllm import LLM, SamplingParams

    print(f"Loading {MODEL_PATH} (quantization=awq, cpu_offload_gb=4)...")
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
    layer0 = model.model.layers[LAYER_ID]
    reference_module = layer0.block_sparse_moe.experts[EXPERT_ID]

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
        print("FAIL: hook never fired")
        sys.exit(1)
    hs_captured = captured["hidden_states"]
    print(f"Captured real hidden_states: shape={tuple(hs_captured.shape)}, "
          f"dtype={hs_captured.dtype}, device={hs_captured.device}")

    hs_synthetic = torch.randn(
        hs_captured.shape, dtype=hs_captured.dtype, device=hs_captured.device,
    )

    test1_ok = _try_forward(reference_module, hs_synthetic, "Test 1: hidden_states SINTETICO, cpu_offload_gb=4")

    if test1_ok:
        print("\nTest 1 NON e' crashato — l'ipotesi si sposta su H1 (lifetime del "
              "tensore catturato). Eseguo Test 2 (hidden_states catturato davvero) "
              "per confermare.")
        _try_forward(reference_module, hs_captured, "Test 2: hidden_states CATTURATO da hook")
    else:
        print("\nTest 1 E' crashato con input sintetico fresco — H1 esclusa, il "
              "problema non e' il lifetime del tensore catturato. Sospetto verso "
              "H2 (offload/chiamata fuori sequenza). Eseguo Test 3 (stessa chiamata "
              "una seconda volta) per determinismo — ma il processo potrebbe essere "
              "gia' in uno stato CUDA corrotto dopo un illegal memory access "
              "(comportamento CUDA noto: gli errori possono essere 'sticky' per il "
              "resto del contesto), quindi un secondo crash qui non e' una prova "
              "pulita di determinismo — solo un secondo tentativo, il risultato va "
              "letto con quella riserva.")
        try:
            _try_forward(reference_module, hs_synthetic, "Test 3: stessa identica chiamata, ripetuta")
        except Exception as e:
            print(f"Test 3 non eseguibile — contesto CUDA probabilmente gia' "
                  f"corrotto dal Test 1: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
