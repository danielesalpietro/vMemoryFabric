#!/usr/bin/env python3
"""Fetta 2 (rivista) — forza pin_memory=True sotto WSL, isola la causa del NaN.

Fetta 1 (vedi LOGBOOK 2026-08-09) ha confermato: il NaN si riproduce
identico SENZA GCSGWorker (vLLM vanilla) -> bug a monte. Il log di quel run
ha rivelato un indizio nuovo:

    WARNING interface.py:236 Using 'pin_memory=False' as WSL is detected.

vLLM disabilita pin_memory per l'host memory di cpu_offload_gb perche'
rileva WSL (vllm.platforms.interface.in_wsl(), chiamata da
Platform.is_pin_memory_available()). Ipotesi: il repacking Marlin/AWQ per
cpu_offload_gb si aspetta un trasferimento CPU->GPU su memoria pinnata
(UVA) — coerente col titolo di vLLM issue #21864 ("UVA CPU Offload") — e
riceve invece memoria paginabile, producendo dati corrotti -> NaN.

NOTA sul monkey-patch: la funzione reale in vllm==0.6.6.post1 e'
vllm.platforms.interface.in_wsl() (modulo-level), NON
vllm.utils.is_in_wsl — verificato via grep + hasattr() prima di scrivere
questo script; quel nome non esiste in questa versione e un patch su di
esso sarebbe stato un no-op silenzioso.

Segnale binario:
  - Token sani -> causa confermata: pin_memory=False + Marlin UVA = NaN.
  - Ancora NaN -> pin_memory non e' la causa; tornare alla Fetta 2
    originale (quantization="awq", zero cpu_offload_gb, 18.83GB pesi
    entrano nei 21.6GB disponibili senza offload).

Usage:
    PYTHONPATH=src python scripts/smoke_test_fetta2_pinmemory.py
"""
from __future__ import annotations

import vllm.platforms.interface as _iface

_original_in_wsl = _iface.in_wsl
_iface.in_wsl = lambda: False
print(f"Patched vllm.platforms.interface.in_wsl: {_original_in_wsl()} -> {_iface.in_wsl()}")

from vllm import LLM, SamplingParams  # noqa: E402  (import dopo il patch, apposta)

MODEL_PATH = "/data/nvme/models/Mixtral-8x7B-Instruct-v0.1-AWQ"


def main() -> None:
    print(f"Loading {MODEL_PATH} — vLLM VANILLA (no worker_cls), "
          f"quantization=awq_marlin, cpu_offload_gb=4, pin_memory forzato ...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq_marlin",
        cpu_offload_gb=4,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=2048,
        hf_overrides={"head_dim": 128},
    )
    print("load_model() completed.")

    prompts = [
        "[INST] What is 2+2? [/INST]",
        "[INST] Write a one-line Python function that reverses a string. [/INST]",
        "[INST] Explain quantum entanglement in one sentence. [/INST]",
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=32, temperature=0.0, logprobs=1),
    )

    print("\n--- risultati per prompt ---")
    non_empty = 0
    for o in outputs:
        out = o.outputs[0]
        token_ids = tuple(out.token_ids)
        print(f"prompt={o.prompt!r}")
        print(f"  text={out.text!r}")
        print(f"  token_ids={token_ids}")
        print(f"  finish_reason={out.finish_reason!r}")
        print(f"  cumulative_logprob={out.cumulative_logprob!r}")
        if out.text.strip():
            non_empty += 1

    print(f"\n{non_empty}/{len(outputs)} outputs non-empty.")

    def _is_nan_signature(o) -> bool:
        out = o.outputs[0]
        cl = out.cumulative_logprob
        cl_is_nan_or_none = cl is None or (isinstance(cl, float) and cl != cl)
        return set(out.token_ids) == {0} and cl_is_nan_or_none

    all_nan = all(_is_nan_signature(o) for o in outputs)
    if non_empty == len(outputs) and not all_nan:
        print(
            "\nSIGNAL: token sani con pin_memory forzato -> CAUSA CONFERMATA: "
            "pin_memory=False (WSL detection) + Marlin/AWQ UVA offload = NaN. "
            "Fix: aggiungere il monkey-patch alla config standard del PoC."
        )
    elif all_nan:
        print(
            "\nSIGNAL: ancora NaN con pin_memory forzato -> pin_memory NON e' "
            "la causa. Tornare alla Fetta 2 originale (quantization='awq', "
            "zero cpu_offload_gb)."
        )
    else:
        print("\nSIGNAL: risultato misto — ispezionare manualmente sopra.")


if __name__ == "__main__":
    main()
