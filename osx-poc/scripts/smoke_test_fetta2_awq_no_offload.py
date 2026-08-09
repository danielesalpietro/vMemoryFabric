#!/usr/bin/env python3
"""Fetta 2 (originale) — isola Marlin/cpu_offload_gb da "awq puro, zero offload".

Fetta 1: NaN si riproduce identico senza GCSGWorker -> bug a monte.
Fetta 2 (rivista): NaN si riproduce identico con pin_memory forzato a True
sotto WSL -> pin_memory scagionato.

Restano due sospetti: il kernel Marlin di dequantizzazione AWQ->Marlin su
Ampere, e/o cpu_offload_gb in generale (indipendentemente dal pinning).
Questo test isola entrambi insieme: quantization="awq" (percorso
mixtral_quant.py, ModuleList di MixtralMLP — NON FusedMoE/Marlin) e zero
cpu_offload_gb. I pesi (18.83GB) entrano nei 21.6GB disponibili
(gpu_memory_utilization=0.90 x 24GiB) senza offload, quindi non dovrebbe
incontrare l'hang osservato in precedenza con awq_marlin+niente offload.

Segnale binario:
  - Token sani -> causa confinata a Marlin<->cpu_offload_gb. Workaround
    immediato: quantization="awq" in produzione (kernel non ottimizzato su
    Ampere ma corretto).
  - Ancora NaN -> il problema e' nel checkpoint stesso o nel setup
    vLLM/CUDA, non nell'offload né nel path Marlin specificamente.

Usage:
    PYTHONPATH=src python scripts/smoke_test_fetta2_awq_no_offload.py
"""
from __future__ import annotations

from vllm import LLM, SamplingParams

MODEL_PATH = "/data/nvme/models/Mixtral-8x7B-Instruct-v0.1-AWQ"


def main() -> None:
    print(f"Loading {MODEL_PATH} — vLLM VANILLA (no worker_cls), "
          f"quantization=awq (NON marlin), cpu_offload_gb=0 ...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq",
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
            "\nSIGNAL: token sani con awq puro + zero offload -> causa "
            "confinata a Marlin<->cpu_offload_gb. Workaround: "
            "quantization='awq' in produzione."
        )
    elif all_nan:
        print(
            "\nSIGNAL: ancora NaN anche con awq puro + zero offload -> "
            "problema nel checkpoint stesso o nel setup vLLM/CUDA, non "
            "nell'offload né nel path Marlin."
        )
    else:
        print("\nSIGNAL: risultato misto — ispezionare manualmente sopra.")


if __name__ == "__main__":
    main()
