#!/usr/bin/env python3
"""Fetta 1 — isola GCSGWorker dal NaN bug (vedi LOGBOOK 2026-08-09).

Stesso checkpoint, stessa config esatta di
scripts/smoke_test_gcsg_mixtral8x7b.py (quantization="awq_marlin",
cpu_offload_gb=4, hf_overrides head_dim=128), ma LLM() SENZA worker_cls —
vLLM vanilla, zero GCSGWorker, zero hook .gate.

Segnale binario:
  - NaN anche qui (token_ids tutti 0, cumulative_logprob=None) -> il bug e'
    a monte di GCSGWorker (quantizzazione/cpu_offload_gb/kernel Marlin),
    coerente con vLLM issue #21864 (AWQ Marlin + UVA cpu offload) e #7204
    (GPTQ Marlin + cpu_offload_gb). GCSGWorker scagionato.
  - Testo sano -> il bug e' negli hook .gate di GCSGWorker, non a monte.

Usage:
    PYTHONPATH=src python scripts/smoke_test_fetta1_vanilla.py
"""
from __future__ import annotations

from vllm import LLM, SamplingParams

MODEL_PATH = "/data/nvme/models/Mixtral-8x7B-Instruct-v0.1-AWQ"


def main() -> None:
    print(f"Loading {MODEL_PATH} — vLLM VANILLA (no worker_cls), "
          f"quantization=awq_marlin, cpu_offload_gb=4 ...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq_marlin",
        cpu_offload_gb=4,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=2048,
        hf_overrides={"head_dim": 128},
    )
    print("load_model() completed — vanilla vLLM, no GCSGWorker attached.")

    cache_config = llm.llm_engine.cache_config
    num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
    print(f"KV-cache blocks: num_gpu_blocks={num_gpu_blocks}")

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
        cum_logprob = out.cumulative_logprob
        print(f"prompt={o.prompt!r}")
        print(f"  text={out.text!r}")
        print(f"  token_ids={token_ids}")
        print(f"  finish_reason={out.finish_reason!r}")
        print(f"  cumulative_logprob={cum_logprob!r}")
        if out.text.strip():
            non_empty += 1

    print(f"\n{non_empty}/{len(outputs)} outputs non-empty.")

    all_nan_signature = all(
        set(o.outputs[0].token_ids) == {0} and o.outputs[0].cumulative_logprob is None
        for o in outputs
    )
    if all_nan_signature:
        print(
            "\nSIGNAL: NaN signature reproduced WITHOUT GCSGWorker — "
            "bug is upstream (quantization/cpu_offload_gb/Marlin kernel). "
            "GCSGWorker is cleared."
        )
    elif non_empty == len(outputs):
        print(
            "\nSIGNAL: clean generation WITHOUT GCSGWorker — "
            "bug is in GCSGWorker's .gate hooks / execute_model(), not upstream."
        )
    else:
        print(
            "\nSIGNAL: mixed/partial result — neither clean signal, "
            "needs manual inspection of the per-prompt output above."
        )


if __name__ == "__main__":
    main()
