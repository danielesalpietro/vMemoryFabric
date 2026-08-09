#!/usr/bin/env python3
"""Fetta 2 (corretta) — isola il kernel Marlin, a parita' di cpu_offload_gb.

Il tentativo precedente (quantization="awq", cpu_offload_gb=0) era
inconcludente: i pesi non-Marlin (22.97GB, non compattati) superano da soli
il budget di 21.6GB (gpu_memory_utilization=0.90 x 24GiB) e il run e'
crashato per overcommit di VRAM dopo ~30 min, non per il bug NaN. quantization
="awq" puro non e' caricabile su questa GPU senza offload, in nessuna
configurazione di gpu_memory_utilization (anche a 0.99 restano solo 23.76GB,
ancora sotto i 22.97GB dei soli pesi).

Confronto corretto: cpu_offload_gb=4 in ENTRAMBI i branch (l'unica config in
cui "awq" puro carica per intero), cosi' la sola variabile che cambia e' il
kernel di dequantizzazione (Marlin vs plain AWQ / mixtral_quant.py).

Fetta 1: NaN identico senza GCSGWorker (awq_marlin + offload) -> bug a monte.
Fetta 2 rivista: NaN identico con pin_memory forzato -> pin_memory scagionato.
Questo test: awq puro (non marlin) + stesso cpu_offload_gb=4, vanilla vLLM.

Segnale binario:
  - Token sani -> il kernel Marlin e' il colpevole specifico. Workaround:
    quantization="awq" in produzione (piu' lento su Ampere ma corretto).
  - Ancora NaN -> il fattore comune resta cpu_offload_gb stesso,
    indipendentemente dal kernel di quantizzazione.

Usage:
    PYTHONPATH=src python scripts/smoke_test_fetta2_awq_with_offload.py
"""
from __future__ import annotations

from vllm import LLM, SamplingParams

MODEL_PATH = "/data/nvme/models/Mixtral-8x7B-Instruct-v0.1-AWQ"


def main() -> None:
    print(f"Loading {MODEL_PATH} — vLLM VANILLA (no worker_cls), "
          f"quantization=awq (NON marlin), cpu_offload_gb=4 ...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq",
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
            "\nSIGNAL: token sani con awq puro + offload -> il kernel "
            "Marlin e' il colpevole specifico. Workaround: "
            "quantization='awq' in produzione."
        )
    elif all_nan:
        print(
            "\nSIGNAL: ancora NaN con awq puro + offload -> il fattore "
            "comune resta cpu_offload_gb stesso, indipendentemente dal "
            "kernel di quantizzazione."
        )
    else:
        print("\nSIGNAL: risultato misto — ispezionare manualmente sopra.")


if __name__ == "__main__":
    main()
