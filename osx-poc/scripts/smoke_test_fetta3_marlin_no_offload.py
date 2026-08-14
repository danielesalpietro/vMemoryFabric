#!/usr/bin/env python3
"""Fetta 3 — awq_marlin, ZERO cpu_offload_gb, vanilla vLLM, con watchdog.

Fetta 1: NaN identico senza GCSGWorker -> GCSGWorker scagionato.
Fetta 2 (rivista): NaN identico con pin_memory forzato -> pin_memory scagionato.
Fetta 2 (corretta): NaN identico con awq puro + offload -> kernel Marlin
scagionato. L'unico fattore rimasto comune a OGNI riproduzione del bug
finora e' cpu_offload_gb stesso.

Questo test isola cpu_offload_gb direttamente: con awq_marlin i pesi
(18.83GB) entrano comodamente nel budget di 21.6GB
(gpu_memory_utilization=0.90 x 24GiB) SENZA bisogno di offload — l'unica
combinazione in cui "zero offload" e' testabile senza saturare la VRAM
come successo con awq puro.

Precedente noto (LOGBOOK 2026-08-09, sotto GCSGWorker, prima di isolare il
worker con Fetta 1): rimuovere cpu_offload_gb con awq_marlin causava un
hang di 28+ minuti, GPU al 100% di utilizzo, nessun progresso — killato
deliberatamente, mai confermato se fosse un hang infinito o solo
estremamente lento. Ipotesi di allora: il repacking AWQ->Marlin potrebbe
avere bisogno dello scratch space in VRAM che l'offload lascia libero
spostando temporaneamente pesi via.

Watchdog: timeout duro a 900s (15 min) via SIGTERM poi SIGKILL, PIU' un
heartbeat ogni 30s — cosi' se si blocca di nuovo, l'ultimo log vLLM
interlacciato con l'ultimo heartbeat dice ESATTAMENTE in quale fase
(caricamento pesi, repacking Marlin, profile_run KV-cache, primo forward)
si e' fermato, invece di essere solo tempo perso.

Segnale binario:
  - Completa e genera testo sano -> cpu_offload_gb CONFERMATO come causa
    del NaN, isolato in modo pulito e definitivo.
  - Completa ma produce NaN -> cpu_offload_gb scagionato anch'esso; il
    problema e' nel checkpoint/setup vLLM+CUDA stesso, non nell'offload.
  - Timeout/hang -> conferma il pattern gia' visto (indipendente da
    GCSGWorker stavolta), fase esatta nota dal log; serve capire la
    causa dell'hang separatamente, non piu' solo "evitarlo".

Usage:
    PYTHONPATH=src python scripts/smoke_test_fetta3_marlin_no_offload.py
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

START = time.monotonic()


def _elapsed() -> str:
    return f"T+{time.monotonic() - START:6.1f}s"


def _log(msg: str) -> None:
    print(f"[{_elapsed()}] {msg}", flush=True)


def _watchdog(timeout: float = 900.0) -> None:
    time.sleep(timeout)
    _log(f"WATCHDOG: timeout di {timeout:.0f}s raggiunto — nessun completamento. "
         f"Ultima fase nota: vedi ultimo log vLLM/heartbeat sopra. Invio SIGTERM.")
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(5)
    _log("WATCHDOG: SIGTERM non ha terminato il processo entro 5s — invio SIGKILL.")
    try:
        os.kill(os.getpid(), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _heartbeat(interval: float = 30.0) -> None:
    while True:
        time.sleep(interval)
        _log("heartbeat — processo ancora vivo, in attesa del prossimo log vLLM")


def main() -> None:
    threading.Thread(target=_watchdog, args=(900.0,), daemon=True).start()
    threading.Thread(target=_heartbeat, args=(30.0,), daemon=True).start()

    _log("Import vllm...")
    from vllm import LLM, SamplingParams

    MODEL_PATH = "/data/nvme/models/Mixtral-8x7B-Instruct-v0.1-AWQ"

    _log(f"Inizializzo LLM — {MODEL_PATH}, quantization=awq_marlin, "
         f"cpu_offload_gb NON impostato (default 0), vanilla (no worker_cls)...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq_marlin",
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=2048,
        hf_overrides={"head_dim": 128},
    )
    _log("LLM pronto — load_model()/profile_run()/KV-cache init completati.")

    cache_config = llm.llm_engine.cache_config
    _log(f"KV-cache blocks: num_gpu_blocks={getattr(cache_config, 'num_gpu_blocks', None)}")

    prompts = [
        "[INST] What is 2+2? [/INST]",
        "[INST] Write a one-line Python function that reverses a string. [/INST]",
        "[INST] Explain quantum entanglement in one sentence. [/INST]",
    ]
    _log("Lancio generate()...")
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=32, temperature=0.0, logprobs=1),
    )
    _log("generate() completato.")

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
            "\nSIGNAL: token sani con awq_marlin + ZERO cpu_offload_gb -> "
            "cpu_offload_gb CONFERMATO come causa del NaN, isolato in modo "
            "definitivo."
        )
    elif all_nan:
        print(
            "\nSIGNAL: ancora NaN anche con ZERO cpu_offload_gb -> "
            "cpu_offload_gb scagionato anch'esso. Il problema e' nel "
            "checkpoint stesso o nel setup vLLM+CUDA, non nell'offload."
        )
    else:
        print("\nSIGNAL: risultato misto — ispezionare manualmente sopra.")

    sys.exit(0)


if __name__ == "__main__":
    main()
