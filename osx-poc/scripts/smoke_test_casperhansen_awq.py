#!/usr/bin/env python3
"""Verifica finale: casperhansen/mixtral-instruct-awq al posto del
checkpoint TheBloke difettoso (vedi LOGBOOK 2026-08-09 — root cause
vllm-project/vllm#2359).

Prima versione di questo script (quantization=awq_marlin, ZERO
cpu_offload_gb) si e' bloccata: GPU al 100%, VRAM quasi satura
(24303/24576 MiB), 21+ minuti senza che il log arrivasse nemmeno alla riga
"Loading model weights took X GB" — stessa firma dell'hang da 28 minuti
gia' visto nella primissima sessione di debug (sotto GCSGWorker, con
cpu_offload_gb rimosso). Stesso pattern su un checkpoint diverso e senza
GCSGWorker: l'hang senza cpu_offload_gb sembra un problema di scratch
space per il repacking AWQ->Marlin su questo stack (vLLM 0.6.6.post1 +
Ampere), indipendente dal checkpoint — non un sintomo del bug NaN del
file TheBloke.

Questa versione usa cpu_offload_gb=4 (la configurazione che in OGNI test
precedente, su qualunque checkpoint, ha sempre completato il caricamento
correttamente) e aggiunge watchdog + heartbeat per non restare bloccati
di nuovo senza un segnale diagnostico.

Segnale binario:
  - Testo sano su tutti i prompt -> checkpoint OK, root cause confermata
    (era il file TheBloke, non lo stack OSX). L'hang senza offload resta
    un problema separato, da tracciare ma non bloccante (si puo' sempre
    usare cpu_offload_gb=4 anche se non strettamente necessario per la
    VRAM).
  - Ancora NaN -> root cause NON confermata, riaprire l'indagine.
  - Timeout -> stessa classe di hang, questa volta anche CON offload:
    servirebbe capire se e' proprio cpu_offload_gb+Marlin il problema
    strutturale, indipendente sia dal checkpoint che dalla sua presenza/
    assenza.

Usage:
    PYTHONPATH=src python scripts/smoke_test_casperhansen_awq.py
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

    MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"

    _log(f"Inizializzo LLM — {MODEL_PATH}, quantization=awq_marlin, "
         f"cpu_offload_gb=4, vanilla (no worker_cls)...")
    llm = LLM(
        model=MODEL_PATH,
        quantization="awq_marlin",
        cpu_offload_gb=4,
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
        print(f"prompt={o.prompt!r}")
        print(f"  text={out.text!r}")
        print(f"  token_ids={tuple(out.token_ids)}")
        print(f"  finish_reason={out.finish_reason!r}")
        print(f"  cumulative_logprob={out.cumulative_logprob!r}")
        if out.text.strip():
            non_empty += 1

    print(f"\n{non_empty}/{len(outputs)} outputs non-empty.")
    if non_empty == len(outputs):
        print("\nSIGNAL: testo sano su tutti i prompt — checkpoint OK, "
              "root cause confermata (era il file TheBloke, non lo stack).")
    else:
        print("\nSIGNAL: ancora vuoto/NaN anche con questo checkpoint — "
              "root cause NON confermata, riaprire l'indagine.")

    sys.exit(0)


if __name__ == "__main__":
    main()
