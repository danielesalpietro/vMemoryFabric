#!/usr/bin/env python3
"""MMLU 5-shot evaluation — GCSGWorker on real Mixtral-8x7B.

First real run of the metric README.md's roadmap table calls out
("GCSG quality degradation < 2% (MMLU-5shot)") and tests/test_scheduler.py
skips ("TODO Sprint 3 — richiede vLLM integration + MMLU dataset").

Scope, stated plainly rather than glossed over:

  - Runs through GCSGWorker (hooks, request_id tracking, contamination
    bookkeeping all real — see LOGBOOK 2026-08-09), NOT plain vLLM.
  - GCSGWorker is currently in hook-only mode: _load_shadow_pool() fails
    on this checkpoint's Marlin-packed FusedMoE weights (GitHub issue #10,
    filed same session) and load_model() catches that and disables shadow
    execution rather than crashing. shadow_activations stays at 0
    structurally — this run measures baseline generation quality THROUGH
    GCSGWorker's hook path, not the quality degradation FROM shadow
    contamination the README's <2% target is actually about. That
    measurement is blocked until issue #10 closes.
  - Subsample, not the full 14042-question test set: 57 subjects x up to
    10 questions/subject (first 10 in dataset order) = up to 570
    questions, to keep a single run tractable. Declared here as a
    same-session validation pass, not a claim of the full-benchmark
    number — same honesty standard PT-PEP's validation used (LOGBOOK
    2026-08-08: "declared limitation ... not presented as independent
    validation").
  - Standard 5-shot protocol (Hendrycks et al. 2020): 5 few-shot examples
    per subject from cais/mmlu's "dev" split (285 rows = 5 x 57 subjects,
    exactly the standard few-shot set), prepended to each test question.
    Scored via next-token logprob comparison among the four answer-letter
    tokens (A/B/C/D), not by generating free text and parsing it — the
    standard MMLU scoring method, avoids parsing ambiguity entirely.

Usage:
    PYTHONPATH=src python scripts/eval_mmlu_gcsg.py [--n-per-subject 10]
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from collections import defaultdict

from datasets import load_dataset

from scheduler.gcsg import GCSGWorker

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"
LETTERS = ["A", "B", "C", "D"]

START = time.monotonic()


def _elapsed() -> str:
    return f"T+{time.monotonic() - START:6.1f}s"


def _log(msg: str) -> None:
    print(f"[{_elapsed()}] {msg}", flush=True)


def _watchdog(timeout: float) -> None:
    time.sleep(timeout)
    _log(f"WATCHDOG: timeout di {timeout:.0f}s raggiunto — nessun completamento. Invio SIGTERM.")
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
        _log("heartbeat — processo ancora vivo, in attesa del prossimo log")


def _format_question(question: str, choices: list[str], answer_letter: str | None) -> str:
    lines = [question]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append(f"Answer:{' ' + answer_letter if answer_letter else ''}")
    return "\n".join(lines)


def _build_prompt(subject: str, few_shot: list[dict], question: dict) -> str:
    header = (
        f"The following are multiple choice questions (with answers) about "
        f"{subject.replace('_', ' ')}.\n\n"
    )
    shots = "\n\n".join(
        _format_question(ex["question"], ex["choices"], LETTERS[ex["answer"]])
        for ex in few_shot
    )
    test_q = _format_question(question["question"], question["choices"], None)
    return header + shots + "\n\n" + test_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-subject", type=int, default=10)
    parser.add_argument("--watchdog-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    threading.Thread(target=_watchdog, args=(args.watchdog_timeout,), daemon=True).start()
    threading.Thread(target=_heartbeat, args=(30.0,), daemon=True).start()

    _log("Loading cais/mmlu (all subjects, dev + test splits)...")
    ds = load_dataset("cais/mmlu", "all")
    dev, test = ds["dev"], ds["test"]

    few_shot_by_subject: dict[str, list[dict]] = defaultdict(list)
    for ex in dev:
        few_shot_by_subject[ex["subject"]].append(ex)

    test_by_subject: dict[str, list[dict]] = defaultdict(list)
    for ex in test:
        subj = ex["subject"]
        if len(test_by_subject[subj]) < args.n_per_subject:
            test_by_subject[subj].append(ex)

    prompts: list[str] = []
    meta: list[tuple[str, int]] = []   # (subject, correct_answer_idx)
    for subject, examples in test_by_subject.items():
        few_shot = few_shot_by_subject[subject]
        for ex in examples:
            prompts.append(_build_prompt(subject, few_shot, ex))
            meta.append((subject, ex["answer"]))

    _log(f"Built {len(prompts)} prompts across {len(test_by_subject)} subjects "
         f"({args.n_per_subject}/subject cap).")

    _log(f"Loading {MODEL_PATH} via GCSGWorker (hook-only mode — see issue #10)...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        worker_cls="scheduler.gcsg.GCSGWorker",
        quantization="awq_marlin",
        cpu_offload_gb=4,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_model_len=4096,
        hf_overrides={"head_dim": 128},
    )
    _log("LLM ready.")

    tokenizer = llm.get_tokenizer()
    letter_token_ids: dict[str, int] = {}
    for letter in LETTERS:
        # " A" style — the natural continuation after "Answer:" in the prompt.
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_token_ids[letter] = ids[-1]
    _log(f"Answer-letter token ids: {letter_token_ids}")

    _log("Running generate() on all prompts (max_tokens=1, logprobs=20)...")
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=1, temperature=0.0, logprobs=20),
    )
    _log("generate() completed.")

    correct = 0
    unresolved = 0
    per_subject_correct: dict[str, int] = defaultdict(int)
    per_subject_total: dict[str, int] = defaultdict(int)

    for (subject, answer_idx), out in zip(meta, outputs):
        per_subject_total[subject] += 1
        top_logprobs = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}

        best_letter = None
        best_logprob = float("-inf")
        for letter, token_id in letter_token_ids.items():
            entry = top_logprobs.get(token_id)
            if entry is not None and entry.logprob > best_logprob:
                best_logprob = entry.logprob
                best_letter = letter

        if best_letter is None:
            unresolved += 1
            continue

        predicted_idx = LETTERS.index(best_letter)
        if predicted_idx == answer_idx:
            correct += 1
            per_subject_correct[subject] += 1

    total = len(meta)
    accuracy = correct / total if total else 0.0

    print(f"\n--- MMLU 5-shot risultati (GCSGWorker, hook-only, {total} domande) ---")
    print(f"Accuratezza complessiva: {accuracy:.1%} ({correct}/{total})")
    print(f"Non risolte (nessuna delle 4 lettere nei top-20 logprob): {unresolved}")

    print("\nPer soggetto (peggiori 10):")
    subject_acc = sorted(
        (
            (subj, per_subject_correct[subj] / per_subject_total[subj], per_subject_total[subj])
            for subj in per_subject_total
        ),
        key=lambda t: t[1],
    )
    for subj, acc, n in subject_acc[:10]:
        print(f"  {subj}: {acc:.1%} ({n} domande)")

    guard_stats = llm.llm_engine.model_executor.driver_worker.guard.stats()
    print(f"\nGCSGGuard stats: {guard_stats}")
    print(
        "\nNOTA: shadow_activations resta 0 per costruzione (shadow pool non "
        "caricato, vedi issue #10) — questo numero è la baseline attraverso "
        "GCSGWorker in modalità hook-only, NON la degradazione di qualità da "
        "contaminazione shadow che il target README (<2%) misura."
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
