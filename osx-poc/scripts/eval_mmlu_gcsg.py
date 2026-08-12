#!/usr/bin/env python3
"""MMLU 5-shot evaluation — GCSGWorker on real Mixtral-8x7B.

First real run of the metric README.md's roadmap table calls out
("GCSG quality degradation < 2% (MMLU-5shot)") and tests/test_scheduler.py
skips ("TODO Sprint 3 — richiede vLLM integration + MMLU dataset").

Scope, stated plainly rather than glossed over:

  - Runs through GCSGWorker (hooks, request_id tracking, contamination
    bookkeeping all real — see LOGBOOK 2026-08-09), NOT plain vLLM.
  - UPDATED 2026-08-10 (issues #10/#16): shadow execution is no longer
    structurally disabled. _load_shadow_pool() now pins shadow-pool experts
    to GPU explicitly before registering them (both the Marlin and AWQ
    ModuleList paths — see LOGBOOK 2026-08-10), so shadow_activations is a
    real, non-zero measurement here, not a stub. This is the actual
    quality-degradation-from-shadow-contamination run the README's <2%
    target is about — not blocked anymore, though the comparison against
    the 72.3% hook-only baseline (LOGBOOK 2026-08-09) still needs to be
    drawn explicitly per run, see the printed GCSGGuard stats.
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
import json
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


def _score_outputs(
    meta_slice: list[tuple[str, int]], outputs_slice: list, letter_token_ids: dict[str, int],
) -> dict:
    """Punteggio su una fetta (meta, outputs) allineata — usata sia per-blocco
    (log incrementale, 2026-08-10, issue #10/#16: serve isolare QUALE blocco
    si blocca senza perdere il progresso dei blocchi già completati) sia per
    l'aggregato finale, stessa logica, non duplicata."""
    correct = 0
    unresolved = 0
    per_subject_correct: dict[str, int] = defaultdict(int)
    per_subject_total: dict[str, int] = defaultdict(int)

    for (subject, answer_idx), out in zip(meta_slice, outputs_slice):
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

    total = len(meta_slice)
    return {
        "total": total,
        "correct": correct,
        "unresolved": unresolved,
        "accuracy": correct / total if total else 0.0,
        "per_subject_correct": dict(per_subject_correct),
        "per_subject_total": dict(per_subject_total),
    }


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
    parser.add_argument(
        "--max-prompts", type=int, default=None,
        help="Tronca al primi N prompt dopo la costruzione (2026-08-10, issue "
             "#10/#16: run incrementali 8/16/32/64/128/252 per isolare a basso "
             "costo dove il throughput degrada, invece di scommettere sull'intero "
             "run da 570 alla cieca). Non cambia il campionamento per-subject.",
    )
    parser.add_argument(
        "--prompt-start", type=int, default=0,
        help="Indice di partenza nella lista prompt (2026-08-10, issue #10/#16: "
             "il blocco riproducibile non è per profondità di coda in UNA "
             "chiamata generate() — anche due chiamate da 32 nello STESSO "
             "processo si bloccano alla seconda — ma per stato che si accumula "
             "nello stesso processo/worker tra chiamate successive. Combinato con "
             "--max-prompts, seleziona una fetta [start:start+max) — usato per "
             "processare i 570 prompt come processi Docker SEPARATI, uno per "
             "fetta, ognuno con un GCSGWorker fresco: ogni run isolato n=8/16/32 "
             "finora ha sempre funzionato, solo il riuso dello stesso processo "
             "per più chiamate ha mai fallito).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Se impostato, chiama llm.generate() a blocchi di questa "
             "dimensione invece che su tutti i prompt insieme (2026-08-10, "
             "issue #10/#16: il run intero da 570 in una sola chiamata si "
             "bloccava in modo riproducibile intorno alla richiesta 27-31 — "
             "n=8/16/32 isolati, chiamati singolarmente, non l'hanno mai "
             "fatto. Sospetto: profondità della coda interna di vLLM con "
             "centinaia di richieste ancora in attesa, non un prompt "
             "specifico. Un blocco per chiamata limita quante richieste sono "
             "sottomesse al motore contemporaneamente — non elimina il "
             "sospetto, lo aggira). Nessun default: comportamento invariato "
             "(una sola chiamata) se non specificato esplicitamente.",
    )
    parser.add_argument(
        "--wire-tier-manager", action="store_true",
        help="Opt-in, default off (zero change to the existing awq_marlin "
             "baseline behavior unless passed): wires a real TierManager/EAT "
             "via GCSGWorker.configure_tier_manager() before LLM(...), same "
             "config as scripts/smoke_test_gcsg_tier_manager.py (issue #17 "
             "sub-goal 1). Forces quantization=awq (NOT awq_marlin) — the "
             "wiring only touches path 3 (AWQ ModuleList); awq_marlin would "
             "silently exercise the untouched Marlin path instead. This is "
             "sub-goal 3 (integrated-path MMLU rerun): the actual comparison "
             "against the 72.28%/72.3% baseline runs through this flag.",
    )
    parser.add_argument(
        "--results-file", type=str, default=None,
        help="Se impostato (solo con --chunk-size), scrive una riga JSON per "
             "blocco completato — indice, range prompt, accuratezza del "
             "blocco, shadow_activations cumulativo, timestamp — appesa e "
             "flushata subito dopo ogni blocco. Se un blocco successivo si "
             "blocca, questo file dice esattamente fin dove si è arrivato e "
             "con quale esito per blocco, senza perdere il lavoro già fatto.",
    )
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

    slice_end = (args.prompt_start + args.max_prompts) if args.max_prompts is not None else None
    if args.prompt_start or slice_end is not None:
        prompts = prompts[args.prompt_start:slice_end]
        meta = meta[args.prompt_start:slice_end]
        _log(f"Fetta [{args.prompt_start}:{slice_end if slice_end is not None else 'fine'}] "
             f"— {len(prompts)} prompt (--prompt-start/--max-prompts).")

    tier_manager = None
    if args.wire_tier_manager:
        from eat import ExpertAccessTable
        from tier import TierManager as _TierManager

        eat = ExpertAccessTable(capacity=1000, n_slots=4)
        tier_manager = _TierManager(eat=eat, nvme_path="/data/nvme", gpu_device=0)
        GCSGWorker.configure_tier_manager(tier_manager)
        _log("--wire-tier-manager: TierManager/EAT wired via "
             "GCSGWorker.configure_tier_manager() — forcing quantization=awq "
             "(path 3, the only one this wiring touches).")

    quantization = "awq" if args.wire_tier_manager else "awq_marlin"
    _log(f"Loading {MODEL_PATH} via GCSGWorker (shadow execution active — issues #10/#16), "
         f"quantization={quantization}...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        worker_cls="scheduler.gcsg.GCSGWorker",
        quantization=quantization,
        cpu_offload_gb=4,
        # gpu_memory_utilization=0.95, max_num_seqs=16, max_model_len=3328:
        # valori per AMBIENTI DI TEST/VALIDAZIONE, NON di produzione — la
        # concorrenza realistica di produzione va rivista separatamente
        # prima di qualunque deploy (issue #10/#16). Necessari perché ora lo
        # shadow pool pinna ~1.02-1.05GiB extra sempre residenti in GPU
        # (_load_shadow_pool(), commit issue #16) — con gpu_memory_utilization
        # =0.90 (il default usato finché lo shadow pool era in hook-only) il
        # budget KV-cache residuo crolla sotto il minimo (osservato: 0 blocchi
        # a max_model_len=3328, -0.22GiB residuo). 0.95 verificato
        # empiricamente (scripts/probe_kv_blocks.py, 2026-08-10): 503
        # blocchi, 8048 token di capacità, margine 2.42x per max_model_len.
        # max_model_len=3328 (non più 4096): calcolato dalla lunghezza reale
        # dei 570 prompt 5-shot MMLU (scripts/measure_mmlu_prompt_lengths.py
        # — min=282, p50=547, p90=1197, p99=2961, max=3306), arrotondato al
        # blocco (16 token) sopra il massimo osservato — copre l'intero
        # dataset di validazione senza escludere le domande più lunghe
        # (escluderle introdurrebbe lo stesso tipo di bias sistematico già
        # discusso per la copertura parziale delle layer Marlin, evitato
        # apposta). max_num_seqs=16: la leva "abbassalo per liberare memoria
        # di attivazione" non ha funzionato come atteso (attivazione scala
        # con max_model_len, non con max_num_seqs — verificato: max_num_seqs=1
        # NON ha ridotto l'attivazione), 16 è quindi solo un valore
        # ragionevole per il throughput di validazione, non una leva di
        # sicurezza memoria.
        gpu_memory_utilization=0.95,
        max_num_seqs=16,
        enforce_eager=True,
        max_model_len=3328,
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

    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    if args.chunk_size:
        n_chunks = -(-len(prompts) // args.chunk_size)
        _log(f"Running generate() a blocchi da {args.chunk_size} "
             f"({len(prompts)} prompt totali, {n_chunks} blocchi)...")
        results_fh = open(args.results_file, "a") if args.results_file else None
        outputs = []
        try:
            for chunk_index, chunk_start in enumerate(range(0, len(prompts), args.chunk_size)):
                chunk_end = min(chunk_start + args.chunk_size, len(prompts))
                chunk_prompts = prompts[chunk_start:chunk_end]
                chunk_meta = meta[chunk_start:chunk_end]
                _log(f"  blocco {chunk_index}/{n_chunks - 1} [{chunk_start}:{chunk_end}] "
                     f"({len(chunk_prompts)} prompt)...")
                chunk_outputs = llm.generate(chunk_prompts, sampling_params)
                outputs.extend(chunk_outputs)

                chunk_score = _score_outputs(chunk_meta, chunk_outputs, letter_token_ids)
                guard_stats_now = llm.llm_engine.model_executor.driver_worker.guard.stats()
                _log(f"  blocco {chunk_index}/{n_chunks - 1} completato — "
                     f"accuratezza {chunk_score['accuracy']:.1%} "
                     f"({chunk_score['correct']}/{chunk_score['total']}), "
                     f"shadow_activations cumulativo={guard_stats_now['shadow_activations']}")
                if results_fh:
                    results_fh.write(json.dumps({
                        "chunk_index": chunk_index,
                        "range": [chunk_start, chunk_end],
                        "n_prompts": len(chunk_prompts),
                        **chunk_score,
                        "shadow_activations_cumulative": guard_stats_now["shadow_activations"],
                        "elapsed_s": time.monotonic() - START,
                        "tier_manager_wired": args.wire_tier_manager,
                    }) + "\n")
                    results_fh.flush()
                    os.fsync(results_fh.fileno())
        finally:
            if results_fh:
                results_fh.close()
        _log(f"generate() a blocchi completato — {len(outputs)}/{len(prompts)} output.")
    else:
        _log("Running generate() on all prompts (max_tokens=1, logprobs=20)...")
        outputs = llm.generate(prompts, sampling_params)
        _log("generate() completed.")

    score = _score_outputs(meta, outputs, letter_token_ids)
    total, correct, unresolved, accuracy = (
        score["total"], score["correct"], score["unresolved"], score["accuracy"],
    )
    per_subject_correct = score["per_subject_correct"]
    per_subject_total = score["per_subject_total"]

    if not args.chunk_size and args.results_file:
        # Percorso non a blocchi (o "blocco esterno" — processo Docker
        # separato per fetta, vedi --prompt-start): una riga per invocazione,
        # stesso schema del logging incrementale a blocchi in-process, per
        # poter concatenare i risultati di più processi separati sullo
        # stesso file (2026-08-10, issue #10/#16).
        guard_stats_now = llm.llm_engine.model_executor.driver_worker.guard.stats()
        with open(args.results_file, "a") as fh:
            fh.write(json.dumps({
                "chunk_index": None,
                "range": [args.prompt_start, args.prompt_start + total],
                "n_prompts": total,
                **score,
                "shadow_activations_cumulative": guard_stats_now["shadow_activations"],
                "elapsed_s": time.monotonic() - START,
                "tier_manager_wired": args.wire_tier_manager,
            }) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    print(f"\n--- MMLU 5-shot risultati (GCSGWorker, shadow execution attiva, {total} domande) ---")
    print(f"Accuratezza complessiva: {accuracy:.1%} ({correct}/{total})")
    print(f"Non risolte (nessuna delle 4 lettere nei top-20 logprob): {unresolved}")

    print("\nPer soggetto (peggiori 10):")
    subject_acc = sorted(
        (
            (subj, per_subject_correct.get(subj, 0) / per_subject_total[subj], per_subject_total[subj])
            for subj in per_subject_total
        ),
        key=lambda t: t[1],
    )
    for subj, acc, n in subject_acc[:10]:
        print(f"  {subj}: {acc:.1%} ({n} domande)")

    guard_stats = llm.llm_engine.model_executor.driver_worker.guard.stats()
    print(f"\nGCSGGuard stats: {guard_stats}")
    if guard_stats.get("shadow_activations", 0) > 0:
        print(
            f"\nShadow execution DAVVERO attiva ({guard_stats['shadow_activations']} "
            f"attivazioni, activation_rate={guard_stats['activation_rate']:.1%}) — "
            f"issues #10/#16 chiuse per questo path. Confronto contro la baseline "
            f"hook-only (72.3%, 412/570, LOGBOOK 2026-08-09 — stesso identico "
            f"protocollo ma su un subset diverso se total != 570, il confronto "
            f"valido richiede lo stesso n): questa run={accuracy:.1%}."
        )
    else:
        print(
            "\nNOTA: shadow_activations è 0 — la shadow execution non è stata "
            "attivata in questa run (shadow pool non caricato, o nessun token ha "
            "superato le soglie di gating). Questo numero è una baseline "
            "hook-only, non la degradazione di qualità da contaminazione shadow "
            "che il target README (<2%) misura."
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
