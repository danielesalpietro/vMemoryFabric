"""Issue #10/#16, 2026-08-10: dump del testo REALE dei prompt alle posizioni
di stallo note, per analisi qualitativa (dominio? notazione? struttura?)
invece di sole feature strutturali (lunghezza, confine soggetto — già
escluse). Nessuna GPU necessaria.
"""
from collections import defaultdict

from datasets import load_dataset

import sys
sys.path.insert(0, "scripts")
from eval_mmlu_gcsg import _build_prompt

STALL_POSITIONS = [33, 79, 86, 97]
GOOD_POSITIONS = list(range(0, 16)) + list(range(64, 80))

ds = load_dataset("cais/mmlu", "all")
dev, test = ds["dev"], ds["test"]

few_shot_by_subject = defaultdict(list)
for ex in dev:
    few_shot_by_subject[ex["subject"]].append(ex)

test_by_subject = defaultdict(list)
for ex in test:
    subj = ex["subject"]
    if len(test_by_subject[subj]) < 10:
        test_by_subject[subj].append(ex)

prompts = []
subjects = []
raw_questions = []
for subject, examples in test_by_subject.items():
    few_shot = few_shot_by_subject[subject]
    for ex in examples:
        prompts.append(_build_prompt(subject, few_shot, ex))
        subjects.append(subject)
        raw_questions.append(ex["question"])

with open("prompts_dump.txt", "w", encoding="utf-8") as f:
    f.write("=== POSIZIONI DI STALLO (solo la domanda finale, non il prompt 5-shot intero) ===\n\n")
    for pos in STALL_POSITIONS:
        f.write(f"--- posizione {pos} (subject={subjects[pos]}) ---\n")
        f.write(raw_questions[pos] + "\n\n")

    f.write("\n=== POSIZIONI BUONE (solo la domanda finale) ===\n\n")
    for pos in GOOD_POSITIONS:
        f.write(f"--- posizione {pos} (subject={subjects[pos]}) ---\n")
        f.write(raw_questions[pos] + "\n\n")

    f.write("\n=== PROMPT COMPLETO alla posizione 33 (5-shot + domanda) ===\n\n")
    f.write(prompts[33] + "\n")

print("Scritto prompts_dump.txt")
