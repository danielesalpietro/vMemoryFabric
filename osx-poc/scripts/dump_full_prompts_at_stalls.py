from collections import defaultdict

from datasets import load_dataset

import sys
sys.path.insert(0, "scripts")
from eval_mmlu_gcsg import _build_prompt

STALL_POSITIONS = [33, 79, 86, 97]

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
for subject, examples in test_by_subject.items():
    few_shot = few_shot_by_subject[subject]
    for ex in examples:
        prompts.append(_build_prompt(subject, few_shot, ex))
        subjects.append(subject)

with open("prompts_full_dump.txt", "w", encoding="utf-8") as f:
    for pos in STALL_POSITIONS:
        f.write(f"\n\n{'='*80}\n=== POSIZIONE {pos} (subject={subjects[pos]}) ===\n{'='*80}\n\n")
        f.write(prompts[pos])
        # conta caratteri/pattern sospetti
        p = prompts[pos]
        f.write(f"\n\n--- stats: len={len(p)} chars, "
                f"underscore_runs(>=3)={p.count('___')}, "
                f"unicode_non_ascii={sum(1 for c in p if ord(c) > 127)}, "
                f"newlines={p.count(chr(10))} ---\n")

print("Scritto prompts_full_dump.txt")
