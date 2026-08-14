"""Issue #10/#16, 2026-08-10: le posizioni che si bloccano (~33, ~79) hanno
prompt anomali per lunghezza in token? Nessuna GPU necessaria — solo
tokenizer + dataset, stessa identica costruzione prompt di eval_mmlu_gcsg.py.
"""
from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

import sys
sys.path.insert(0, "scripts")
from eval_mmlu_gcsg import _build_prompt

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"
STALL_POSITIONS = [27, 28, 29, 30, 31, 32, 33, 34, 35, 78, 79, 80, 81, 82]

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

print(f"{len(prompts)} prompt totali, tokenizzo con {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
lengths = [len(tokenizer.encode(p)) for p in prompts]

print(f"\nlunghezza media generale: {sum(lengths)/len(lengths):.1f} token")
print(f"\n--- Posizioni intorno agli stall noti (~27-35, ~78-82) ---")
for pos in STALL_POSITIONS:
    if pos < len(prompts):
        print(f"  posizione {pos:3d}: subject={subjects[pos]:30s} "
              f"token={lengths[pos]:5d}")

print(f"\n--- Tutte le posizioni con token > 2500 (outlier lunghi) ---")
for pos, (n, subj) in enumerate(zip(lengths, subjects)):
    if n > 2500:
        print(f"  posizione {pos:3d}: subject={subj:30s} token={n:5d}")

print(f"\n--- Cambio di subject vicino alle posizioni di stall (confine tra soggetti?) ---")
prev_subj = None
for pos, subj in enumerate(subjects):
    if subj != prev_subj:
        print(f"  posizione {pos:3d}: inizio subject={subj}")
        prev_subj = subj
