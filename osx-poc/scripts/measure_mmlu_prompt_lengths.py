"""Misura one-off: lunghezza reale in token dei prompt 5-shot MMLU costruiti
da eval_mmlu_gcsg.py, con il tokenizer reale del checkpoint — per calcolare
max_model_len/max_num_seqs da dati reali (issue #10/#16 punto 3), non da un
numero tondo scelto a caso. Nessuna GPU necessaria (solo tokenizer + dataset).
"""
from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

import sys
sys.path.insert(0, "scripts")
from eval_mmlu_gcsg import _build_prompt  # riusa la stessa identica costruzione prompt

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"

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
for subject, examples in test_by_subject.items():
    few_shot = few_shot_by_subject[subject]
    for ex in examples:
        prompts.append(_build_prompt(subject, few_shot, ex))

print(f"{len(prompts)} prompt costruiti, tokenizzo con {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
lengths = [len(tokenizer.encode(p)) for p in prompts]
lengths.sort()

n = len(lengths)
print(f"n={n}")
print(f"min={lengths[0]}")
print(f"p50={lengths[n // 2]}")
print(f"p90={lengths[int(n * 0.90)]}")
print(f"p99={lengths[int(n * 0.99)]}")
print(f"max={lengths[-1]}")
