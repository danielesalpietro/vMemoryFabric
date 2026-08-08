#!/usr/bin/env python3
"""OSX-PoC — Build PT-PEP TF-IDF classifier (Sprint 3 baseline)

Extracts per-domain vocabularies via TF-IDF from public datasets instead of
manual keyword curation (subjective, slow, hard to defend in the paper).
For each of the 8 PT-PEP domains, pulls a fixed-size slice from a public
HuggingFace dataset, splits it 80/20 (train/held-out), fits a single shared
TfidfVectorizer on the pooled training text, and computes one centroid per
domain (mean TF-IDF vector of that domain's training docs).

predict() at inference time is then just: vectorize the prompt, cosine
similarity against the 8 centroids, normalize to a distribution.

Outputs:
    osx-poc/models/ptpep_tfidf_v1.joblib   — {vectorizer, centroids, domains}
    osx-poc/tests/fixtures/ptpep_validation.json — 400 held-out (text, domain)
        pairs (50/domain), same-distribution held-out per domain, NOT a
        different-source OOD set (see LOGBOOK / paper limitations section).

Usage:
    PYTHONPATH=src python scripts/build_ptpep_classifier.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

N_PER_DOMAIN = 250   # 200 train + 50 held-out, per domain (2000 total, matches
                      # the ~2000-prompt scale in the original module docstring)
TRAIN_FRACTION = 0.8

# (domain, hf_dataset, config, split, text_field)
# Sourced from a Sprint-3 probe (see LOGBOOK 2026-08-08, Oskarshamn) confirming
# all 8 load cleanly. `text_field` is the column closest to "a user prompt" —
# not the full document (e.g. legalbench's `question`, not `contract`).
SOURCES = [
    ("coding",   "sahil2801/CodeAlpaca-20k", None,                  "train", "instruction"),
    ("math",     "meta-math/MetaMathQA",      None,                  "train", "query"),
    ("medical",  "qiaojin/PubMedQA",           "pqa_labeled",        "train", "question"),
    ("legal",    "nguha/legalbench",           "consumer_contracts_qa", "test", "question"),
    ("science",  "allenai/sciq",               None,                  "train", "question"),
    ("language", "grammarly/coedit",           None,                  "train", "src"),
    ("creative", "euclaise/writingprompts",    None,                  "train", "prompt"),
    ("general",  "tatsu-lab/alpaca",           None,                  "train", "instruction"),
]


def load_domain_texts(domain: str, name: str, config, split: str, field: str) -> list[str]:
    ds = load_dataset(name, config, split=f"{split}[:{N_PER_DOMAIN}]")
    texts = [str(row[field]).strip() for row in ds if str(row[field]).strip()]
    if len(texts) < N_PER_DOMAIN:
        raise RuntimeError(
            f"{domain}: only {len(texts)} non-empty rows from {name}, need {N_PER_DOMAIN}"
        )
    return texts[:N_PER_DOMAIN]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models"
    fixtures_dir = repo_root / "tests" / "fixtures"
    models_dir.mkdir(exist_ok=True)
    fixtures_dir.mkdir(exist_ok=True, parents=True)

    train_by_domain: dict[str, list[str]] = {}
    test_records: list[dict] = []

    for domain, name, config, split, field in SOURCES:
        print(f"[{domain}] loading {name} ({config}) split={split} field={field} ...")
        texts = load_domain_texts(domain, name, config, split, field)
        n_train = int(N_PER_DOMAIN * TRAIN_FRACTION)
        train_by_domain[domain] = texts[:n_train]
        for t in texts[n_train:]:
            test_records.append({"text": t, "domain": domain})
        print(f"[{domain}] {n_train} train / {len(texts) - n_train} held-out")

    domains = [d for d, *_ in SOURCES]
    pooled_train = [t for d in domains for t in train_by_domain[d]]

    print(f"\nFitting TfidfVectorizer on {len(pooled_train)} pooled training docs ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    vectorizer.fit(pooled_train)

    centroids = {}
    for domain in domains:
        vecs = vectorizer.transform(train_by_domain[domain])
        centroids[domain] = np.asarray(vecs.mean(axis=0)).ravel()

    artifact = {"vectorizer": vectorizer, "centroids": centroids, "domains": domains}
    artifact_path = models_dir / "ptpep_tfidf_v1.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"\nSaved classifier artifact -> {artifact_path} "
          f"({artifact_path.stat().st_size / 1024:.0f} KB)")

    fixture_path = fixtures_dir / "ptpep_validation.json"
    fixture_path.write_text(json.dumps(test_records, indent=2, ensure_ascii=False))
    print(f"Saved validation fixture -> {fixture_path} ({len(test_records)} examples)")

    # ── quick sanity check: hit rate on the held-out set, using the exact ──
    # ── same cosine-similarity scoring predict() will use ──────────────────
    centroid_matrix = np.stack([centroids[d] for d in domains])
    test_vecs = vectorizer.transform([r["text"] for r in test_records])
    sims = cosine_similarity(test_vecs, centroid_matrix)
    predicted = [domains[i] for i in sims.argmax(axis=1)]
    correct = sum(p == r["domain"] for p, r in zip(predicted, test_records))
    hit_rate = correct / len(test_records)
    print(f"\nHeld-out hit rate: {correct}/{len(test_records)} = {hit_rate:.1%}")

    from collections import Counter
    confusion = Counter()
    for p, r in zip(predicted, test_records):
        if p != r["domain"]:
            confusion[(r["domain"], p)] += 1
    if confusion:
        print("Top confusions (true -> predicted: count):")
        for (true_d, pred_d), n in confusion.most_common(10):
            print(f"  {true_d} -> {pred_d}: {n}")


if __name__ == "__main__":
    main()
