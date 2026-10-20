"""
train_classifier.py
-------------------
"Trains" the ANDM notification classifier on realistic everyday notifications.

How training works here: the classifier is an exemplar (kNN) model, so training
means adding labelled examples to rules.json and re-embedding everything into
centroids.pt (the same thing build.py does). No gradient descent involved.

Steps
  1. BASELINE  - score the current model on realistic_notifications.json
                 (these examples are unseen, so this is an honest accuracy).
  2. TRAIN     - back up rules.json, merge the examples in (deduplicated),
                 re-embed every exemplar, and overwrite centroids.pt.
  3. RE-CHECK  - score again. NOTE: this is on the training examples, so it is
                 optimistic. Test with your own new notifications afterwards:
                     py test_classifier.py "your text here"

Usage
  py train_classifier.py                         # evaluate, train, re-evaluate
  py train_classifier.py --eval-only             # just show the baseline
  py train_classifier.py --data my_more.json     # use another dataset file
Dataset format: a JSON list of {"title","body","sender_id","label"}
                label is one of Immediate | Adaptive | Ignore
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import classifier as clf

BASE = Path(__file__).parent
RULES_FILE = BASE / "rules.json"
VALID = {"Immediate", "Adaptive", "Ignore"}


def to_text(item: dict) -> str:
    """Same text shape the server builds: 'title. body'."""
    parts = [str(item.get("title", "")).strip(), str(item.get("body", "")).strip()]
    return ". ".join(p for p in parts if p)


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clean = []
    for i, item in enumerate(data):
        if item.get("label") not in VALID:
            print(f"  skipping #{i}: bad label {item.get('label')!r}")
            continue
        if not to_text(item):
            print(f"  skipping #{i}: empty text")
            continue
        clean.append(item)
    return clean


def evaluate(data: list[dict], title: str) -> float:
    print(f"\n=== {title} ===")
    correct, misses = 0, []
    per_class = {k: [0, 0] for k in VALID}
    for item in data:
        r = clf.classify(to_text(item))
        ok = r["category"] == item["label"]
        per_class[item["label"]][1] += 1
        if ok:
            correct += 1
            per_class[item["label"]][0] += 1
        else:
            misses.append((item, r))
    acc = correct / len(data) * 100
    print(f"Accuracy: {correct}/{len(data)} = {acc:.1f}%")
    for k, (c, n) in per_class.items():
        if n:
            print(f"  {k:<10} {c}/{n}")
    if misses:
        print("Misclassified:")
        for item, r in misses:
            print(f"  expected {item['label']:<9} got {r['category']:<9} "
                  f"({r['source']}, {r['confidence']:.2f}) | {to_text(item)[:90]}")
    return acc


def embed_batch(texts: list[str], batch_size: int = 64) -> torch.Tensor:
    """Same pooling + normalisation as classifier.get_embedding, but batched."""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        inputs = clf.tokenizer(chunk, return_tensors="pt", truncation=True,
                               max_length=64, padding=True)
        with torch.no_grad():
            hs = clf.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hs.size()).float()
            pooled = (hs * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
            out.append(F.normalize(pooled, p=2, dim=1))
    return torch.cat(out, dim=0)


def train(data: list[dict]) -> None:
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        rules = json.load(f)

    backup = RULES_FILE.with_name(f"rules.backup.{int(time.time())}.json")
    shutil.copy(RULES_FILE, backup)
    print(f"\nBacked up rules.json -> {backup.name}")

    added = 0
    for item in data:
        text = to_text(item)
        bucket = rules.setdefault(item["label"], [])
        if text not in bucket:
            bucket.append(text)
            added += 1
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    print(f"Added {added} new exemplars to rules.json "
          f"({', '.join(f'{k}={len(v)}' for k, v in rules.items())})")

    print("Re-embedding all exemplars (about a minute on CPU)...")
    centroids = {cat: embed_batch(texts) for cat, texts in rules.items()}
    torch.save(centroids, BASE / "centroids.pt")
    clf.centroids = centroids          # use the new matrices right away
    print("Saved centroids.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "realistic_notifications.json"))
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    dataset = load_dataset(Path(args.data))
    if not dataset:
        sys.exit("No valid examples found.")
    print(f"Loaded {len(dataset)} examples from {Path(args.data).name}")

    evaluate(dataset, "BASELINE (before training, unseen examples)")
    if args.eval_only:
        sys.exit(0)

    train(dataset)
    evaluate(dataset, "AFTER TRAINING (on training examples, optimistic)")
    print("\nDone. Restart the server (py main.py) so it loads the new centroids.pt.")