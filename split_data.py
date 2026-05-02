# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
split_data.py — Stratified train/val/test split of the Tunisian dataset.

Split per category: 8 train / 1 val / 1 test
Run this script once before each training run as the dataset grows.
The test set must never be trained on.

Output:
  data/splits/train.csv  — 8 pairs per category
  data/splits/val.csv    — 1 pair per category
  data/splits/test.csv   — 1 pair per category (locked)
"""

import csv
import os
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV    = os.path.join(PROJECT_ROOT, "data/raw/tunisian_dataset.csv")
SPLITS_DIR   = os.path.join(PROJECT_ROOT, "data/splits")

TRAIN_PER_CAT = 8
VAL_PER_CAT   = 1
TEST_PER_CAT  = 1
TOTAL_PER_CAT = TRAIN_PER_CAT + VAL_PER_CAT + TEST_PER_CAT  # 10

os.makedirs(SPLITS_DIR, exist_ok=True)

# ── Load and group by category ─────────────────────────────────────────────────
by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)

with open(INPUT_CSV, encoding="utf-8") as f:
    for i, row in enumerate(csv.DictReader(f), start=2):
        if None in row or any(v is None for v in row.values()):
            print(f"  WARNING: skipping malformed row at line {i}: {dict(row)}")
            continue
        cat     = row["category"].strip()
        darija  = row["darija"].strip()
        english = row["english"].strip()
        if cat and darija and english:
            by_category[cat].append((darija, english))

# ── Split and write ────────────────────────────────────────────────────────────
FIELDNAMES = ["category", "darija", "english"]

train_rows, val_rows, test_rows = [], [], []
skipped = []

for cat, pairs in by_category.items():
    if len(pairs) < TOTAL_PER_CAT:
        skipped.append((cat, len(pairs)))
        continue

    # Deterministic: first 8 → train, 9th → val, 10th → test
    for darija, english in pairs[:TRAIN_PER_CAT]:
        train_rows.append({"category": cat, "darija": darija, "english": english})
    darija, english = pairs[TRAIN_PER_CAT]
    val_rows.append({"category": cat, "darija": darija, "english": english})
    darija, english = pairs[TRAIN_PER_CAT + VAL_PER_CAT]
    test_rows.append({"category": cat, "darija": darija, "english": english})

def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

write_csv(os.path.join(SPLITS_DIR, "train.csv"), train_rows)
write_csv(os.path.join(SPLITS_DIR, "val.csv"),   val_rows)
write_csv(os.path.join(SPLITS_DIR, "test.csv"),  test_rows)

# ── Report ─────────────────────────────────────────────────────────────────────
print("=" * 50)
print("SPLIT COMPLETE")
print("=" * 50)
print(f"  Categories processed : {len(by_category) - len(skipped)}")
print(f"  train.csv            : {len(train_rows)} pairs")
print(f"  val.csv              : {len(val_rows)} pairs")
print(f"  test.csv             : {len(test_rows)} pairs  ← locked, never train on this")

if skipped:
    print(f"\n  WARNING: {len(skipped)} categories skipped (fewer than {TOTAL_PER_CAT} pairs):")
    for cat, count in skipped:
        print(f"    {cat}: {count} pairs")
