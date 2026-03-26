# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
data_loader.py — Dataset explorer and vocabulary builder for Tunisian Darija Transformer.

JOB 1: Explore the atlasia/darija_english HuggingFace dataset.
JOB 2: Rebuild DarijaTokenizer vocabulary from real dataset sentences.

Dataset note: 'atlasia/darija_english' has 5 sub-configs.
  - web_data      :  3,000 rows  | Arabic-script darija + English
  - comments      : 10,000 rows  | Arabic-script darija + English
  - stories       :  3,000 rows  | Arabic-script darija + English
  - doda          : 45,103 rows  | Arabizi darija + English  <-- used here
  - transliteration: 67,186 rows | Arabizi <-> Arabic script (no English)

We use 'doda' because it is the largest translation config AND uses Arabizi
(Latin + numeric markers 3/7/9/5), matching the tokenizer in vocab.py.
"""

import sys
import torch
from datasets import load_dataset

# ── GPU Check ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("GPU CHECK")
print("=" * 60)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  Device  : {gpu_name}")
    print(f"  VRAM    : {vram_total:.2f} GB total")
    if vram_total < 4.0:
        print(f"  WARNING : VRAM < 4GB — stay under 3.5GB allocation limit")
else:
    print("  Device  : CPU only (no CUDA GPU detected)")
print()

# ── JOB 1: Explore Dataset ─────────────────────────────────────────────────────
print("=" * 60)
print("JOB 1 — DATASET EXPLORATION  (config: 'doda')")
print("=" * 60)

# doda columns: id | darija (Arabizi) | en
DARIJA_COL  = "darija"
ENGLISH_COL = "en"

print("Loading 'atlasia/darija_english', config='doda', split='train'...")
dataset = load_dataset("atlasia/darija_english", "doda", split="train")

# Total sentence pairs
total = len(dataset)
print(f"\n  Total sentence pairs : {total:,}")

# Column names
print(f"  Column names         : {dataset.column_names}")
print(f"  Darija column        : '{DARIJA_COL}'")
print(f"  English column       : '{ENGLISH_COL}'")

# First 5 darija/english pairs
print()
print("  First 5 sentence pairs:")
print("  " + "-" * 54)
for i in range(min(5, total)):
    row = dataset[i]
    print(f"  [{i}] Darija  : {row[DARIJA_COL]}")
    print(f"      English : {row[ENGLISH_COL]}")
    print()

# Missing / empty rows
empty_darija = sum(
    1 for row in dataset
    if row[DARIJA_COL] is None or str(row[DARIJA_COL]).strip() == ""
)
empty_english = sum(
    1 for row in dataset
    if row[ENGLISH_COL] is None or str(row[ENGLISH_COL]).strip() == ""
)
either_empty = sum(
    1 for row in dataset
    if (row[DARIJA_COL]  is None or str(row[DARIJA_COL]).strip()  == "")
    or (row[ENGLISH_COL] is None or str(row[ENGLISH_COL]).strip() == "")
)

print("  Missing / Empty rows:")
print(f"    Empty darija  rows : {empty_darija:,}")
print(f"    Empty english rows : {empty_english:,}")
print(f"    Either empty       : {either_empty:,}")

# ── JOB 2: Rebuild Vocabulary ──────────────────────────────────────────────────
print()
print("=" * 60)
print("JOB 2 — VOCABULARY REBUILD")
print("=" * 60)

sys.path.insert(0, "/workspace")
from vocab import DarijaTokenizer

# Collect all non-empty darija sentences
all_darija = [
    str(row[DARIJA_COL])
    for row in dataset
    if row[DARIJA_COL] is not None and str(row[DARIJA_COL]).strip() != ""
]

print(f"  Sentences used to build vocab : {len(all_darija):,}")
print(f"  (old hardcoded corpus had 8 example sentences)")

tokenizer = DarijaTokenizer()
print(f"  BPE vocabulary size           : {tokenizer.vocab_size():,} tokens")

# Encode/decode first darija sentence
first_sentence = all_darija[0]
print(f"\n  Test sentence : {first_sentence!r}")

ids = tokenizer.encode(first_sentence)
print(f"  Token IDs     : {ids}")

decoded = tokenizer.decode(ids)
print(f"  Decoded tokens: {decoded}")

unk_count = ids.count(tokenizer.unk_idx)
print(f"  UNK tokens    : {unk_count}  (BPE should produce 0 — unknown words decompose into pieces)")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
