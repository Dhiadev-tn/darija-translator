# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
clean_data.py — Dataset cleaner for atlasia/darija_english (doda config).

Cleaning pipeline:
  1. Remove duplicate sentence pairs
  2. Remove pairs where darija or english is under 3 words
  3. Remove pairs where english still contains Arabizi digits (3, 7, 9, 5)
     mixed with letters — indicating an untranslated Arabizi string
  4. Remove pairs containing URLs, emojis, or symbols (@, #, http)
  5. Remove pairs containing Moroccan-dialect markers
     (bzzaf / bel zaf / bil zaf / kif 7alak / kif halak)

Output: /workspace/data/clean/clean_darija_english.csv
"""

import os
import re
import csv
from datasets import load_dataset

# ── Output path ────────────────────────────────────────────────────────────────
OUT_DIR  = "/workspace/data/clean"
OUT_FILE = os.path.join(OUT_DIR, "clean_darija_english.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load dataset ───────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)
print("  Source : atlasia/darija_english  |  config=doda  |  split=train")

raw = load_dataset("atlasia/darija_english", "doda", split="train")

# Pull out non-empty darija/english pairs into plain Python tuples.
pairs = [
    (str(row["darija"]).strip(), str(row["en"]).strip())
    for row in raw
    if row["darija"] and str(row["darija"]).strip()
    and row["en"]     and str(row["en"]).strip()
]

original_count = len(pairs)
print(f"  Loaded : {original_count:,} non-empty pairs\n")


# ── Helper: word count ─────────────────────────────────────────────────────────
def word_count(text: str) -> int:
    return len(text.split())


# ── Helper: contains Arabizi digit mixed with letters ─────────────────────────
# Matches a word that has at least one Arabizi digit (3, 7, 9, 5) AND at least
# one alphabetic character — i.e. the digit is embedded in a word, not standalone.
_ARABIZI_MIXED = re.compile(r'\b(?=[a-zA-Z0-9]*[37950])(?=[a-zA-Z0-9]*[a-zA-Z])[a-zA-Z0-9]+\b')

def has_arabizi_digits(text: str) -> bool:
    """Return True if any word in text mixes Arabizi digits with Latin letters."""
    return bool(_ARABIZI_MIXED.search(text))


# ── Helper: contains URL / emoji / banned symbols ─────────────────────────────
# Emoji range covers the main Unicode emoji blocks.
_URL_PATTERN   = re.compile(r'https?://|www\.|\.com\b|\.net\b|\.org\b', re.IGNORECASE)
_AT_HASH       = re.compile(r'[@#]')
_HTTP_BARE     = re.compile(r'\bhttp\b', re.IGNORECASE)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002700-\U000027BF"   # dingbats
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U00002600-\U000026FF"   # misc symbols
    "]+",
    flags=re.UNICODE,
)

def has_url_emoji_symbol(text: str) -> bool:
    return bool(
        _URL_PATTERN.search(text)
        or _AT_HASH.search(text)
        or _HTTP_BARE.search(text)
        or _EMOJI_PATTERN.search(text)
    )


# ── Helper: contains Moroccan-dialect markers ──────────────────────────────────
_MOROCCAN = re.compile(
    r'\b(bzzaf|bel\s*zaf|bil\s*zaf|kif\s*7alak|kif\s*halak)\b',
    re.IGNORECASE,
)

def has_moroccan_dialect(text: str) -> bool:
    return bool(_MOROCCAN.search(text))


# ══════════════════════════════════════════════════════════════════════════════
# CLEANING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("CLEANING PIPELINE")
print("=" * 60)

# ── Step 1: Remove duplicates ──────────────────────────────────────────────────
seen    = set()
step1   = []
for pair in pairs:
    key = (pair[0].lower(), pair[1].lower())
    if key not in seen:
        seen.add(key)
        step1.append(pair)

removed_dupes = original_count - len(step1)
print(f"  Step 1 — Duplicates removed      : {removed_dupes:,}")

# ── Step 2: Remove pairs with < 3 words ───────────────────────────────────────
step2 = [
    p for p in step1
    if word_count(p[0]) >= 3 and word_count(p[1]) >= 3
]
removed_short = len(step1) - len(step2)
print(f"  Step 2 — Too short (< 3 words)   : {removed_short:,}")

# ── Step 3: Remove pairs where english contains Arabizi digits ─────────────────
step3 = [
    p for p in step2
    if not has_arabizi_digits(p[1])
]
removed_arabizi = len(step2) - len(step3)
print(f"  Step 3 — Arabizi digits in EN    : {removed_arabizi:,}")

# ── Step 4: Remove pairs with URLs, emojis, @, # ──────────────────────────────
step4 = [
    p for p in step3
    if not has_url_emoji_symbol(p[0]) and not has_url_emoji_symbol(p[1])
]
removed_noise = len(step3) - len(step4)
print(f"  Step 4 — URLs / emojis / symbols : {removed_noise:,}")

# ── Step 5: Remove Moroccan-dialect pairs ─────────────────────────────────────
step5 = [
    p for p in step4
    if not has_moroccan_dialect(p[0]) and not has_moroccan_dialect(p[1])
]
removed_moroccan = len(step4) - len(step5)
print(f"  Step 5 — Moroccan dialect words  : {removed_moroccan:,}")

clean_pairs = step5
final_count = len(clean_pairs)

total_removed = original_count - final_count
print()
print(f"  Original count  : {original_count:,}")
print(f"  Total removed   : {total_removed:,}  ({total_removed / original_count * 100:.1f}%)")
print(f"  Final clean     : {final_count:,}  ({final_count / original_count * 100:.1f}%)")


# ── Sample pairs ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("5 SAMPLE CLEAN PAIRS")
print("=" * 60)
for i, (darija, english) in enumerate(clean_pairs[:5], 1):
    print(f"  [{i}] Darija  : {darija}")
    print(f"      English : {english}")
    print()


# ── Save to CSV ────────────────────────────────────────────────────────────────
print("=" * 60)
print("SAVING")
print("=" * 60)

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["darija", "english"])
    writer.writerows(clean_pairs)

file_size_kb = os.path.getsize(OUT_FILE) / 1024
print(f"  Saved  : {OUT_FILE}")
print(f"  Rows   : {final_count:,}")
print(f"  Size   : {file_size_kb:.1f} KB")
print("=" * 60)
