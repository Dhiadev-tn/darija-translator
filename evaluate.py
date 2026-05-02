# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
evaluate.py — BLEU evaluation on the held-out test set.

Loads the fine-tuned checkpoint, runs greedy decoding on every sentence
in data/splits/test.csv, and computes a BLEU score against the expected
English translations.

Run this once after every training run to track real progress:
  python /workspace/evaluate.py

Never train on test.csv. This file is the ground truth.
"""

import sys
import math
import csv

import torch
import sacrebleu

sys.path.insert(0, "/workspace")
from vocab import DarijaTokenizer
from model import DarijaTransformer, EMB_DIM, MAX_SEQ_LEN

CHECKPOINT_PATH = "/workspace/models/checkpoints/finetune/best_model.pt"
TEST_CSV        = "/workspace/data/splits/test.csv"

PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("EVALUATE — held-out test set")
print("=" * 60)
print(f"  Device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── Load checkpoint ────────────────────────────────────────────────────────────
print(f"  Checkpoint : {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
print(f"  Epoch      : {ckpt['epoch']}")
if "avg_val_loss" in ckpt:
    print(f"  Val loss   : {ckpt['avg_val_loss']:.4f}")
print()

# ── Load tokenizer ─────────────────────────────────────────────────────────────
tokenizer      = DarijaTokenizer()
SRC_VOCAB_SIZE = tokenizer.src_vocab_size()
TGT_VOCAB_SIZE = tokenizer.tgt_vocab_size()

# ── Load model ─────────────────────────────────────────────────────────────────
model = DarijaTransformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

# ── Greedy decode ──────────────────────────────────────────────────────────────
def translate(darija_text: str, max_tokens: int = 64) -> str:
    src_ids = [BOS_IDX] + tokenizer.encode_src(darija_text) + [EOS_IDX]
    src     = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_pad = (src == PAD_IDX)
    scale   = math.sqrt(EMB_DIM)

    with torch.no_grad():
        src_emb = model.src_pos_enc(model.src_embedding(src) * scale)
        memory  = model.encoder(src_emb, src_key_padding_mask=src_pad)

        tgt_ids = [BOS_IDX]
        for _ in range(max_tokens):
            tgt     = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
            tgt_emb = model.tgt_pos_enc(model.tgt_embedding(tgt) * scale)
            causal  = model._causal_mask(tgt.size(1)).to(DEVICE)
            dec_out = model.decoder(tgt_emb, memory,
                                    tgt_mask=causal,
                                    memory_key_padding_mask=src_pad)
            next_id = model.output_projection(dec_out)[0, -1, :].argmax(dim=-1).item()
            if next_id == EOS_IDX:
                break
            tgt_ids.append(next_id)

    tokens = tokenizer.decode_tgt(tgt_ids[1:])
    return tokens[0] if tokens else ""


# ── Load test set ──────────────────────────────────────────────────────────────
test_pairs: list[tuple[str, str, str]] = []  # (category, darija, expected_english)
with open(TEST_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        test_pairs.append((
            row["category"].strip(),
            row["darija"].strip(),
            row["english"].strip(),
        ))

print(f"  Test pairs : {len(test_pairs)}")
print()

# ── Run evaluation ─────────────────────────────────────────────────────────────
hypotheses: list[str] = []
references: list[str] = []

print(f"  {'Category':<28} {'Darija':<35} {'Expected':<30} {'Model'}")
print("  " + "-" * 120)

for category, darija, expected in test_pairs:
    prediction = translate(darija)
    hypotheses.append(prediction)
    references.append(expected)
    print(f"  {category:<28} {darija:<35} {expected:<30} {prediction}")

# ── BLEU score ─────────────────────────────────────────────────────────────────
bleu = sacrebleu.corpus_bleu(hypotheses, [references])

print()
print("=" * 60)
print(f"  BLEU score : {bleu.score:.2f}")
print("=" * 60)
print()
print("  Interpretation:")
print("   0–10  : almost no overlap with reference translations")
print("  10–20  : some structure but mostly wrong words")
print("  20–30  : understandable, getting meaningful translations")
print("  30+    : good — approaching human-level for short sentences")
print()
print("  Note: with a small dataset expect low scores early on.")
print("  What matters is the score going UP after each retrain.")
