# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
inference_finetune.py — Inference for the fine-tuned Tunisian Darija checkpoint.

Loads the BPE tokenizer from the path saved inside the checkpoint, then runs
greedy decoding on input Darija (Arabizi) sentences.

Usage:
  python /workspace/inference_finetune.py
"""

import sys
import math
import torch

sys.path.insert(0, "/workspace")
from vocab import DarijaTokenizer
from model import DarijaTransformer, EMB_DIM, MAX_SEQ_LEN

CHECKPOINT_PATH = "/workspace/models/checkpoints/finetune/best_model.pt"

PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3

# ── GPU Check ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("GPU CHECK")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  Device : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {vram_total:.2f} GB total")
else:
    print("  Device : CPU (no CUDA GPU detected)")
print()

# ── Load checkpoint ────────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
print(f"  Epoch  : {ckpt['epoch']}")
print(f"  Loss   : {ckpt['avg_loss']:.4f}")
print()

# ── Load BPE tokenizer ─────────────────────────────────────────────────────────
tokenizer     = DarijaTokenizer()
SRC_VOCAB_SIZE = tokenizer.src_vocab_size()
TGT_VOCAB_SIZE = tokenizer.tgt_vocab_size()
print(f"  BPE vocab size : {SRC_VOCAB_SIZE:,} (shared src/tgt)")

# ── Load model ─────────────────────────────────────────────────────────────────
model = DarijaTransformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
).to(DEVICE)

model.load_state_dict(ckpt["model"])
model.eval()
print("  Model loaded → eval mode")
print()

# ── Greedy decode ──────────────────────────────────────────────────────────────
def translate(darija_text: str, max_tokens: int = 64) -> str:
    """Greedy-decode a Darija (Arabizi) sentence to English."""
    src_ids = [BOS_IDX] + tokenizer.encode_src(darija_text) + [EOS_IDX]
    src     = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_pad = (src == PAD_IDX)

    scale = math.sqrt(EMB_DIM)

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


# ── Test sentences ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_sentences = [
        # Tunisian greetings / daily phrases
        ("cha7welek lyoum",              "what do you want today"),
        ("ma3inich fil mekla",           "I don't feel like eating"),
        ("9rit kteb ma7leh",             "I read a nice book"),
        ("mchit na9ra fil sbe7",         "I went to study in the morning"),
        ("rba7t fi sibe9 m3a sa7bi",     "I won a race with my friend"),
        # Greetings from dataset
        ("aslema",                       "hello"),
        ("kifech 7alek",                 "how are you"),
        ("labess 3leek",                 "you are fine"),
        # Slang
        ("ma7lek",                        "you are amazing"),
        ("fi blesti",                    "in my place / mind your business"),
    ]

    print("=" * 60)
    print("INFERENCE  (fine-tuned Tunisian model — epoch 20, loss 2.6264)")
    print("=" * 60)
    for darija, expected in test_sentences:
        translation = translate(darija)
        print(f"  Darija   : {darija}")
        print(f"  Expected : {expected}")
        print(f"  Model    : {translation}")
        print()
