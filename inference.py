# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
inference.py — Greedy-decode inference for the Tunisian Darija-to-English Transformer.
"""

import sys
import torch

sys.path.insert(0, "/workspace")
from vocab import DarijaTokenizer
from model import DarijaTransformer, EMB_DIM, NUM_HEADS, ENC_LAYERS, DEC_LAYERS, FFN_DIM, MAX_SEQ_LEN, DROPOUT

# ── GPU Check ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("GPU CHECK")
print("=" * 60)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    gpu_name   = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  Device : {gpu_name}")
    print(f"  VRAM   : {vram_total:.2f} GB total")
    if vram_total < 4.0:
        print(f"  WARNING: VRAM < 4GB — enforcing 3.5GB safety limit")
else:
    DEVICE = torch.device("cpu")
    print("  Device : CPU only (no CUDA GPU detected)")
print()

# ── Load Tokenizer (rebuilt from clean corpus) ─────────────────────────────────
import csv

CLEAN_CSV = "/workspace/data/clean/clean_darija_english.csv"
print(f"Loading tokenizer from {CLEAN_CSV} ...")

all_darija, all_english = [], []
with open(CLEAN_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["darija"].strip() and row["english"].strip():
            all_darija.append(row["darija"].strip())
            all_english.append(row["english"].strip())

tokenizer = DarijaTokenizer(src_corpus=all_darija, tgt_corpus=all_english)
print(f"  src_vocab_size : {tokenizer.src_vocab_size():,}  (Darija)")
print(f"  tgt_vocab_size : {tokenizer.tgt_vocab_size():,}  (English)")

PAD_IDX = tokenizer.pad_idx
BOS_IDX = tokenizer.bos_idx
EOS_IDX = tokenizer.eos_idx

# ── Load Model ─────────────────────────────────────────────────────────────────
print("Loading model from /workspace/models/checkpoints/best_model.pt...")
model = DarijaTransformer(
    src_vocab_size = tokenizer.src_vocab_size(),
    tgt_vocab_size = tokenizer.tgt_vocab_size(),
    emb_dim        = EMB_DIM,
    num_heads      = NUM_HEADS,
    enc_layers     = ENC_LAYERS,
    dec_layers     = DEC_LAYERS,
    ffn_dim        = FFN_DIM,
    max_seq_len    = MAX_SEQ_LEN,
    dropout        = DROPOUT,
).to(DEVICE)

checkpoint = torch.load(
    "/workspace/models/checkpoints/best_model.pt",
    map_location=DEVICE,
    weights_only=True,
)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()
print("  Model loaded and set to eval mode.")
print()


# ── translate() ────────────────────────────────────────────────────────────────
def translate(darija_text: str, max_tokens: int = 64) -> str:
    """
    Translate a Darija (Arabizi) string to English using greedy decoding.

    Steps:
      1. Tokenize the input and wrap with BOS / EOS.
      2. Run the encoder once to get memory.
      3. Decode token-by-token: at each step feed the growing target sequence,
         take the last logit, argmax → next token.
      4. Stop when EOS is produced or max_tokens is reached.
    """
    # 1. Encode source
    src_ids = [BOS_IDX] + tokenizer.encode_src(darija_text) + [EOS_IDX]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, S)

    # Source padding mask — no padding here (single sentence, no batching)
    src_pad_mask = (src == PAD_IDX)  # (1, S), all False

    with torch.no_grad():
        # 2. Encode once
        import math
        scale   = math.sqrt(EMB_DIM)
        src_emb = model.src_pos_enc(model.src_embedding(src) * scale)  # (1, S, D)
        memory  = model.encoder(src_emb, src_key_padding_mask=src_pad_mask)  # (1, S, D)

        # 3. Greedy decode
        tgt_ids = [BOS_IDX]

        for _ in range(max_tokens):
            tgt = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, T)

            tgt_emb   = model.tgt_pos_enc(model.tgt_embedding(tgt) * scale)        # (1, T, D)
            tgt_len   = tgt.size(1)
            causal    = model._causal_mask(tgt_len).to(DEVICE)                     # (T, T)

            dec_out   = model.decoder(
                tgt_emb,
                memory,
                tgt_mask=causal,
                memory_key_padding_mask=src_pad_mask,
            )                                                                        # (1, T, D)

            logits    = model.output_projection(dec_out)                            # (1, T, V)
            next_id   = logits[0, -1, :].argmax(dim=-1).item()

            if next_id == EOS_IDX:
                break

            tgt_ids.append(next_id)

    # 4. Decode — skip BOS
    output_tokens = tokenizer.decode_tgt(tgt_ids[1:])
    return " ".join(output_tokens)


# ── Test Sentences ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_sentences = [
        "ara lia dak sac.",
        "hadchi kollo lghalaT dyalk",
        "khssni nmchi daba",
        "wach nti 7m9a!",
        "Tab3an rah mkta2eb!",
    ]

    print("=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    for sentence in test_sentences:
        translation = translate(sentence)
        print(f"Darija  : {sentence}")
        print(f"English : {translation}")
        print()
