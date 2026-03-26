# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
vocab.py — BPE tokenizer for Darija-to-English translation using SentencePiece.

Replaces the word-level whitespace tokenizer with a BPE (Byte Pair Encoding) model
trained on the combined Darija + English corpus.

Why BPE over word-level:
  - Arabizi words have many spelling variants: m3a / m3ak / m3aha / m3ahom
    With word-level tokenization each variant is a separate token, most seen rarely.
    BPE learns shared subword pieces (m3a, hom, k, ha) so variants share structure.
  - Unseen word forms get decomposed into known pieces instead of collapsing to <UNK>.
  - Vocabulary stays compact (16k pieces vs 40k+ words) so every token is well-trained.

Special token indices — identical to the old tokenizer so train.py needs no changes:
  0 = <PAD>   1 = <UNK>   2 = <BOS>   3 = <EOS>

Arabizi phoneme markers (3, 7, 9, 5) are added as user_defined_symbols so they
always get their own vocabulary entries and are never absorbed into other pieces.
"""

import os
import csv
import sentencepiece as spm

# ── Paths ──────────────────────────────────────────────────────────────────────
VOCAB_DIR        = "/workspace/vocab"
BPE_MODEL_PREFIX = os.path.join(VOCAB_DIR, "darija_bpe")
BPE_MODEL_FILE   = BPE_MODEL_PREFIX + ".model"
DEFAULT_CSV      = "/workspace/data/clean/clean_darija_english.csv"

# ── Vocabulary configuration ───────────────────────────────────────────────────
BPE_VOCAB_SIZE   = 16_000   # shared vocabulary for both Darija and English

# Special token indices — must stay fixed; train.py hardcodes PAD=0 BOS=2 EOS=3
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# Arabizi phoneme markers: protect as user_defined_symbols so they always get
# dedicated vocabulary slots and are never merged away during BPE training.
ARABIZI_MARKERS = ["3", "7", "9", "5"]


# ── BPE trainer ───────────────────────────────────────────────────────────────

def train_bpe(csv_path: str = DEFAULT_CSV) -> None:
    """
    Train a SentencePiece BPE model on the combined Darija + English corpus
    and save it to VOCAB_DIR/darija_bpe.model.

    Both languages are combined so the shared 16k vocabulary covers Arabizi
    subword pieces AND English subword pieces in one model.
    """
    os.makedirs(VOCAB_DIR, exist_ok=True)

    # Collect every non-empty sentence from both columns.
    texts: list[str] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row["darija"].strip()
            e = row["english"].strip()
            if d:
                texts.append(d)
            if e:
                texts.append(e)

    # Write to a temporary plain-text file (SentencePiece reads from file).
    temp_txt = os.path.join(VOCAB_DIR, "_train_corpus.txt")
    with open(temp_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))

    print(f"  Training BPE on {len(texts):,} sentences "
          f"({len(texts)//2:,} Darija + {len(texts)//2:,} English) ...")

    spm.SentencePieceTrainer.train(
        input=temp_txt,
        model_prefix=BPE_MODEL_PREFIX,
        vocab_size=BPE_VOCAB_SIZE,
        model_type="bpe",
        # Special token layout — must match PAD/UNK/BOS/EOS indices above.
        pad_id=PAD_ID,   pad_piece="<PAD>",
        unk_id=UNK_ID,   unk_piece="<UNK>",
        bos_id=BOS_ID,   bos_piece="<BOS>",
        eos_id=EOS_ID,   eos_piece="<EOS>",
        # Protect Arabizi phoneme markers so they always have vocab entries.
        user_defined_symbols=ARABIZI_MARKERS,
        # Cover virtually all characters including Arabic script + Latin + digits.
        character_coverage=0.9995,
        # Treat whitespace consistently across languages.
        normalization_rule_name="nmt_nfkc_cf",
    )

    os.remove(temp_txt)
    print(f"  BPE model saved → {BPE_MODEL_FILE}")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class DarijaTokenizer:
    """
    BPE tokenizer for Darija→English translation.

    Interface is identical to the old word-level DarijaTokenizer so train.py,
    inference.py, and any other consumers need no changes.

    Parameters
    ----------
    src_corpus / tgt_corpus : accepted for backward compatibility, not used.
        BPE is trained from the CSV file, not from in-memory lists.
    csv_path : path to the clean CSV used to train BPE if model is missing.

    Shared vocabulary:
        Both encode_src and encode_tgt use the same BPE model because the 16k
        vocabulary covers both Darija subword pieces and English subword pieces.
    """

    def __init__(
        self,
        src_corpus=None,   # kept for backward compatibility
        tgt_corpus=None,   # kept for backward compatibility
        corpus=None,       # kept for backward compatibility (data_loader.py)
        csv_path: str = DEFAULT_CSV,
    ):
        # Train BPE model on first run; subsequent runs load the saved file.
        if not os.path.exists(BPE_MODEL_FILE):
            print(f"  BPE model not found at {BPE_MODEL_FILE}")
            train_bpe(csv_path)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(BPE_MODEL_FILE)

        # Shared special-token indices (identical in old tokenizer).
        self.pad_idx = PAD_ID
        self.unk_idx = UNK_ID
        self.bos_idx = BOS_ID
        self.eos_idx = EOS_ID

    # ── Source (Darija) side ───────────────────────────────────────────────────

    def encode_src(self, text: str) -> list[int]:
        """Encode a Darija sentence → list of BPE token IDs."""
        return self.sp.encode(text, out_type=int)

    def decode_src(self, ids: list[int]) -> list[str]:
        """Convert source token IDs back to BPE piece strings (for debugging)."""
        return [self.sp.id_to_piece(i) for i in ids]

    # ── Target (English) side ──────────────────────────────────────────────────

    def encode_tgt(self, text: str) -> list[int]:
        """Encode an English sentence → list of BPE token IDs."""
        return self.sp.encode(text, out_type=int)

    def decode_tgt(self, ids: list[int]) -> list[str]:
        """
        Convert target token IDs back to an English string.
        Returns a single-element list so inference.py's ' '.join() call works
        identically to before.
        """
        return [self.sp.decode(ids)]

    # ── Unified interface (used by data_loader.py) ─────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Unified encode — same BPE model for any language."""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> list[str]:
        """Unified decode → list of BPE piece strings."""
        return [self.sp.id_to_piece(i) for i in ids]

    # ── Sizes ──────────────────────────────────────────────────────────────────

    def src_vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def tgt_vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def vocab_size(self) -> int:
        """Unified size — used by data_loader.py."""
        return self.sp.get_piece_size()


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BPE TOKENIZER TEST")
    print("=" * 60)

    tok = DarijaTokenizer()

    old_word_vocab = 40_962   # word-level src vocab from Phase 1
    new_bpe_vocab  = tok.vocab_size()
    print(f"\n  Before (word-level) src vocab : {old_word_vocab:,} tokens")
    print(f"  After  (BPE)        shared    : {new_bpe_vocab:,} tokens")
    print(f"  Reduction                     : {old_word_vocab - new_bpe_vocab:,} fewer tokens")

    print()
    print("  Arabizi variant test — do ma3andich / ma3endich share pieces?")
    print("  " + "-" * 50)

    for word in ["ma3andich", "ma3endich"]:
        ids    = tok.encode(word)
        pieces = tok.decode(ids)
        print(f"  {word:15s} → ids={ids}  pieces={pieces}")

    shared = set(tok.encode("ma3andich")) & set(tok.encode("ma3endich"))
    shared_pieces = [tok.sp.id_to_piece(i) for i in shared]
    print(f"\n  Shared piece IDs    : {shared}")
    print(f"  Shared pieces       : {shared_pieces}")
    if shared:
        print("  RESULT: PASS — variants share common BPE subword pieces ✓")
    else:
        print("  RESULT: WARN — no shared pieces found (may need more training data)")

    print()
    print("  Arabizi marker test — are 3 / 7 / 9 / 5 protected?")
    print("  " + "-" * 50)
    for marker in ARABIZI_MARKERS:
        marker_id = tok.sp.piece_to_id(marker)
        print(f"  '{marker}' → vocab ID {marker_id}  "
              f"({'protected' if marker_id > 3 else 'WARNING: overlaps special token'})")

    print()
    enc = tok.encode_src("ara lia dak sac")
    dec = tok.decode_tgt(enc)
    print(f"  encode_src('ara lia dak sac') → {enc}")
    print(f"  decode_tgt(ids)               → {dec}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
