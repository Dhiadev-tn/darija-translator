# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
model.py — Nano-Transformer for Tunisian Darija-to-English translation.

Architecture: full encoder-decoder Transformer, sized to fit within 3500MB VRAM
on an RTX 3050 Laptop (4GB).
"""

import math
import torch
import torch.nn as nn

# ── GPU Check ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("GPU CHECK")
print("=" * 60)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    gpu_name  = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  Device : {gpu_name}")
    print(f"  VRAM   : {vram_total:.2f} GB total")
    if vram_total < 4.0:
        print(f"  WARNING: VRAM < 4GB — enforcing 3.5GB safety limit")
else:
    DEVICE = torch.device("cpu")
    print("  Device : CPU only (no CUDA GPU detected)")
print()

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Vocab sizes are set at runtime from the tokenizer; these are safe defaults.
SRC_VOCAB_SIZE = 16_000   # shared BPE vocabulary size  [updated from word-level 40,962]
TGT_VOCAB_SIZE = 16_000   # shared BPE vocabulary size  [updated from word-level 12,001]
EMB_DIM        = 256      # width of every embedding vector        [was 128]
NUM_HEADS      = 8        # parallel attention heads (EMB_DIM must be divisible by NUM_HEADS)
ENC_LAYERS     = 4        # number of encoder blocks stacked       [was 3]
DEC_LAYERS     = 4        # number of decoder blocks stacked       [was 3]
FFN_DIM        = 1024     # inner size of the feed-forward sublayer [was 512]
MAX_SEQ_LEN    = 32       # p99 of clean data is 14 words; 32 covers 100% + BOS/EOS headroom
DROPOUT        = 0.2      # fraction of activations randomly zeroed [was 0.1 — larger model needs more reg]


# Injects information about the position of each token into its embedding.
# Without this the Transformer treats "I love you" and "you love I" identically
# because attention has no built-in sense of order.
# We use fixed sine/cosine waves (no learnable parameters) so this never
# needs gradient updates.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build a (max_len, emb_dim) table of position encodings once.
        pe    = torch.zeros(max_len, emb_dim)                    # (L, D)
        pos   = torch.arange(max_len).unsqueeze(1).float()       # (L, 1)
        denom = torch.exp(
            torch.arange(0, emb_dim, 2).float()
            * (-math.log(10_000.0) / emb_dim)
        )                                                         # (D/2,)
        pe[:, 0::2] = torch.sin(pos * denom)   # even dimensions → sine
        pe[:, 1::2] = torch.cos(pos * denom)   # odd  dimensions → cosine

        # Register as a buffer so it moves to GPU with .to(device) but is
        # never treated as a learnable parameter.
        self.register_buffer("pe", pe.unsqueeze(0))              # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, emb_dim)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# The full encoder-decoder Transformer.
#
# Encoder: reads the Darija source sentence and compresses it into a sequence
#          of context vectors (one per token).
# Decoder: reads the English tokens produced so far, attends to the encoder
#          context, and predicts the next English token at every position.
# Linear:  projects the decoder output to a score over the entire vocabulary.
class DarijaTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int   = SRC_VOCAB_SIZE,
        tgt_vocab_size: int   = TGT_VOCAB_SIZE,
        emb_dim:        int   = EMB_DIM,
        num_heads:      int   = NUM_HEADS,
        enc_layers:     int   = ENC_LAYERS,
        dec_layers:     int   = DEC_LAYERS,
        ffn_dim:        int   = FFN_DIM,
        max_seq_len:    int   = MAX_SEQ_LEN,
        dropout:        float = DROPOUT,
    ):
        super().__init__()
        self._tgt_vocab_size = tgt_vocab_size

        # Separate embedding tables: src uses the Darija vocab, tgt uses English.
        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=0)

        # Positional encoders for source and target sequences.
        self.src_pos_enc = PositionalEncoding(emb_dim, max_seq_len, dropout)
        self.tgt_pos_enc = PositionalEncoding(emb_dim, max_seq_len, dropout)

        # A single TransformerEncoderLayer bundles:
        #   multi-head self-attention → add & norm → feed-forward → add & norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,   # expect (batch, seq, dim) not (seq, batch, dim)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # A single TransformerDecoderLayer bundles:
        #   masked self-attention → add & norm
        #   cross-attention (to encoder output) → add & norm
        #   feed-forward → add & norm
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # Final linear layer: maps each decoder output vector to a score for
        # every token in the target vocabulary.
        self.output_projection = nn.Linear(emb_dim, tgt_vocab_size)

        # Weight tying: share weights between the target embedding table and
        # the output projection. This halves that part of memory (~23 MB saved)
        # and often improves training stability.
        self.output_projection.weight = self.tgt_embedding.weight

        self._init_weights()

    # Initialise weights with small random values so gradients flow well at
    # the very start of training (Xavier uniform for linear layers).
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=EMB_DIM ** -0.5)

    # Build a causal (look-ahead) mask for the decoder so that position i
    # cannot attend to any position j > i. Without this the decoder would
    # "cheat" by reading future target tokens during training.
    def _causal_mask(self, size: int) -> torch.Tensor:
        # Returns an (size, size) boolean tensor; True = "block this position".
        return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)

    # One full forward pass through encoder then decoder.
    # src:         (batch, src_len)  — token IDs of the Darija input
    # tgt:         (batch, tgt_len)  — token IDs of the English output so far
    # src_key_padding_mask: (batch, src_len) bool — True where src is <PAD>
    # tgt_key_padding_mask: (batch, tgt_len) bool — True where tgt is <PAD>
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Embed + add positional encoding; scale by sqrt(emb_dim) to keep
        #    embedding magnitudes in a stable range as emb_dim grows.
        scale = math.sqrt(EMB_DIM)
        src_emb = self.src_pos_enc(self.src_embedding(src) * scale)  # (B, S, D)
        tgt_emb = self.tgt_pos_enc(self.tgt_embedding(tgt) * scale)  # (B, T, D)

        # 2. Encode the source sentence.
        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )                                                              # (B, S, D)

        # 3. Build causal mask so the decoder can't peek at future tokens.
        tgt_len   = tgt.size(1)
        causal    = self._causal_mask(tgt_len).to(tgt.device)        # (T, T)

        # 4. Decode: attend to target history and encoder memory.
        dec_out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )                                                              # (B, T, D)

        # 5. Project to vocabulary scores.
        logits = self.output_projection(dec_out)                      # (B, T, V)
        return logits


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MODEL TEST")
    print("=" * 60)

    # Instantiate the model and move it to the available device.
    model = DarijaTransformer().to(DEVICE)

    # Count every learnable scalar in the model.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters   : {total_params:,}")

    # Estimate VRAM for model weights only (float32 = 4 bytes per parameter).
    # Training would add ~2× for gradients + ~2–8× for optimiser states, but
    # inference only needs the weights.
    weight_vram_mb = (total_params * 4) / (1024 ** 2)
    print(f"  Model weight VRAM  : {weight_vram_mb:.2f} MB  (float32, inference)")
    print(f"  Safety limit       : 3500 MB")
    assert weight_vram_mb < 3500, "VRAM limit exceeded — shrink the model!"
    print(f"  VRAM check         : PASSED (headroom: {3500 - weight_vram_mb:.1f} MB)")

    # Build dummy input tensors.
    # In real training these would come from the DataLoader.
    BATCH   = 4
    SRC_LEN = 20   # tokens in the Darija sentence
    TGT_LEN = 18   # tokens produced so far by the decoder

    src_ids = torch.randint(1, SRC_VOCAB_SIZE, (BATCH, SRC_LEN)).to(DEVICE)
    tgt_ids = torch.randint(1, TGT_VOCAB_SIZE, (BATCH, TGT_LEN)).to(DEVICE)

    # Simulate some padding in the source (last 3 positions are <PAD> = 0).
    src_pad_mask = torch.zeros(BATCH, SRC_LEN, dtype=torch.bool).to(DEVICE)
    src_pad_mask[:, -3:] = True   # mark last 3 tokens as padding

    print()
    print(f"  Dummy src shape    : {list(src_ids.shape)}   (batch, src_len)")
    print(f"  Dummy tgt shape    : {list(tgt_ids.shape)}   (batch, tgt_len)")

    # Run one forward pass (no gradient needed for the test).
    model.eval()
    with torch.no_grad():
        logits = model(src_ids, tgt_ids, src_key_padding_mask=src_pad_mask)

    expected_shape = (BATCH, TGT_LEN, TGT_VOCAB_SIZE)
    print()
    print(f"  Output logits shape: {list(logits.shape)}")
    print(f"  Expected shape     : {list(expected_shape)}")
    assert tuple(logits.shape) == expected_shape, \
        f"Shape mismatch: got {tuple(logits.shape)}, expected {expected_shape}"
    print(f"  Shape check        : PASSED")

    print()
    print("=" * 60)
    print("DONE — model is ready for training.")
    print("=" * 60)
