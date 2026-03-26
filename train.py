# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
train.py — Training loop for the Darija-to-English Nano-Transformer.

Reads from:  vocab.py  (DarijaTokenizer)
             model.py  (DarijaTransformer)
Reads from:  data/clean/clean_darija_english.csv  (pre-cleaned corpus)
Writes to:   /workspace/models/checkpoints/  (epoch snapshots + best model)
"""

import os
import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# ── GPU Check — exit immediately if no GPU is found ───────────────────────────
print("=" * 60)
print("GPU CHECK")
print("=" * 60)
if not torch.cuda.is_available():
    print("  ERROR: No CUDA GPU detected.")
    print("  Training requires a GPU. Exiting.")
    sys.exit(1)

DEVICE     = torch.device("cuda")
gpu_name   = torch.cuda.get_device_name(0)
vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"  Device     : {gpu_name}")
print(f"  VRAM total : {vram_total:.2f} GB")
if vram_total < 4.0:
    print(f"  WARNING    : VRAM < 4GB — 3500MB safety limit is active")
print()

# ── Training Configuration ────────────────────────────────────────────────────
BATCH_SIZE        = 32      # sentences processed in parallel per step
EPOCHS            = 15      # full passes over the dataset            [was 10]
LEARNING_RATE     = 3e-4    # peak LR — reached after warmup, then cosine decay [was flat 1e-4]
WARMUP_STEPS      = 400     # steps to linearly ramp LR from 0 → peak before cosine decay
GRAD_ACCUM_STEPS  = 4       # accumulate gradients for N steps before updating
                            # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 128
USE_AMP           = True    # mixed precision (fp16) — halves VRAM for activations
CHECKPOINT_DIR    = "/workspace/models/checkpoints"
CLEAN_CSV         = "/workspace/data/clean/clean_darija_english.csv"
VRAM_LIMIT_MB     = 3500    # hard stop if GPU usage exceeds this

# Sequence length: p99 of clean corpus is 14 words; 32 covers 100% with BOS/EOS headroom.
# Cutting from 64 → 32 makes attention 4× cheaper (O(n²)) and halves padding waste.
MAX_SEQ_LEN = 32
PAD_IDX     = 0   # <PAD>
BOS_IDX     = 2   # <BOS>  start-of-sequence token
EOS_IDX     = 3   # <EOS>  end-of-sequence token
# SRC_VOCAB_SIZE and TGT_VOCAB_SIZE are set after the tokenizer is built below.

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Imports from project files ────────────────────────────────────────────────
sys.path.insert(0, "/workspace")
from vocab  import DarijaTokenizer
from model  import DarijaTransformer

# ── Load Pre-cleaned Dataset ──────────────────────────────────────────────────
# Reads from clean_data.py output — duplicates, short pairs, Arabizi leakage,
# noise, and Moroccan dialect markers already removed.
import csv

print("=" * 60)
print("DATA PREPARATION")
print("=" * 60)
print(f"Loading clean corpus from {CLEAN_CSV} ...")

dataset_clean = []
with open(CLEAN_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        d, e = row["darija"].strip(), row["english"].strip()
        if d and e:
            dataset_clean.append((d, e))

print(f"  Loaded            : {len(dataset_clean):,} clean pairs")

# ── Build Dual Tokenizer ──────────────────────────────────────────────────────
# Separate src vocab (Darija) and tgt vocab (English) so English words are
# never encoded as <UNK> during training.
all_darija  = [pair[0] for pair in dataset_clean]
all_english = [pair[1] for pair in dataset_clean]

print(f"\nBuilding dual tokenizer ...")
tokenizer = DarijaTokenizer(src_corpus=all_darija, tgt_corpus=all_english)

SRC_VOCAB_SIZE = tokenizer.src_vocab_size()
TGT_VOCAB_SIZE = tokenizer.tgt_vocab_size()

print(f"  src_vocab_size    : {SRC_VOCAB_SIZE:,}  (Darija)")
print(f"  tgt_vocab_size    : {TGT_VOCAB_SIZE:,}  (English)")
print()


# A PyTorch Dataset that converts raw sentence pairs into fixed-length tensors.
# Each item is one (src, tgt) pair the DataLoader will batch together.
class DarijaDataset(Dataset):
    def __init__(
        self,
        pairs:      list[tuple[str, str]],
        tokenizer:  DarijaTokenizer,
        max_len:    int = MAX_SEQ_LEN,
    ):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    # Convert one sentence pair to padded integer tensors.
    # Sentences are wrapped in BOS…EOS then padded (or truncated) to max_len.
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        darija_text, english_text = self.pairs[idx]
        src = self._pack(self.tokenizer.encode_src(darija_text))
        tgt = self._pack(self.tokenizer.encode_tgt(english_text))
        return src, tgt

    # Add BOS/EOS, truncate to max_len, then right-pad with <PAD>.
    def _pack(self, ids: list[int]) -> torch.Tensor:
        max_body = self.max_len - 2
        ids = ids[:max_body]
        ids = [BOS_IDX] + ids + [EOS_IDX]
        ids = ids + [PAD_IDX] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


# ── DataLoader ────────────────────────────────────────────────────────────────
# Wrap the dataset in a DataLoader that shuffles and batches automatically.
# num_workers=0 keeps everything in the main process — safer inside Docker.
# drop_last=True discards the final incomplete batch so batch size is constant.
torch_dataset = DarijaDataset(dataset_clean, tokenizer)
dataloader    = DataLoader(
    torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    pin_memory=True,   # speed up CPU→GPU transfers
)
steps_per_epoch = len(dataloader)
print(f"Dataset   : {len(torch_dataset):,} samples")
print(f"Batches   : {steps_per_epoch:,} per epoch  (batch_size={BATCH_SIZE})")
print(f"Eff. batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}  (×{GRAD_ACCUM_STEPS} grad accumulation)")
print()


# ── Helper: Build Padding Mask ────────────────────────────────────────────────
# The Transformer ignores positions where the mask is True.
# We mark <PAD> tokens (index 0) so attention doesn't waste capacity on them.
def make_pad_mask(tensor: torch.Tensor) -> torch.Tensor:
    # tensor shape: (batch, seq_len)
    # returns:      (batch, seq_len)  bool, True = this position is padding
    return tensor == PAD_IDX


# ── Model, Optimizer, Loss ────────────────────────────────────────────────────
print("=" * 60)
print("MODEL SETUP")
print("=" * 60)

model = DarijaTransformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {total_params:,}")
print(f"  Weight VRAM: {total_params * 4 / 1024**2:.1f} MB  (fp32)")
print()

# AdamW is Adam with weight decay — better than plain Adam for Transformers
# because it decouples the decay from the gradient update step.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

# LR schedule: linear warmup → cosine decay.
# Warmup prevents large early updates from destabilising fresh embeddings.
# Cosine decay smoothly reduces LR so the model refines rather than oscillates late in training.
total_steps = EPOCHS * (35_977 // BATCH_SIZE // GRAD_ACCUM_STEPS)  # rough estimate

def lr_lambda(current_step: int) -> float:
    if current_step < WARMUP_STEPS:
        return current_step / max(1, WARMUP_STEPS)
    progress = (current_step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# CrossEntropyLoss measures how wrong the predicted token distribution is.
# ignore_index=PAD_IDX means padding positions don't contribute to the loss —
# we don't want the model to "learn" that padding is a real output.
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# GradScaler multiplies the loss by a large factor before backward() to keep
# fp16 gradients from flushing to zero (underflow). It then divides back before
# the optimizer step. This is the core of mixed-precision training.
scaler = GradScaler(enabled=USE_AMP)


# ── Training Loop ─────────────────────────────────────────────────────────────
# One full training run: iterate over every epoch, every batch inside it.
def train():
    best_loss   = math.inf
    global_step = 0

    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Mixed prec.   : {USE_AMP} (fp16)")
    print(f"  Checkpoints   : {CHECKPOINT_DIR}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss   = 0.0
        batches_seen = 0

        optimizer.zero_grad()   # clear any leftover gradients at epoch start

        for step, (src, tgt) in enumerate(dataloader):
            # Move tensors from CPU RAM to GPU VRAM.
            src = src.to(DEVICE, non_blocking=True)   # (B, MAX_SEQ_LEN)
            tgt = tgt.to(DEVICE, non_blocking=True)   # (B, MAX_SEQ_LEN)

            # Teacher forcing: the decoder sees the target shifted one step left.
            # tgt_input  = [BOS, w1, w2, … wN]   ← what the decoder reads
            # tgt_output = [w1,  w2, … wN,  EOS]  ← what we want it to predict
            tgt_input  = tgt[:, :-1]   # (B, MAX_SEQ_LEN-1)
            tgt_output = tgt[:, 1:]    # (B, MAX_SEQ_LEN-1)

            # Build padding masks so attention ignores <PAD> positions.
            src_pad_mask = make_pad_mask(src)           # (B, S)
            tgt_pad_mask = make_pad_mask(tgt_input)     # (B, T)

            # Forward pass inside autocast: PyTorch automatically uses fp16
            # for supported ops and fp32 where precision matters (e.g. softmax).
            with autocast(enabled=USE_AMP):
                logits = model(
                    src, tgt_input,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                )
                # logits: (B, T, V) → flatten to (B*T, V) for loss
                # tgt_output: (B, T) → flatten to (B*T,)
                loss = criterion(
                    logits.reshape(-1, TGT_VOCAB_SIZE),
                    tgt_output.reshape(-1),
                )
                # Divide by accumulation steps so the effective loss magnitude
                # is the same whether we accumulate 1 step or 4.
                loss = loss / GRAD_ACCUM_STEPS

            # Backward pass: scaler keeps fp16 gradients from underflowing.
            scaler.scale(loss).backward()

            # ── VRAM safety check after the very first batch ──────────────────
            if global_step == 0:
                vram_used_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
                vram_reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
                print(f"  [VRAM] Allocated : {vram_used_mb:.1f} MB")
                print(f"  [VRAM] Reserved  : {vram_reserved_mb:.1f} MB")
                print(f"  [VRAM] Limit     : {VRAM_LIMIT_MB} MB")
                if vram_reserved_mb > VRAM_LIMIT_MB:
                    print(f"\n  WARNING: VRAM usage {vram_reserved_mb:.1f} MB exceeds "
                          f"{VRAM_LIMIT_MB} MB limit. Stopping training.")
                    sys.exit(1)
                print()

            # ── Optimizer step every GRAD_ACCUM_STEPS mini-batches ────────────
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                # Unscale before clipping so clip threshold is in real units.
                scaler.unscale_(optimizer)
                # Gradient clipping prevents exploding gradients — caps the
                # total gradient norm at 1.0.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Accumulate loss for epoch average (undo the /GRAD_ACCUM_STEPS scaling).
            epoch_loss   += loss.item() * GRAD_ACCUM_STEPS
            batches_seen += 1
            global_step  += 1

            # ── Print progress every 100 steps ────────────────────────────────
            if (step + 1) % 100 == 0:
                running_avg = epoch_loss / batches_seen
                print(f"  Epoch {epoch:2d} | Step {step+1:5d}/{steps_per_epoch} "
                      f"| Loss {loss.item() * GRAD_ACCUM_STEPS:.4f} "
                      f"| Avg {running_avg:.4f}")

        # ── Epoch summary ──────────────────────────────────────────────────────
        avg_loss = epoch_loss / batches_seen
        vram_mb  = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print()
        print(f"  ── Epoch {epoch:2d} complete ──")
        print(f"     Avg loss : {avg_loss:.4f}")
        print(f"     VRAM     : {vram_mb:.1f} MB / {VRAM_LIMIT_MB} MB")

        # ── Save checkpoint for this epoch ────────────────────────────────────
        # Each epoch snapshot lets you roll back if later epochs overfit.
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:02d}.pt")
        torch.save(
            {
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scaler":     scaler.state_dict(),
                "avg_loss":   avg_loss,
            },
            ckpt_path,
        )
        print(f"     Checkpoint: {ckpt_path}")

        # ── Save best model separately ─────────────────────────────────────────
        # "Best" = lowest average training loss seen so far.
        if avg_loss < best_loss:
            best_loss      = avg_loss
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(
                {
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "avg_loss": best_loss,
                },
                best_ckpt_path,
            )
            print(f"     Best model: {best_ckpt_path}  (loss={best_loss:.4f})")

        # ── Ongoing VRAM guard ─────────────────────────────────────────────────
        if vram_mb > VRAM_LIMIT_MB:
            print(f"\n  WARNING: VRAM {vram_mb:.1f} MB exceeded {VRAM_LIMIT_MB} MB limit.")
            print("  Stopping training to protect the GPU.")
            break

        print()

    print("=" * 60)
    print(f"TRAINING COMPLETE — best loss: {best_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train()
