# Author: Dhia (@dhiadev-tn) | github.com/dhiadev-tn
# Project: Tunisian Darija-to-English NLP Pipeline
# License: CC BY-NC-SA 4.0
"""
finetune.py — Phase 2: Fine-tune the Moroccan pre-trained checkpoint on Tunisian Darija.

Load sequence (fresh start):
  1. Load Moroccan CSV  (35k pairs, Phase 1 pre-training data)
  2. Load hand-crafted Tunisian dataset from data/raw/tunisian_dataset.csv
  3. Combine both → rebuild tokenizer with expanded vocabulary
  4. Load models/checkpoints/best_model.pt → transfer weights to new model
  5. Fine-tune on combined data (Tunisian repeated 20x to compensate small size)
  6. Save checkpoints to models/checkpoints/finetune/

Load sequence (resume):
  — Automatically detects the latest epoch_XX.pt in models/checkpoints/finetune/
  — Restores model, optimizer, scheduler, and scaler states
  — Continues training from where it left off

Data strategy with small Tunisian dataset (120 pairs):
  Tunisian repeated 20x = 2,400 samples — enough signal for the model to learn
  Moroccan kept at 1x = 35k samples — prevents catastrophic forgetting
"""

import os
import sys
import math
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# ── GPU Check ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("GPU CHECK")
print("=" * 60)
if not torch.cuda.is_available():
    print("  ERROR: No CUDA GPU detected. Exiting.")
    sys.exit(1)

DEVICE     = torch.device("cuda")
vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"  Device : {torch.cuda.get_device_name(0)}")
print(f"  VRAM   : {vram_total:.2f} GB total")
if vram_total < 4.0:
    print("  WARNING: VRAM < 4 GB — 3500 MB safety limit is active")
print()

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT       = os.path.dirname(os.path.abspath(__file__))
MOROCCAN_CSV       = os.path.join(PROJECT_ROOT, "data/clean/clean_darija_english.csv")
TUNISIAN_CSV       = os.path.join(PROJECT_ROOT, "data/raw/tunisian_dataset.csv")
TUNISIAN_REPEAT    = 20        # repeat 120 pairs × 20 = 2,400 Tunisian samples
MOROCCAN_REPEAT    = 1         # keep Moroccan at 1x — prevents catastrophic forgetting
PRETRAINED_CKPT    = os.path.join(PROJECT_ROOT, "models/checkpoints/best_model.pt")
FINETUNE_DIR       = os.path.join(PROJECT_ROOT, "models/checkpoints/finetune")

BATCH_SIZE       = 32
EPOCHS           = 20
LEARNING_RATE    = 2e-5    # 5× lower than pre-training — protect pre-trained weights
WARMUP_STEPS     = 200     # gradual LR warmup so early steps don't corrupt embeddings
GRAD_ACCUM_STEPS = 4       # effective batch = 32 × 4 = 128
USE_AMP          = True
VRAM_LIMIT_MB    = 3500
MAX_SEQ_LEN      = 32
PAD_IDX          = 0
BOS_IDX          = 2
EOS_IDX          = 3

os.makedirs(FINETUNE_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_ROOT)
import vocab
from vocab import DarijaTokenizer
from model import DarijaTransformer

# ── Step 1: Load Moroccan data ─────────────────────────────────────────────────
print("=" * 60)
print("DATA LOADING")
print("=" * 60)

moroccan_pairs: list[tuple[str, str]] = []
with open(MOROCCAN_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        d, e = row["darija"].strip(), row["english"].strip()
        if d and e:
            moroccan_pairs.append((d, e))
print(f"  Moroccan pairs   : {len(moroccan_pairs):,}")

# ── Step 2: Load hand-crafted Tunisian dataset ─────────────────────────────────
tunisian_pairs: list[tuple[str, str]] = []
with open(TUNISIAN_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        d, e = row["darija"].strip(), row["english"].strip()
        if d and e:
            tunisian_pairs.append((d, e))
print(f"  Tunisian pairs   : {len(tunisian_pairs):,}  (hand-crafted, native-corrected)")
print(f"  Tunisian repeat  : ×{TUNISIAN_REPEAT}  → {len(tunisian_pairs) * TUNISIAN_REPEAT:,} effective samples")

# ── Step 3: Build combined vocabulary ─────────────────────────────────────────
combined_pairs = tunisian_pairs * TUNISIAN_REPEAT + moroccan_pairs * MOROCCAN_REPEAT
random.shuffle(combined_pairs)

all_darija  = [p[0] for p in combined_pairs]
all_english = [p[1] for p in combined_pairs]

print(f"\n  Combined total   : {len(combined_pairs):,} pairs")
print(f"  Breakdown        : {len(tunisian_pairs):,} Tunisian + "
      f"{len(moroccan_pairs) * MOROCCAN_REPEAT:,} Moroccan (×{MOROCCAN_REPEAT})")

tokenizer    = DarijaTokenizer(src_corpus=all_darija, tgt_corpus=all_english)
NEW_SRC_VOCAB = tokenizer.src_vocab_size()
NEW_TGT_VOCAB = tokenizer.tgt_vocab_size()
print(f"\n  New src_vocab    : {NEW_SRC_VOCAB:,}")
print(f"  New tgt_vocab    : {NEW_TGT_VOCAB:,}")

# ── Dataset + DataLoader ───────────────────────────────────────────────────────

class DarijaDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tok: DarijaTokenizer,
                 max_len: int = MAX_SEQ_LEN):
        self.pairs   = pairs
        self.tok     = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        d, e = self.pairs[idx]
        return self._pack(self.tok.encode_src(d)), self._pack(self.tok.encode_tgt(e))

    def _pack(self, ids: list[int]) -> torch.Tensor:
        ids = ids[:self.max_len - 2]
        ids = [BOS_IDX] + ids + [EOS_IDX]
        ids += [PAD_IDX] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


def make_pad_mask(t: torch.Tensor) -> torch.Tensor:
    return t == PAD_IDX


torch_dataset   = DarijaDataset(combined_pairs, tokenizer)
dataloader      = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0, drop_last=True, pin_memory=True)
steps_per_epoch = len(dataloader)
total_steps     = EPOCHS * steps_per_epoch

print(f"\n  Training samples : {len(torch_dataset):,}")
print(f"  Steps/epoch      : {steps_per_epoch:,}")
print(f"  Total steps      : {total_steps:,}")
print(f"  Eff. batch size  : {BATCH_SIZE * GRAD_ACCUM_STEPS}")

# ── Model ──────────────────────────────────────────────────────────────────────
model = DarijaTransformer(
    src_vocab_size=NEW_SRC_VOCAB,
    tgt_vocab_size=NEW_TGT_VOCAB,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
weight_mb    = total_params * 4 / 1024 ** 2

# ── Optimizer + LR Schedule ───────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
scaler    = GradScaler(enabled=USE_AMP)


def lr_lambda(current_step: int) -> float:
    if current_step < WARMUP_STEPS:
        return current_step / max(1, WARMUP_STEPS)
    progress = (current_step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ── Step 4: Resume from finetune checkpoint OR load pre-trained weights ────────
print("\n" + "=" * 60)

# Look for the latest finetune epoch checkpoint to resume from.
resume_ckpt = None
for ep in range(EPOCHS, 0, -1):
    candidate = os.path.join(FINETUNE_DIR, f"epoch_{ep:02d}.pt")
    if os.path.exists(candidate):
        resume_ckpt = candidate
        break

if resume_ckpt:
    print("RESUMING FROM CHECKPOINT")
    print("=" * 60)
    ckpt        = torch.load(resume_ckpt, map_location="cpu")
    start_epoch = ckpt["epoch"]
    best_loss   = ckpt.get("best_loss", ckpt["avg_loss"])

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        # Old checkpoint — fast-forward scheduler to the correct position.
        sched_steps = start_epoch * (steps_per_epoch // GRAD_ACCUM_STEPS)
        for _ in range(sched_steps):
            scheduler.step()

    global_step = start_epoch * steps_per_epoch

    print(f"  Resumed from     : {resume_ckpt}")
    print(f"  Resuming at      : epoch {start_epoch + 1} / {EPOCHS}")
    print(f"  Best loss so far : {best_loss:.4f}")

else:
    print("LOADING PRE-TRAINED CHECKPOINT")
    print("=" * 60)
    ckpt      = torch.load(PRETRAINED_CKPT, map_location="cpu")
    old_state = ckpt["model"]
    print(f"  Checkpoint       : epoch {ckpt['epoch']}, loss {ckpt['avg_loss']:.4f}")

    with torch.no_grad():
        for key, old_tensor in old_state.items():
            if key not in model.state_dict():
                continue
            new_tensor = model.state_dict()[key]
            if old_tensor.shape == new_tensor.shape:
                new_tensor.copy_(old_tensor)
            else:
                # Vocab size changed: partial row copy for embedding tables.
                rows = min(old_tensor.size(0), new_tensor.size(0))
                new_tensor[:rows].copy_(old_tensor[:rows])
                print(f"  Partial copy: {key}  "
                      f"{list(old_tensor.shape)} → {list(new_tensor.shape)}, {rows} rows")

    transferred = sum(
        1 for k, v in old_state.items()
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape
    )
    print(f"  Weights loaded   : {transferred} / {len(old_state)} tensors (full shape match)")

    start_epoch = 0
    best_loss   = math.inf
    global_step = 0

print(f"\n  Total parameters : {total_params:,}")
print(f"  Weight VRAM      : {weight_mb:.1f} MB  (fp32)")
print(f"  VRAM headroom    : {VRAM_LIMIT_MB - weight_mb:.0f} MB remaining")

# ── Fine-tuning Loop ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINE-TUNING")
print("=" * 60)
print(f"  Epochs        : {start_epoch + 1} → {EPOCHS}")
print(f"  LR            : {LEARNING_RATE:.1e}  (+cosine schedule, {WARMUP_STEPS} warmup steps)")
print(f"  Mixed prec.   : {USE_AMP}")
print(f"  Checkpoints   : {FINETUNE_DIR}")
print()

for epoch in range(start_epoch + 1, EPOCHS + 1):
    model.train()
    epoch_loss   = 0.0
    batches_seen = 0
    optimizer.zero_grad()

    for step, (src, tgt) in enumerate(dataloader):
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_pad_mask = make_pad_mask(src)
        tgt_pad_mask = make_pad_mask(tgt_input)

        with autocast(enabled=USE_AMP):
            logits = model(
                src, tgt_input,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
            )
            loss = criterion(
                logits.reshape(-1, NEW_TGT_VOCAB),
                tgt_output.reshape(-1),
            )
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        # VRAM check after the very first forward pass of the run.
        if global_step == start_epoch * steps_per_epoch:
            vram_used = torch.cuda.memory_reserved(0) / (1024 ** 2)
            print(f"  [VRAM] Reserved : {vram_used:.1f} MB / {VRAM_LIMIT_MB} MB")
            if vram_used > VRAM_LIMIT_MB:
                print("  ERROR: VRAM limit exceeded. Reduce BATCH_SIZE and re-run.")
                sys.exit(1)
            print()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        epoch_loss   += loss.item() * GRAD_ACCUM_STEPS
        batches_seen += 1
        global_step  += 1

        if (step + 1) % 200 == 0:
            running_avg = epoch_loss / batches_seen
            current_lr  = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:2d} | Step {step+1:5d}/{steps_per_epoch} "
                  f"| Loss {loss.item() * GRAD_ACCUM_STEPS:.4f} "
                  f"| Avg {running_avg:.4f} "
                  f"| LR {current_lr:.2e}")

    avg_loss = epoch_loss / batches_seen
    vram_mb  = torch.cuda.memory_reserved(0) / (1024 ** 2)
    print()
    print(f"  ── Epoch {epoch:2d} complete ──")
    print(f"     Avg loss  : {avg_loss:.4f}")
    print(f"     VRAM      : {vram_mb:.1f} MB / {VRAM_LIMIT_MB} MB")

    if avg_loss < best_loss:
        best_loss = avg_loss

    # Save epoch checkpoint — includes scheduler + scaler for clean resume.
    ckpt_path = os.path.join(FINETUNE_DIR, f"epoch_{epoch:02d}.pt")
    torch.save(
        {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "avg_loss":  avg_loss,
            "best_loss": best_loss,
            "bpe_model": vocab.BPE_MODEL_FILE,
        },
        ckpt_path,
    )
    print(f"     Checkpoint: {ckpt_path}")

    if avg_loss <= best_loss:
        best_path = os.path.join(FINETUNE_DIR, "best_model.pt")
        torch.save(
            {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "avg_loss":  best_loss,
                "bpe_model": vocab.BPE_MODEL_FILE,
            },
            best_path,
        )
        print(f"     Best model: {best_path}  (loss={best_loss:.4f})")

    if vram_mb > VRAM_LIMIT_MB:
        print(f"\n  WARNING: VRAM {vram_mb:.1f} MB exceeded {VRAM_LIMIT_MB} MB limit.")
        print("  Stopping training to protect the GPU.")
        break

    print()

print("=" * 60)
print(f"FINE-TUNING COMPLETE — best loss: {best_loss:.4f}")
print("=" * 60)
