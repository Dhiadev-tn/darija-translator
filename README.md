# Darija-Translator
### The First Open-Source Tunisian Darija-to-English NLP Pipeline

---

## Why This Exists

I felt a pang in my chest the day I realized how underrepresented my Tunisian dialect is in the digital world.

I searched. I found nothing , only uncleaned Moroccan datasets that didn't speak my language, my culture, or my identity. And in that moment it became clear to me: the door to digital representation for Tunisian Darija hadn't been built yet. So I decided to build it myself.

This project is not just a translation model. It is a statement. Tunisia is a true standing culture that deserves recognition — and it has minds capable of fighting each day despite limited resources and restricted horizons.

Previous work explored Tunisian Darija translation using pre-trained models. 
I am intending to build the first ever pipeline from scratch 
---

## The Problem

**Tunisian Darija is spoken by 12 million people. It has near-zero representation in modern NLP research.**

- No clean open-source Tunisian Darija dataset exists
- Existing Arabic NLP tools fail on Tunisian dialect
- Arabizi ,the way Tunisians write their dialect using Latin letters and numbers (3→ع, 7→ح, 9→ق, 5→خ) ,is completely unsupported by standard tokenizers
- The digital world has ignored an entire identity

This project is the first step toward changing that.

---

## What Was Built

A complete NLP pipeline built from scratch on an RTX 3050 Laptop (4GB VRAM):

```
vocab.py          → Custom BPE tokenizer with Arabizi support
                    Handles 3, 7, 9, 5 as protected markers
                    16,000 token vocabulary

data_loader.py    → Dataset pipeline
                    35,977 cleaned Darija-English pairs

clean_data.py     → Data cleaning pipeline
                    Removed duplicates, noise, and Moroccan-specific terms

model.py          → Nano-Transformer architecture
                    15.6M parameters
                    Built entirely in PyTorch

train.py          → Full training loop
                    Mixed precision (fp16)
                    Gradient accumulation
                    VRAM-safe (peak: 628MB / 3500MB limit)

finetune.py       → Fine-tuning pipeline
                    120 hand-crafted Tunisian sentence pairs
                    Built manually by a native speaker

inference.py      → Translation interface
```

---

## The Dataset

### Foundation Layer — Cleaned Moroccan Darija
- **Source**: atlasia/darija_english (HuggingFace)
- **After cleaning**: 35,977 pairs
- **Removed**: 8,736 pairs , duplicates, short pairs, untranslated sentences, Moroccan-specific vocabulary

### Tunisian Layer — Hand-Crafted by a native tunisian speaker 
- **120 sentence pairs** built manually
- Authentic Tunisian Darija across 12 categories: greetings, farewells, family, food & drinks, shopping & money, time & directions, emotions & feelings, compliments & insults, school & studying, health & illness, Tunisian slang, Tunisian proverbs
- **Zero automated generation** , every pair written and validated by a native Tunisian speaker

### Next Phase — Field Collection (Summer 2026)
- Multi-region recordings across Tunisia (Tunis, Sfax, Sousse, Djerba, Kairouan)
- Multiple age groups and regional dialects
- Target: 3,000–5,000 validated pairs
- This will be the first field-recorded Tunisian Darija dataset in existence

---

## Technical Highlights

| Component | Detail |
|-----------|--------|
| Architecture | Encoder-Decoder Transformer |
| Parameters | ~15.6M |
| Tokenizer | SentencePiece BPE — 16,000 tokens, trained on combined Darija+English corpus, shared between source and target, Arabizi-aware |
| Embedding dim | 256 (source + target, weight-tied output layer) |
| Encoder / Decoder | 4 layers each |
| Training data | ~35,000 cleaned Moroccan Darija pairs |
| Pre-training | 15 epochs — final loss 2.8393 |
| Fine-tuning data | 120 hand-crafted Tunisian pairs |
| Fine-tuning | 20 epochs — final loss 2.6264 |
| Model size | ~59.4 MB (float32) |
| Peak VRAM | 628MB out of 3,500MB limit |
| Hardware | NVIDIA RTX 3050 Laptop (4GB VRAM) |
| Framework | PyTorch 2.1 + CUDA 12.1 |

> **Architecture note**: The output projection layer shares weights with the target embedding (weight tying) ,a deliberate design decision that reduces parameters and improves generalization on low-resource translation tasks.

---

## Current Results

The model demonstrates partial understanding of Darija structure. Simple greetings translate correctly. Complex sentences reveal the core limitation: **no large clean Tunisian dataset exists yet.**

This is not a failure. This is the finding.

The pipeline works. The architecture is proven. The data gap is documented firsthand. Phase 2 , field collection , is the answer.

---

## How To Run It

### Requirements
```bash
Python 3.10+
PyTorch 2.1.0 (CUDA 12.1)
sentencepiece
datasets (HuggingFace)
```

### Setup
```bash
git clone https://github.com/dhiadev-tn/darija-translator
cd darija-translator
pip install torch sentencepiece datasets
```

### Clean The Data
```bash
python clean_data.py
```

### Train
```bash
python train.py
```

### Translate
```bash
python inference.py
```

---

## Roadmap

```
 Phase 1 — Foundation
   Arabizi tokenizer, data pipeline, Nano-Transformer

 Phase 2 — Training
   35,977 cleaned pairs, BPE tokenization, fine-tuning

⏳ Phase 3 — Field Collection (Summer 2026)
   Multi-region Tunisian recordings
   Native speaker validation
   3,000–5,000 authentic pairs

⏳ Phase 4 — Production Model
   Retrain on clean Tunisian foundation
   HuggingFace public release
   Open for community contributions
```

---

## License

This project is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

You are free to use, share, and build on this work for non-commercial purposes — as long as you credit the author and share your improvements under the same license.

---

## About

Built by **Dhia** ([@dhiadev-tn](https://github.com/dhiadev-tn)) — Tunisia

*This project is a bridge. A link between Tunisia and the digital world. Proof that identity deserves representation ,and that the people who carry that identity are capable of building that representation themselves.*

---
## Links
- 📊 Dataset: https://huggingface.co/datasets/Dhiadev-tn/tunisian-darija-english
- 🤗 Model: https://huggingface.co/Dhiadev-tn/darija-translator

---

> *"I didn't wait for the door to be built. I built it."*
