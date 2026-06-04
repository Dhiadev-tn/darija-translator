# Darija-Translator
### The First Open-Source Tunisian Darija-to-English NLP Pipeline

---

## Why This Exists

I felt a pang in my chest the day I realized how underrepresented my Tunisian dialect is in the digital world.

I searched. I found nothing , only uncleaned Moroccan datasets that didn't speak my language, my culture, or my identity. And in that moment it became clear to me: the door to digital representation for Tunisian Darija hadn't been built yet. So I decided to build it myself.

This project is not just a translation model. It is a statement. Tunisia is a true standing culture that deserves recognition , and it has minds capable of fighting each day despite limited resources and restricted horizons.

Previous work explored Tunisian Darija translation using pre-trained models. 
This is the first pipeline built from scratch — no shortcuts, no pre-trained foundations, every component written by hand.
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

A complete NLP pipeline built from scratch — starting on an RTX 3050 Laptop (4GB VRAM):

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
                    500 hand-crafted Tunisian sentence pairs
                    Built manually by a native speaker

split_data.py     → Stratified train/val/test split
                    8 train / 1 val / 1 test per category
                    Ensures honest evaluation on unseen data

evaluate.py       → BLEU scoring on locked test set
                    First honest metric — test set never trained on

inference.py      → Translation interface
```

---

## The Dataset

### Foundation Layer : Cleaned Moroccan Darija
- **Source**: atlasia/darija_english (HuggingFace)
- **After cleaning**: 35,977 pairs
- **Removed**: 8,736 pairs , duplicates, short pairs, untranslated sentences, Moroccan-specific vocabulary

### Tunisian Layer : Hand-Crafted by a native Tunisian speaker
- **500 sentence pairs** built manually across **50 categories**
- **Zero automated generation** — every pair written and validated by a native Tunisian speaker
- Categories: greetings, farewells, family, food & drinks, shopping & money, time & directions, emotions & feelings, compliments & insults, school & studying, health & illness, Tunisian slang, Tunisian proverbs, weather, seasons, islamic expressions, transportation, cafe culture, code switching, wedding celebrations, hammam culture, Ramadan culture, market & medina, barbershop culture, bureaucracy & paperwork, football culture, bac exam culture, work & job seeking, louage culture, 3aylet gathering, 3rouset el 7ouma, 7ouma life, coffee shop arguments, French loanwords in Darija, police & dawla, el ghorba, el 3aza & death, summer & beach culture, 3id el kbir, el hajj & el omra, taxi culture, military service, university life, fishing culture, political talk, social pressure, farming life, sbitar culture, pharmacy & medicine, olive harvest season, friday culture

### Next Phase : Growing the Dataset (Summer 2026)
- Collecting from family, community, and everyday Tunisian life — cafes, markets, gatherings
- Multiple sources: self, family members, friends, strangers
- Target: **1,000 pairs** by August 2026
- Every milestone triggers a retrain and a new BLEU measurement — the score going up is the story

---

## Technical Highlights

| Component | Detail |
|-----------|--------|
| Architecture | Encoder-Decoder Transformer |
| Parameters | ~15.6M |
| Tokenizer | SentencePiece BPE : 16,000 tokens, trained on combined Darija+English corpus, shared between source and target, Arabizi-aware |
| Embedding dim | 256 (source + target, weight-tied output layer) |
| Encoder / Decoder | 4 layers each |
| Training data | ~35,000 cleaned Moroccan Darija pairs |
| Pre-training | 15 epochs on Moroccan data (RTX 3050 Laptop, 4GB VRAM) |
| Fine-tuning data | 500 hand-crafted Tunisian pairs, stratified 8/1/1 split |
| Fine-tuning | 20 epochs, best checkpoint by val loss (RTX 4070 Desktop) |
| BLEU score | **3.89** — v1 baseline, June 2026 |
| Model size | ~59.4 MB (float32) |
| Peak VRAM (pre-training) | 628MB out of 3,500MB limit |
| Framework | PyTorch 2.1 + CUDA 12.1 |

> **Architecture note**: The output projection layer shares weights with the target embedding (weight tying) ,a deliberate design decision that reduces parameters and improves generalization on low-resource translation tasks.

---

## Current Results

| Metric | Value |
|--------|-------|
| BLEU score | **3.89** (v1 baseline — 500 pairs, June 2026) |
| Best checkpoint | Epoch 7 / 20 (by validation loss) |
| Test set | 50 pairs, locked — never trained on |

The model demonstrates partial understanding of Darija structure. Simple sentences translate with recognizable meaning. Complex cultural expressions reveal the core limitation: **not enough Tunisian data yet.**

This is not a failure. This is the finding.

The pipeline works. The architecture is proven. The data gap is documented firsthand. 3.89 is the number to beat — and it will be beaten, retrain by retrain, as the dataset grows toward 1,000 pairs.

---

## How To Run It

### Requirements
```bash
Python 3.10+
PyTorch 2.1.0 (CUDA 12.1)
sentencepiece
sacrebleu
```

### Setup
```bash
git clone https://github.com/dhiadev-tn/darija-translator
cd darija-translator
pip install sentencepiece sacrebleu
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Train
```bash
python train.py
```

### Translate
```bash
python inference.py
```

### Evaluate
```bash
python evaluate.py
```

---

## Roadmap

```
✅ Phase 1 : Foundation
   Arabizi tokenizer, data pipeline, Nano-Transformer
   Pre-trained on 35,977 Moroccan pairs (RTX 3050 Laptop)

✅ Phase 2 : v1 Baseline (June 2026)
   500 hand-crafted Tunisian pairs, 50 categories
   Proper train/val/test splits — honest evaluation
   First official BLEU score: 3.89

⏳ Phase 3 : Dataset Growth (Summer 2026)
   Collecting from family, community, everyday Tunisian life
   Target: 1,000 authentic pairs by August 2026
   Retrain after milestone → new BLEU score

⏳ Phase 4 : Demo & Visibility
   HuggingFace Spaces live demo (Gradio)
   Blog posts written in real time during the field mission and the collection process, the cultural moments, the data growing live
   Final BLEU score — measurable progress from v1 to v2
```

---

## License

This project is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

You are free to use, share, and build on this work for non-commercial purposes , as long as you credit the author and share your improvements under the same license.

---

## About

Built by **Dhia** ([@dhiadev-tn](https://github.com/dhiadev-tn)) , Tunisia

*This project is a bridge. A link between Tunisia and the digital world. Proof that identity deserves representation ,and that the people who carry that identity are capable of building that representation themselves.*

---
## Links
- 📊 Dataset: https://huggingface.co/datasets/Dhiadev-tn/tunisian-darija-english
- 🤗 Model: https://huggingface.co/Dhiadev-tn/darija-translator

---

> *"I didn't wait for the door to be built. I built it."*
