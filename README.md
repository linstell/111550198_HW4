# NYCU Visual Recognition HW4 — Image Restoration

**Student ID:** 111550198  
**Name:** Lina Tarnueva  
**Course:** Visual Recognition using Deep Learning, NYCU Spring 2026

---

## Introduction

This repository implements an image restoration model for the HW4 competition on CodaBench. The task is to train a **single model** that restores images degraded by two types of corruption: **rain streaks** and **snow particles**.

### Method Overview

We use a modified **PromptIR** architecture — a transformer-based model that uses learnable prompt vectors to adapt restoration behavior per degradation type without requiring explicit degradation labels at inference time. Key components:

- **Backbone:** Restormer-style Multi-Dconv Head Transposed Attention (MDTA) + Gated Depthwise FFN (GDFN)
- **Architecture:** 3-scale encoder-decoder with PromptBlocks at each decoder level
- **Training:** Progressive multi-stage training (128px → 192px → 256px patches)
- **Loss:** Charbonnier + Edge loss (0.05 weight)
- **TTA:** Flip-only (horizontal flip) — rotation TTA hurts rain images by up to 11 dB PSNR

### Key Findings

| Finding | Impact |
|---|---|
| 8-mode TTA with rotations hurts rain PSNR by 6-11 dB | Critical: use flip-only TTA |
| EMA weights vs raw weights | +0.4 dB PSNR improvement |
| Rain-aware augmentation (no 90° rotations for rain) | Prevents directional artifacts |
| Brightness augmentation [0.6, 1.5x] | Handles bright snow test images |
| Edge loss (Charbonnier + 0.05 × Edge) | +0.11 dB PSNR improvement |

---

## Environment Setup

```bash
# Clone repository
git clone https://github.com/linstell/111550198_HW4.git
cd 111550198_HW4

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install Pillow tqdm numpy
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+ with CUDA
- GPU with 6GB+ VRAM (tested on RTX 3060 Laptop)

---

## Dataset Structure

```
hw4_realse_dataset/
├── train/
│   ├── degraded/
│   │   ├── rain-1.png ... rain-1600.png
│   │   └── snow-1.png ... snow-1600.png
│   └── clean/
│       ├── rain_clean-1.png ... rain_clean-1600.png
│       └── snow_clean-1.png ... snow_clean-1600.png
└── test/
    └── degraded/
        └── 0.png ... 99.png
```

---

## Usage

### Training

Run progressive stages sequentially. Each stage fine-tunes from the previous checkpoint.

```bash
# Stage 1: 128px patches, lr=3e-4, 150 epochs (~1.5 days on RTX 3060)
python src/train.py --stage 1

# Stage 2: 192px patches, lr=1e-4, 60 epochs
python src/train.py --stage 2

# Stage 3: full 256px, lr=3e-5, 40 epochs
python src/train.py --stage 3

# Stage 4: fine-tuning, lr=1e-5, 20 epochs
python src/train.py --stage 4
```

Checkpoints are saved to `checkpoints/best_model.pth` (EMA weights).

### Inference

```bash
python src/inference.py
```

Output is saved to `outputs/pred.npz`. TTA uses modes `[0, 1]` (original + horizontal flip).

### Create Submission

```bash
python src/check_submission.py
cd outputs
# Windows
Compress-Archive -Path pred.npz -DestinationPath submission.zip -Force
# Linux/Mac
zip submission.zip pred.npz
```

### Validate Dataset

```bash
python src/check_dataset.py
```

---

## Model Architecture

```
Input (3, 256, 256)
    │
    ▼
PatchEmbed (3 → C)
    │
    ├── Enc1 (C,   2 blocks, 1 head)  ──────────────────────────────────┐
    │       │ ↓ Down1                                                    │
    ├── Enc2 (2C,  3 blocks, 2 heads) ──────────────────────────┐       │
    │       │ ↓ Down2                                            │       │
    ├── Enc3 (4C,  4 blocks, 4 heads) ──────────────────┐       │       │
    │       │ ↓ Down3                                    │       │       │
    └── Bottleneck (4C, 4 blocks, 8 heads)               │       │       │
            │ ↑ Up3                                      │       │       │
        Dec3 (4C, 4 blocks, 4 heads) + PromptBlock ◄─────┘       │       │
            │ ↑ Up2                                              │       │
        Dec2 (2C, 3 blocks, 2 heads) + PromptBlock ◄─────────────┘       │
            │ ↑ Up1                                                      │
        Dec1 (C,  2 blocks, 1 head)  + PromptBlock ◄────────────────────┘
            │
    Output Conv (C → 3)
            │
    clamp(Input + Residual, 0, 1)
```

**Parameters:** ~15.4M (base_channels=64) or ~9.2M (base_channels=48)

---

## Performance Snapshot

| Submission | Model | Stage | Leaderboard PSNR |
|---|---|---|---|
| Baseline (old) | PromptIRSmall 26M | Stage 5 | 29.90 |
| New model 48ch | PromptIRNet 9.2M | Stage 1 | 30.01 |
| New model 48ch | PromptIRNet 9.2M | Stage 2 | 30.28 |
| New model 48ch | PromptIRNet 9.2M | Stage 3 | 30.44 |
| New model 48ch | PromptIRNet 9.2M | Stage 4 | **30.47** |
| New model 64ch | PromptIRNet 15.4M | Stage 1 | 30.31 |
| New model 64ch | PromptIRNet 15.4M | Stage 2 | 30.43 |

> **Current best: 30.47** (above strong baseline of 30.0)

*[Insert leaderboard screenshot here]*

---

## File Structure

```
src/
├── model.py          # PromptIRNet architecture
├── train.py          # Progressive stage training pipeline
├── dataset.py        # RestorationDataset with rain-aware augmentation
├── inference.py      # Inference with auto-detect architecture + TTA
├── validate_inference.py  # Pipeline validation tool
├── check_dataset.py  # Dataset sanity check
└── check_submission.py    # Submission format validator
checkpoints/          # (not committed — too large)
outputs/              # Generated predictions
```

---

## References

1. Potlapalli et al., "PromptIR: Prompting for All-in-One Blind Image Restoration," arXiv:2306.13090
2. Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration," CVPR 2022
3. PromptIR GitHub: https://github.com/va1shn9v/PromptIR
