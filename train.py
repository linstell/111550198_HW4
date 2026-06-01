"""
PromptIR training pipeline - optimized for RTX 3060 Laptop (6GB VRAM).

Stages:
  Stage 1: 128px patches, batch=8,  lr=3e-4, 150ep  (~10 min/epoch)
  Stage 2: 192px patches, batch=4,  lr=1e-4, 60ep   (~20 min/epoch)
  Stage 3: 256px full,    batch=2,  lr=3e-5, 40ep   (~25 min/epoch)
  Stage 4: 256px full,    batch=1,  lr=1e-5, 20ep   (fine-tune)

Usage:
  python train.py --stage 1
  python train.py --stage 2
  python train.py --stage 3
  python train.py --stage 4 --best_psnr 29.5
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import RestorationDataset
from model import SimpleRestorationNet


# ---------------------------------------------------------------------------
# Stage configuration — tuned for RTX 3060 Laptop 6GB
# ---------------------------------------------------------------------------

STAGE_CONFIG = {
    1: dict(crop_size=128,  batch_size=8, lr=3e-4,  epochs=150, accum_steps=1),
    2: dict(crop_size=192,  batch_size=4, lr=1e-4,  epochs=60,  accum_steps=2),
    3: dict(crop_size=None, batch_size=2, lr=3e-5,  epochs=40,  accum_steps=2),
    4: dict(crop_size=None, batch_size=1, lr=1e-5,  epochs=20,  accum_steps=4),
}

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LATEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "latest_model.pth")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = prediction - target
        return torch.sqrt(diff * diff + self.epsilon ** 2).mean()


def edge_loss(prediction, target):
    pred_gray = prediction.mean(dim=1, keepdim=True)
    target_gray = target.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=prediction.dtype, device=prediction.device,
    ).unsqueeze(0)
    sobel_y = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
        dtype=prediction.dtype, device=prediction.device,
    ).unsqueeze(0)

    return (
        F.l1_loss(F.conv2d(pred_gray, sobel_x, padding=1),
                  F.conv2d(target_gray, sobel_x, padding=1))
        + F.l1_loss(F.conv2d(pred_gray, sobel_y, padding=1),
                    F.conv2d(target_gray, sobel_y, padding=1))
    )


def calculate_psnr(prediction, target):
    mse = torch.mean((prediction - target) ** 2)
    if mse.item() == 0:
        return 100.0
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach(), alpha=1.0 - self.decay
                )

    def apply_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, train_loader, criterion, optimizer,
    scaler, device, use_amp, ema, accum_steps=1,
):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress_bar):
        degraded, clean, _ = batch
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            restored = model(degraded)
            image_loss = criterion(restored, clean)
            detail_loss = edge_loss(restored, clean)
            raw_loss = image_loss + 0.05 * detail_loss
            loss = raw_loss / accum_steps

        if not torch.isfinite(loss):
            print("Warning: non-finite loss, skipping.")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.item()
        progress_bar.set_postfix(
            loss=f"{raw_loss.item():.4f}",
            img=f"{image_loss.item():.4f}",
            edge=f"{detail_loss.item():.4f}",
        )

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, use_amp):
    model.eval()
    total_loss, total_psnr = 0.0, 0.0
    rain_psnr, snow_psnr = 0.0, 0.0
    rain_count, snow_count = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            degraded, clean, labels = batch
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                restored = model(degraded)
                loss = criterion(restored, clean)

            total_loss += loss.item()

            for i in range(degraded.shape[0]):
                p = calculate_psnr(
                    restored[i:i+1].float(), clean[i:i+1].float()
                )
                total_psnr += p
                if labels[i].item() == 0:
                    rain_psnr += p
                    rain_count += 1
                elif labels[i].item() == 1:
                    snow_psnr += p
                    snow_count += 1

    n = len(val_loader.dataset)
    return (
        total_loss / len(val_loader),
        total_psnr / n,
        rain_psnr / max(rain_count, 1),
        snow_psnr / max(snow_count, 1),
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

DEGRADED_DIR = "hw4_realse_dataset/train/degraded"
CLEAN_DIR = "hw4_realse_dataset/train/clean"


def create_datasets(crop_size):
    train_ds = RestorationDataset(
        degraded_dir=DEGRADED_DIR, clean_dir=CLEAN_DIR,
        image_size=256, augment=True, crop_size=crop_size,
    )
    val_ds = RestorationDataset(
        degraded_dir=DEGRADED_DIR, clean_dir=CLEAN_DIR,
        image_size=256, augment=False, crop_size=None,
    )
    n = len(train_ds)
    split = int(0.9 * n)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    return Subset(train_ds, idx[:split]), Subset(val_ds, idx[split:])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=int, default=1, choices=[1, 2, 3, 4],
        help="1=patch128, 2=patch192, 3=full256, 4=finetune",
    )
    parser.add_argument(
        "--best_psnr", type=float, default=0.0,
        help="Known best val PSNR — prevents overwriting a better checkpoint.",
    )
    args = parser.parse_args()
    cfg = STAGE_CONFIG[args.stage]

    print(f"\n{'=' * 55}")
    print(f"  PromptIR Stage {args.stage}  |  "
          f"crop={cfg['crop_size']}  batch={cfg['batch_size']}  "
          f"lr={cfg['lr']}  epochs={cfg['epochs']}")
    print(f"{'=' * 55}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- Datasets ----
    train_dataset, val_dataset = create_datasets(cfg["crop_size"])
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    # ---- Model ----
    model = SimpleRestorationNet(base_channels=64).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    best_psnr = args.best_psnr

    if os.path.exists(BEST_MODEL_PATH) and args.stage > 1:
        print(f"Resuming from {BEST_MODEL_PATH}")
        state = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("Training from scratch.")

    ema = EMA(model, decay=0.999)

    # ---- Optimiser + scheduler ----
    criterion = CharbonnierLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Training loop ----
    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}  "
              f"|  LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, use_amp, ema, cfg["accum_steps"],
        )

        # Validate with EMA weights
        normal_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.apply_to(model)
        val_loss, val_psnr, rain_psnr, snow_psnr = validate(
            model, val_loader, criterion, device, use_amp,
        )
        model.load_state_dict(normal_state)

        scheduler.step()

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val   Loss: {val_loss:.6f}")
        print(f"Val   PSNR: {val_psnr:.4f}  (best: {best_psnr:.4f})")
        print(f"  Rain PSNR: {rain_psnr:.4f}  Snow PSNR: {snow_psnr:.4f}")

        torch.save(model.state_dict(), LATEST_MODEL_PATH)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ema.apply_to(model)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            model.load_state_dict(normal_state)
            print(f"  --> New best EMA checkpoint  (PSNR {best_psnr:.4f})")

    print(f"\nStage {args.stage} done.  Best val PSNR: {best_psnr:.4f}")
    if args.stage < 4:
        print(f"Next: python train.py --stage {args.stage + 1}")
    else:
        print("All stages complete. Run inference.py.")


if __name__ == "__main__":
    main()