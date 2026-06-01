"""
Validates that inference.py pipeline matches training validation PSNR.

Run from hw4 root: python src/validate_inference.py

This tests:
1. Model loads correctly
2. Normalization is correct
3. PSNR computed same way as training
4. No RGB/BGR issues
5. Whether latest_model vs best_model matters

If inference_psnr ~ training_val_psnr -> pipeline is correct
If inference_psnr << training_val_psnr -> inference pipeline is broken
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import importlib.util
import os
import sys

sys.path.insert(0, "src")


def calculate_psnr_float(pred, target):
    """PSNR on float [0,1] tensors - same as training."""
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return 100.0
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def calculate_psnr_uint8(pred_uint8, target_uint8):
    """PSNR on uint8 [0,255] arrays - as leaderboard might compute."""
    pred = pred_uint8.astype(np.float64)
    target = target_uint8.astype(np.float64)
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    # Load inference module
    spec = importlib.util.spec_from_file_location("inf", "src/inference.py")
    inf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test best_model.pth
    print("=" * 60)
    print("Testing: checkpoints/best_model.pth")
    print("=" * 60)
    model = inf.load_model("checkpoints/best_model.pth", device)

    # Load validation pairs from training data
    from dataset import RestorationDataset
    val_ds = RestorationDataset(
        degraded_dir="hw4_realse_dataset/train/degraded",
        clean_dir="hw4_realse_dataset/train/clean",
        image_size=256,
        augment=False,
    )

    # Use fixed val indices (same as training)
    n = len(val_ds)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    val_indices = idx[int(0.9 * n):]  # same 10% split as training

    print(f"Validation images: {len(val_indices)}")
    print()

    psnrs_float = []
    psnrs_uint8 = []
    psnrs_tta = []

    with torch.no_grad():
        for i, val_idx in enumerate(val_indices[:50]):  # test first 50
            degraded, clean, label = val_ds[val_idx]

            inp = degraded.unsqueeze(0).to(device)
            clean_t = clean.unsqueeze(0)

            # Method 1: direct forward (no TTA) — float PSNR
            out = model(inp)
            psnr_f = calculate_psnr_float(out.cpu(), clean_t)
            psnrs_float.append(psnr_f)

            # Method 2: convert to uint8 then compute PSNR
            out_uint8 = (out.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            clean_uint8 = (clean.numpy() * 255).clip(0, 255).astype(np.uint8)
            psnr_u = calculate_psnr_uint8(out_uint8, clean_uint8)
            psnrs_uint8.append(psnr_u)

            # Method 3: with TTA
            out_tta = inf.predict_with_tta(model, inp, [0, 1])
            psnr_tta = calculate_psnr_float(out_tta.cpu(), clean_t)
            psnrs_tta.append(psnr_tta)

            if i < 5:
                lab = "rain" if label == 0 else "snow"
                print(f"  [{lab}] direct={psnr_f:.2f} uint8={psnr_u:.2f} tta={psnr_tta:.2f}")

    print()
    print(f"Average PSNR (direct float):  {np.mean(psnrs_float):.4f}")
    print(f"Average PSNR (uint8 convert): {np.mean(psnrs_uint8):.4f}")
    print(f"Average PSNR (with TTA):      {np.mean(psnrs_tta):.4f}")
    print()

    diff = np.mean(psnrs_float) - np.mean(psnrs_uint8)
    print(f"Float vs uint8 PSNR gap: {diff:.4f} dB")
    if abs(diff) > 0.5:
        print("WARNING: Large gap between float and uint8 PSNR!")
        print("Leaderboard may use uint8 comparison -> this explains score drop")
    else:
        print("Float/uint8 gap is small -> normalization is OK")

    print()
    # Also test latest_model.pth if exists
    if os.path.exists("checkpoints/latest_model.pth"):
        print("=" * 60)
        print("Testing: checkpoints/latest_model.pth (raw weights)")
        print("=" * 60)
        model_latest = inf.load_model("checkpoints/latest_model.pth", device)
        psnrs_latest = []
        with torch.no_grad():
            for val_idx in val_indices[:50]:
                degraded, clean, _ = val_ds[val_idx]
                inp = degraded.unsqueeze(0).to(device)
                out = model_latest(inp)
                psnr = calculate_psnr_float(out.cpu(), clean.unsqueeze(0))
                psnrs_latest.append(psnr)
        print(f"Average PSNR (latest_model): {np.mean(psnrs_latest):.4f}")
        print(f"Average PSNR (best_model):   {np.mean(psnrs_float):.4f}")
        diff2 = np.mean(psnrs_float) - np.mean(psnrs_latest)
        print(f"EMA vs raw weights gap: {diff2:.4f} dB")


if __name__ == "__main__":
    main()
    
