"""
Inference script — auto-detects model architecture from checkpoint.

Supports:
  - PromptIRSmall (base_channels=64, has enc1.prompt) -> old 29.90 model
  - PromptIRNet   (base_channels=48, no enc1.prompt)  -> new model
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3, channels * 3,
            kernel_size=3, padding=1, groups=channels * 3, bias=False,
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        batch, channels, height, width = x.shape
        head_dim = channels // self.num_heads
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(batch, self.num_heads, head_dim, height * width)
        k = k.reshape(batch, self.num_heads, head_dim, height * width)
        v = v.reshape(batch, self.num_heads, head_dim, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (torch.matmul(q, k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return self.proj(torch.matmul(attn, v).reshape(batch, channels, height, width))


class GDFN(nn.Module):
    def __init__(self, channels, expansion=2.66):
        super().__init__()
        hidden = int(channels * expansion)
        self.proj_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(
            hidden * 2, hidden * 2,
            kernel_size=3, padding=1, groups=hidden * 2, bias=False,
        )
        self.gate = SimpleGate()
        self.proj_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj_out(self.gate(self.dw(self.proj_in(x))))


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GDFN(channels, expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PromptBlock(nn.Module):
    def __init__(self, channels, prompt_len=5, prompt_size=32):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompts = nn.Parameter(
            torch.randn(prompt_len, channels, prompt_size, prompt_size) * 0.02
        )
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, prompt_len, kernel_size=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        weights = F.softmax(
            self.weight_generator(x).view(b, self.prompt_len), dim=1
        )
        prompt = torch.einsum("bp,pchw->bchw", weights, self.prompts)
        prompt = F.interpolate(
            prompt, size=(h, w), mode="bilinear", align_corners=False
        )
        return self.fusion(torch.cat([x, prompt], dim=1))


# ---------------------------------------------------------------------------
# OLD model: PromptIRSmall
# Exact match for FINAL_best_29_90.pth:
#   base_channels=64, enc1/enc2 have prompts, enc3=6 blocks,
#   bottleneck=9 items (4 TB + PromptBlock + 4 TB), prompt_len=6,
#   heads: enc1=2, enc2=4, enc3/bottleneck/dec3=8, dec2=4, dec1=2
# ---------------------------------------------------------------------------

class _OldEncoderLevel(nn.Module):
    def __init__(self, channels, num_blocks, num_heads,
                 use_prompt=False, prompt_len=6):
        super().__init__()
        self.blocks = nn.Sequential(
            *[TransformerBlock(channels, num_heads) for _ in range(num_blocks)]
        )
        self.prompt = (
            PromptBlock(channels, prompt_len=prompt_len)
            if use_prompt else nn.Identity()
        )

    def forward(self, x):
        return self.prompt(self.blocks(x))


class _OldDecoderLevel(nn.Module):
    def __init__(self, channels, num_blocks, num_heads, prompt_len=6):
        super().__init__()
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(
            *[TransformerBlock(channels, num_heads) for _ in range(num_blocks)]
        )
        self.prompt = PromptBlock(channels, prompt_len=prompt_len)

    def forward(self, x, skip):
        x = self.reduce(torch.cat([x, skip], dim=1))
        return self.prompt(self.blocks(x))


class PromptIRSmall(nn.Module):
    """Old model — FINAL_best_29_90.pth (base_channels=64)."""

    def __init__(self, base_channels=64):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.patch_embed = nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False)

        self.enc1 = _OldEncoderLevel(c1, 2, num_heads=2,
                                     use_prompt=True, prompt_len=6)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2, bias=False)

        self.enc2 = _OldEncoderLevel(c2, 3, num_heads=4,
                                     use_prompt=True, prompt_len=6)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2, bias=False)

        self.enc3 = _OldEncoderLevel(c3, 6, num_heads=8, use_prompt=False)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=2, stride=2, bias=False)

        self.bottleneck = nn.Sequential(
            TransformerBlock(c3, num_heads=8),
            TransformerBlock(c3, num_heads=8),
            TransformerBlock(c3, num_heads=8),
            TransformerBlock(c3, num_heads=8),
            PromptBlock(c3, prompt_len=6),
            TransformerBlock(c3, num_heads=8),
            TransformerBlock(c3, num_heads=8),
            TransformerBlock(c3, num_heads=8),
            TransformerBlock(c3, num_heads=8),
        )

        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2, bias=False)
        self.dec3 = _OldDecoderLevel(c3, 6, num_heads=8, prompt_len=6)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=False)
        self.dec2 = _OldDecoderLevel(c2, 3, num_heads=4, prompt_len=6)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=False)
        self.dec1 = _OldDecoderLevel(c1, 2, num_heads=2, prompt_len=6)

        self.output = nn.Conv2d(c1, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        feat = self.patch_embed(x)
        e1 = self.enc1(feat)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(self.up3(b), e3)
        d2 = self.dec2(self.up2(d3), e2)
        d1 = self.dec1(self.up1(d2), e1)
        return torch.clamp(x + self.output(d1), 0.0, 1.0)


# ---------------------------------------------------------------------------
# NEW model: PromptIRNet
# Matches best_stage3_new.pth:
#   base_channels=48, no enc prompts, enc3/dec3=4 blocks,
#   bottleneck=4 TB, prompt_len=5,
#   heads: enc1=1, enc2=2, enc3/bottleneck/dec3=4, dec2=2, dec1=1
# ---------------------------------------------------------------------------

class _NewEncoderLevel(nn.Module):
    def __init__(self, channels, num_blocks, num_heads):
        super().__init__()
        self.blocks = nn.Sequential(
            *[TransformerBlock(channels, num_heads) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class _NewDecoderLevel(nn.Module):
    def __init__(self, channels, num_blocks, num_heads, prompt_len=5):
        super().__init__()
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(
            *[TransformerBlock(channels, num_heads) for _ in range(num_blocks)]
        )
        self.prompt = PromptBlock(channels, prompt_len=prompt_len)

    def forward(self, x, skip):
        x = self.reduce(torch.cat([x, skip], dim=1))
        return self.prompt(self.blocks(x))


class PromptIRNet(nn.Module):
    """New model — best_stage3_new.pth (base_channels=48)."""

    def __init__(self, base_channels=48):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.patch_embed = nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False)

        self.enc1 = _NewEncoderLevel(c1, num_blocks=2, num_heads=1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2, bias=False)

        self.enc2 = _NewEncoderLevel(c2, num_blocks=3, num_heads=2)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2, bias=False)

        self.enc3 = _NewEncoderLevel(c3, num_blocks=4, num_heads=4)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=2, stride=2, bias=False)

        self.bottleneck = nn.Sequential(
            *[TransformerBlock(c3, num_heads=8) for _ in range(4)]
        )

        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2, bias=False)
        self.dec3 = _NewDecoderLevel(c3, num_blocks=4, num_heads=4, prompt_len=5)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=False)
        self.dec2 = _NewDecoderLevel(c2, num_blocks=3, num_heads=2, prompt_len=5)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=False)
        self.dec1 = _NewDecoderLevel(c1, num_blocks=2, num_heads=1, prompt_len=5)

        self.output = nn.Conv2d(c1, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        feat = self.patch_embed(x)
        e1 = self.enc1(feat)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(self.up3(b), e3)
        d2 = self.dec2(self.up2(d3), e2)
        d1 = self.dec1(self.up1(d2), e1)
        return torch.clamp(x + self.output(d1), 0.0, 1.0)


# Keep alias for train.py compatibility
SimpleRestorationNet = PromptIRNet


# ---------------------------------------------------------------------------
# Auto-detect and load correct model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    keys = list(state_dict.keys())
    base_channels = state_dict["patch_embed.weight"].shape[0]
    has_enc_prompt = any(k.startswith("enc1.prompt.") for k in keys)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"base_channels: {base_channels}, enc_prompt: {has_enc_prompt}")

    if has_enc_prompt:
        print("Architecture: PromptIRSmall (old)")
        model = PromptIRSmall(base_channels=base_channels).to(device)
    else:
        print("Architecture: PromptIRNet (new)")
        model = PromptIRNet(base_channels=base_channels).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n/1e6:.1f}M  |  Loaded OK")
    return model


# ---------------------------------------------------------------------------
# TTA
# ---------------------------------------------------------------------------

def apply_tta_transform(x, mode):
    if mode == 0: return x
    if mode == 1: return torch.flip(x, dims=[-1])
    if mode == 2: return torch.flip(x, dims=[-2])
    if mode == 3: return torch.rot90(x, k=1, dims=[-2, -1])
    if mode == 4: return torch.rot90(x, k=2, dims=[-2, -1])
    if mode == 5: return torch.rot90(x, k=3, dims=[-2, -1])
    if mode == 6: return torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1])
    if mode == 7: return torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-2])
    raise ValueError(f"Unknown TTA mode: {mode}")


def reverse_tta_transform(x, mode):
    if mode == 0: return x
    if mode == 1: return torch.flip(x, dims=[-1])
    if mode == 2: return torch.flip(x, dims=[-2])
    if mode == 3: return torch.rot90(x, k=-1, dims=[-2, -1])
    if mode == 4: return torch.rot90(x, k=-2, dims=[-2, -1])
    if mode == 5: return torch.rot90(x, k=-3, dims=[-2, -1])
    if mode == 6:
        x = torch.flip(x, dims=[-1])
        return torch.rot90(x, k=-1, dims=[-2, -1])
    if mode == 7:
        x = torch.flip(x, dims=[-2])
        return torch.rot90(x, k=-1, dims=[-2, -1])
    raise ValueError(f"Unknown TTA mode: {mode}")


def predict_with_tta(model, input_tensor, modes):
    predictions = []
    for mode in modes:
        aug = apply_tta_transform(input_tensor, mode)
        pred = model(aug)
        pred = reverse_tta_transform(pred, mode)
        predictions.append(pred)
    return torch.clamp(torch.stack(predictions).mean(0), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_sorted_test_files(test_dir):
    files = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return sorted(files, key=lambda name: int(os.path.splitext(name)[0]))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta_modes = [0, 1]

    print(f"Device: {device}  |  TTA modes: {tta_modes}")

    test_dir = "hw4_realse_dataset/test/degraded"
    output_dir = "outputs/restored"
    checkpoint_path = "checkpoints/best_model.pth"

    os.makedirs(output_dir, exist_ok=True)

    model = load_model(checkpoint_path, device)

    test_files = get_sorted_test_files(test_dir)
    print(f"Test images: {len(test_files)}")

    predictions = {}

    with torch.no_grad():
        for file_name in tqdm(test_files, desc="Restoring"):
            image_path = os.path.join(test_dir, file_name)
            degraded_image = Image.open(image_path).convert("RGB")
            original_size = degraded_image.size

            input_tensor = transforms.ToTensor()(degraded_image)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            restored_tensor = predict_with_tta(model, input_tensor, tta_modes)
            restored_tensor = restored_tensor.squeeze(0).cpu()
            restored_image = transforms.ToPILImage()(restored_tensor)

            if restored_image.size != original_size:
                restored_image = restored_image.resize(original_size, Image.LANCZOS)

            restored_image.save(os.path.join(output_dir, file_name))

            arr = np.transpose(
                np.array(restored_image, dtype=np.uint8), (2, 0, 1)
            )
            predictions[file_name] = arr

    np.savez("outputs/pred.npz", **predictions)
    print("Saved: outputs/pred.npz")


if __name__ == "__main__":
    main()
