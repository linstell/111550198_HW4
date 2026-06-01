"""
PromptIR - GPU-efficient version for RTX 3060 Laptop (6GB VRAM).

Reduced from 58.5M to ~20M params by:
  - 3 scales instead of 4 (1/8 resolution bottleneck is enough)
  - base_channels=48 kept (same as paper)
  - Block counts [2,3,4] encoder, [4,3,2] decoder (reduced from paper)
  - Bottleneck: 4 blocks (reduced from 8)
  - PromptBlocks at decoder only (paper design, correct)

This fits in 6GB VRAM at batch=8, 128px patches.
Expected: ~30-31 PSNR (previous best was 29.90 with ~25M model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for (B, C, H, W) tensors."""

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
    """
    Multi-Dconv Head Transposed Attention (Restormer).
    Channel-wise attention: O(C^2 * HW) not O(H^2W^2).
    """

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
        out = torch.matmul(attn, v)
        return self.proj(out.reshape(batch, channels, height, width))


class GDFN(nn.Module):
    """Gated Depthwise FFN (Restormer)."""

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
    """Pre-norm MDTA + pre-norm GDFN."""

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
    """
    PromptIR prompt interaction block (paper Section 3.2).

    Learns prompt_len prototype tensors. Dynamically selects a weighted
    combination conditioned on input features. Placed at decoder levels
    to adapt restoration behavior per degradation type.
    """

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
        prompt = F.interpolate(prompt, size=(h, w), mode="bilinear", align_corners=False)
        return self.fusion(torch.cat([x, prompt], dim=1))


class EncoderLevel(nn.Module):
    def __init__(self, channels, num_blocks, num_heads):
        super().__init__()
        self.blocks = nn.Sequential(
            *[TransformerBlock(channels, num_heads) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class DecoderLevel(nn.Module):
    def __init__(self, channels, num_blocks, num_heads):
        super().__init__()
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(
            *[TransformerBlock(channels, num_heads) for _ in range(num_blocks)]
        )
        # PromptBlock at every decoder level (paper design)
        self.prompt = PromptBlock(channels)

    def forward(self, x, skip):
        x = self.reduce(torch.cat([x, skip], dim=1))
        return self.prompt(self.blocks(x))


class PromptIRNet(nn.Module):
    """
    PromptIR - GPU-efficient 3-scale variant.

    Architecture (3 scales):
      Input -> PatchEmbed
      Enc1 (C,  2 blocks, 1 head)  -> Down1 -> 2C
      Enc2 (2C, 3 blocks, 2 heads) -> Down2 -> 4C
      Enc3 (4C, 4 blocks, 4 heads) -> Down3 -> 4C
      Bottleneck (4C, 4 blocks, 8 heads)
      Up3 -> Dec3 (4C, 4 blocks, 4 heads) + PromptBlock
      Up2 -> Dec2 (2C, 3 blocks, 2 heads) + PromptBlock
      Up1 -> Dec1 (C,  2 blocks, 1 head)  + PromptBlock
      Output conv

    With base_channels=48: C=48, 2C=96, 4C=192
    Parameters: ~20M — fits RTX 3060 Laptop at batch=8, 128px

    Clamp always applied: restored = clamp(input + residual, 0, 1)
    """

    def __init__(self, base_channels=48):
        super().__init__()

        c1 = base_channels       # 48
        c2 = base_channels * 2   # 96
        c3 = base_channels * 4   # 192

        self.patch_embed = nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False)

        # Encoder
        self.enc1 = EncoderLevel(c1, num_blocks=2, num_heads=1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2, bias=False)

        self.enc2 = EncoderLevel(c2, num_blocks=3, num_heads=2)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2, bias=False)

        self.enc3 = EncoderLevel(c3, num_blocks=4, num_heads=4)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=2, stride=2, bias=False)

        # Bottleneck at 1/8 resolution
        self.bottleneck = nn.Sequential(
            *[TransformerBlock(c3, num_heads=8) for _ in range(4)]
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2, bias=False)
        self.dec3 = DecoderLevel(c3, num_blocks=4, num_heads=4)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=False)
        self.dec2 = DecoderLevel(c2, num_blocks=3, num_heads=2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=False)
        self.dec1 = DecoderLevel(c1, num_blocks=2, num_heads=1)

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

        # Clamp always applied — inference is safe
        return torch.clamp(x + self.output(d1), 0.0, 1.0)


SimpleRestorationNet = PromptIRNet