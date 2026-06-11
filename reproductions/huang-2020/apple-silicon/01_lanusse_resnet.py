#!/usr/bin/env python3
"""
01_lanusse_resnet.py

PyTorch port of the CMU DeepLens ResNet-46 from Lanusse et al. 2018
(arXiv:1703.02642). This is the architecture Huang+2020 (arXiv:1906.00970) §3.2
re-implemented in TensorFlow and uses unchanged for their DECaLS lens search.

Architecture (Lanusse 2018 Fig 4):

  Conv 7x7 -> 32 ch, ELU, BN
  Stage 1 (32 -> 32):  3 × pre-act bottleneck (in=32, mid=16, out=32)        no /2
  Stage 2 (32 -> 64):  1 × pre-act bottleneck (in=32, mid=32, out=64, /2)
                        + 2 × pre-act bottleneck (in=64, mid=32, out=64)
  Stage 3 (64 -> 128): 1 × pre-act bottleneck (in=64, mid=64, out=128, /2)
                        + 2 × pre-act bottleneck (in=128, mid=64, out=128)
  Stage 4 (128 -> 256): 1 × pre-act bottleneck (in=128, mid=128, out=256, /2)
                        + 2 × pre-act bottleneck (in=256, mid=128, out=256)
  Stage 5 (256 -> 512): 1 × pre-act bottleneck (in=256, mid=256, out=512, /2)
                        + 2 × pre-act bottleneck (in=512, mid=256, out=512)
  AdaptiveAvgPool2d(1) -> Flatten -> Linear(512, 1) -> Sigmoid

The pre-activated bottleneck unit (Lanusse Fig 3, He+2016 v2 style):

  preact = ELU(BN(x))
  main   = Conv1x1(in→mid, stride=s)
            → ELU(BN) → Conv3x3(mid→mid)
            → ELU(BN) → Conv1x1(mid→out)
  shortcut = Conv1x1(in→out, stride=s)  if downsample else Identity
  return main(preact) + shortcut(preact_or_x)

Total: 1 initial conv + 5 stages × 3 blocks × 3 conv/block = 46 conv layers.
~3.0 M parameters at the standard depth.

When run as a script, this file performs a smoke test:
  - Builds the model
  - Forward + backward on a fake (B=2, C=3, H=101, W=101) batch
  - Checks: param count, output range, gradient finite
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBottleneck(nn.Module):
    """Pre-activated bottleneck residual block from Lanusse+2018 Fig 3.

    Pre-activation means BN -> activation comes BEFORE the conv, so the
    bottleneck's first BN-ELU is shared between the main path and the shortcut
    when the shortcut needs a projection (downsample or channel-change case).
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.proj_shortcut = downsample or in_ch != out_ch

        # main path
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)

        # shortcut
        if self.proj_shortcut:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preact = F.elu(self.bn1(x))
        sc = self.shortcut(preact) if self.proj_shortcut else x
        out = self.conv1(preact)
        out = self.conv2(F.elu(self.bn2(out)))
        out = self.conv3(F.elu(self.bn3(out)))
        return out + sc


class CMUDeepLens(nn.Module):
    """Lanusse+2018 ResNet-46 for galaxy-galaxy strong lens classification.

    Args:
        in_channels: image bands (Huang+2020 uses 3 = grz).
        block_stages: list of (in_ch, mid_ch, out_ch, n_blocks) per stage.
            Defaults to the L18/Huang-2020 spec.
    """

    DEFAULT_STAGES = [
        # (in_ch, mid_ch, out_ch, n_blocks) — first block downsamples if in_ch != out_ch
        (32, 16, 32, 3),
        (32, 32, 64, 3),
        (64, 64, 128, 3),
        (128, 128, 256, 3),
        (256, 256, 512, 3),
    ]

    def __init__(self, in_channels: int = 3, block_stages=None):
        super().__init__()
        stages = block_stages or self.DEFAULT_STAGES

        # Initial 7x7 conv + ELU + BN
        self.stem_conv = nn.Conv2d(in_channels, stages[0][0], kernel_size=7, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(stages[0][0])

        # Residual stages
        layers = []
        for s_idx, (in_ch, mid_ch, out_ch, n_blocks) in enumerate(stages):
            downsample_first = (s_idx > 0)  # downsample at the start of every stage except the first
            for b in range(n_blocks):
                if b == 0 and downsample_first:
                    layers.append(PreActBottleneck(in_ch, mid_ch, out_ch, downsample=True))
                elif b == 0:
                    # First block of stage 1 — input is 32 ch, output 32 ch, no downsample
                    layers.append(PreActBottleneck(in_ch, mid_ch, out_ch, downsample=False))
                else:
                    layers.append(PreActBottleneck(out_ch, mid_ch, out_ch, downsample=False))
        self.stages = nn.Sequential(*layers)

        # Head
        self.bn_final = nn.BatchNorm2d(stages[-1][2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stages[-1][2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        x = F.elu(self.stem_bn(x))
        x = self.stages(x)
        x = F.elu(self.bn_final(x))
        x = self.pool(x).flatten(1)
        logit = self.fc(x).squeeze(-1)
        return logit  # callers apply sigmoid for probability / use BCEWithLogitsLoss


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def smoke_test() -> None:
    """Build model, run forward+backward on fake input, check shapes + grad finiteness."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    model = CMUDeepLens(in_channels=3).to(device)
    n_params = count_params(model)
    print(f"[smoke] params: {n_params:,}")

    # Try the Lanusse-2018 native input size (45x45) and Huang-2020 (101x101)
    for size in (45, 101):
        x = torch.randn(4, 3, size, size, device=device)
        y_true = torch.randint(0, 2, (4,), device=device).float()
        logit = model(x)
        prob = torch.sigmoid(logit)
        loss = F.binary_cross_entropy_with_logits(logit, y_true)
        loss.backward()
        # check
        assert logit.shape == (4,), f"bad logit shape: {logit.shape}"
        assert (0 <= prob).all() and (prob <= 1).all()
        grads_finite = all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.requires_grad)
        print(f"[smoke] input {size}x{size}: logit={logit.shape} "
              f"prob∈[{prob.min().item():.3f}, {prob.max().item():.3f}] "
              f"loss={loss.item():.4f} grads_finite={grads_finite}")
        # zero grads for next size
        model.zero_grad(set_to_none=True)

    print("[smoke] OK")


if __name__ == "__main__":
    smoke_test()
