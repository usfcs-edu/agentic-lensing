#!/usr/bin/env python3
"""
01b_shielded_resnet.py

The "shielded" ResNet of Huang+2021 (arXiv:2005.04730, §3.3) — the headline
architectural novelty of that paper relative to the Huang+2020 / Lanusse+2018
(L18) ResNet-46 reproduced in `01_lanusse_resnet.py`.

What the paper says (§3.3, verbatim where quoted):
  "These 'shields' are 1×1 convolutional layers inserted between every three
   blocks of the L18 architecture ... they have the effect of reducing
   dimensionality and mitigating the exponential increase in computational
   complexity ... we reduce the number of trainable parameters by a factor of
   50 (from 3 million to 60 thousand) ... the validation AUC has increased from
   0.992 (using the original L18 model) to 0.997 ... in the final block of the
   architecture in L18 we experimented with reducing the output from 512
   channels to 256, 128, 64, 32, and 16 channels. We find that 'shields' that
   keep the output to 32 channels perform the best."

Reconstruction (the paper does not publish exact intermediate channel counts):
  The L18 body is 5 stages × 3 pre-activated bottleneck blocks (15 blocks).
  We insert one pre-activated 1×1 "shield" after each of the FIRST FOUR
  3-block stages (i.e. "between every three blocks") — 4 shields — and cap the
  per-stage block width at 32 output channels. Each shield squeezes the channel
  count to SHIELD_CH before the next stage. The final stage outputs `final_out`
  (the paper's swept parameter; default 32, their best). This preserves L18's
  depth (15 blocks) and spatial-downsampling schedule while collapsing the wide
  256/512-channel final stages that dominate L18's 3.5M params, landing at
  ~60K params at final_out=32 — matching the paper's "60 thousand" and the
  ~50× reduction.

The `Shield` is pre-activated (BN → ELU → 1×1 conv, no bias) to match the
He-2016-v2 / L18 pre-activation convention used by `PreActBottleneck`.

When run as a script this performs a smoke test that ASSERTS the verifiable
reproduction targets: ~60K params, exactly 4 shields, 15 bottleneck blocks,
≥50× reduction vs L18, and prints the final_out channel-sweep param ladder
(reproducing the paper's 512→…→16 experiment).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse PreActBottleneck (and CMUDeepLens, for the param-ratio check) from the
# L18 port, loaded by path because the filename starts with a digit.
_spec = importlib.util.spec_from_file_location(
    "lanusse_resnet", str(Path(__file__).resolve().parent / "01_lanusse_resnet.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PreActBottleneck = _mod.PreActBottleneck
CMUDeepLens = _mod.CMUDeepLens
count_params = _mod.count_params


class Shield(nn.Module):
    """Pre-activated 1×1 'shield' (InceptionNet-style dimensionality reduction).

    BN → ELU → Conv1x1(in→out). No spatial change; only channel squeeze.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.elu(self.bn(x)))


class ShieldedDeepLens(nn.Module):
    """Huang+2021 shielded ResNet for galaxy-galaxy strong-lens classification.

    Args:
        in_channels: image bands (grz = 3).
        final_out:   output channels of the final 3-block stage (paper sweeps
                     512→256→128→64→32→16; 32 is their best, the default).
        stage_out:   block output width of stages 1-4 (capped low — this is the
                     dimensionality compression the shields enable).
        stage_mid:   bottleneck width of stages 1-4.
        shield_ch:   channels each shield squeezes to between stages.
    """

    STAGE_OUT = 32
    STAGE_MID = 16
    SHIELD_CH = 16

    def __init__(self, in_channels: int = 3, final_out: int = 32,
                 stage_out: int | None = None, stage_mid: int | None = None,
                 shield_ch: int | None = None):
        super().__init__()
        stage_out = stage_out or self.STAGE_OUT
        stage_mid = stage_mid or self.STAGE_MID
        shield_ch = shield_ch or self.SHIELD_CH
        final_mid = max(stage_mid, final_out // 2)

        # (mid, out, n_blocks, shield_out|None, downsample_first)
        stages = [
            (stage_mid, stage_out, 3, shield_ch, False),  # stage 1 (no /2)
            (stage_mid, stage_out, 3, shield_ch, True),   # stage 2 (/2)
            (stage_mid, stage_out, 3, shield_ch, True),   # stage 3 (/2)
            (stage_mid, stage_out, 3, shield_ch, True),   # stage 4 (/2)
            (final_mid, final_out, 3, None,      True),   # stage 5 (/2), no shield
        ]

        # Stem: 7×7 conv → BN, 32 channels (unchanged from L18).
        stem_ch = stages[0][1]
        self.stem_conv = nn.Conv2d(in_channels, stem_ch, kernel_size=7, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(stem_ch)

        body = []
        prev = stem_ch
        for (mid, out, n_blocks, shield_out, downsample_first) in stages:
            for b in range(n_blocks):
                if b == 0:
                    body.append(PreActBottleneck(prev, mid, out, downsample=downsample_first))
                else:
                    body.append(PreActBottleneck(out, mid, out, downsample=False))
            if shield_out is not None:
                body.append(Shield(out, shield_out))
                prev = shield_out
            else:
                prev = out
        self.body = nn.Sequential(*body)

        # Head
        self.bn_final = nn.BatchNorm2d(final_out)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(x)
        x = F.elu(self.stem_bn(x))
        x = self.body(x)
        x = F.elu(self.bn_final(x))
        x = self.pool(x).flatten(1)
        logit = self.fc(x).squeeze(-1)
        return logit  # callers apply sigmoid / use BCEWithLogitsLoss


def smoke_test() -> None:
    """Build the model, verify the reproduction targets, sweep final_out."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    l18 = CMUDeepLens(in_channels=3)
    n_l18 = count_params(l18)
    print(f"[smoke] L18 baseline params:     {n_l18:,}")

    model = ShieldedDeepLens(in_channels=3, final_out=32).to(device)
    n_shielded = count_params(model)
    n_shields = sum(isinstance(m, Shield) for m in model.modules())
    n_blocks = sum(isinstance(m, PreActBottleneck) for m in model.modules())
    ratio = n_l18 / n_shielded
    print(f"[smoke] shielded (final_out=32): {n_shielded:,}")
    print(f"[smoke] shields={n_shields}  bottleneck_blocks={n_blocks}  "
          f"reduction={ratio:.1f}x")

    # Verifiable reproduction targets (paper §3.3).
    assert 55_000 <= n_shielded <= 65_000, f"param count {n_shielded} not ~60K"
    assert n_shields == 4, f"expected 4 shields, got {n_shields}"
    assert n_blocks == 15, f"expected 15 bottleneck blocks, got {n_blocks}"
    assert ratio >= 50.0, f"reduction {ratio:.1f}x < 50x"

    # Forward + backward at the Huang 101×101 input size.
    x = torch.randn(4, 3, 101, 101, device=device)
    y_true = torch.randint(0, 2, (4,), device=device).float()
    logit = model(x)
    prob = torch.sigmoid(logit)
    loss = F.binary_cross_entropy_with_logits(logit, y_true)
    loss.backward()
    grads_finite = all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad)
    assert logit.shape == (4,), f"bad logit shape: {logit.shape}"
    assert (0 <= prob).all() and (prob <= 1).all()
    assert grads_finite
    print(f"[smoke] forward/backward OK: logit={tuple(logit.shape)} "
          f"prob∈[{prob.min().item():.3f},{prob.max().item():.3f}] "
          f"loss={loss.item():.4f} grads_finite={grads_finite}")

    # Channel-sweep param ladder (reproduces paper's 512→…→16 experiment).
    print("[smoke] final_out channel sweep (paper Fig.: 32 is best):")
    for fo in (512, 256, 128, 64, 32, 16):
        m = ShieldedDeepLens(in_channels=3, final_out=fo)
        print(f"           final_out={fo:>3d}  params={count_params(m):>9,d}")

    print("[smoke] OK")


if __name__ == "__main__":
    smoke_test()
