#!/usr/bin/env python3
"""
02_efficientnet.py

The EfficientNetV2 base model of Inchausti+2025 (arXiv:2508.20087, §3.2.2) — the
second architecture in that paper's two-model ensemble (the first being the
shielded ResNet of `01b_shielded_resnet.py`, reused from the Huang+2021 repro).

What the paper says (§3.2.2, verbatim where quoted):
  "For the first time with respect to Papers I through III, we integrate the
   EfficientNetV2 ... Unlike our ResNet model, the EfficientNet was pre-trained.
   In this work, we fine-tune the EfficientNet model using the same cross-entropy
   loss function as for the ResNet ... The EfficientNet model contains a total of
   20,542,883 trainable parameters." (val AUC 0.9987 at epoch 50.)

The paper does NOT specify the EfficientNetV2 variant, the input adaptation for
the 3-band grz 101x101 cutouts, the pretraining corpus, or the fine-tuning head.
Reconstruction (documented in the Phase-5 report):
  * Variant = EfficientNetV2-S (`tf_efficientnetv2_s` in timm 1.0.x). Its
    ImageNet-pretrained backbone (num_classes=0) is 20,177,488 params in the
    lensfinder venv; EfficientNetV2-S is the only timm V2 variant near 20.5M.
  * A dense fine-tuning head `Linear(1280 -> HEAD_DIM) -> ReLU -> Linear(-> 2)`
    with HEAD_DIM=285 lands the total within ~260 params (0.0013%) of the
    paper's 20,542,883. The head width is unpublished, so this is the closest
    principled fit; HEAD_DIM is exposed as a knob.
  * Input: the grz cube feeds the 3 input channels (in_chans=3) in place of RGB;
    we keep Huang's native 101x101 field of view (EfficientNetV2-S is fully
    convolutional + global-pooled, so it accepts 101x101 without a resize) rather
    than upsampling to the 384px pretrain resolution. Normalisation uses the same
    per-band mean/std + clamp +/-250 as the shielded/L18 nets (NOT ImageNet
    stats) so all three models share an identical Dataset — fine-tuning adapts
    the pretrained stem to the astronomical-flux domain.

Output convention: forward(x) -> (B, num_classes) logits. With num_classes=2
(the paper's cross-entropy framing) the lens-class probability is
softmax(logits)[:, 1]. The training script (06) uses CrossEntropyLoss.

Smoke test (run as a script) ASSERTS: param count within tolerance of
20,542,883 and a finite 101x101x3 forward+backward. It builds with
pretrained=False so the smoke test needs no network (param count is independent
of the pretrained weight values).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

try:
    import timm
    _HAVE_TIMM = True
except Exception:  # pragma: no cover
    _HAVE_TIMM = False

VARIANT = "tf_efficientnetv2_s"
HEAD_DIM = 285
PAPER_PARAMS = 20_542_883


class EfficientNetV2Lens(nn.Module):
    """EfficientNetV2-S adapted to 101x101x3 grz binary lens classification."""

    def __init__(self, variant: str = VARIANT, pretrained: bool = True,
                 in_channels: int = 3, num_classes: int = 2,
                 head_dim: int = HEAD_DIM):
        super().__init__()
        if not _HAVE_TIMM:
            raise ImportError("timm is required for EfficientNetV2Lens "
                              "(pip install timm in the lensfinder venv)")
        self.variant = variant
        self.num_classes = num_classes
        self.head_dim = head_dim
        self.pretrained_loaded = False
        try:
            self.backbone = timm.create_model(
                variant, pretrained=pretrained, num_classes=0, in_chans=in_channels)
            self.pretrained_loaded = bool(pretrained)
        except Exception as e:  # offline / weight-download failure
            warnings.warn(f"[effnet] pretrained load failed ({e}); "
                          f"falling back to RANDOM init", RuntimeWarning)
            self.backbone = timm.create_model(
                variant, pretrained=False, num_classes=0, in_chans=in_channels)
            self.pretrained_loaded = False
        feat = self.backbone.num_features  # 1280 for EfficientNetV2-S
        self.head = nn.Sequential(
            nn.Linear(feat, head_dim), nn.ReLU(inplace=True),
            nn.Linear(head_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def smoke_test() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}  variant={VARIANT}  head_dim={HEAD_DIM}")
    # pretrained=False -> no network needed; param count is identical either way.
    model = EfficientNetV2Lens(pretrained=False).to(device)
    n = count_params(model)
    gap = n - PAPER_PARAMS
    print(f"[smoke] EfficientNetV2Lens params: {n:,}  "
          f"(paper {PAPER_PARAMS:,}; gap {gap:+,} = {100 * gap / PAPER_PARAMS:+.4f}%)")
    assert 20_400_000 <= n <= 20_650_000, f"param count {n:,} not ~20.54M"
    assert model.num_classes == 2

    x = torch.randn(4, 3, 101, 101, device=device)
    y = torch.randint(0, 2, (4,), device=device)
    logits = model(x)
    assert logits.shape == (4, 2), f"bad logits shape {logits.shape}"
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    grads_finite = all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad)
    prob = torch.softmax(logits, dim=1)[:, 1]
    assert (0 <= prob).all() and (prob <= 1).all()
    assert grads_finite
    print(f"[smoke] forward/backward OK: logits={tuple(logits.shape)} "
          f"prob in [{prob.min().item():.3f},{prob.max().item():.3f}] "
          f"loss={loss.item():.4f} grads_finite={grads_finite}")
    print("[smoke] OK")


if __name__ == "__main__":
    smoke_test()
