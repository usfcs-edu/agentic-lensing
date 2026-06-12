#!/usr/bin/env python3
"""141_pilot_escnn_c4.py — Phase 140: the CHEAP gate for the expensive
D4-equivariant member (runs LOCALLY, one GPU; pin with CUDA_VISIBLE_DEVICES).

Question: does built-in rotation equivariance (weight-tied escnn convs, a
genuinely different inductive bias from the v1 roster's augmentation-trained
nets) buy matched-FPR recovery at fixed capacity? Phase 7 (80_equivariance)
showed TEST-TIME D4 pooling helps slightly; this pilot tests TRAINED-IN C4
equivariance against a parameter-matched ordinary-CNN twin before paying for
the full D4 member on Perlmutter (142).

DEPENDENCY: escnn is NOT in the claudenet venv. If missing this script prints
the exact install command and exits 2 (the orchestrator installs):
    /home2/benson/.venvs/claudenet/bin/pip install escnn

Models (~1-2M free params each, the shielded-ResNet scale x ~6, small enough
to train in minutes on the 25% subsample):
  equivariant  EquivLens('c4'): 5 escnn R2Conv blocks (regular reprs of C4,
        3x3 stride-2, InnerBatchNorm+ReLU), activation widths 96-192-384-512-512,
        GroupPooling -> global avg pool -> Linear(128,64)-ReLU-Linear(64,1)
        single logit (the shielded/l18 output convention, sigmoid scoring).
  twin         the SAME template as plain Conv2d/BatchNorm2d/ReLU with widths
        HALVED (48-96-192-256-256): a C4 R2Conv between regular fields of
        activation width w has w_in*w_out*k^2/4 free params (4-fold weight
        tying), a plain conv w_in*w_out*k^2 — halving both widths equalises
        the free-parameter counts, so the comparison isolates the inductive
        bias, not capacity. Both counts are printed and compared.

Training: BOTH models use _train.train_supervised directly — it CAN consume
the escnn module because EquivLens is a plain nn.Module whose forward wraps
the batch into a GeometricTensor and unwraps at the end (arch='shielded' ->
BCEWithLogits + sigmoid scoring, the v1 single-logit recipe: Adam lr 1e-3,
StepLR decay_ep=max(8,epochs//3), best-val-AUC checkpoint, identical
LensDataset grz pipeline). Data = a label-stratified 25% subsample (seed
C.SEED=2026) of the staged TRAIN split + the full staged val split; the SAME
rows and the SAME aug_seed for both arms (no per-arm bootstrap — the pilot
isolates architecture, and the PU guard is skipped because both arms see
identical rows). 15 epochs.

GATE (written to data/v2/escnn_pilot.json with all numbers): storfer/inchausti
recovery@1%FPR with thresholds on the OLD v1 staged test negatives scored by
each pilot model itself (_ensemble.recovery_at_fpr — the exact 28_eval
arithmetic);
    equivariant storfer@1% >= twin storfer@1% + 0.02  ->  'RUN-D4'
    else                                              ->  'SKIP-D4'

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      /home2/benson/.venvs/claudenet/bin/python 141_pilot_escnn_c4.py
    # CPU-only construction check (needs escnn): 141_pilot_escnn_c4.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import _clib as C

try:
    import escnn
    from escnn import gspaces
    from escnn import nn as enn
    _HAVE_ESCNN = True
except Exception:  # noqa: BLE001 — any import failure means "not installed"
    _HAVE_ESCNN = False

V2 = C.DATA / "v2"
INSTALL_CMD = "/home2/benson/.venvs/claudenet/bin/pip install escnn"
WIDTHS_C4 = (96, 192, 384, 512, 512)          # activation channels per block
HEAD_HIDDEN = 64
GATE_MARGIN = 0.02


class EquivLens(nn.Module):
    """Group-equivariant conv stack (escnn R2Conv over regular reprs) ->
    GroupPooling -> global avg pool -> small MLP -> single logit (B,).

    Plain nn.Module from the outside: forward wraps the (B,3,H,W) batch into a
    GeometricTensor and unwraps after group pooling, so _train.train_supervised
    and _scorelib.score_paths consume it unchanged with arch='shielded'.
    group='c4' (rotations) for the pilot; group='d4' (rotations+flips, 142).
    widths are ACTIVATION channel counts and must divide the group order."""

    def __init__(self, group: str, widths, in_channels: int = 3,
                 head_hidden: int = HEAD_HIDDEN):
        super().__init__()
        if not _HAVE_ESCNN:
            raise ImportError(f"escnn is required — {INSTALL_CMD}")
        if group == "c4":
            gs = gspaces.rot2dOnR2(N=4)
        elif group == "d4":
            gs = gspaces.flipRot2dOnR2(N=4)
        else:
            raise ValueError(f"unknown group {group!r}")
        self.group, self.widths = group, tuple(int(w) for w in widths)
        order = gs.fibergroup.order()
        self.in_type = enn.FieldType(gs, in_channels * [gs.trivial_repr])
        layers, prev = [], self.in_type
        for w in self.widths:
            assert w % order == 0, f"width {w} not divisible by group order {order}"
            ft = enn.FieldType(gs, (w // order) * [gs.regular_repr])
            layers += [enn.R2Conv(prev, ft, kernel_size=3, stride=2, padding=1),
                       enn.InnerBatchNorm(ft), enn.ReLU(ft, inplace=True)]
            prev = ft
        self.body = enn.SequentialModule(*layers)
        self.gpool = enn.GroupPooling(prev)
        n_feat = self.gpool.out_type.size            # channels after group pooling
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(n_feat, head_hidden),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(head_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = enn.GeometricTensor(x, self.in_type)
        t = self.gpool(self.body(t))
        return self.head(t.tensor).squeeze(-1)       # (B,) single logit


class TwinCNN(nn.Module):
    """The ordinary-CNN control: identical template (3x3 stride-2 Conv2d ->
    BatchNorm2d -> ReLU per block, same depth/kernel/stride/head) at the
    param-matching widths (see module docstring). Single logit (B,)."""

    def __init__(self, widths, in_channels: int = 3, head_hidden: int = HEAD_HIDDEN):
        super().__init__()
        self.widths = tuple(int(w) for w in widths)
        layers, prev = [], in_channels
        for w in self.widths:
            layers += [nn.Conv2d(prev, w, 3, stride=2, padding=1),
                       nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            prev = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(prev, head_hidden),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(head_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_pilot_table(frac: float) -> pd.DataFrame:
    """Label-stratified `frac` subsample (seed C.SEED) of the staged TRAIN split
    + the full staged val split; fits_dir re-pointed to the local layout
    (19_build_member_subsets.remap, imported not copied)."""
    M19 = C._load("cn_141_m19", C.ROOT / "19_build_member_subsets.py")
    split = M19.remap(pd.read_parquet(C.DATA / "training_split_staged.parquet"))
    tr = split[split.split == "train"]
    rng = np.random.default_rng(C.SEED)
    parts = []
    for lab in (0, 1):
        sub = tr[tr.label == lab]
        n = int(round(frac * len(sub)))
        parts.append(sub.iloc[rng.choice(len(sub), size=n, replace=False)])
    tr_sub = pd.concat(parts).assign(split="train")
    val = split[split.split == "val"]
    df = pd.concat([tr_sub, val], ignore_index=True)
    print(f"[141] pilot table: train={len(tr_sub)} "
          f"({int(tr_sub.label.sum())} pos, {frac:.0%} stratified, seed {C.SEED}) "
          f"+ val={len(val)}")
    return df[["row_id", "label", "RA", "DEC", "fits_dir", "split"]]


def eval_recovery(model, mean, std, device, batch: int) -> dict:
    """storfer/inchausti recovery@1%FPR, thresholds on the v1 staged test
    negatives scored by THIS model (the 28_eval_flagship arithmetic)."""
    import _ensemble as E
    import _train as T
    ev = {}
    for sp in ("testneg", "storfer", "inchausti"):
        d = pd.read_parquet(C.DATA / f"eval_{sp}.parquet").copy()
        if "cutouts" not in str(d.fits_dir.iloc[0]):     # 80_equivariance guard
            d["fits_dir"] = d["fits_dir"].apply(
                lambda p: str(C.DATA / Path(str(p)).name))
        d["p"] = T.score_df(model, "shielded", d, mean, std, device, batch=batch)
        ev[sp] = d["p"].to_numpy()
        print(f"[score] {sp:10s} n={len(d)} mean_p={np.nanmean(ev[sp]):.3f}")
    out = {}
    for cat in ("storfer", "inchausti"):
        out[f"{cat}_1"] = E.recovery_at_fpr(ev["testneg"], ev[cat],
                                            fprs=(0.01,))[0.01]["recovery"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frac", type=float, default=0.25,
                    help="stratified train-split subsample fraction (default 0.25)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--margin", type=float, default=GATE_MARGIN,
                    help="gate margin on storfer@1%% (default 0.02)")
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS_C4),
                    help="equivariant ACTIVATION widths; twin uses half each")
    ap.add_argument("--out", default=str(V2 / "escnn_pilot.json"))
    ap.add_argument("--force", action="store_true",
                    help="downgrade the capacity-match assert (param ratio in "
                         "[0.8,1.25]) to a warning")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU-only: build both models + 4x3x101x101 forward, exit")
    args = ap.parse_args()

    if not _HAVE_ESCNN:
        print(f"[141] FATAL: escnn is not installed in this venv.\n"
              f"[141] install it with:  {INSTALL_CMD}\n"
              f"[141] then re-run this script (exit 2 = missing dependency)")
        return 2

    widths = tuple(int(w) for w in args.widths.split(","))
    equiv = EquivLens("c4", widths)
    pe = n_params(equiv)
    # self-calibrating twin: search the width scale whose plain-CNN param count
    # matches the equivariant net (the w//2 heuristic missed by ~1.5x)
    best = None
    for s in np.linspace(0.30, 0.60, 31):
        tw = tuple(max(8, int(round(w * s))) for w in widths)
        d = abs(n_params(TwinCNN(tw)) - pe) / pe
        if best is None or d < best[0]:
            best = (d, tw)
    twin_widths = best[1]
    twin = TwinCNN(twin_widths)
    pt = n_params(twin)
    ratio = pe / pt
    print(f"[141] params: equivariant(C4 {widths})={pe:,} "
          f"twin({twin_widths})={pt:,} ratio={ratio:.2f}")
    if not 0.8 <= ratio <= 1.25:
        msg = (f"param ratio {ratio:.2f} outside [0.8,1.25] — capacity matching "
               f"is broken and the gate must isolate equivariance, not capacity "
               f"(adjust --widths)")
        assert args.force, f"{msg}; --force downgrades this to a warning"
        print(f"[141] WARNING (--force): {msg}")

    if args.dry_run:
        x = torch.randn(4, 3, 101, 101)
        with torch.no_grad():
            oe, ot = equiv.eval()(x), twin.eval()(x)
        assert oe.shape == ot.shape == (4,), (oe.shape, ot.shape)
        print(f"[141:dry] equiv logits={np.round(oe.numpy(), 3).tolist()} "
              f"twin logits={np.round(ot.numpy(), 3).tolist()} -> OK")
        return 0

    import _train as T
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = build_pilot_table(args.frac)
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / "ckpt").mkdir(parents=True, exist_ok=True)

    results = {}
    for tag, model in (("equivariant", equiv), ("twin", twin)):
        print(f"[train] pilot {tag}: epochs={args.epochs} batch={args.batch} "
              f"lr=1e-3 aug_seed={C.SEED} device={device}")
        t0 = time.time()
        model, val_auc, mean, std = T.train_supervised(
            model, "shielded", df, device, epochs=args.epochs, batch=args.batch,
            lr=1e-3, decay_ep=max(8, args.epochs // 3), accum=1, aug_seed=C.SEED)
        mins = (time.time() - t0) / 60
        print(f"[train] {tag} best_val_auc={val_auc:.4f} ({mins:.1f}m)")
        rec = eval_recovery(model, mean, std, device, args.batch)
        results[tag] = {"val_auc": float(val_auc), "minutes": round(mins, 1),
                        "params": n_params(model), **rec}
        ck = V2 / "ckpt" / f"escnn_pilot_{'c4' if tag == 'equivariant' else 'twin'}.pt"
        torch.save({"state_dict": model.state_dict(), "arch": f"escnn_pilot_{tag}",
                    "score_arch": "shielded", "widths": list(model.widths),
                    "mean": mean.tolist(), "std": std.tolist(),
                    "val_auc": float(val_auc)}, ck)
        print(f"[141] {tag}: storfer@1%={rec['storfer_1']:.3f} "
              f"inchausti@1%={rec['inchausti_1']:.3f} (ckpt {ck.name})")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    delta = results["equivariant"]["storfer_1"] - results["twin"]["storfer_1"]
    verdict = "RUN-D4" if delta >= args.margin else "SKIP-D4"
    out = {"config": {"frac": args.frac, "epochs": args.epochs,
                      "batch": args.batch, "seed": C.SEED,
                      "widths_equiv": list(widths), "widths_twin": list(twin_widths),
                      "escnn_version": getattr(escnn, "__version__", "?")},
           "equivariant": results["equivariant"], "twin": results["twin"],
           "delta_storfer_1": float(delta), "gate_margin": args.margin,
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[141] storfer@1%: equivariant={results['equivariant']['storfer_1']:.3f} "
          f"twin={results['twin']['storfer_1']:.3f} delta={delta:+.3f} "
          f"(margin {args.margin}) -> {verdict}")
    print(f"[141] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
