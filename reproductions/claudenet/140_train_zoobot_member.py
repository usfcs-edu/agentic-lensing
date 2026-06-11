#!/usr/bin/env python3
"""140_train_zoobot_member.py — Phase 140: train the 7th decorrelated ensemble
member, a ConvNeXT-Nano initialised from Zoobot's GalaxyZoo-pretrained encoder
(runs LOCALLY, one GPU; pin with CUDA_VISIBLE_DEVICES from the caller).

Why Zoobot: every v1 member starts from ImageNet weights or random init; a
galaxy-morphology pretrain is a genuinely different prior, so its ERRORS should
decorrelate from the roster (the active ensemble ingredient, cf. 27/13).

Backbone: timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
pretrained=True, num_classes=0). Hub id VERIFIED 2026-06-11 via
huggingface_hub.model_info (metadata only, no weight download): repo exists,
sha 494bae25, tags base_model:timm/convnext_nano.in12k, weights file
pytorch_model.bin. --hub-id overrides; --pretrained {zoobot,imagenet,none} is
a LOUD fallback chain (zoobot -> timm-default ImageNet convnext_nano -> random
init) so an offline/hub failure degrades visibly, never silently.
Head: Linear(feat_dim, 2) on the pooled num_features output — the
02_efficientnet pattern (backbone(num_classes=0) + linear head, softmax[:,1]
scoring) minus the hidden layer (Zoobot's encoder is already galaxy-tuned).

Recipe fidelity (EXACT v1 member machinery):
  * training table = 19_build_member_subsets' recipe via direct import of its
    pu_drop: ALL staged-train positives + a bootstrap resample of the
    PU-guarded staged-train negatives with boot_seed=7 + the staged val split
    (the next free seeds after the v1 roster's 1-6: boot_seed=7, aug_seed=707);
  * _train.train_supervised with the v1 'efficientnet' (pretrained-backbone)
    recipe: epochs=20_train_member.EPOCHS['efficientnet']=25, batch 128,
    accum 2, Adam lr 1e-3, StepLR decay_ep=max(8, epochs//3), CrossEntropy,
    best-val-AUC checkpoint;
  * preprocessing = the identical _trainlib.LensDataset grz pipeline (per-band
    flux mean/std + clamp +/-250, rot/flip/zoom aug). Deliberately NO ImageNet
    normalisation — v1 keeps astronomical flux normalisation and lets the
    pretrained stem adapt (see the 02_efficientnet docstring).

Scoring convention: score_arch='efficientnet' — VERIFIED: _trainlib.model_prob
applies softmax(out, dim=1)[:, 1] for arch=='efficientnet', which matches the
(B,2)-logit head here. The checkpoint uses arch='timm' (a NEW generic branch
added to 112_score_pool.load_member_checkpoint) with variant=<plain timm
architecture name resolved from the backbone's pretrained_cfg, so 112 can
rebuild it OFFLINE with pretrained=False>; the hub id is kept as provenance.

Writes (an --epochs override appends '_smoke' to ALL artifact names, the 121
convention, so smoke runs can never overwrite production artifacts):
  data/v2/member_zoobot_N_train.parquet      (audit copy of the training table)
  data/v2/ckpt/member_zoobot_N.pt            (112-loadable: state_dict, arch=
        'timm', variant, score_arch='efficientnet', mean, std, val_auc, + hub_id/
        pretrained_source/boot_seed/aug_seed provenance)
  data/v2/scores_member_zoobot_N.parquet     [split,row_id,label,p,pc]
        (v1 schema; val/testneg/storfer/inchausti; pc = isotonic fit on val,
         exactly as 121/25 do)

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      /home2/benson/.venvs/claudenet/bin/python 140_train_zoobot_member.py
    # CPU-only model-construction check (no GPU, no network):
    CUDA_VISIBLE_DEVICES= /home2/benson/.venvs/claudenet/bin/python \
      140_train_zoobot_member.py --dry-run --pretrained none
    # hub-id metadata check only (CPU, no weight download):
    140_train_zoobot_member.py --check-hub
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import _clib as C

try:
    import timm
    _HAVE_TIMM = True
except Exception:  # pragma: no cover
    _HAVE_TIMM = False

V2 = C.DATA / "v2"
NAME = "zoobot_N"
HUB_ID = "hf_hub:mwalmsley/zoobot-encoder-convnext_nano"
FALLBACK_VARIANT = "convnext_nano"      # timm-default ImageNet weights
BOOT_SEED, AUG_SEED = 7, 707            # next free seeds after the v1 roster (1-6)
BATCH, ACCUM = 128, 2                   # the v1 efficientnet-member setting


class TimmLens(nn.Module):
    """Generic timm backbone (num_classes=0 -> pooled features) + a single
    Linear(feat_dim, num_classes) head; forward -> (B, num_classes) logits,
    lens prob = softmax[:, 1] (the 02_efficientnet output convention).
    Raises on backbone-construction failure — the LOUD fallback chain in
    build_model() owns the degradation policy, not this class."""

    def __init__(self, variant: str, pretrained: bool, in_channels: int = 3,
                 num_classes: int = 2):
        super().__init__()
        if not _HAVE_TIMM:
            raise ImportError("timm is required for TimmLens "
                              "(the claudenet venv has it)")
        self.variant = variant
        self.num_classes = num_classes
        self.backbone = timm.create_model(variant, pretrained=pretrained,
                                          num_classes=0, in_chans=in_channels)
        # plain architecture name (e.g. 'convnext_nano') so 112 can rebuild the
        # module OFFLINE with pretrained=False even when built from an hf_hub id
        cfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        self.arch_name = cfg.get("architecture") or variant
        self.feat_dim = int(self.backbone.num_features)
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def verify_hub(hub_id: str) -> bool:
    """CPU-only metadata existence check (huggingface_hub.model_info; NO weight
    download). Failure is non-fatal: the fallback chain handles it loudly."""
    repo = hub_id.removeprefix("hf_hub:")
    try:
        from huggingface_hub import model_info
        mi = model_info(repo)
        print(f"[140] hub id VERIFIED: {repo} (sha {str(mi.sha)[:8]}; "
              f"metadata only, no weights downloaded)")
        return True
    except Exception as e:
        print(f"[140] WARNING: could not verify hub id {repo} "
              f"({type(e).__name__}: {e}) — hub unreachable or id wrong; the "
              f"--pretrained fallback chain below will degrade LOUDLY if the "
              f"weight download also fails")
        return False


def build_model(hub_id: str, pretrained: str) -> tuple[TimmLens, str]:
    """LOUD fallback chain: zoobot -> imagenet -> none. Returns (model, source)."""
    order = {"zoobot": ("zoobot", "imagenet", "none"),
             "imagenet": ("imagenet", "none"),
             "none": ("none",)}[pretrained]
    last = None
    for src in order:
        variant = hub_id if src == "zoobot" else FALLBACK_VARIANT
        if src == "zoobot":
            verify_hub(hub_id)
        try:
            model = TimmLens(variant=variant, pretrained=src != "none")
            n = sum(p.numel() for p in model.parameters())
            print(f"[140] built TimmLens source={src} variant={variant} "
                  f"(arch_name={model.arch_name}) feat_dim={model.feat_dim} "
                  f"params={n:,}")
            if src != pretrained:
                print(f"[140] *** NOTE: requested --pretrained {pretrained} but "
                      f"USING '{src}' weights (see fallback prints above) ***")
            return model, src
        except Exception as e:
            last = e
            print(f"[140] *** FALLBACK: pretrained source '{src}' FAILED "
                  f"({type(e).__name__}: {e}) -> trying next in chain ***")
    raise SystemExit(f"[140] FATAL: every pretrained source failed: {last}")


def build_member_table(boot_seed: int, fits_root=None) -> pd.DataFrame:
    """The 19_build_member_subsets recipe for ONE new member: all staged-train
    positives + a same-size bootstrap resample of the PU-guarded staged-train
    negatives (seed C.SEED+boot_seed) + the shared staged val split. pu_drop is
    imported from 19 (no copy). fits_root re-points every fits_dir basename
    (default C.DATA, the v1 local layout; 142 passes $SCRATCH on Perlmutter)."""
    M19 = C._load("cn_140_m19", C.ROOT / "19_build_member_subsets.py")
    root = Path(fits_root) if fits_root else C.DATA
    split = pd.read_parquet(C.DATA / "training_split_staged.parquet").copy()
    split["fits_dir"] = split["fits_dir"].apply(lambda p: str(root / Path(str(p)).name))
    tr = split[split.split == "train"]
    tr_pos = tr[tr.label == 1].reset_index(drop=True)
    tr_neg = tr[tr.label == 0].reset_index(drop=True)
    tr_neg, ndrop = M19.pu_drop(tr_neg)
    print(f"[140] pu guard dropped {ndrop} train negatives -> {len(tr_neg)} clean")
    val = split[split.split == "val"]
    rng = np.random.default_rng(C.SEED + boot_seed)
    boot = tr_neg.iloc[rng.integers(0, len(tr_neg), size=len(tr_neg))]
    dfm = pd.concat([tr_pos.assign(split="train"), boot.assign(split="train"),
                     val.assign(split="val")], ignore_index=True)
    print(f"[140] member table: train={len(tr_pos)}pos+{len(boot)}neg "
          f"val={len(val)} (boot_seed={boot_seed})")
    return dfm[["row_id", "label", "RA", "DEC", "fits_dir", "split"]]


def dry_run(args) -> int:
    """CPU-only construction + forward check (use with --pretrained none for a
    fully offline run; never touches a GPU)."""
    import _trainlib as TL
    model, src = build_model(args.hub_id, args.pretrained)
    model.eval()
    x = torch.randn(4, 3, 101, 101)
    with torch.no_grad():
        logits = model(x)
        prob = TL.model_prob(model, x, "efficientnet")   # the 112 scoring path
    assert logits.shape == (4, 2), f"bad logits shape {tuple(logits.shape)}"
    assert prob.shape == (4,) and bool((prob >= 0).all() and (prob <= 1).all())
    print(f"[140:dry] source={src} logits={tuple(logits.shape)} "
          f"model_prob(efficientnet)={np.round(prob.numpy(), 4).tolist()} -> OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hub-id", default=HUB_ID,
                    help=f"timm hf_hub id for the Zoobot encoder (default {HUB_ID})")
    ap.add_argument("--pretrained", choices=("zoobot", "imagenet", "none"),
                    default="zoobot",
                    help="start of the LOUD fallback chain zoobot->imagenet->none")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the v1 epoch count (SMOKE TESTS ONLY; "
                         "appends '_smoke' to every artifact name)")
    ap.add_argument("--build-only", action="store_true",
                    help="build + write the member training table, no training (no GPU)")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU-only model construction + 4x3x101x101 forward, then exit")
    ap.add_argument("--check-hub", action="store_true",
                    help="hub-id metadata existence check only (CPU, no download)")
    args = ap.parse_args()

    if args.check_hub:
        return 0 if verify_hub(args.hub_id) else 1
    if args.dry_run:
        return dry_run(args)

    M20 = C._load("cn_140_m20", C.ROOT / "20_train_member.py")  # EPOCHS recipe
    epochs = args.epochs if args.epochs else M20.EPOCHS["efficientnet"]
    art = NAME + ("_smoke" if args.epochs is not None else "")
    if art != NAME:
        print(f"[140] --epochs {args.epochs} override -> SMOKE artifact names "
              f"(member_{art}_train.parquet / member_{art}.pt / "
              f"scores_member_{art}.parquet)")

    V2.mkdir(parents=True, exist_ok=True)
    dfm = build_member_table(BOOT_SEED)
    table_f = V2 / f"member_{art}_train.parquet"
    dfm.to_parquet(table_f, index=False)
    print(f"[140] wrote {table_f}")
    if args.build_only:
        print("[140] --build-only: stopping before training")
        return 0

    import _train as T
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src = build_model(args.hub_id, args.pretrained)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {art} arch=timm({model.arch_name}) params={n_params:,} "
          f"epochs={epochs} batch={BATCH} accum={ACCUM} aug_seed={AUG_SEED} "
          f"n_train={(dfm.split == 'train').sum()} device={device}")
    t0 = time.time()
    # arch='efficientnet' selects the v1 pretrained-backbone recipe inside
    # train_supervised: CrossEntropy loss + softmax[:,1] val scoring.
    model, val_auc, mean, std = T.train_supervised(
        model, "efficientnet", dfm, device, epochs=epochs, batch=BATCH, lr=1e-3,
        decay_ep=max(8, epochs // 3), accum=ACCUM, aug_seed=AUG_SEED)
    print(f"[train] {art} best_val_auc={val_auc:.4f} ({(time.time() - t0) / 60:.1f}m)")

    ckpt_dir = V2 / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": "timm",
                "score_arch": "efficientnet",
                "variant": model.arch_name,          # offline-rebuildable timm name
                "hub_id": args.hub_id, "pretrained_source": src,
                "mean": mean.tolist(), "std": std.tolist(), "val_auc": val_auc,
                "boot_seed": BOOT_SEED, "aug_seed": AUG_SEED},
               ckpt_dir / f"member_{art}.pt")
    print(f"[train] saved {ckpt_dir / f'member_{art}.pt'}")

    # score the shared v1 eval manifests + isotonic pc (121_retrain verbatim)
    import _ensemble as E
    rows = []
    for sp in ("val", "testneg", "storfer", "inchausti"):
        d = pd.read_parquet(C.DATA / f"eval_{sp}.parquet").copy()
        d["p"] = T.score_df(model, "efficientnet", d, mean, std, device)
        d["split"] = sp
        rows.append(d[["split", "row_id", "label", "p"]])
        print(f"[score] {sp:10s} n={len(d)} mean_p={d['p'].mean():.3f}")
    sc = pd.concat(rows, ignore_index=True)
    sc["p"] = sc["p"].astype(np.float32)
    val = sc[sc.split == "val"]
    cal = E.make_calibrator("isotonic").fit(val["p"].to_numpy(), val["label"].to_numpy())
    sc["pc"] = cal.transform(sc["p"].to_numpy())
    out_f = V2 / f"scores_member_{art}.parquet"
    sc.to_parquet(out_f, index=False)
    print(f"[140] {art} done -> {out_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
