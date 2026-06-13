#!/usr/bin/env python3
"""142_train_escnn_d4.py — Phase 140: the FULL D4-equivariant ensemble member
(runs on PERLMUTTER, 1x A100 via nersc/shared_gpu.slurm; ONLY if
141_pilot_escnn_c4.py's gate said 'RUN-D4').

DEPENDENCY: escnn is not in the NERSC pytorch module. One-time on Perlmutter
(with `module load pytorch/2.8.0` active, as in nersc/shared_gpu.slurm):
    pip install --user escnn
If missing this script prints that command and exits 2 (orchestrator installs).

Model: EquivLens('d4', ...) imported from 141_pilot_escnn_c4.py (one source of
truth) — 5 escnn R2Conv blocks over regular reprs of D4 (rotations+flips,
8-fold weight tying), activation widths 128-256-512-768-768 (~1.3M free
params), GroupPooling -> global avg pool -> MLP -> single logit; sigmoid
scoring (score_arch='shielded', the v1 single-logit convention).

Recipe fidelity: the member training table is 140_train_zoobot_member.
build_member_table (19's recipe: all staged-train positives + a bootstrap
resample of the PU-guarded staged-train negatives + the staged val split) with
boot_seed=8, aug_seed=808 (the next free seeds after zoobot_N's 7/707), trained
via _train.train_supervised with the v1 SCRATCH-NET recipe (EPOCHS['shielded']
=45 from 20_train_member, batch 128, accum 1, Adam lr 1e-3, StepLR
decay_ep=max(8,epochs//3), best-val-AUC checkpoint, identical LensDataset grz
pipeline). TF32 is forced OFF before any conv (the 112/100 TITAN-parity
contract) for both training and scoring. NOTE: train_supervised has no
mid-epoch resume — if 45 escnn epochs exceed the 4h shared-QOS wall, submit
with `-q regular -t 12:00:00`; --score-only resumes a job that died AFTER the
checkpoint was written.

Checkpoint/scores conventions = 140, with ONE documented exception: arch=
'escnn_d4' is NOT loadable by 112_score_pool (112's generic branches rebuild
timm/torchvision-style modules; escnn modules need the escnn package and the
widths/group config). So this script saves TWO things and exports predictions
itself:
  1. the escnn state_dict + full config (group/widths/seeds/escnn version) in
     data/v2/ckpt_escnn/escnn_D4.pt — its OWN dir, NOT member_-prefixed, so
     112's --extra-ckpt-dir member_*.pt glob can never pick it up — and
  2. the scores directly: the v1 eval splits (val/testneg/storfer/inchausti,
     v1 schema + isotonic pc exactly as 140/121) AND, with --score-pool
     <cutout-root>, the NegEval shards — by REUSING 112_score_pool.run_pass
     (imported, not copied: same memmap shard iteration, same normalise/clamp/
     model_prob math, same resume-safe fingerprinted .partial cache).

Writes (an --epochs override appends '_smoke' to every artifact name, the 121
convention):
  data/v2/member_escnn_D4_train.parquet      (audit copy of the training table)
  data/v2/ckpt_escnn/escnn_D4.pt
  data/v2/scores_member_escnn_D4.parquet     [split,row_id,label,p,pc]
  <--pool-out>                               [row_id, ok, member_escnn_D4]
                       (default data/v2/scores_pool_escnn_D4.parquet; rsync it
                        back and merge into scores_negeval_pool.parquet for 113)

    # Perlmutter (repo rsync'd -L; data/training_split_staged.parquet, the
    # eval_*.parquet manifests and the FITS cutout dirs staged under
    # $SCRATCH/claudenet/fits/<dirname>):
    sbatch -q regular -t 12:00:00 --export=ALL,CMD='python 142_train_escnn_d4.py \
        --fits-root $SCRATCH/claudenet/fits \
        --score-pool $SCRATCH/claudenet/cutouts/negeval' nersc/shared_gpu.slurm
    # rescore-only from an existing checkpoint (e.g. after a wall-clock death
    # during pool scoring; resumes via 112's .partial cache):
    ... CMD='python 142_train_escnn_d4.py --score-only --fits-root ... --score-pool ...'
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _clib as C

V2 = C.DATA / "v2"
NAME = "escnn_D4"
BOOT_SEED, AUG_SEED = 8, 808            # next free seeds after zoobot_N (7/707)
WIDTHS_D4 = (128, 256, 512, 768, 768)   # ~1.3M free params (8-fold D4 tying)
INSTALL_CMD = "pip install --user escnn   # on Perlmutter, pytorch module loaded"

# one source of truth for the model + the install gate (escnn import is guarded
# inside 141, so loading it never crashes a venv without escnn)
M141 = C._load("cn_142_m141", C.ROOT / "141_pilot_escnn_c4.py")


def repoint(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Re-point every fits_dir basename at `root` (local C.DATA layout or the
    Perlmutter $SCRATCH staging dir)."""
    df = df.copy()
    df["fits_dir"] = df["fits_dir"].apply(lambda p: str(root / Path(str(p)).name))
    return df


def score_eval_splits(model, mean, std, device, art: str, fits_root: Path,
                      batch: int) -> Path:
    """v1 eval manifests -> data/v2/scores_member_<art>.parquet
    [split,row_id,label,p,pc] (the 140/121 block verbatim, fits_dir re-pointed)."""
    import _ensemble as E
    import _train as T
    rows = []
    for sp in ("val", "testneg", "storfer", "inchausti"):
        d = repoint(pd.read_parquet(C.DATA / f"eval_{sp}.parquet"), fits_root)
        d["p"] = T.score_df(model, "shielded", d, mean, std, device, batch=batch)
        d["split"] = sp
        rows.append(d[["split", "row_id", "label", "p"]])
        print(f"[score] {sp:10s} n={len(d)} mean_p={np.nanmean(d['p']):.3f} "
              f"finite={int(np.isfinite(d['p']).sum())}")
    sc = pd.concat(rows, ignore_index=True)
    sc["p"] = sc["p"].astype(np.float32)
    val = sc[sc.split == "val"]
    cal = E.make_calibrator("isotonic").fit(val["p"].to_numpy(), val["label"].to_numpy())
    sc["pc"] = cal.transform(sc["p"].to_numpy())
    out_f = V2 / f"scores_member_{art}.parquet"
    sc.to_parquet(out_f, index=False)
    print(f"[142] eval scores -> {out_f}")
    return out_f


def score_pool(model, mean, std, device, art: str, root: Path, out: Path,
               batch: int) -> None:
    """Score the NegEval shards with 112_score_pool.run_pass (imported: same
    shard iteration, normalise/clamp/model_prob math and resume-safe
    fingerprinted partial cache as every other pool scorer)."""
    M112 = C._load("cn_142_m112", C.ROOT / "112_score_pool.py")
    index = (pd.read_parquet(root / "index.parquet")
             .sort_values(["shard", "idx_in_shard"]).reset_index(drop=True))
    col = f"member_{art}"
    print(f"[142] pool: {len(index):,} rows in {index.shard.nunique()} shards; "
          f"{int(index.ok.sum()):,} ok -> column {col}")
    partial = out.parent / f"{out.name}.partial_{col}.npz"
    probs = M112.run_pass(col, model, "shielded", mean, std, index, root,
                          device, batch, partial)
    df = pd.DataFrame({"row_id": index.row_id, "ok": index.ok, col: probs})
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out)
    partial.unlink(missing_ok=True)
    f = np.isfinite(probs)
    print(f"[142] pool scores -> {out} (n={len(df):,} finite={int(f.sum()):,} "
          f"mean_p={np.nanmean(probs):.4f} p99.9={np.nanquantile(probs, 0.999):.4f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS_D4),
                    help="ACTIVATION widths (each divisible by 8 for D4)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the v1 epoch count (SMOKE TESTS ONLY; "
                         "appends '_smoke' to every artifact name)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--fits-root", default=None,
                    help="dir holding the FITS cutout dirs by basename "
                         "(default the local data/ layout; on Perlmutter e.g. "
                         "$SCRATCH/claudenet/fits)")
    ap.add_argument("--score-pool", default=None,
                    help="111 cutout root (cutouts_<k>.npy + index.parquet) to "
                         "score the NegEval shards after training")
    ap.add_argument("--pool-out", default=None,
                    help="pool scores parquet (default "
                         "data/v2/scores_pool_<art>.parquet)")
    ap.add_argument("--pool-batch", type=int, default=512)
    ap.add_argument("--score-only", action="store_true",
                    help="skip training; load the existing checkpoint and "
                         "(re)score the eval splits and --score-pool shards")
    ap.add_argument("--build-only", action="store_true",
                    help="build + write the member training table, no training")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU-only: build the D4 model + 4x3x101x101 forward, exit")
    args = ap.parse_args()

    if not M141._HAVE_ESCNN:
        print(f"[142] FATAL: escnn is not importable here.\n"
              f"[142] install it with:  {INSTALL_CMD}\n"
              f"[142] (exit 2 = missing dependency)")
        return 2

    widths = tuple(int(w) for w in args.widths.split(","))
    if args.dry_run:
        model = M141.EquivLens("d4", widths).eval()
        print(f"[142:dry] EquivLens(d4, {widths}) params={M141.n_params(model):,}")
        with torch.no_grad():
            o = model(torch.randn(4, 3, 101, 101))
        assert o.shape == (4,), o.shape
        print(f"[142:dry] logits={np.round(o.numpy(), 3).tolist()} -> OK")
        return 0

    # TITAN-parity: TF32 off BEFORE any conv/matmul (112/100 contract)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fits_root = Path(args.fits_root) if args.fits_root else C.DATA
    print(f"[142] device={device} tf32=off seed={C.SEED} fits_root={fits_root}")

    M20 = C._load("cn_142_m20", C.ROOT / "20_train_member.py")    # EPOCHS recipe
    epochs = args.epochs if args.epochs else M20.EPOCHS["shielded"]  # scratch net
    art = NAME + ("_smoke" if args.epochs is not None else "")
    if art != NAME:
        print(f"[142] --epochs {args.epochs} override -> SMOKE artifact names "
              f"(member_{art}_train.parquet / ckpt_escnn/{art}.pt / "
              f"scores_member_{art}.parquet / scores_pool_{art}.parquet)")
    ckpt_f = V2 / "ckpt_escnn" / f"{art}.pt"
    pool_out = Path(args.pool_out) if args.pool_out else V2 / f"scores_pool_{art}.parquet"

    if args.score_only:
        assert ckpt_f.exists(), f"--score-only but {ckpt_f} is missing"
        ck = torch.load(str(ckpt_f), map_location="cpu", weights_only=False)
        model = M141.EquivLens(ck["group"], ck["widths"],
                               head_hidden=ck.get("head_hidden", M141.HEAD_HIDDEN))
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()
        mean = np.array(ck["mean"], dtype=np.float32)
        std = np.array(ck["std"], dtype=np.float32)
        print(f"[142] --score-only: loaded {ckpt_f} "
              f"(val_auc={ck.get('val_auc', float('nan')):.4f})")
    else:
        M140 = C._load("cn_142_m140", C.ROOT / "140_train_zoobot_member.py")
        dfm = M140.build_member_table(BOOT_SEED, fits_root=fits_root)
        V2.mkdir(parents=True, exist_ok=True)
        table_f = V2 / f"member_{art}_train.parquet"
        dfm.to_parquet(table_f, index=False)
        print(f"[142] wrote {table_f}")
        if args.build_only:
            print("[142] --build-only: stopping before training")
            return 0

        import _train as T
        model = M141.EquivLens("d4", widths)
        print(f"[train] {art} arch=escnn_d4 widths={widths} "
              f"params={M141.n_params(model):,} epochs={epochs} "
              f"batch={args.batch} accum={args.accum} aug_seed={AUG_SEED} "
              f"n_train={(dfm.split == 'train').sum()}")
        t0 = time.time()
        # arch='shielded' selects the v1 scratch-net recipe inside
        # train_supervised: BCEWithLogits + sigmoid val scoring (single logit).
        model, val_auc, mean, std = T.train_supervised(
            model, "shielded", dfm, device, epochs=epochs, batch=args.batch,
            lr=1e-3, decay_ep=max(8, epochs // 3), accum=args.accum,
            aug_seed=AUG_SEED)
        print(f"[train] {art} best_val_auc={val_auc:.4f} "
              f"({(time.time() - t0) / 60:.1f}m)")
        ckpt_f.parent.mkdir(parents=True, exist_ok=True)
        import escnn
        torch.save({"state_dict": model.state_dict(), "arch": "escnn_d4",
                    "score_arch": "shielded", "group": "d4",
                    "widths": list(widths), "head_hidden": M141.HEAD_HIDDEN,
                    "mean": mean.tolist(), "std": std.tolist(),
                    "val_auc": float(val_auc),
                    "boot_seed": BOOT_SEED, "aug_seed": AUG_SEED,
                    "escnn_version": getattr(escnn, "__version__", "?"),
                    "note": "NOT 112-loadable (escnn module); this script "
                            "exports eval-split and pool scores itself"},
                   ckpt_f)
        print(f"[train] saved {ckpt_f}")

    score_eval_splits(model, mean, std, device, art, fits_root, args.batch)
    if args.score_pool:
        score_pool(model, mean, std, device, art, Path(args.score_pool),
                   pool_out, args.pool_batch)
    else:
        print("[142] NOTE: --score-pool not given -> NegEval pool NOT scored; "
              "the 113 refit needs the pool column (rerun with --score-only "
              "--score-pool <root> if the member is gated into the ensemble)")
    print(f"[142] {art} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
