#!/usr/bin/env python3
"""112_score_pool.py — Phase 110: score the NegEval-1M / mining-pool cutout
shards (111 output) with EVERY v1 scorer on ONE GPU (runs on PERLMUTTER; the
--self-test mode also runs locally on a TITAN).

Scorers (one sequential pass per checkpoint over all shards; TF32 is forced
OFF so A100 numbers match the TITAN harness, cf. 100_nersc_smoke.py):
  member_shielded_A / member_effnet_B / member_effnet_B3 / member_effnet_S2 /
  member_resnet46_C                      <ckpt-dir>/ckpt/member_<name>.pt
  baseline_resnet                        checkpoint_best_shielded194k_staged.pt
  baseline_effnet                        checkpoint_best_efficientnet_staged.pt
  baseline_meta = sigmoid(MetaLearner([p_resnet, p_effnet]))  (exactly
                  03_reproduce_baseline.meta_prob, checkpoint_best_meta_staged.pt)
  member_aion                            only with --aion score (subprocess to
                                         112b_score_aion_pool.py; needs `aion`)

Fidelity contract: per-row math is identical to _scorelib.score_paths — the
float32 cutout is normalised on CPU with the checkpoint's OWN mean/std (3,1,1),
clamped to +/-250, moved to the GPU, then _trainlib.model_prob(model, x, arch)
(sigmoid for shielded/l18 single-logit nets, softmax[:,1] for EfficientNet).
The only difference is the input source (memmapped .npy shard rows instead of
FITS files), which --self-test proves is score-identical (<1e-4).

Checkpoint-schema note: the member checkpoints from 20_train_member.py do NOT
store variant/head_dim/num_classes (the EfficientNet keys _scorelib.
load_checkpoint_model requires), so load_member_checkpoint() below infers
head_dim/num_classes from the state_dict head shapes and takes the timm
variant from members.json (fallback: hardcoded v1 roster). Everything else
(arch/mean/std/state_dict/shielded_cfg) matches _scorelib's schema. Phase-140
checkpoints with arch='timm' (140_train_zoobot_member.py: variant + Linear
head stored in the ckpt) load via the generic timm branch; arch='escnn_d4'
is NOT loadable here — 142_train_escnn_d4.py exports its own scores.

Inputs: --cutout-root with cutouts_<k>.npy shards (n,3,101,101 float32) +
index.parquet [row_id,shard,idx_in_shard,ok,nan_frac]; rows with ok=False are
skipped -> NaN in every score column. Per-scorer partial results are cached as
<out>.partial_<col>.npy (resume-safe; deleted after the final atomic write).

    # Perlmutter (repo synced with symlinks resolved, rsync -L; data/ckpt +
    # the three checkpoint_best_*_staged.pt + members.json synced under data/):
    sbatch --export=ALL,CMD='python 112_score_pool.py \
        --cutout-root $SCRATCH/claudenet/cutouts/negeval \
        --out $SCRATCH/claudenet/scores/negeval_scores.parquet' \
        nersc/shared_gpu.slurm

    # local array-path == file-path proof (writes data/v2/smoke_scores_selftest.parquet,
    # compares vs data/v2/smoke_scores_local.parquet when present):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      /home2/benson/.venvs/claudenet/bin/python 112_score_pool.py --self-test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _clib as C
import _scorelib as SL
import _trainlib as TL

TOL = 1e-4                      # smoke-parity tolerance (same as 100_nersc_smoke)
MEMBERS = ("shielded_A", "effnet_B", "effnet_B3", "effnet_S2", "resnet46_C")
# v1 roster (data/members.json) fallback: ckpt files lack the timm variant key.
VARIANT_FALLBACK = {"effnet_B": "tf_efficientnetv2_s",
                    "effnet_S2": "tf_efficientnetv2_s",
                    "effnet_B3": "tf_efficientnet_b3"}
BASE_SH = "checkpoint_best_shielded194k_staged.pt"
BASE_EF = "checkpoint_best_efficientnet_staged.pt"
BASE_META = "checkpoint_best_meta_staged.pt"


# ----- checkpoint loading ----------------------------------------------------

def member_variant(name: str, ckpt_dir: Path) -> str | None:
    """timm variant for an effnet member: members.json first, then fallback."""
    for d in (ckpt_dir, C.DATA):
        f = Path(d) / "members.json"
        if f.exists():
            for m in json.load(open(f)):
                if m["name"] == name and m.get("variant"):
                    return m["variant"]
    return VARIANT_FALLBACK.get(name)


def load_member_checkpoint(path: Path, device, variant_hint: str | None):
    """_scorelib.load_checkpoint_model, tolerant of the 20_train_member.py
    schema (no variant/head_dim/num_classes keys on efficientnet members:
    head_dim/num_classes come from the state_dict head shapes, the variant
    from members.json). Returns (model, score_arch, mean(3,1,1), std(3,1,1))."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "shielded")
    mean = np.array(ckpt["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(ckpt["std"], dtype=np.float32).reshape(3, 1, 1)
    sd = ckpt["state_dict"]
    if arch == "efficientnet":
        eff = SL._load_module("efficientnet_112", "02_efficientnet.py")
        variant = ckpt.get("variant") or variant_hint
        if variant is None:
            raise ValueError(f"{path.name}: efficientnet member without a variant "
                             "(need members.json or VARIANT_FALLBACK)")
        head_dim = int(sd["head.0.weight"].shape[0])      # Sequential(Linear,ReLU,Linear)
        num_classes = int(sd["head.2.weight"].shape[0])
        model = eff.EfficientNetV2Lens(pretrained=False, variant=variant,
                                       head_dim=head_dim, num_classes=num_classes)
    elif arch == "timm":
        # generic timm member (140_train_zoobot_member.py): variant is the plain
        # timm architecture name stored IN the checkpoint (offline-rebuildable,
        # pretrained=False), head = Linear(num_features, num_classes); the
        # num_classes comes from the state_dict head shape, like efficientnet.
        # NB: C._load (repo root), NOT SL._load_module — _scorelib is a symlink
        # into inchausti-2025, so its loader resolves relative to THAT dir.
        zm = C._load("timm_member_112", C.ROOT / "140_train_zoobot_member.py")
        variant = ckpt.get("variant") or variant_hint
        if variant is None:
            raise ValueError(f"{path.name}: timm member without a 'variant' key")
        num_classes = int(sd["head.weight"].shape[0])
        model = zm.TimmLens(variant=variant, pretrained=False,
                            num_classes=num_classes)
    elif arch == "shielded":
        cfg = ckpt.get("shielded_cfg") or {"final_out": int(ckpt.get("final_out", 32))}
        model = SL.ShieldedDeepLens(in_channels=3, **cfg)
    elif arch == "l18":
        model = SL.CMUDeepLens(in_channels=3)
    else:
        raise ValueError(f"{path.name}: unknown arch {arch!r}")
    model.load_state_dict(sd)
    model.to(device).eval()
    score_arch = ckpt.get("score_arch") or (
        "efficientnet" if arch in ("efficientnet", "timm") else "shielded")
    return model, score_arch, mean, std


def member_ckpt_path(ckpt_dir: Path, name: str) -> Path:
    for p in (ckpt_dir / "ckpt" / f"member_{name}.pt", ckpt_dir / f"member_{name}.pt"):
        if p.exists():
            return p
    raise FileNotFoundError(f"member checkpoint member_{name}.pt not under {ckpt_dir}")


# ----- core array-path scoring (the math of _scorelib.score_paths) ----------

@torch.no_grad()
def score_array(model, score_arch, mean_t, std_t, x_np: np.ndarray, device):
    """(B,3,101,101) float32 -> (B,) float32 probs; identical math to
    _scorelib.score_paths: CPU normalise+clamp, then model_prob on device."""
    x = torch.from_numpy(np.ascontiguousarray(x_np))
    x = torch.clamp((x - mean_t) / std_t, -250.0, 250.0).to(device)
    return TL.model_prob(model, x, score_arch).cpu().numpy()


def index_fingerprint(index: pd.DataFrame) -> str:
    """Binds a resume sidecar to the exact index it was computed against."""
    h = pd.util.hash_pandas_object(index["row_id"].astype(str), index=False).to_numpy()
    return f"{len(index)}:{index.row_id.iloc[0]}:{index.row_id.iloc[-1]}:{int(h.sum()) & 0xFFFFFFFFFFFF:x}"


def run_pass(col, model, score_arch, mean, std, index, root: Path, device,
             batch: int, partial: Path) -> np.ndarray:
    """One full pass of one checkpoint over all shards -> probs aligned with
    `index` rows (NaN where ok=False). Cached in `partial` (resume), bound to
    the index by fingerprint — a stale/foreign sidecar is reprocessed."""
    fp = index_fingerprint(index)
    if partial.exists():
        try:
            z = np.load(partial, allow_pickle=False)
            probs, fp_stored = z["probs"], str(z["fp"])
        except Exception:
            probs, fp_stored = None, None
        if probs is not None and fp_stored == fp and len(probs) == len(index):
            print(f"[112] {col}: resume from {partial.name}")
            return probs
        print(f"[112] {col}: stale partial (fingerprint mismatch) -> rescoring")
    t0 = time.time()
    probs = np.full(len(index), np.nan, dtype=np.float32)
    mean_t = torch.from_numpy(np.asarray(mean, dtype=np.float32).reshape(3, 1, 1))
    std_t = torch.from_numpy(np.asarray(std, dtype=np.float32).reshape(3, 1, 1))
    for k, sub in index.groupby("shard", sort=True):
        mm = np.load(root / f"cutouts_{k}.npy", mmap_mode="r")
        if mm.shape[1:] != (3, 101, 101):
            raise ValueError(f"cutouts_{k}.npy has shape {mm.shape}; expected (*,3,101,101)")
        ok = sub[sub.ok]
        gidx = ok.index.to_numpy()                      # positions in `index`
        sidx = ok.idx_in_shard.to_numpy()               # positions in the shard
        for s in range(0, len(ok), batch):
            x = np.asarray(mm[sidx[s:s + batch]], dtype=np.float32)
            probs[gidx[s:s + batch]] = score_array(model, score_arch,
                                                   mean_t, std_t, x, device)
        del mm
    n_ok = int(np.isfinite(probs).sum())
    np.savez(partial, probs=probs, fp=fp)
    print(f"[112] {col}: scored {n_ok:,}/{len(index):,} rows "
          f"({(time.time() - t0) / 60:.1f} min)")
    return probs


def meta_prob(meta, p_sh, p_ef, device):
    """copied from 03_reproduce_baseline.meta_prob (the v1 baseline_meta math)."""
    P = np.stack([p_sh, p_ef], 1).astype(np.float32)
    ok = np.isfinite(P).all(1)
    out = np.full(len(P), np.nan, np.float32)
    with torch.no_grad():
        out[ok] = torch.sigmoid(meta(torch.from_numpy(P[ok]).to(device))).cpu().numpy()
    return out


# ----- self-test: array-path == file-path ------------------------------------

def self_test(args, device) -> int:
    man = pd.read_parquet(C.DATA / "v2" / "smoke_manifest.parquet")
    fits_dir = Path(args.smoke_fits_dir) if args.smoke_fits_dir else (
        C.DATA / "smoke_fits" if (C.DATA / "smoke_fits").exists()
        else C.DATA / "cutouts_fits_neg_dr9")   # the 1000 smoke FITS live here locally
    print(f"[112:selftest] {len(man)} rows, fits_dir={fits_dir}")

    # tiny in-memory shard built with the v1 FITS reader
    cube = np.zeros((len(man), 3, 101, 101), np.float32)
    ok = np.zeros(len(man), bool)
    for i, rid in enumerate(man.row_id):
        try:
            a = TL.load_fits_cube(fits_dir / f"{rid}.fits")
        except Exception:
            continue
        if a.shape == (3, 101, 101):
            cube[i] = a
            ok[i] = True
    print(f"[112:selftest] loaded {int(ok.sum())}/{len(man)} cutouts")

    model, arch, mean, std, _ = SL.load_checkpoint_model(
        Path(args.ckpt_dir) / BASE_EF, device)
    mean_t = torch.from_numpy(np.asarray(mean, np.float32).reshape(3, 1, 1))
    std_t = torch.from_numpy(np.asarray(std, np.float32).reshape(3, 1, 1))
    probs = np.full(len(man), np.nan, np.float32)
    pos = np.where(ok)[0]
    for s in range(0, len(pos), args.batch):
        b = pos[s:s + args.batch]
        probs[b] = score_array(model, arch, mean_t, std_t, cube[b], device)

    out = Path(args.out) if args.out else C.DATA / "v2" / "smoke_scores_selftest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": man.row_id, "p": probs}).to_parquet(out, index=False)
    print(f"[112:selftest] wrote {out}")

    ref_f = C.DATA / "v2" / "smoke_scores_local.parquet"
    if not ref_f.exists():
        print("[112:selftest] no smoke_scores_local.parquet here -> wrote scores only "
              "(compare on the host that has it)")
        return 0
    ref = pd.read_parquet(ref_f).set_index("row_id")["p"]
    got = pd.Series(probs, index=man.row_id, name="p")
    j = ref.to_frame("pa").join(got.to_frame("pb"), how="inner").dropna()
    d = (j.pa - j.pb).abs()
    verdict = "PASS" if (len(j) == len(man) and d.max() < TOL) else "FAIL"
    print(f"[112:selftest] n={len(j)} max|dp|={d.max():.2e} mean={d.mean():.2e} "
          f"tol={TOL} -> {verdict}")
    return 0 if verdict == "PASS" else 1


# ----- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutout-root", help="111 output dir (cutouts_<k>.npy + index.parquet)")
    ap.add_argument("--out", help="output scores parquet path")
    ap.add_argument("--ckpt-dir", default=str(C.DATA),
                    help="dir with checkpoint_best_*_staged.pt (+ ckpt/member_*.pt)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--aion", choices=("skip", "score"), default="skip",
                    help="'score' shells out to 112b_score_aion_pool.py (needs `aion`)")
    ap.add_argument("--extra-ckpt-dir", default=None,
                    help="also score every member_*.pt in this dir (e.g. the 121 "
                         "retrained variants in data/v2/ckpt); column = file stem")
    ap.add_argument("--only-extra", action="store_true",
                    help="score ONLY --extra-ckpt-dir checkpoints (skip the v1 "
                         "roster, baselines, meta, aion)")
    ap.add_argument("--self-test", action="store_true",
                    help="prove array-path == file-path on the 1000-row smoke set")
    ap.add_argument("--smoke-fits-dir", default=None,
                    help="--self-test FITS dir (default data/smoke_fits, falling "
                         "back to data/cutouts_fits_neg_dr9)")
    args = ap.parse_args()

    # TITAN-parity: A100 TF32 off BEFORE any conv/matmul (cf. 100_nersc_smoke.py)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[112] device={device} tf32=off seed={C.SEED}")

    if args.self_test:
        return self_test(args, device)
    if not (args.cutout_root and args.out):
        ap.error("--cutout-root and --out are required (or use --self-test)")

    root, out, ckpt_dir = Path(args.cutout_root), Path(args.out), Path(args.ckpt_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    index = (pd.read_parquet(root / "index.parquet")
             .sort_values(["shard", "idx_in_shard"]).reset_index(drop=True))
    print(f"[112] {len(index):,} rows in {index.shard.nunique()} shards; "
          f"{int(index.ok.sum()):,} ok")

    def partial(col):
        return out.parent / f"{out.name}.partial_{col}.npz"

    cols: dict[str, np.ndarray] = {}

    # 1. the five v1 ensemble members
    for name in (() if args.only_extra else MEMBERS):
        col = f"member_{name}"
        path = member_ckpt_path(ckpt_dir, name)
        model, score_arch, mean, std = load_member_checkpoint(
            path, device, member_variant(name, ckpt_dir))
        print(f"[112] {col}: {path.name} score_arch={score_arch}")
        cols[col] = run_pass(col, model, score_arch, mean, std, index, root,
                             device, args.batch, partial(col))
        del model
        torch.cuda.empty_cache()

    # 1b. extra checkpoints (121 retrained variants etc.); column = file stem,
    #     timm variant read from the checkpoint itself (121 stores it)
    if args.extra_ckpt_dir:
        extras = sorted(Path(args.extra_ckpt_dir).glob("member_*.pt"))
        for p in [p for p in extras if p.name.endswith("_smoke.pt")]:
            print(f"[112] skipping smoke checkpoint {p.name}")
        extras = [p for p in extras if not p.name.endswith("_smoke.pt")]
        if not extras:
            print(f"[112] FATAL: --extra-ckpt-dir {args.extra_ckpt_dir} has no "
                  f"non-smoke member_*.pt")
            return 1
        skipped = []
        for path in extras:
            col = path.stem
            try:                     # a non-loadable ckpt (e.g. arch='escnn_d4')
                model, score_arch, mean, std = load_member_checkpoint(path, device, None)
                print(f"[112] {col}: {path.name} score_arch={score_arch} (extra)")
                cols[col] = run_pass(col, model, score_arch, mean, std, index, root,
                                     device, args.batch, partial(col))
                del model
            except Exception as e:   # must not kill the rest of the pass
                print(f"[112] SKIP {path.name}: {e!r}")
                skipped.append(path.name)
            torch.cuda.empty_cache()
        if skipped:
            print(f"[112] extra checkpoints skipped {len(skipped)}/{len(extras)}: "
                  f"{skipped}")
            if len(skipped) == len(extras):
                print("[112] FATAL: ALL --extra-ckpt-dir checkpoints failed")
                return 1

    # 2. the reproduced Inchausti baselines (bases for baseline_meta)
    for col, fname, arch in (() if args.only_extra else
                             (("baseline_resnet", BASE_SH, "shielded"),
                              ("baseline_effnet", BASE_EF, "efficientnet"))):
        model, _, mean, std, _ = SL.load_checkpoint_model(ckpt_dir / fname, device)
        print(f"[112] {col}: {fname}")
        cols[col] = run_pass(col, model, arch, mean, std, index, root,
                             device, args.batch, partial(col))
        del model
        torch.cuda.empty_cache()

    # 3. baseline_meta over [p_resnet, p_effnet] (03_reproduce_baseline math)
    if not args.only_extra:
        MetaLearner = SL._load_module("meta_learner_112", "03_meta_learner.py").MetaLearner
        meta = MetaLearner().to(device)
        meta.load_state_dict(torch.load(str(ckpt_dir / BASE_META), map_location="cpu",
                                        weights_only=False)["state_dict"])
        meta.eval()
        cols["baseline_meta"] = meta_prob(meta, cols["baseline_resnet"],
                                          cols["baseline_effnet"], device)

    # 4. optional degraded-AION member (separate script: needs the aion package)
    if args.aion == "score":
        aion_out = out.parent / f"{out.stem}_aion.parquet"
        cmd = [sys.executable, str(C.ROOT / "112b_score_aion_pool.py"),
               "--cutout-root", str(root), "--out", str(aion_out)]
        print(f"[112] aion subprocess: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        a = pd.read_parquet(aion_out).set_index("row_id")["member_aion"]
        cols["member_aion"] = a.reindex(index.row_id).to_numpy(np.float32)

    df = pd.DataFrame({"row_id": index.row_id, "ok": index.ok, **cols})
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out)
    for col in cols:
        partial(col).unlink(missing_ok=True)
    print(f"[112] wrote {out} ({len(df):,} rows x {len(cols)} scorers)")
    if args.aion != "score" and not args.only_extra:
        print("[112] NOTE: --aion skip -> pool lacks member_aion; 113 will DISABLE "
              "the combiners and emit NO flagship verdict. Use --aion score for "
              "the production NegEval run.")
    for col, v in cols.items():
        f = np.isfinite(v)
        print(f"[112]   {col:20s} n={int(f.sum()):,} mean_p={np.nanmean(v):.4f} "
              f"p99.9={np.nanquantile(v, 0.999):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
