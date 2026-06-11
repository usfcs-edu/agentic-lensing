#!/usr/bin/env python3
"""150_distill_student.py — Phase 150: distill the v2 ensemble into ONE
EfficientNetV2-S student for scan throughput (runs LOCALLY, one GPU; pin with
CUDA_VISIBLE_DEVICES from the caller).

WHY: the shared-load 5-member stage-1 runs at ~865 cutouts/s/GPU (v1 paper,
Table "throughput"); a single EfficientNetV2-S runs at ~5,594. A student that
matches the ensemble's recovery@0.1%FPR within 0.02 makes the 45M-cutout DR9
sweep a one-model job (Phase-150 gate, checked by 151_throughput_bench.py).

TEACHER-TARGET CONTRACT (--teacher-scores, assembled by the orchestrator):
  a parquet with EXACTLY the columns
      row_id     str   (unique — one row per object)
      p_teacher  float (in [0,1] — the v2 flagship probability)
  p_teacher = the 145-calibrated flagship combiner ('average' over the
  isotonic-calibrated v2 member probs) evaluated on (i) the v1 staged-TRAIN
  rows (members scored via _train.score_df, then calibrated+combined exactly
  as 145 does) and (ii) the mined/NegEval rows that exist locally as FITS
  (their member raw scores already live in data/v2/scores_*_pool.parquet;
  calibrate+combine the same way). Training rows whose row_id is absent are
  DROPPED (count printed; >=50% per-source coverage is asserted).

TRAINING SET (local mode — the only implemented mode):
  v1 staged-train rows (data/training_split_staged.parquet split=='train';
  hard labels + teacher soft targets) UNION the locally-extracted mined FITS
  rows from 120/120b (--mined-manifests, default the 20k hard+random mined
  sets; hard label 0 + teacher soft targets), capped at --n-unlabeled extra
  rows. Validation = the untouched staged val split (hard-label AUC selects
  the best checkpoint, exactly the v1 recipe).
  A Perlmutter mode (training on ~400k pool cutouts from the npy shards) is
  deliberately NOT implemented: Phase 150 first tries local mode, and the 151
  gate decides. If recovery falls short, extract more mined rows to FITS with
  120b on Perlmutter, rsync them local, and feed them in via
  --extra-fits-manifests (comma list of 120b-style parquets
  [row_id, fits_dir(, label)]; rows without a label column or with label<0 /
  NaN become SOFT-TARGET-ONLY rows — the loss already supports them).

LOSS (Hinton KD): KD = alpha*CE(hard) + (1-alpha)*T^2*KL(teacher_T||student_T)
  with alpha=--alpha (0.5), T=--temperature (2.0). The teacher prob is
  softened through its logit (q_T = sigmoid(logit(p)/T), the 2-class softmax
  of teacher logits [0, logit(p)] at temperature T); the CE term averages over
  rows that HAVE a hard label. KD == CE exactly at alpha=1 (asserted by
  --self-test). Everything else is the v1 effnet member recipe verbatim
  (20_train_member: tf_efficientnetv2_s pretrained, epochs 25, batch 128,
  accum 2, Adam lr 1e-3, StepLR decay_ep=max(8,epochs//3) gamma 0.1,
  _trainlib.LensDataset aug, band_stats normalisation, best-val-AUC ckpt).

Writes (an --epochs override appends '_smoke' to ALL artifact names, as 121):
  data/v2/student_distilled_train.parquet      (the audit table)
  data/v2/ckpt/student_distilled.pt            (112-loadable schema:
        state_dict/arch='efficientnet'/score_arch/mean/std/val_auc/
        shielded_cfg=None/variant='tf_efficientnetv2_s' + distill metadata)
  data/v2/ckpt_student/member_student_distilled.pt   (symlink — lets
        `112_score_pool.py --extra-ckpt-dir data/v2/ckpt_student --only-extra`
        add a member_student_distilled column to the pool scores)
  data/v2/scores_student_distilled.parquet     [split,row_id,label,p,pc]
        over the 4 v1 eval splits (v2 member-score schema; pc = isotonic on val)

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
      /home2/benson/.venvs/claudenet/bin/python 150_distill_student.py \\
        --teacher-scores data/v2/teacher_targets.parquet
    # table-only sanity (no GPU): add --build-only
    # KD-loss math self-test (no data, no GPU, CPU tiny tensors):
    python 150_distill_student.py --self-test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import _clib as C
import _train as TR
import _trainlib as TL

V2 = C.DATA / "v2"
VARIANT = "tf_efficientnetv2_s"
DEFAULT_MINED = "data/v2/mined_hard_fits_manifest.parquet,data/v2/mined_random_fits_manifest.parquet"


# ===== KD loss (importable; --self-test asserts its math on CPU) ==============

def kd_loss(logits, y, p_teacher, has_hard, alpha: float, T: float):
    """KD = alpha*CE(hard) + (1-alpha)*T^2*KL(teacher_T || student_T).

    logits (B,2) student logits; y (B,) hard labels (used only where
    has_hard>0.5); p_teacher (B,) teacher lens-class prob in [0,1];
    has_hard (B,) {0,1}. Teacher softening: q_T = sigmoid(logit(p)/T) =
    softmax([0, logit(p)]/T)[1]. KL uses the standard Hinton direction
    KL(teacher || student) = F.kl_div(log_softmax(z/T), teacher_T), scaled by
    T^2 so its gradient magnitude stays comparable to CE. CE averages over
    hard-labelled rows only (0 if the batch has none). At alpha=1 the loss is
    exactly CE. Returns (loss, ce.detach(), kl.detach())."""
    eps = 1e-6
    q = torch.clamp(p_teacher, eps, 1.0 - eps)
    qT = torch.sigmoid(torch.log(q / (1.0 - q)) / T)
    teacher = torch.stack([1.0 - qT, qT], dim=1)
    logp = F.log_softmax(logits / T, dim=1)
    kl = F.kl_div(logp, teacher, reduction="batchmean") * (T * T)
    m = has_hard > 0.5
    ce = F.cross_entropy(logits[m], y[m].long()) if bool(m.any()) else logits.new_zeros(())
    return alpha * ce + (1.0 - alpha) * kl, ce.detach(), kl.detach()


class KDLensDataset(TL.LensDataset):
    """LensDataset (identical FITS load / normalise / clamp / augmentation)
    that also yields the per-row teacher prob and hard-label mask."""

    def __getitem__(self, i: int):
        x, y = super().__getitem__(i)
        r = self.df.iloc[i]
        return (x, y, torch.tensor(float(r["p_teacher"]), dtype=torch.float32),
                torch.tensor(float(r["has_hard"]), dtype=torch.float32))


# ===== distillation table ======================================================

def _read_manifest(f: Path) -> pd.DataFrame:
    m = pd.read_parquet(f)
    assert {"row_id", "fits_dir"} <= set(m.columns), \
        f"{f}: 120b-style manifest needs row_id+fits_dir (has {list(m.columns)})"
    m = m.copy()
    m["row_id"] = m["row_id"].astype(str)
    if "label" not in m.columns:
        m["label"] = np.nan
    m["source"] = Path(f).name
    return m[["row_id", "label", "fits_dir", "source"]]


def build_distill_table(teacher_f: Path, mined_fs: list[Path],
                        extra_fs: list[Path], n_unlabeled: int) -> pd.DataFrame:
    """Staged train+val rows (hard) + capped mined/extra rows (hard-or-soft),
    teacher targets merged on; returns [row_id,label,fits_dir,split,has_hard,
    p_teacher,source]."""
    staged = pd.read_parquet(C.DATA / "training_split_staged.parquet")
    staged["row_id"] = staged["row_id"].astype(str)
    base = staged[staged["split"].isin(("train", "val"))][
        ["row_id", "label", "fits_dir", "split"]].copy()
    # staged fits_dir can carry stale phoenix-era absolute paths — re-point each
    # basename at the local layout (the 19/30 remap convention). The mined/extra
    # manifest rows are NOT remapped: 120b writes absolute-local fits_dir.
    base["fits_dir"] = base["fits_dir"].apply(lambda p: str(C.DATA / Path(str(p)).name))
    for r in base.sample(n=min(5, len(base)), random_state=C.SEED).itertuples():
        f = Path(r.fits_dir) / f"{r.row_id}.fits"
        assert f.exists(), (f"remapped staged FITS missing on disk: {f} — "
                            f"are the local data/ cutout dirs staged?")
    base["has_hard"] = 1.0
    base["source"] = np.where(base["split"] == "val", "staged_val", "staged_train")

    extras = [_read_manifest(f) for f in (*mined_fs, *extra_fs)]
    ex = (pd.concat(extras, ignore_index=True) if extras
          else pd.DataFrame(columns=["row_id", "label", "fits_dir", "source"]))
    n_dup = int(ex["row_id"].duplicated().sum())
    if n_dup:
        print(f"[table] dropping {n_dup:,} duplicate extra rows (manifests overlap, "
              f"e.g. hard ∩ random mined sets; first manifest wins)")
        ex = ex.drop_duplicates("row_id", keep="first").reset_index(drop=True)
    if len(ex) > n_unlabeled:
        print(f"[table] capping extra rows {len(ex):,} -> --n-unlabeled "
              f"{n_unlabeled:,} (manifest order kept: 120 ranks hardest first)")
        ex = ex.head(n_unlabeled).copy()
    if len(ex):
        clash = set(ex["row_id"]) & set(base["row_id"])
        assert not clash, (f"{len(clash)} extra row_ids collide with the staged "
                           f"table (pools are brick-disjoint), e.g. {sorted(clash)[:3]}")
        exists = ex.apply(lambda r: (Path(r["fits_dir"]) / f"{r['row_id']}.fits").exists(),
                          axis=1)
        if (~exists).any():
            print(f"[table] dropping {int((~exists).sum()):,} extra rows without a "
                  f"FITS file on disk")
            ex = ex[exists].copy()
        soft = ex["label"].isna() | (ex["label"] < 0)
        ex["has_hard"] = np.where(soft, 0.0, 1.0)
        ex["label"] = ex["label"].fillna(0).clip(lower=0).astype(int)
        ex["split"] = "train"

    df = pd.concat([base, ex], ignore_index=True)

    teacher = pd.read_parquet(teacher_f)
    assert {"row_id", "p_teacher"} <= set(teacher.columns), \
        f"{teacher_f}: needs columns row_id+p_teacher (has {list(teacher.columns)})"
    teacher = teacher[["row_id", "p_teacher"]].copy()
    teacher["row_id"] = teacher["row_id"].astype(str)
    assert teacher["row_id"].is_unique, f"{teacher_f}: duplicate row_ids"
    pt = teacher["p_teacher"].to_numpy(dtype=np.float64)
    assert np.isfinite(pt).all() and pt.min() >= -1e-9 and pt.max() <= 1 + 1e-9, \
        f"{teacher_f}: p_teacher must be finite in [0,1]"

    df = df.merge(teacher, on="row_id", how="left")
    is_train = df["split"] == "train"
    miss = is_train & df["p_teacher"].isna()
    for src, sub in df[is_train].groupby("source"):
        cov = float(sub["p_teacher"].notna().mean())
        print(f"[table] teacher coverage {src:40s} {cov:6.1%} of {len(sub):,} rows")
        assert cov >= 0.5, (f"teacher coverage for {src} is {cov:.1%} < 50% — "
                            f"is {teacher_f} the right assembly?")
    if miss.any():
        print(f"[table] dropping {int(miss.sum()):,} train rows without a teacher target")
        df = df[~miss].reset_index(drop=True)
    df["p_teacher"] = df["p_teacher"].fillna(0.5)        # val rows: never used in the loss

    tr = df[df.split == "train"]
    print(f"[table] train={len(tr):,} (hard={int((tr.has_hard > 0).sum()):,}, "
          f"soft-only={int((tr.has_hard == 0).sum()):,}, "
          f"pos={int((tr.label == 1).sum()):,}) val={int((df.split == 'val').sum()):,}")
    return df


# ===== training loop (the v1 effnet recipe with the KD objective) =============

def train_distill(model, df, device, *, epochs, batch, lr, decay_ep, accum,
                  alpha, T, aug_seed, workers=4, val_sample=0):
    """Mirrors _train.train_supervised (Adam, StepLR gamma 0.1, best-val-AUC
    checkpoint, LensDataset aug) with kd_loss in place of plain CE. Returns
    (model, best_val_auc, mean, std)."""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(aug_seed)
    np.random.seed(aug_seed)
    mean, std = TR.band_stats(df[df.split == "train"])
    dtr, dva = df[df.split == "train"], df[df.split == "val"]
    if val_sample and val_sample < len(dva):
        dva = dva.sample(n=val_sample, random_state=C.SEED)
        print(f"[train] subsampled val to {len(dva):,} rows (--val-sample)")
    assert dva["label"].nunique() == 2, "val split needs both classes for AUC"
    dl_tr = DataLoader(KDLensDataset(dtr, dtr["fits_dir"].iloc[0], mean, std, True),
                       batch_size=batch, shuffle=True, num_workers=workers,
                       drop_last=True, pin_memory=True)
    dl_va = DataLoader(TL.LensDataset(dva, dva["fits_dir"].iloc[0], mean, std, False),
                       batch_size=256, shuffle=False, num_workers=workers,
                       pin_memory=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_ep, gamma=0.1)

    best_auc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        ce_s = kl_s = 0.0; nb = 0
        for i, (x, y, pt, hh) in enumerate(dl_tr):
            x = x.to(device, non_blocking=True)
            loss, ce, kl = kd_loss(model(x), y.to(device), pt.to(device),
                                   hh.to(device), alpha, T)
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                opt.step(); opt.zero_grad()
            ce_s += float(ce); kl_s += float(kl); nb += 1
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, y in dl_va:
                ps.append(TL.model_prob(model, x.to(device), "efficientnet").cpu().numpy())
                ys.append(y.numpy())
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(ps))
        print(f"[train] ep{ep:03d} ce={ce_s / max(nb, 1):.4f} kl={kl_s / max(nb, 1):.4f} "
              f"val_auc={auc:.4f}{'  *best*' if auc > best_auc else ''}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
    model.load_state_dict(best_state); model.eval()
    return model, float(best_auc), mean, std


# ===== --self-test: KD math on CPU tiny tensors (no data, no GPU) =============

def self_test() -> int:
    torch.manual_seed(C.SEED)
    checks = []

    def check(name, ok, detail=""):
        checks.append(bool(ok))
        print(f"[selftest] {'PASS' if ok else 'FAIL'}  {name}{(': ' + detail) if detail else ''}")

    B, T = 64, 2.0
    logits = torch.randn(B, 2)
    y = torch.randint(0, 2, (B,)).float()
    p_t = torch.rand(B)
    ones = torch.ones(B)

    # 1. KD -> CE as alpha -> 1 (exact at alpha=1; gap = (1-alpha)*kl)
    ce_ref = F.cross_entropy(logits, y.long())
    gaps = []
    for a in (0.5, 0.9, 0.99, 1.0):
        loss, ce, kl = kd_loss(logits, y, p_t, ones, a, T)
        gaps.append(abs(float(loss - ce_ref)))
        assert abs(float(loss - (a * ce + (1 - a) * kl))) < 1e-6
    check("KD -> CE as alpha -> 1", gaps[-1] < 1e-7 and all(
        gaps[i + 1] <= gaps[i] + 1e-12 for i in range(len(gaps) - 1)),
        "gaps " + ", ".join(f"{g:.2e}" for g in gaps))
    check("CE term == nn CE at alpha=1", abs(float(
        kd_loss(logits, y, p_t, ones, 1.0, T)[0] - ce_ref)) < 1e-7)

    # 2. KL >= 0 over random draws
    kls = [float(kd_loss(torch.randn(B, 2), y, torch.rand(B), ones, 0.0, T)[2])
           for _ in range(200)]
    check("KL >= 0 (200 random draws)", min(kls) >= -1e-9, f"min={min(kls):.2e}")

    # 3. self-consistent teacher (p = softmax(z)[:,1]) -> KL ~ 0
    p_self = torch.softmax(logits, dim=1)[:, 1]
    kl0 = float(kd_loss(logits, y, p_self, ones, 0.0, T)[2])
    check("KL ~ 0 when teacher == student", kl0 < 1e-6, f"kl={kl0:.2e}")

    # 4. soft-only rows: no hard labels -> CE term 0 -> alpha=1 loss is 0
    l_soft = float(kd_loss(logits, y, p_t, torch.zeros(B), 1.0, T)[0])
    check("CE masked out for soft-only rows", abs(l_soft) < 1e-9)

    # 5. tiny end-to-end: student forward + KD backward on CPU (offline weights)
    try:
        M = C.models()
        model = M["EfficientNetV2Lens"](pretrained=False, variant=VARIANT)
        xb = torch.randn(4, 3, 101, 101)
        loss, _, _ = kd_loss(model(xb), torch.tensor([0., 1., 0., 1.]),
                             torch.rand(4), torch.tensor([1., 1., 0., 0.]), 0.5, T)
        loss.backward()
        g = [p.grad for p in model.parameters() if p.grad is not None]
        ok = bool(torch.isfinite(loss)) and g and all(torch.isfinite(t).all() for t in g)
        check("student forward + KD backward finite (4x3x101x101, CPU)", ok,
              f"loss={float(loss):.4f}")
    except Exception as e:
        check("student forward + KD backward finite (4x3x101x101, CPU)", False, repr(e))

    n_ok = sum(checks)
    print(f"[selftest] {n_ok}/{len(checks)} checks passed -> "
          f"{'SELF-TEST PASS' if n_ok == len(checks) else 'SELF-TEST FAIL'}")
    return 0 if n_ok == len(checks) else 1


# ===== main ====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher-scores", default=str(V2 / "teacher_targets.parquet"),
                    help="parquet [row_id, p_teacher] — see the contract in the docstring")
    ap.add_argument("--mined-manifests", default=DEFAULT_MINED,
                    help="comma list of 120b manifests with LOCAL FITS ('' = none)")
    ap.add_argument("--extra-fits-manifests", default="",
                    help="comma list of EXTRA 120b-style manifests "
                         "[row_id,fits_dir(,label)]; label<0/NaN/absent = soft-only")
    ap.add_argument("--n-unlabeled", type=int, default=400_000,
                    help="cap on total non-staged rows taken from the manifests")
    ap.add_argument("--alpha", type=float, default=0.5, help="hard-CE weight in KD")
    ap.add_argument("--temperature", type=float, default=2.0, help="KD temperature T")
    ap.add_argument("--aug-seed", type=int, default=909,
                    help="augmentation seed (next free after zoobot_N's 707 and "
                         "escnn_D4's 808)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the v1 epoch count (SMOKE TESTS ONLY -> '_smoke' names)")
    ap.add_argument("--val-sample", type=int, default=0,
                    help="subsample val to N rows per epoch (0 = full, v1-faithful)")
    ap.add_argument("--build-only", action="store_true",
                    help="build + write the distillation table, no training (no GPU)")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU KD-loss math assertions (no data, no GPU)")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    art = "student_distilled" + ("_smoke" if args.epochs is not None else "")
    if args.epochs is not None:
        print(f"[150] --epochs {args.epochs} override -> SMOKE artifact names "
              f"({art}_train.parquet / ckpt/{art}.pt / scores_{art}.parquet)")

    teacher_f = Path(args.teacher_scores)
    assert teacher_f.exists(), (f"{teacher_f} missing — the orchestrator must "
                                f"assemble the teacher targets first (see docstring)")
    mined_fs = [Path(s) for s in args.mined_manifests.split(",") if s.strip()]
    extra_fs = [Path(s) for s in args.extra_fits_manifests.split(",") if s.strip()]
    for f in (*mined_fs, *extra_fs):
        assert f.exists(), f"manifest {f} not found"
    df = build_distill_table(teacher_f, mined_fs, extra_fs, args.n_unlabeled)
    V2.mkdir(parents=True, exist_ok=True)
    table_f = V2 / f"{art}_train.parquet"
    df.to_parquet(table_f, index=False)
    print(f"[150] wrote {table_f}")
    if args.build_only:
        print("[150] --build-only: stopping before training")
        return 0

    # ---- v1 effnet member recipe (20_train_member) with the KD objective ----
    M20 = C._load("cn_150_m20", C.ROOT / "20_train_member.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs if args.epochs else M20.EPOCHS["efficientnet"]
    batch, accum = 128, 2
    model = M20.build_model("efficientnet", VARIANT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] student arch=efficientnet variant={VARIANT} params={n_params:,} "
          f"epochs={epochs} batch={batch} accum={accum} lr=1e-3 alpha={args.alpha} "
          f"T={args.temperature} aug_seed={args.aug_seed} device={device}")
    t0 = time.time()
    model, val_auc, mean, std = train_distill(
        model, df, device, epochs=epochs, batch=batch, lr=1e-3,
        decay_ep=max(8, epochs // 3), accum=accum, alpha=args.alpha,
        T=args.temperature, aug_seed=args.aug_seed, val_sample=args.val_sample)
    print(f"[train] student best_val_auc={val_auc:.4f} ({(time.time() - t0) / 60:.1f}m)")

    ckpt_dir = V2 / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tr = df[df.split == "train"]
    ckpt_f = ckpt_dir / f"{art}.pt"
    torch.save({"state_dict": model.state_dict(), "arch": "efficientnet",
                "score_arch": "efficientnet", "mean": mean.tolist(),
                "std": std.tolist(), "val_auc": val_auc, "shielded_cfg": None,
                "variant": VARIANT,                       # timm variant for 112
                "distill": {"alpha": args.alpha, "temperature": args.temperature,
                            "teacher_scores": str(teacher_f), "epochs": epochs,
                            "aug_seed": args.aug_seed,
                            "n_train": int(len(tr)),
                            "n_soft_only": int((tr.has_hard == 0).sum()),
                            "sources": tr["source"].value_counts().to_dict()}},
               ckpt_f)
    print(f"[train] saved {ckpt_f}")
    # member_-prefixed symlink so 112 --extra-ckpt-dir can score the pool with it
    link_dir = V2 / "ckpt_student"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / f"member_{art}.pt"
    link.unlink(missing_ok=True)
    link.symlink_to(Path("..") / "ckpt" / ckpt_f.name)
    print(f"[train] symlinked {link} -> {ckpt_f}")

    # ---- score the shared v1 eval manifests (20/121 verbatim) + isotonic pc ----
    import _ensemble as E
    rows = []
    for sp in ("val", "testneg", "storfer", "inchausti"):
        d = pd.read_parquet(C.DATA / f"eval_{sp}.parquet").copy()
        d["p"] = TR.score_df(model, "efficientnet", d, mean, std, device)
        d["split"] = sp
        rows.append(d[["split", "row_id", "label", "p"]])
        print(f"[score] {sp:10s} n={len(d)} mean_p={d['p'].mean():.3f}")
    sc = pd.concat(rows, ignore_index=True)
    sc["p"] = sc["p"].astype(np.float32)
    val = sc[sc.split == "val"]
    cal = E.make_calibrator("isotonic").fit(val["p"].to_numpy(), val["label"].to_numpy())
    sc["pc"] = cal.transform(sc["p"].to_numpy())
    out_f = V2 / f"scores_{art}.parquet"
    sc.to_parquet(out_f, index=False)
    print(f"[150] student done -> {out_f} (next: 151_throughput_bench.py for the gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
