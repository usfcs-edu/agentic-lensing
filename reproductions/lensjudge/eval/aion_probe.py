"""
AION-1 frozen-encoder probe on corpus_v5 (LensJudge v5, Phase E experiment 1).

Question: does the Claude-free HSC corpus carry signal that a *small* probe head
can learn from frozen foundation-model embeddings? (A probe cannot memorize the
way a 51M-param LoRA can, so this discriminates corpus signal from memorization.)

Pipeline (two stages, two venvs; data passed via .npz):

  stage 1 `extract`  (run with /home2/benson/.venvs/aion/bin/python, GPU):
      corpus_v5/{split}.csv -> HSC grizy FITS from the warm cache
      -> HSCImage modality (native coadd units, zeropoint 27; the AION image
         codec itself center-crops to 96x96, clamps, rescales, arcsinh-compresses)
      -> CodecManager.encode -> AION.encode (frozen) -> mean-pool 576 tokens
      -> outputs/aion_probe/embeddings_{split}.npz         (variant=base)
         outputs/aion_probe/embeddings_{split}_large.npz   (variant=large)

  stage 2 `probe`  (run with /home2/benson/.venvs/lensjudge/bin/python, CPU):
      TRAIN rows with label in {lens, nonlens} (softmid excluded), standardized.
      (a) logistic regression, C grid selected on valsel AUC
      (b) MLP head in->256->2, adam, early stop on valsel AUC (partial_fit loop)
      Metrics: AUC on valsel + test2; recovery at rejection=0.91 on test2
      (threshold = 91st percentile of test2 nonlens scores, matching how the
      VLM numbers are quoted). Appends to outputs/aion_probe/RESULTS.md.

Faithfulness notes (vs the AION model card / released code):
  * We feed raw HSC pdr3_wide coadd pixels (zp 27) exactly as the released
    ImageCodec preprocessing expects (RescaleToLegacySurvey divides by
    10**((27-22.5)/2.5) for survey == "HSC").
  * Our cached cutouts are ~167-168 px; the codec's own CenterCrop(96) needs
    equal-shaped bands, so we center-crop each band to 96x96 ourselves with the
    codec's exact formula (start = (dim - 96) // 2). For bands differing by
    1 px (167 vs 168) this shifts centers by <= 0.5 px.
  * Missing bands are allowed (the codec's ImagePadder zero-fills + masks
    channels); we require i or r and batch rows grouped by band signature.
  * Pooling: mean over all 576 encoder tokens (same recipe the local AION-1
    reproduction validated on the paper's HSC lens-retrieval task).

Run:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home2/benson/git/agentic-lensing/reproductions \
    /home2/benson/.venvs/aion/bin/python aion_probe.py extract --variant base
  PYTHONPATH=... /home2/benson/.venvs/lensjudge/bin/python aion_probe.py probe --variant base
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

LJ = Path("/home2/benson/git/agentic-lensing/reproductions/lensjudge")
CORPUS = LJ / "outputs" / "corpus_v5"
CACHE = LJ / "cache" / "hsc_v5" / "pdr3_wide"
OUT = LJ / "outputs" / "aion_probe"

SPLITS = ["train", "valsel", "test2"]
BANDS = "grizy"
BAND_NAME = {b: f"HSC-{b.upper()}" for b in BANDS}
CROP = 96          # the codec's CenterCrop size; 96 px * 0.168"/px = 16.1"
N_TOKENS = 576     # HSCImage.num_tokens (24 x 24 latents)
MODELS = {"base": "polymathic-ai/aion-base", "large": "polymathic-ai/aion-large"}
DEFAULT_BATCH = {"base": 64, "large": 32}
SEED = 2026


def emb_path(split: str, variant: str) -> Path:
    suffix = "" if variant == "base" else f"_{variant}"
    return OUT / f"embeddings_{split}{suffix}.npz"


def heads_path(variant: str) -> Path:
    return OUT / f"heads_{variant}.joblib"


def _key(ra: float, dec: float) -> str:
    # identical to reproductions/lensjudge/common/hsc_fetch.py::_key
    return f"{float(ra):.6f}_{float(dec):+.6f}"


# --------------------------------------------------------------------------- #
# stage 1: embedding extraction (aion venv)
# --------------------------------------------------------------------------- #
def _read_image(path: Path):
    """First 2D image HDU as finite float32 (mirrors hsc_fetch._read_image)."""
    from astropy.io import fits

    try:
        with fits.open(path) as h:
            for hd in h:
                if hd.data is not None and getattr(hd.data, "ndim", 0) == 2:
                    return np.nan_to_num(np.asarray(hd.data, np.float32))
    except Exception:
        return None
    return None


def _center_crop(a: np.ndarray, size: int = CROP) -> np.ndarray | None:
    """The codec's CenterCrop formula (start = (dim - size) // 2), per band."""
    h, w = a.shape
    if h < size or w < size:
        return None
    y0, x0 = (h - size) // 2, (w - size) // 2
    return a[y0:y0 + size, x0:x0 + size]


def _load_row_pixels(ra: float, dec: float, cache: Path = CACHE):
    """-> (stack (nb,96,96) float32, band_sig str) or (None, reason)."""
    d = cache / _key(ra, dec)
    if not d.is_dir():
        return None, "no_cache_dir"
    chans, sig = [], ""
    for b in BANDS:
        arr = _read_image(d / f"{b}.fits") if (d / f"{b}.fits").exists() else None
        if arr is None:
            continue
        crop = _center_crop(arr)
        if crop is None:                       # cutout smaller than 96 px
            continue
        chans.append(crop)
        sig += b
    if not chans:
        return None, "no_pixels"
    if "i" not in sig and "r" not in sig:
        return None, "no_i_or_r"
    return np.stack(chans).astype(np.float32), sig


def cmd_extract(args):
    import torch
    from aion.codecs import CodecManager
    from aion.model import AION
    from aion.modalities import HSCImage

    os.environ.setdefault("HF_HOME", "/home2/benson/.cache/huggingface")
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] variant={args.variant} device={device} batch={args.batch}")

    cm = CodecManager(device=device)
    model = AION.from_pretrained(MODELS[args.variant]).to(device).eval()

    cache = Path(args.cache) if args.cache else CACHE
    for split in args.splits.split(","):
        t0 = time.time()
        csv_path = Path(args.manifest) if args.manifest else CORPUS / f"{split}.csv"
        rows = _read_csv(csv_path)
        pix, sigs, keep, skipped = [], [], [], {}
        for i, r in enumerate(rows):
            stack, sig = _load_row_pixels(float(r["ra"]), float(r["dec"]), cache)
            if stack is None:
                skipped[r["name"]] = sig      # sig is the reason string here
                continue
            pix.append(stack)
            sigs.append(sig)
            keep.append(i)
        print(f"[{split}] {len(keep)}/{len(rows)} rows with pixels; "
              f"skipped {len(skipped)} ({_count(skipped.values())})")

        embs = np.zeros((len(keep), model.dim), np.float32)
        by_sig: dict[str, list[int]] = {}
        for j, s in enumerate(sigs):
            by_sig.setdefault(s, []).append(j)
        for sig, idxs in sorted(by_sig.items()):
            bands = [BAND_NAME[b] for b in sig]
            for lo in range(0, len(idxs), args.batch):
                sel = idxs[lo:lo + args.batch]
                flux = torch.as_tensor(np.stack([pix[j] for j in sel]), device=device)
                tokens = cm.encode(HSCImage(flux=flux, bands=bands))
                ntok = sum(v.shape[1] if v.dim() == 2 else 1 for v in tokens.values())
                assert ntok == N_TOKENS, ntok
                emb = model.encode(tokens, num_encoder_tokens=ntok)  # (b, 576, D)
                embs[sel] = emb.float().mean(dim=1).cpu().numpy()
                del tokens, emb, flux
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            emb_path(split, args.variant),
            emb=embs,
            name=np.array([rows[i]["name"] for i in keep]),
            label=np.array([rows[i]["label"] for i in keep]),
            grade=np.array([rows[i].get("grade", "") for i in keep]),
            src=np.array([rows[i].get("src", rows[i].get("kind", "")) for i in keep]),
            soft=np.array([float(rows[i].get("soft") or "nan") for i in keep],
                          np.float32),
            band_sig=np.array(sigs),
            skipped_name=np.array(list(skipped.keys())),
            skipped_reason=np.array(list(skipped.values())),
        )
        print(f"[{split}] saved {embs.shape} -> {emb_path(split, args.variant).name} "
              f"({time.time() - t0:.0f}s)")


def _read_csv(path: Path):
    import csv

    with open(path) as f:
        return list(csv.DictReader(f))


def _count(vals):
    from collections import Counter

    return dict(Counter(vals))


# --------------------------------------------------------------------------- #
# stage 2: probes (lensjudge venv; sklearn only)
# --------------------------------------------------------------------------- #
def _load_split(split: str, variant: str):
    z = np.load(emb_path(split, variant), allow_pickle=False)
    return z["emb"].astype(np.float64), z["label"], z


def _auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def _recovery_at_rejection(y, s, rejection=0.91):
    """TPR at the score threshold that rejects `rejection` of the negatives."""
    thr = float(np.quantile(s[y == 0], rejection))
    rec = float(np.mean(s[y == 1] >= thr))
    rej = float(np.mean(s[y == 0] < thr))
    return rec, rej, thr


def _fit_lr(Xtr, ytr, Xva, yva, seed=SEED):
    from sklearn.linear_model import LogisticRegression

    best = None
    for C in [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
        clf.fit(Xtr, ytr)
        auc = _auc(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return best  # (val_auc, C, clf)


def _fit_mlp(Xtr, ytr, Xva, yva, hidden=256, max_epochs=400, patience=30, seed=SEED):
    """MLP in->hidden->2; one partial_fit pass per epoch; early stop on val AUC."""
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(hidden_layer_sizes=(hidden,), solver="adam",
                        learning_rate_init=1e-3, alpha=1e-4, batch_size=128,
                        random_state=seed)
    rng = np.random.default_rng(seed)
    best_auc, best_clf, best_ep, since = -1.0, None, 0, 0
    for ep in range(1, max_epochs + 1):
        order = rng.permutation(len(Xtr))
        clf.partial_fit(Xtr[order], ytr[order], classes=np.array([0, 1]))
        auc = _auc(yva, clf.predict_proba(Xva)[:, 1])
        if auc > best_auc:
            best_auc, best_clf, best_ep, since = auc, copy.deepcopy(clf), ep, 0
        else:
            since += 1
            if since >= patience:
                break
    return best_auc, best_ep, best_clf


def cmd_probe(args):
    from sklearn.preprocessing import StandardScaler

    variant = args.variant
    Xtr_all, ltr, ztr = _load_split("train", variant)
    Xva_all, lva, _ = _load_split("valsel", variant)
    Xte_all, lte, _ = _load_split("test2", variant)

    m_tr = np.isin(ltr, ["lens", "nonlens"])          # exclude softmid
    Xtr, ytr = Xtr_all[m_tr], (ltr[m_tr] == "lens").astype(int)
    yva, yte = (lva == "lens").astype(int), (lte == "lens").astype(int)

    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva_all), sc.transform(Xte_all)
    print(f"[probe:{variant}] train {Xtr.shape} ({ytr.sum()} lens / "
          f"{(1 - ytr).sum()} nonlens; softmid excluded: {(~m_tr).sum()}), "
          f"valsel {Xva.shape} ({yva.sum()}/{(1 - yva).sum()}), "
          f"test2 {Xte.shape} ({yte.sum()}/{(1 - yte).sum()})")

    res = {"variant": variant, "dim": Xtr.shape[1],
           "n_train": len(ytr), "n_valsel": len(yva), "n_test2": len(yte)}

    va_auc, C, lr = _fit_lr(Xtr, ytr, Xva, yva)
    s_te = lr.predict_proba(Xte)[:, 1]
    rec, rej, thr = _recovery_at_rejection(yte, s_te)
    res["logreg"] = {"C": C, "valsel_auc": round(va_auc, 4),
                     "test2_auc": round(_auc(yte, s_te), 4),
                     "test2_recovery@rej0.91": round(rec, 4),
                     "test2_rejection_achieved": round(rej, 4)}
    print(f"  logreg  C={C:<5} valsel AUC={va_auc:.4f}  "
          f"test2 AUC={res['logreg']['test2_auc']:.4f}  recovery@0.91={rec:.3f}")

    va_auc, ep, mlp = _fit_mlp(Xtr, ytr, Xva, yva)
    s_te = mlp.predict_proba(Xte)[:, 1]
    rec, rej, thr = _recovery_at_rejection(yte, s_te)
    res["mlp256"] = {"best_epoch": ep, "valsel_auc": round(va_auc, 4),
                     "test2_auc": round(_auc(yte, s_te), 4),
                     "test2_recovery@rej0.91": round(rec, 4),
                     "test2_rejection_achieved": round(rej, 4)}
    print(f"  mlp256  ep={ep:<4} valsel AUC={va_auc:.4f}  "
          f"test2 AUC={res['mlp256']['test2_auc']:.4f}  recovery@0.91={rec:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"results_{variant}.json", "w") as f:
        json.dump(res, f, indent=2)
    import joblib

    joblib.dump({"scaler": sc, "logreg": lr, "mlp256": mlp,
                 "logreg_C": res["logreg"]["C"],
                 "mlp_best_epoch": res["mlp256"]["best_epoch"]},
                heads_path(variant))
    print(f"[probe:{variant}] wrote results_{variant}.json + {heads_path(variant).name}")
    return res


def cmd_gate(args):
    """Score ALREADY-TRAINED heads on a frozen held-out set (no retraining)."""
    import joblib

    variant = args.variant
    heads = joblib.load(heads_path(variant))
    X, lab, z = _load_split(args.split, variant)
    y = (lab == "lens").astype(int)
    Xs = heads["scaler"].transform(X)
    res = {"variant": variant, "split": args.split, "rejection_target": args.rejection,
           "n": len(y), "n_lens": int(y.sum()), "n_nonlens": int((1 - y).sum())}
    for head in ("logreg", "mlp256"):
        s = heads[head].predict_proba(Xs)[:, 1]
        rec, rej, thr = _recovery_at_rejection(y, s, rejection=args.rejection)
        res[head] = {"auc": round(_auc(y, s), 4),
                     f"recovery@rej{args.rejection}": round(rec, 4),
                     "rejection_achieved": round(rej, 4)}
        print(f"  {head:<7} {args.split} AUC={res[head]['auc']:.4f}  "
              f"recovery@{args.rejection}={rec:.3f} (rej achieved {rej:.3f})")
    with open(OUT / f"results_{args.split}_{variant}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[gate:{variant}] wrote results_{args.split}_{variant}.json")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="AION embedding extraction (aion venv, GPU)")
    ex.add_argument("--variant", default="base", choices=list(MODELS))
    ex.add_argument("--splits", default=",".join(SPLITS),
                    help="corpus splits, or the output tag when --manifest is given")
    ex.add_argument("--batch", type=int, default=None)
    ex.add_argument("--manifest", default=None,
                    help="CSV with name,ra,dec,label to embed instead of corpus splits")
    ex.add_argument("--cache", default=None, help="alternate pdr3_wide cache root")
    pr = sub.add_parser("probe", help="LR + MLP probes (lensjudge venv, CPU)")
    pr.add_argument("--variant", default="base", choices=list(MODELS))
    ga = sub.add_parser("gate", help="score saved heads on a frozen set (no retrain)")
    ga.add_argument("--variant", default="base", choices=list(MODELS))
    ga.add_argument("--split", default="gate", help="embeddings tag to score")
    ga.add_argument("--rejection", type=float, default=0.89)
    args = ap.parse_args()
    if args.cmd == "extract":
        args.batch = args.batch or DEFAULT_BATCH[args.variant]
        cmd_extract(args)
    elif args.cmd == "gate":
        cmd_gate(args)
    else:
        cmd_probe(args)


if __name__ == "__main__":
    main()
