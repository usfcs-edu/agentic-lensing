#!/usr/bin/env python
"""
Frozen-encoder linear-probe harness  —  "one encoder, six classes" capability.

This is a SEPARATE capability test from the redshift-regression Metric-1
(eval_per_class.py). WS1 found that Metric-1 FAILs for galaxies at prototype
scale (z-precision poor). But class discrimination and z-regression are
different questions: even a z-imprecise encoder may still carry
class-discriminative features (continuum shape, 4000A break, broad emission
lines). This probe isolates THAT signal.

Method (standard frozen-feature linear probe):
  1. Freeze the trained SpectrumFM encoder. For each spectrum we build the
     approach-'a' encoder sequence [SOS, RZ, SPECTRUM.., EOS] and run
     model.encode() (NOT the full forward / decoder).
  2. We MASK the redshift token (encoder position 1) with MASK_TOKEN. This is
     MANDATORY: approach-'a' encoder normally sees the true redshift, which
     would let the probe cheat (stars sit at z~=0, QSOs at high z), so the
     probe would classify off the *label-correlated redshift* instead of off
     the spectrum. Masking it forces the features to come from the spectrum.
  3. Mean-pool the encoder output over the SPECTRUM positions only and fit a
     multinomial logistic regression (sklearn; torch nn.Linear fallback) on a
     HEALPIX-DISJOINT train split, then score on the held-out split.

The split is by healpix file (split_records_by_healpix), so probe-train and
probe-test share NO pointing — there is no same-coadd leakage.

A class-prior / majority-class baseline is reported alongside. On the --smoke
run (RANDOM features, REAL labels) the probe macro-F1 should sit near the
class-prior baseline — that is the honesty check that the metric + confusion
plumbing is not silently inflating the score.

Usage:
  # CPU smoke (no GPU, no transformer) — validates probe + metric plumbing:
  ~/.venvs/redshifty/bin/python tools/spectrumfm/probe_six_class.py --smoke

  # real probe on one L4 (orchestrator runs this, not this subagent):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 \
    ~/.venvs/redshifty/bin/python tools/spectrumfm/probe_six_class.py
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/raid/benson/git/agentic-lensing/lensing-repos/redshifty")
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nersc"))
sys.path.insert(0, str(TOOLS))

import desi_targets  # noqa: E402
from dr1_dataset import (  # noqa: E402
    DR1IndexedDataset,
    collate_dr1_with_labels,
)
from src.training.data_split import split_records_by_healpix  # noqa: E402

# Reuse the just-built per-class primitives verbatim (do not reinvent).
from eval_per_class import (  # noqa: E402
    build_spec_tok,
    build_z_tok,
    load_records,
    _slice_batch,
)

MANIFEST = "/raid/benson/data/desi_dr1_medium/manifest_mix.jsonl"
CKPT_DIR = Path("/raid/benson/data/desi_dr1_medium/checkpoints/checkpoints")
DEFAULT_CKPTS = [
    CKPT_DIR / "approach_a_mix_l4x2_v1" / "best.pt",
    CKPT_DIR / "approach_a_mix_l4x2_v2noskip" / "best.pt",
]

# Probe classes = the five real (non-OTHER) DESI classes. OTHER is dropped from
# the probe (it is a grab-bag of secondary/sky/standard targets with no single
# spectral signature) but its count is always reported for transparency.
PROBE_CLASSES = ("LRG", "ELG", "QSO", "MWS", "BGS")


# ----------------------------------------------------------------------------
# Frozen-encoder feature extraction (the critical correctness part).
# ----------------------------------------------------------------------------
def extract_features(ckpt_path, batch, device, chunk=256):
    """Mean-pooled frozen-encoder features for one checkpoint (chunked).

    Returns (N, 768) float64 numpy array. We load the model ONCE and loop over
    `chunk`-sized mini-batches (mirroring eval_per_class.predict) so a large N
    does not OOM a single L4.

    Encoder layout for approach 'a' is [SOS=0, RZ=1, SPECTRUM.., EOS=last]
    (src/training/sequences.py:tokenize_and_build). We:
      * build the sequence with encoder_mask_ratio=0.0 (deterministic),
      * MASK position 1 (the redshift token) -> MASK_TOKEN so the probe cannot
        read the true redshift (mandatory; see module docstring),
      * model.encode() the masked sequence (frozen, no decoder),
      * mean-pool the SPECTRUM positions ONLY: feats_seq[:, 2:-1, :].mean(1).
        Position 0 = SOS, position 1 = the now-masked RZ token, the final
        position = EOS — all three are excluded so the pooled vector is a pure
        spectrum summary.
    """
    from src.models.transformer import (
        SpectrumTransformer, TOTAL_VOCAB_SIZE, MASK_TOKEN, SOS_TOKEN,
    )
    from src.training.sequences import tokenize_and_build

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    approach = ck.get("approach", "a")
    assert approach == "a", f"probe assumes approach a (rz at enc pos 1), got {approach}"
    z_tok = build_z_tok(ck["z_tokenizer"])
    spec_tok = build_spec_tok(Path(ck["tokenizer_ckpt_path"]), device)
    # Read model architecture from the checkpoint if persisted (ladder arms
    # with d_model!=768); fall back to the historical 768/6/6/12 defaults so
    # OLD checkpoints lacking "model_config" build byte-identically to today.
    cfg = ck.get("model_config", {})
    d_model = cfg.get("d_model", 768)
    n_encoder_layers = cfg.get("n_encoder_layers", 6)
    n_decoder_layers = cfg.get("n_decoder_layers", 6)
    n_heads = cfg.get("n_heads", 12)
    max_seq_len = cfg.get("max_seq_len", int(ck.get("max_seq_len", 1024)))  # admits finer (seq 547); 1024≡512 for V1
    model = SpectrumTransformer(
        vocab_size=TOTAL_VOCAB_SIZE, d_model=d_model,
        n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers,
        n_heads=n_heads, max_seq_len=max_seq_len,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    use_cuda = device.type == "cuda"

    feats_parts = []
    N = batch["flux"].shape[0]
    with torch.no_grad():
        for lo in range(0, N, chunk):
            sub = _slice_batch(batch, lo, min(lo + chunk, N))
            enc, _dec, _tgt, _, _ = tokenize_and_build(
                sub, spec_tok, z_tok, "a", device, encoder_mask_ratio=0.0)
            enc = enc.clone()
            # Structural guard: pos 0 must be SOS, so pos 1 is the redshift
            # token we are about to hide (catches a layout mismatch).
            assert int(enc[0, 0]) == SOS_TOKEN, "unexpected encoder layout (pos0 != SOS)"
            enc[:, 1] = MASK_TOKEN  # MANDATORY: hide true redshift from the probe
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                feats_seq = model.encode(enc)  # (B, L_enc, 768)
            # Pool SPECTRUM positions only: drop SOS(0), masked-RZ(1), EOS(-1).
            pooled = feats_seq[:, 2:-1, :].float().mean(dim=1)  # (B, 768)
            feats_parts.append(pooled.cpu().numpy().astype(np.float64))
    return np.concatenate(feats_parts, axis=0)


# ----------------------------------------------------------------------------
# Labeled-feature assembly for one record split (frozen features + labels).
# ----------------------------------------------------------------------------
def _strided_pick(n_total, n_want):
    """Evenly-spaced (monotone) indices over [0, n_total) — the eval_per_class
    stratified-strided idiom (np.linspace over len(ds)). Spans every record so
    the per-class sample is representative across bright/dark pointings."""
    n = min(n_want, n_total)
    return sorted(set(int(i) for i in np.linspace(0, n_total - 1, n).round().astype(int)))


def build_labeled_batch(records, n_want, tag):
    """Build a labeled DR1 batch (ZWARN==0 only) from `records`, strided to
    ~`n_want` spectra. Returns (batch, classes, n_other) or (None, None, 0).

    We over-sample the strided pool, then drop ZWARN!=0 and OTHER, so the final
    count may be < n_want; the strided spacing keeps it representative.
    """
    ds = DR1IndexedDataset(
        records, require_good_zwarn=False, require_nonzero_flux=True,
        return_labels=True, cache_size=4)
    total = len(ds)
    sel = _strided_pick(total, n_want)
    print(f"[{tag}] {len(records)} records, {total} spectra -> {len(sel)} strided picks")
    batch = collate_dr1_with_labels([ds[i] for i in sel])
    if batch is None:
        return None, None, 0

    zwarn = batch["zwarn"].numpy()
    classes = desi_targets.decode_class_array(
        batch["desi_target"].numpy(), batch["mws_target"].numpy(), batch["bgs_target"].numpy())

    clean = zwarn == 0                                  # restrict to clean labels
    n_other = int(((classes == "OTHER") & clean).sum())
    keep = clean & np.isin(classes, PROBE_CLASSES)      # drop OTHER from the probe
    keep_idx = np.nonzero(keep)[0]
    if keep_idx.size == 0:
        return None, None, n_other
    batch = _slice_batch(batch, 0, len(classes))        # no-op; keep dict shape
    batch = {k: (v[keep_idx] if hasattr(v, "__getitem__") and not isinstance(v, list)
                 else [v[i] for i in keep_idx]) for k, v in batch.items()}
    classes = classes[keep_idx]
    print(f"[{tag}] kept {len(classes)} ZWARN==0 probe spectra "
          f"(dropped {int((~clean).sum())} ZWARN!=0, {n_other} OTHER); "
          f"class counts {dict(Counter(classes.tolist()))}")
    return batch, classes, n_other


# ----------------------------------------------------------------------------
# Probe backends: sklearn LogisticRegression, or torch nn.Linear fallback.
# ----------------------------------------------------------------------------
def _try_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.preprocessing import StandardScaler     # noqa: F401
        return True
    except Exception:
        return False


def fit_predict_sklearn(Xtr, ytr, Xte, n_classes):
    """Standardize (fit on train) then multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", multi_class="auto")
    clf.fit(Xtr_s, ytr)
    return clf.predict(Xte_s)


def fit_predict_torch(Xtr, ytr, Xte, n_classes, epochs=300, seed=42):
    """Fallback: standardize (train stats) + nn.Linear(768, n_classes) trained
    with Adam and class-balanced cross-entropy for ~`epochs` on CPU. Used
    automatically when sklearn is unavailable; features are small so this is
    cheap. Equivalent to the sklearn multinomial-logistic probe."""
    torch.manual_seed(seed)
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd

    xt = torch.tensor(Xtr_s, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    xe = torch.tensor(Xte_s, dtype=torch.float32)

    # class-balanced CE weights = N / (n_classes * count_c) (sklearn 'balanced').
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(ytr) / (n_classes * counts)
    weight = torch.tensor(w, dtype=torch.float32)

    lin = nn.Linear(xt.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    lin.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(lin(xt), yt)
        loss.backward()
        opt.step()
    lin.eval()
    with torch.no_grad():
        pred = lin(xe).argmax(dim=1).numpy()
    return pred


# ----------------------------------------------------------------------------
# Metrics + reporting.
# ----------------------------------------------------------------------------
def _confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _prf_from_confusion(cm):
    """Per-class precision/recall/f1/support from a confusion matrix
    (rows=true, cols=pred). Returns dicts keyed by class index."""
    n = cm.shape[0]
    prec, rec, f1, sup = {}, {}, {}, {}
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        sup[c] = int(cm[c, :].sum())
        prec[c] = tp / (tp + fp) if (tp + fp) else 0.0
        rec[c] = tp / (tp + fn) if (tp + fn) else 0.0
        f1[c] = (2 * prec[c] * rec[c] / (prec[c] + rec[c])
                 if (prec[c] + rec[c]) else 0.0)
    return prec, rec, f1, sup


def _macro_weighted(prec, rec, f1, sup):
    classes = list(sup.keys())
    total = sum(sup.values()) or 1
    macro_f1 = float(np.mean([f1[c] for c in classes])) if classes else 0.0
    weighted_f1 = float(sum(f1[c] * sup[c] for c in classes) / total)
    return macro_f1, weighted_f1


def baseline_metrics(y_train, y_test, n_classes):
    """No-skill baseline: take the majority class from the TRAIN distribution
    and always predict it on the TEST set, then score on test. This is the
    standard ML baseline (train-derived, applied to test) — honest under a
    healpix-disjoint split where train/test class priors can differ, unlike a
    test-majority baseline which would be optimistically high.
    Returns (accuracy, macro_f1, maj_class_idx)."""
    maj = int(np.bincount(y_train, minlength=n_classes).argmax())
    y_pred = np.full_like(y_test, maj)
    cm = _confusion(y_test, y_pred, n_classes)
    prec, rec, f1, sup = _prf_from_confusion(cm)
    macro_f1, _ = _macro_weighted(prec, rec, f1, sup)
    acc = float((y_pred == y_test).mean())
    return acc, macro_f1, maj


def report(name, y_train, y_true, y_pred, class_names, backend, n_other_train, n_other_test):
    n_classes = len(class_names)
    cm = _confusion(y_true, y_pred, n_classes)
    prec, rec, f1, sup = _prf_from_confusion(cm)
    macro_f1, weighted_f1 = _macro_weighted(prec, rec, f1, sup)
    acc = float((y_pred == y_true).mean())
    base_acc, base_macro_f1, maj = baseline_metrics(y_train, y_true, n_classes)

    print(f"\n================  {name}  ================")
    print(f"  probe backend: {backend}   classes: {list(class_names)}")
    print(f"  N(test)={len(y_true)}   OTHER dropped: train={n_other_train}, test={n_other_test}")

    hdr = f"  {'class':<7} {'prec':>7} {'recall':>7} {'f1':>7} {'support':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in range(n_classes):
        print(f"  {class_names[c]:<7} {prec[c]:>7.3f} {rec[c]:>7.3f} "
              f"{f1[c]:>7.3f} {sup[c]:>8d}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'macro':<7} {'':>7} {'':>7} {macro_f1:>7.3f} {sum(sup.values()):>8d}")
    print(f"  {'weighted':<7} {'':>7} {'':>7} {weighted_f1:>7.3f}")
    print(f"  overall accuracy: {acc:.3f}")

    print(f"\n  baseline (train-majority='{class_names[maj]}' predicted on test): "
          f"accuracy={base_acc:.3f}  macro-F1={base_macro_f1:.3f}")
    print(f"  PROBE vs BASELINE:  acc {acc:.3f} vs {base_acc:.3f} "
          f"({acc - base_acc:+.3f}),  macro-F1 {macro_f1:.3f} vs "
          f"{base_macro_f1:.3f} ({macro_f1 - base_macro_f1:+.3f})")

    print("\n  confusion matrix (rows=true, cols=pred):")
    colhdr = "      " + " ".join(f"{c:>6}" for c in class_names)
    print(colhdr)
    for r in range(n_classes):
        row = " ".join(f"{cm[r, j]:>6d}" for j in range(n_classes))
        print(f"  {class_names[r]:>4} {row}")

    return dict(acc=acc, macro_f1=macro_f1, weighted_f1=weighted_f1,
                base_acc=base_acc, base_macro_f1=base_macro_f1)


def fit_and_report(name, Xtr, ytr, Xte, yte, class_names,
                   n_other_train, n_other_test):
    n_classes = len(class_names)
    use_sklearn = _try_sklearn()
    backend = "sklearn LogisticRegression" if use_sklearn else "torch nn.Linear fallback"
    if use_sklearn:
        y_pred = fit_predict_sklearn(Xtr, ytr, Xte, n_classes)
    else:
        y_pred = fit_predict_torch(Xtr, ytr, Xte, n_classes)
    return report(name, ytr, yte, y_pred, class_names, backend, n_other_train, n_other_test)


# ----------------------------------------------------------------------------
# CPU smoke — no GPU, no transformer: RANDOM features, REAL labels.
# ----------------------------------------------------------------------------
def run_smoke(args):
    print("[SMOKE] CPU only — no checkpoint, no GPU, no transformer. RANDOM "
          "(N,768) features with REAL decoded labels: validates the probe + "
          "metric + confusion plumbing. Random features -> macro-F1 should sit "
          "NEAR the class-prior baseline (the honesty check).")
    records = load_records(args.manifest)
    sv3 = [r for r in records if str(r.get("survey", "")).lower() == "sv3"]
    main = [r for r in records if str(r.get("survey", "")).lower() != "sv3"]
    # One sv3 + one main coadd (same idiom as eval_per_class.run_smoke): index
    # explicitly into each record's row range so BOTH target-column families are
    # exercised (a single coadd holds thousands of rows).
    mix = (sv3[:1] + main[:1]) or records[:2]
    ds = DR1IndexedDataset(
        mix, require_good_zwarn=False, require_nonzero_flux=True,
        return_labels=True, cache_size=4)
    per = 200
    n0 = int(ds.records[0]["n_rows"])
    idxs = list(range(0, min(per, n0)))
    if len(mix) > 1:
        idxs += list(range(n0, min(n0 + per, len(ds))))
    batch = collate_dr1_with_labels([ds[i] for i in idxs])
    if batch is None:
        print("[SMOKE] no spectra survived collation — check the manifest paths.")
        return

    zwarn = batch["zwarn"].numpy()
    classes = desi_targets.decode_class_array(
        batch["desi_target"].numpy(), batch["mws_target"].numpy(), batch["bgs_target"].numpy())
    clean = zwarn == 0
    n_other = int(((classes == "OTHER") & clean).sum())
    keep = clean & np.isin(classes, PROBE_CLASSES)
    classes = classes[keep]
    print(f"[SMOKE] {int(keep.sum())} ZWARN==0 probe spectra "
          f"(dropped {int((~clean).sum())} ZWARN!=0, {n_other} OTHER); "
          f"class counts {dict(Counter(classes.tolist()))}")

    # Restrict the class label space to the classes actually present, so the
    # confusion matrix and baseline are well-defined on this tiny mix.
    present = [c for c in PROBE_CLASSES if c in set(classes.tolist())]
    cls_to_idx = {c: i for i, c in enumerate(present)}
    y = np.array([cls_to_idx[c] for c in classes], dtype=np.int64)

    # RANDOM features: deterministic, independent of the labels by construction.
    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((len(y), 768)).astype(np.float64)

    # Healpix-free random 70/30 split for the smoke (the real run splits by
    # healpix; here we only test the probe/metric plumbing).
    perm = rng.permutation(len(y))
    n_te = max(len(present), int(round(0.30 * len(y))))
    te_idx, tr_idx = perm[:n_te], perm[n_te:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    print(f"[SMOKE] random-feature split: {len(ytr)} train / {len(yte)} test, "
          f"{len(present)} present classes {present}")

    res = fit_and_report("SMOKE (random features, real labels)",
                         Xtr, ytr, Xte, yte, present, n_other, n_other)
    # Honesty assertion: random features must NOT beat the no-skill baseline by
    # a meaningful margin. If they do, the metric or the split is leaking.
    delta = res["macro_f1"] - res["base_macro_f1"]
    if delta > 0.10:
        print(f"\n[SMOKE] FAIL: random-feature macro-F1 {res['macro_f1']:.3f} beats "
              f"baseline {res['base_macro_f1']:.3f} by {delta:+.3f} (>0.10) — "
              f"the metric or train/test split may be leaking.")
    else:
        print(f"\n[SMOKE] OK — random-feature macro-F1 {res['macro_f1']:.3f} sits at "
              f"the no-skill baseline {res['base_macro_f1']:.3f} (Δ{delta:+.3f}); "
              f"features carry no class signal, plumbing is honest.")


def run_probe(args):
    device = torch.device(args.device)
    print(f"[device] {device}")
    records = load_records(args.manifest)
    train_records, test_records = split_records_by_healpix(
        records, args.holdout_frac, args.seed)
    print(f"[split] {len(records)} records -> {len(train_records)} probe-train / "
          f"{len(test_records)} probe-test (healpix-disjoint, "
          f"holdout_frac={args.holdout_frac}, seed={args.seed})")

    tr_batch, tr_classes, n_other_tr = build_labeled_batch(
        train_records, args.n_train, "train")
    te_batch, te_classes, n_other_te = build_labeled_batch(
        test_records, args.n_test, "test")
    if tr_batch is None or te_batch is None:
        print("[probe] empty train or test probe set — aborting.")
        return

    class_names = list(PROBE_CLASSES)
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    ytr = np.array([cls_to_idx[c] for c in tr_classes], dtype=np.int64)
    yte = np.array([cls_to_idx[c] for c in te_classes], dtype=np.int64)

    results = {}
    for ckpt in args.checkpoints:
        ckpt = Path(ckpt)
        if not ckpt.exists():
            print(f"[skip] missing checkpoint: {ckpt}")
            continue
        print(f"\n[extract] {ckpt.parent.name}: frozen features for "
              f"{len(ytr)} train + {len(yte)} test spectra (chunk={args.chunk})")
        Xtr = extract_features(ckpt, tr_batch, device, chunk=args.chunk)
        Xte = extract_features(ckpt, te_batch, device, chunk=args.chunk)
        results[ckpt.parent.name] = fit_and_report(
            ckpt.parent.name, Xtr, ytr, Xte, yte, class_names,
            n_other_tr, n_other_te)

    print("\n========== SUMMARY ==========")
    for name, r in results.items():
        print(f"  {name:<32} acc={r['acc']:.3f} (base {r['base_acc']:.3f})  "
              f"macro-F1={r['macro_f1']:.3f} (base {r['base_macro_f1']:.3f})  "
              f"weighted-F1={r['weighted_f1']:.3f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", default=[str(p) for p in DEFAULT_CKPTS])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=256, help="forward-pass mini-batch size")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU-only random-feature probe + metric plumbing smoke")
    args = ap.parse_args()
    if args.smoke:
        run_smoke(args)
    else:
        run_probe(args)


if __name__ == "__main__":
    main()
