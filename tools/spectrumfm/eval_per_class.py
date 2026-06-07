#!/usr/bin/env python
"""
Per-class redshift evaluation vs Redrock  —  SpectrumFM go/no-go **Metric 1**.

Forks eval_redshift_dz.py. For each existing Approach-A checkpoint it computes,
PER CLASS (LRG/ELG/QSO/MWS, plus BGS/OTHER context), the binning-fair
catastrophic-outlier rate against the Redrock production redshift, using the
honest encoder-masked readout (the model must predict z from the spectrum, not
copy it). Output is the proposal Metric-1 verdict: *no class degrades >5%*.

What "degrades >5%" means here (documented because the choice matters):
  * Reference set = ZWARN==0 (Redrock-confident). On that set we treat Redrock's
    Z as ground truth, so a catastrophic outlier is |z_pred - z_redrock|/(1+z)
    >= 0.0033 (the DESI good-z threshold).
  * PRIMARY GATE: a class PASSES if SpectrumFM's per-class catastrophic rate on
    the ZWARN==0 set is <= 5 percentage points (absolute). On the confident set
    Redrock-vs-Redrock dz is identically 0, so 5pp absolute is the honest
    "no degradation beyond 5%" reading.
  * CONTEXT (printed, not the gate): Redrock's OWN per-class failure fraction =
    N(ZWARN!=0)/N(all-in-class). Shows how often Redrock itself is unconfident
    per class, for transparency alongside the SpectrumFM rate.

Caveats (also printed at runtime):
  * QSO comparator is Redrock-pipeline-only (no QuasarNET / broad-MgII
    afterburners), so the QSO row is INDICATIVE.
  * Classes with N(ZWARN==0) < N_FLOOR (=30) are reported but excluded from the
    overall PASS/FAIL vote (flagged INDICATIVE).
  * Checkpoints do not persist the train/val split frac/seed; we reproduce the
    trainer default (0.05 / 42), overridable via --holdout-frac / --seed. If a
    checkpoint used a non-default split, some "val" rows may have been trained on.

Usage:
  # CPU smoke (no checkpoint, no GPU) — exercises the labels path + decoder:
  ~/.venvs/redshifty/bin/python tools/spectrumfm/eval_per_class.py --smoke

  # real eval on one L4:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 \
    ~/.venvs/redshifty/bin/python tools/spectrumfm/eval_per_class.py
"""
import argparse
import json
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

MANIFEST = "/raid/benson/data/desi_dr1_medium/manifest_mix.jsonl"
CKPT_DIR = Path("/raid/benson/data/desi_dr1_medium/checkpoints/checkpoints")
DEFAULT_CKPTS = [
    CKPT_DIR / "approach_a_mix_l4x2_v1" / "best.pt",
    CKPT_DIR / "approach_a_mix_l4x2_v2noskip" / "best.pt",
]

# HEADLINE_CLASSES = the proposal's four label-rich classes (the go/no-go focus).
# GATE_CLASSES = every real DESI class that votes when well-populated; BGS is
# included because it is a genuine galaxy-redshift capability signal and an
# MWS-only "pass" (stars at z~=0 are trivial) is not parity. OTHER = context.
HEADLINE_CLASSES = ("LRG", "ELG", "QSO", "MWS")
GATE_CLASSES = ("LRG", "ELG", "QSO", "MWS", "BGS")
REPORT_CLASSES = ("LRG", "ELG", "QSO", "MWS", "BGS", "OTHER")
N_FLOOR = 30          # below this N(ZWARN==0) a class is INDICATIVE, not voted
GATE_PP = 0.05        # 5 percentage-point absolute degradation gate


def load_records(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ----------------------------------------------------------------------------
# Tokenizer / model reconstruction (copied from eval_redshift_dz.py so this
# fork is a standalone __main__ script — see that file for the rationale).
# ----------------------------------------------------------------------------
def build_spec_tok(path, device):
    from src.tokenizers.spectrum import SpectrumTokenizer
    from src.tokenizers.spectrum_v2 import SpectrumTokenizerV2

    raw = torch.load(path, map_location=device, weights_only=False)
    # Per-stage downsampling strides persisted by pretrain_tokenizer; read only
    # when the ckpt is a dict (vs a bare state_dict). Absent key (old ckpts) ->
    # historical 32x (1,2,2,2) -> identical behavior.
    strides = raw.get("downsample_strides", (1, 2, 2, 2)) if isinstance(raw, dict) else (1, 2, 2, 2)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    is_v2 = "v2" in Path(path).parent.name.lower()
    if is_v2:
        has_skip = any(k.startswith("skip_proj.") for k in sd)
        has_ca = any(k.startswith("cross_attn.") for k in sd)
        tok = SpectrumTokenizerV2(use_skip_connections=has_skip, use_cross_attention=has_ca)
    else:
        tok = SpectrumTokenizer(downsample_strides=tuple(strides))
    tok = tok.to(device)
    tok.load_state_dict(sd)
    tok.eval()
    return tok


def build_z_tok(z_state):
    from src.tokenizers.redshift import RedshiftTokenizer
    from src.tokenizers.redshift_v2 import RedshiftTokenizerV2

    n = int(z_state["n_levels"])
    cls = RedshiftTokenizerV2 if n == 1024 else RedshiftTokenizer
    zt = cls(n_levels=n, gaussian_range=float(z_state["gaussian_range"]))
    zt._sorted_z = z_state["sorted_z"].cpu()
    zt._min_z = float(zt._sorted_z[0])
    zt._max_z = float(zt._sorted_z[-1])
    if isinstance(zt, RedshiftTokenizerV2):
        zt._embedding = nn.Linear(zt.n_levels, zt.d_model, bias=False)
    return zt


def _slice_batch(batch, lo, hi):
    """Slice a collated batch (tensors and python-list fields) to [lo:hi)."""
    return {k: v[lo:hi] for k, v in batch.items()}


def predict(ckpt_path, batch, device, chunk=256):
    """Honest encoder-masked redshift readout for one checkpoint (chunked).

    Returns per-spectrum arrays: z_pred (continuous), conf (top-1 softmax prob
    over the n_levels redshift bins), margin (top1 - top2 prob, a DELTACHI2
    analogue). Encoder layout for approach 'a' is [SOS=0, RZ=1, SPECTRUM..,
    EOS=last] (src/training/sequences.py); we mask position 1 (the redshift
    token) and read the decoder output at position 0. The forward pass is
    chunked so large N does not OOM a single L4.
    """
    from src.models.transformer import (
        SpectrumTransformer, TOTAL_VOCAB_SIZE, MASK_TOKEN, SOS_TOKEN,
        REDSHIFT_TOKEN_OFFSET,
    )
    from src.training.sequences import tokenize_and_build

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    approach = ck.get("approach", "a")
    assert approach == "a", f"readout assumes approach a (rz at enc pos 1), got {approach}"
    z_tok = build_z_tok(ck["z_tokenizer"])
    spec_tok = build_spec_tok(Path(ck["tokenizer_ckpt_path"]), device)
    # max_seq_len is the RoPE base length + the forward() length-assertion
    # ceiling; it does not change weights or RoPE values for used positions, so
    # 1024 reproduces V1 (seq 275) exactly while admitting a finer-tokenizer arm
    # (seq 547). Read from the ckpt if persisted, else default to 1024.
    # Read model architecture from the checkpoint if persisted (ladder arms
    # with d_model!=768); fall back to the historical 768/6/6/12 defaults so
    # OLD checkpoints lacking "model_config" build byte-identically to today.
    cfg = ck.get("model_config", {})
    d_model = cfg.get("d_model", 768)
    n_encoder_layers = cfg.get("n_encoder_layers", 6)
    n_decoder_layers = cfg.get("n_decoder_layers", 6)
    n_heads = cfg.get("n_heads", 12)
    max_seq_len = cfg.get("max_seq_len", int(ck.get("max_seq_len", 1024)))
    model = SpectrumTransformer(
        vocab_size=TOTAL_VOCAB_SIZE, d_model=d_model,
        n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers,
        n_heads=n_heads, max_seq_len=max_seq_len,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_lvl = z_tok.n_levels
    use_cuda = device.type == "cuda"

    z_pred_parts, conf_parts, margin_parts = [], [], []
    N = batch["flux"].shape[0]
    with torch.no_grad():
        for lo in range(0, N, chunk):
            sub = _slice_batch(batch, lo, min(lo + chunk, N))
            enc, dec, tgt, _, _ = tokenize_and_build(
                sub, spec_tok, z_tok, "a", device, encoder_mask_ratio=0.0)
            enc = enc.clone()
            # Structural guard: position 0 must be SOS, so position 1 is the
            # redshift token we are about to hide (catches a layout mismatch).
            assert int(enc[0, 0]) == SOS_TOKEN, "unexpected encoder layout (pos0 != SOS)"
            enc[:, 1] = MASK_TOKEN
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                logits, _ = model(enc, dec)
            rs_logits = logits[:, 0, REDSHIFT_TOKEN_OFFSET:REDSHIFT_TOKEN_OFFSET + n_lvl].float()
            probs = torch.softmax(rs_logits, dim=-1)
            pred_bin = rs_logits.argmax(-1).cpu()
            sorted_p, _ = probs.sort(dim=-1, descending=True)
            z_pred_parts.append(np.asarray(z_tok.decode(pred_bin), dtype=np.float64).flatten())
            conf_parts.append(sorted_p[:, 0].cpu().numpy().astype(np.float64))
            margin_parts.append((sorted_p[:, 0] - sorted_p[:, 1]).cpu().numpy().astype(np.float64))
    return dict(
        z_pred=np.concatenate(z_pred_parts),
        conf=np.concatenate(conf_parts),
        margin=np.concatenate(margin_parts),
        n_levels=n_lvl,
    )


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _fmt(x, nd=3):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:>5.{nd}f}"


def report_checkpoint(name, pred, z_true, zwarn, classes, spectype, thr):
    z_pred = pred["z_pred"]
    conf = pred["conf"]
    dz = np.abs(z_pred - z_true) / (1.0 + z_true)
    cata = dz >= thr
    confident = zwarn == 0                      # Redrock-confident reference set

    # --- integrity: every spectrum belongs to exactly one class ---
    counts = Counter(classes.tolist())
    assert sum(counts.values()) == len(classes), "class partition is not exhaustive"

    print(f"\n================  {name}  ================")
    print(f"  N={len(classes)}  ({int(confident.sum())} ZWARN==0 / "
          f"{int((~confident).sum())} ZWARN!=0)   z-bins={pred['n_levels']}   "
          f"catastrophic thr |dz|/(1+z) >= {thr}")

    hdr = (f"  {'class':<7} {'N(z0)':>6} {'SFM cat%':>9} {'med|dz|':>8} "
           f"{'RR fail%':>9} {'verdict':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    vote = {}
    for c in REPORT_CLASSES:
        in_c = classes == c
        n_all = int(in_c.sum())
        if n_all == 0:
            continue
        ref = in_c & confident
        n_ref = int(ref.sum())
        sfm_cata = float(cata[ref].mean()) if n_ref else float("nan")
        med = float(np.median(dz[ref])) if n_ref else float("nan")
        rr_fail = float((in_c & ~confident).sum()) / n_all      # Redrock's own fail frac

        if c in GATE_CLASSES:
            if n_ref < N_FLOOR:
                verdict = "INDIC"          # too few to vote
            else:
                passed = sfm_cata <= GATE_PP
                verdict = "PASS" if passed else "FAIL"
                vote[c] = passed
        else:
            verdict = "ctx"               # OTHER is context only

        star = " *" if c == "QSO" else ""
        print(f"  {c:<7} {n_ref:>6d} {_fmt(100*sfm_cata,1):>9} {_fmt(med,4):>8} "
              f"{_fmt(100*rr_fail,1):>9} {verdict:>9}{star}")

    # --- aggregate cross-check vs eval_redshift_dz.py ---
    agg_all = float((dz >= thr).mean())
    agg_conf = float(cata[confident].mean()) if confident.any() else float("nan")
    print(f"  {'-'*(len(hdr)-2)}")
    print(f"  aggregate catastrophic rate: all spectra={100*agg_all:5.1f}% "
          f"(good-z={100*(1-agg_all):4.1f}%),  ZWARN==0 only={100*agg_conf:5.1f}% "
          f"(good-z={100*(1-agg_conf):4.1f}%)   "
          f"[cf. eval_redshift_dz.py good-z~19% on its records[:96] subset]")

    # --- confidence calibration (reliability deciles) ---
    print("\n  confidence calibration (top-1 softmax prob over z-bins):")
    edges = np.linspace(0.0, 1.0, 11)
    print(f"    {'conf bin':>12} {'N':>5} {'meanConf':>9} {'obs.acc':>8}")
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < 9 else (conf >= lo) & (conf <= hi + 1e-9)
        n = int(m.sum())
        if n == 0:
            continue
        obs_acc = float((~cata[m]).mean())     # observed within-threshold fraction
        print(f"    [{lo:.1f},{hi:.1f}){'' :>1} {n:>5d} {conf[m].mean():>9.3f} {obs_acc:>8.3f}")

    # threshold whose selected fraction == Redrock ZWARN==0 selection fraction
    f0 = float(confident.mean())
    k = max(1, int(round(f0 * len(conf))))
    order = np.argsort(-conf)
    sel = order[:k]
    sfm_sel_cata = float(cata[sel].mean())
    print(f"  self-selected confident subset (top {100*f0:.0f}% by conf, matching "
          f"Redrock's ZWARN==0 selection): SFM catastrophic rate = {100*sfm_sel_cata:.2f}%")

    # --- overall verdict ---
    if not vote:
        overall = "NO VOTE (all gate classes below N_FLOOR)"
    else:
        overall = "PASS" if all(vote.values()) else "FAIL"
    voted_str = ", ".join(f"{c}={'P' if v else 'F'}" for c, v in vote.items()) or "none"
    missing = [c for c in GATE_CLASSES if c not in vote]
    missing_str = ("; indicative/absent: " + ", ".join(missing)) if missing else ""
    print(f"\n  >>> METRIC-1 VERDICT: {overall}   (voted: {voted_str}{missing_str})")
    # Honesty guard: stars at z~=0 are trivial; a pass driven only by MWS while
    # every galaxy/QSO class fails or is under-populated is NOT redshift parity.
    galaxyish = [c for c in ("LRG", "ELG", "QSO", "BGS") if vote.get(c)]
    if vote.get("MWS") and not galaxyish:
        print("      NOTE: only the stellar class (MWS, z~=0) meets the gate; "
              "galaxy/QSO redshift parity is NOT achieved at this scale.")
    return overall


# ----------------------------------------------------------------------------
# CPU smoke — no checkpoint, no GPU: exercise the labels path + decoder.
# ----------------------------------------------------------------------------
def run_smoke(args):
    print("[SMOKE] CPU only — no checkpoint, no GPU. Exercising the labels path "
          "+ target-bit decoder on a sv3+main mix.")
    records = load_records(args.manifest)
    sv3 = [r for r in records if str(r.get("survey", "")).lower() == "sv3"]
    main = [r for r in records if str(r.get("survey", "")).lower() != "sv3"]
    # One sv3 + one main coadd, NO max_spectra cap, then index explicitly into
    # each record's row range so BOTH target-column families are exercised
    # (a single coadd holds thousands of rows, so a global cap would never
    # reach the second record).
    mix = (sv3[:1] + main[:1]) or records[:2]
    ds = DR1IndexedDataset(
        mix, require_good_zwarn=False, require_nonzero_flux=True,
        return_labels=True, cache_size=4)
    per = max(8, min(48, args.n_spectra // 2))
    n0 = int(ds.records[0]["n_rows"])
    idxs = list(range(0, min(per, n0)))                       # record 0 (sv3)
    if len(mix) > 1:
        idxs += list(range(n0, min(n0 + per, len(ds))))       # record 1 (main)
    batch = collate_dr1_with_labels([ds[i] for i in idxs])
    if batch is None:
        print("[SMOKE] no spectra survived collation — check the manifest paths.")
        return
    classes = desi_targets.decode_class_array(
        batch["desi_target"].numpy(), batch["mws_target"].numpy(), batch["bgs_target"].numpy())
    zwarn = batch["zwarn"].numpy()
    surveys = batch["survey"]
    print(f"[SMOKE] {len(classes)} spectra "
          f"({sum(1 for s in surveys if s == 'sv3')} sv3 / "
          f"{sum(1 for s in surveys if s != 'sv3')} main); "
          f"{int((zwarn == 0).sum())} ZWARN==0")
    print(f"[SMOKE] decoded-class counts: {dict(Counter(classes.tolist()))}")

    # decoded-class vs SPECTYPE agreement on the ZWARN==0 reference set
    agree = tot = 0
    for c, st, zw in zip(classes, batch["spectype"], zwarn):
        a = desi_targets.spectype_agreement(c, st)
        if a is None or zw != 0:
            continue
        tot += 1
        agree += int(a)
    rate = (agree / tot) if tot else float("nan")
    print(f"[SMOKE] decoded-class vs SPECTYPE agreement (ZWARN==0): "
          f"{agree}/{tot} = {100*rate:.1f}%  (expect ~98-99%)")
    if not (sv3[:1] and main[:1]):
        print("[SMOKE] WARNING: manifest lacked both sv3 and main records; "
              "only one target-column family was exercised.")
    print("[SMOKE] OK." if tot and rate > 0.9 else "[SMOKE] CHECK: agreement low or no rows.")


def run_eval(args):
    device = torch.device(args.device)
    print(f"[device] {device}")
    records = load_records(args.manifest)
    _, val_records = split_records_by_healpix(records, args.holdout_frac, args.seed)
    print(f"[split] {len(records)} records -> {len(val_records)} held-out val "
          f"(holdout_frac={args.holdout_frac}, seed={args.seed})")

    # Stratify across ALL val records (not just whichever sort first) so the
    # per-class sample is representative — bright records are BGS/MWS-heavy,
    # dark records carry LRG/ELG/QSO. We build the full val index then take
    # evenly-spaced (monotone) indices, which span every record.
    ds = DR1IndexedDataset(
        val_records, require_good_zwarn=False, require_nonzero_flux=True,
        return_labels=True, cache_size=4)
    total = len(ds)
    n = min(args.n_spectra, total)
    sel = sorted(set(int(i) for i in np.linspace(0, total - 1, n).round().astype(int)))
    print(f"[sample] {total} val spectra -> {len(sel)} strided picks")
    batch = collate_dr1_with_labels([ds[i] for i in sel])
    if batch is None:
        print("[eval] no spectra — aborting.")
        return

    z_true = batch["z"].numpy().astype(np.float64).flatten()
    zwarn = batch["zwarn"].numpy()
    classes = desi_targets.decode_class_array(
        batch["desi_target"].numpy(), batch["mws_target"].numpy(), batch["bgs_target"].numpy())
    print(f"[data] {len(z_true)} spectra; z in [{z_true.min():.3f}, {z_true.max():.3f}]; "
          f"class counts {dict(Counter(classes.tolist()))}")
    print("[caveat] QSO row is INDICATIVE (Redrock-only comparator, no QuasarNET/broad-MgII).")
    print(f"[caveat] classes with N(ZWARN==0) < {N_FLOOR} are reported but not voted.")

    verdicts = {}
    for ckpt in args.checkpoints:
        ckpt = Path(ckpt)
        if not ckpt.exists():
            print(f"[skip] missing checkpoint: {ckpt}")
            continue
        pred = predict(ckpt, batch, device, chunk=args.chunk)
        verdicts[ckpt.parent.name] = report_checkpoint(
            ckpt.parent.name, pred, z_true, zwarn, classes, batch["spectype"],
            args.catastrophic_threshold)

    print("\n========== SUMMARY ==========")
    for name, v in verdicts.items():
        print(f"  {name:<32} Metric-1: {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", default=[str(p) for p in DEFAULT_CKPTS])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--n-spectra", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256, help="forward-pass mini-batch size")
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--catastrophic-threshold", type=float, default=0.0033)
    ap.add_argument("--smoke", action="store_true", help="CPU-only labels-path + decoder smoke")
    args = ap.parse_args()
    if args.smoke:
        run_smoke(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
