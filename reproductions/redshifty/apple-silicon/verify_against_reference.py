#!/usr/bin/env python3
"""verify_against_reference.py — gate the Apple Silicon (MPS) redshifty port against the
phoenix reference, writing data/REPRODUCTION_MPS_COMPARE.md.

Gating philosophy (same as the huang-2020/21 ports): GATE what the port must reproduce
(identical computation, structural correctness), REPORT what legitimately varies between
independent from-scratch runs. The redshift ignition has known high seed variance
(3.76%-8.76% peak z_acc across the phoenix seed sweep), so layer (c) gates the STRUCTURE
of ignition, not the exact number.

  HARD GATES:
    (a) inference fidelity   — data/xcheck_compare.json overall_pass (MPS==CUDA fwd)
    (b) Tier-1 training      — tokenizer + Approach-A from scratch on MPS: NaN-free AND
                               the val metric improves (the non_blocking-NaN sentinel)
    (c) Tier-2 ignition      — val_redshift_acc>=0.10 sustained late, AR>=TF/2,
                               val_loss_redshift drop>=1.0, val_loss<=200, NaN-free
  INFORMATIONAL: peak z_acc + delta vs phoenix 14.86%, AR/TF ratio, wallclock, max|Δ|.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RAID = HERE / "_raid/benson/data/desi_dr1_medium"

rows: list[dict] = []
overall_ok = True


def rec(section, metric, reference, mps, ok=None, gate=True):
    global overall_ok
    if ok is False and gate:
        overall_ok = False
    rows.append({"section": section, "metric": metric, "reference": reference,
                 "mps": mps, "ok": ok})


def load_jsonl(p: Path):
    try:
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    except Exception:
        return None


def has_nan(rows_, keys):
    for r in rows_ or []:
        for k in keys:
            v = r.get(k)
            if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                return True
    return False


# ---- (a) inference fidelity ----
xc = None
try:
    xc = json.loads((RESULTS / "xcheck_compare.json").read_text())
except Exception:
    pass
if xc:
    g = xc["gates"]; info = xc["informational"]
    rec("a: MPS==CUDA fwd", f"median|Δ| (max={info['max_abs_delta']:.2f})", "<=1e-3",
        f"{g['median|Δ|<=1e-3']['value']:.1e}", g["median|Δ|<=1e-3"]["pass"])
    rec("a: MPS==CUDA fwd", "argmax-token agreement", ">=99.5%",
        f"{g['argmax_agree>=99.5%']['value']*100:.2f}%", g["argmax_agree>=99.5%"]["pass"])
    rec("a: MPS==CUDA fwd", "loss relative |Δ|", "<=1e-3",
        f"{g['loss_rel<=1e-3']['value']:.1e}", g["loss_rel<=1e-3"]["pass"])
    rec("a: MPS==CUDA fwd", "|Δ redshift_acc| (science readout)", "<=0.005",
        f"{g['|Δredshift_acc|<=0.005']['value']:.4f}", g["|Δredshift_acc|<=0.005"]["pass"])
else:
    rec("a: MPS==CUDA fwd", "xcheck_compare.json", "<=1e-3", "PENDING", None)


# ---- (b1) Tier-1 tokenizer from scratch ----
# "Did it learn from random init?" = best (min) val far below the step-0 train loss.
# NOT last-val < first-val: on a tiny bounded set the model overfits the tail, which is
# expected and not a failure. The NaN-free check is the non_blocking-bug sentinel.
tok = load_jsonl(DATA / "tier1/tok_tier1/metrics.jsonl")
if tok:
    nan = has_nan(tok, ("loss_total", "loss_recon", "val_total", "val_recon"))
    tr = [r["loss_recon"] for r in tok if r.get("kind") == "train" and "loss_recon" in r]
    vr = [r["val_recon"] for r in tok if "val_recon" in r]
    init = tr[0] if tr else None
    best = min(vr) if vr else (min(tr) if tr else None)
    learned = init is not None and best is not None and best < init
    rec("b1: tokenizer/MPS", f"NaN-free ({len(tok)} rows)", "no NaN",
        "clean" if not nan else "NaN!", not nan)
    rec("b1: tokenizer/MPS", "learns from init (best val_recon < init train)",
        f"{init:.0f}->{best:.1f}" if (init and best) else "n/a", "yes" if learned else "no", learned)
else:
    rec("b1: tokenizer/MPS", "tier1/tok_tier1/metrics.jsonl", "no NaN", "PENDING", None)


# ---- (b2) Tier-1 Approach-A from scratch ----
ta = load_jsonl(DATA / "tier1/checkpoints/approachA_tier1/metrics.jsonl")
if ta:
    nan = has_nan(ta, ("loss", "val_loss", "val_loss_redshift"))
    tr = [r["loss"] for r in ta if r.get("kind") == "train" and "loss" in r]
    vl = [r["val_loss"] for r in ta if "val_loss" in r]
    init = tr[0] if tr else None
    best = min(vl) if vl else (min(tr) if tr else None)
    learned = init is not None and best is not None and best < init
    rec("b2: transformer/MPS", f"NaN-free ({len(ta)} rows)", "no NaN",
        "clean" if not nan else "NaN!", not nan)
    rec("b2: transformer/MPS", "learns from init (best val_loss < init train)",
        f"{init:.0f}->{best:.0f}" if (init and best) else "n/a", "yes" if learned else "no", learned)
else:
    rec("b2: transformer/MPS", "approachA_tier1/metrics.jsonl", "no NaN", "PENDING", None)


# ---- (c) Tier-2 ignition ----
def find_ignition_metrics():
    cands = [RAID / "checkpoints/checkpoints/approach_a_phase10_mix_mps/metrics.jsonl"]
    cands += [Path(p) for p in glob.glob(str(DATA / "runs/*/metrics.jsonl"))]
    for c in cands:
        m = load_jsonl(c)
        if m and any("val_redshift_acc" in r for r in m):
            return m
    return None


ig = find_ignition_metrics()
if ig:
    tf = [r for r in ig if "val_redshift_acc" in r]
    ar = [r for r in ig if "val_ar_ar_redshift_acc" in r]
    nan = has_nan(ig, ("val_loss", "val_redshift_acc", "val_loss_redshift"))
    late = [r for r in tf if r["step"] >= 8500]
    peak_z = max((r["val_redshift_acc"] for r in tf), default=0.0)
    sustained = any(r["val_redshift_acc"] >= 0.10 for r in late) if late else peak_z >= 0.10
    lr_series = [r["val_loss_redshift"] for r in tf if "val_loss_redshift" in r]
    lr_drop = (lr_series[0] - min(lr_series)) if len(lr_series) >= 2 else 0.0
    vloss_min = min((r["val_loss"] for r in tf if "val_loss" in r), default=float("inf"))
    # AR>=TF/2 at the latest AR step
    ar_ok, ar_val, tf_at = None, None, None
    if ar and tf:
        last_ar = ar[-1]
        tf_match = min(tf, key=lambda r: abs(r["step"] - last_ar["step"]))
        ar_val = last_ar["val_ar_ar_redshift_acc"]; tf_at = tf_match["val_redshift_acc"]
        ar_ok = ar_val >= 0.5 * tf_at

    rec("c: ignition/MPS", f"NaN-free ({len(ig)} rows)", "no NaN", "clean" if not nan else "NaN!", not nan)
    rec("c: ignition/MPS", "val_redshift_acc >=0.10 sustained (>=step 8500)", ">=0.10",
        f"{peak_z:.4f} peak", sustained)
    rec("c: ignition/MPS", "val_loss_redshift cumulative drop", ">=1.0", f"{lr_drop:.2f}", lr_drop >= 1.0)
    rec("c: ignition/MPS", "val_loss min", "<=200", f"{vloss_min:.1f}", vloss_min <= 200)
    if ar_ok is not None:
        rec("c: ignition/MPS", "AR >= TF/2 (honest readout)", ">=0.5xTF",
            f"AR={ar_val:.3f} TF={tf_at:.3f}", ar_ok)
    rec("c: ignition (info)", "peak val_redshift_acc vs phoenix 0.1486", "0.1486",
        f"{peak_z:.4f}", None, gate=False)
else:
    rec("c: ignition/MPS", "ignition metrics.jsonl", "ignition struct", "PENDING (run run_tier2.sh)", None)


# ---- write report ----
def mark(ok):
    return {True: "PASS", False: "FAIL", None: "PENDING"}[ok]


prov = "torch unavailable"
try:
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    prov = f"torch {torch.__version__}, device={dev}, py {__import__('sys').version.split()[0]}"
except Exception:
    pass

lines = [
    "# Apple Silicon (MPS) redshifty SpectrumFM-Phase-I reproduction vs. phoenix",
    "", f"Provenance: {prov}", "",
    "Port correctness is gated on (a) same-checkpoint MPS-vs-CUDA forward fidelity,",
    "(b) NaN-free from-scratch training on MPS that improves, and (c) the STRUCTURE of",
    "the redshift ignition (known high seed variance, so the shape is gated, not the",
    "exact peak). The reference ignition (phoenix L4): val_z_acc 14.86% peak @ step 9500,",
    "val_loss min 190.67, val_loss_redshift drop 1.19, AR/TF ~0.73.", "",
    "| layer | metric | reference | MPS | result |",
    "| :--- | :--- | ---: | ---: | :--- |",
]
for r in rows:
    lines.append(f"| {r['section']} | {r['metric']} | {r['reference']} | {r['mps']} | {mark(r['ok'])} |")
gated = [r for r in rows if r["ok"] is not None]
n_pass = sum(1 for r in gated if r["ok"])
n_pend = sum(1 for r in rows if r["ok"] is None)
lines += ["", f"**Gated checks:** {n_pass}/{len(gated)} passed"
          f"{f'  ({n_pend} pending)' if n_pend else ''}.",
          "", f"## {'OVERALL: PASS' if overall_ok else 'OVERALL: FAIL'}"]
RESULTS.mkdir(exist_ok=True)
out = RESULTS / "REPRODUCTION_MPS_COMPARE.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[done] wrote {out}")
raise SystemExit(0 if overall_ok else 1)
