#!/usr/bin/env python3
"""compare_xcheck.py — score the MPS-vs-CUDA same-checkpoint fidelity xcheck.

Consumes data/xcheck_mps.json + data/xcheck_cuda.json (written by xcheck_mps_inference.py
on the Mac and on phoenix over the SAME spectra + weights, fp32) and writes
data/xcheck_compare.json with the verdict.

Gating philosophy (same as the huang-2020/21 ports): a real MPS inference bug corrupts
the BULK (median/argmax/science metrics blow up — cf. the non_blocking-NaN). Benign
fp32-backend rounding leaves the bulk bit-faithful and only perturbs a thin tail of
near-tie / high-magnitude raw logits. So gate the ROBUST + DISCRETE + SCIENCE metrics;
report max|Δ| informationally.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# gates
LOSS_REL_TOL = 1e-3       # relative |Δ loss|
MEDIAN_TOL = 1e-3         # median elementwise |Δ| (bulk bit-faithfulness)
ARGMAX_MIN = 0.995        # discrete-prediction agreement fraction
SCIENCE_TOL = 0.005       # |Δ redshift_acc| — the cross-attention readout


def main() -> None:
    m = json.loads((DATA / "xcheck_mps.json").read_text())
    c = json.loads((DATA / "xcheck_cuda.json").read_text())
    assert m["n_spectra"] == c["n_spectra"], "spectrum-count mismatch — inputs not identical"

    a, b = np.array(m["sample"]), np.array(c["sample"])
    d = np.abs(a - b)
    am, ac = np.array(m["argmax_tokens"]), np.array(c["argmax_tokens"])
    argmax_agree = float((am == ac).mean())
    loss_rel = abs(m["loss"] - c["loss"]) / abs(c["loss"])
    dz = abs(m["metrics"]["redshift_acc"] - c["metrics"]["redshift_acc"])
    dspec = abs(m["metrics"]["spectrum_acc"] - c["metrics"]["spectrum_acc"])
    median = float(np.median(d))

    gates = {
        "loss_rel<=1e-3": (loss_rel, loss_rel <= LOSS_REL_TOL),
        "median|Δ|<=1e-3": (median, median <= MEDIAN_TOL),
        "argmax_agree>=99.5%": (argmax_agree, argmax_agree >= ARGMAX_MIN),
        "|Δredshift_acc|<=0.005": (dz, dz <= SCIENCE_TOL),
    }
    ok = all(p for _, p in gates.values())

    out = {
        "n_spectra": m["n_spectra"],
        "logits_shape": m["logits_shape"],
        "devices": {"mps": m.get("torch"), "cuda": c.get("torch")},
        "gates": {k: {"value": float(v), "pass": bool(p)} for k, (v, p) in gates.items()},
        "informational": {
            "max_abs_delta": float(d.max()),
            "p99_9_abs_delta": float(np.percentile(d, 99.9)),
            "median_abs_delta": median,
            "argmax_agreement": argmax_agree,
            "loss_mps": m["loss"], "loss_cuda": c["loss"], "loss_rel_delta": loss_rel,
            "redshift_acc_mps": m["metrics"]["redshift_acc"], "redshift_acc_cuda": c["metrics"]["redshift_acc"],
            "spectrum_acc_delta": dspec,
            "logits_sum_rel_delta": abs(m["logits_sum"] - c["logits_sum"]) / abs(c["logits_sum"]),
            "logits_absmax_mps": m["logits_absmax"], "logits_absmax_cuda": c["logits_absmax"],
        },
        "overall_pass": ok,
    }
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / "xcheck_compare.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n[xcheck-compare] {'PASS' if ok else 'FAIL'}  "
          f"(median|Δ|={median:.1e}, argmax={argmax_agree*100:.2f}%, "
          f"loss_rel={loss_rel:.1e}, Δz_acc={dz:.4f}; informational max|Δ|={d.max():.2f})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
