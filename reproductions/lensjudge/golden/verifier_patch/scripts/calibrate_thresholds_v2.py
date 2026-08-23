"""Calibrate the v2 letter thresholds for one model key from a calibration run.

Letters are set by FPR on a fixed negative set, never by a vote count: t_A is the smallest S
with FPR <= 1% on the calibration negatives, t_B the smallest S with FPR <= 5%. The
shipped thresholds_v2.json carries `provisional` numbers from the Sonnet-API design half;
until this script has been run for `opus_claude_code` every letter is stamped
letter_source=sonnet_thresholds_uncalibrated.

Run `08d --select ids-file --ids-file scripts/calibration_ids.csv` through the three
workflow stages first, then:

  python scripts/calibrate_thresholds_v2.py --model-key opus_claude_code [--write]

Reads results/results_v2.csv; the `negative` rows of calibration_ids.csv set the thresholds,
the `cowls` rows report recall at each threshold (never used to set them).
"""
import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import util  # noqa: E402

BASE = util.BASE
HERE = os.path.dirname(os.path.abspath(__file__))


def threshold_at_fpr(neg_scores, fpr):
    """Smallest score t such that the fraction of negatives with S >= t is <= fpr."""
    s = sorted(float(x) for x in neg_scores if x == x)
    if not s:
        return None
    n = len(s)
    allowed = int(math.floor(fpr * n))
    desc = s[::-1]
    if allowed >= n:
        return round(min(s), 4)
    t = desc[allowed] + 1e-6                 # just above the (allowed+1)-th highest negative
    return math.ceil(t * 1e4) / 1e4


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--results", default=f"{BASE}/results/results_v2.csv")
    ap.add_argument("--ids", default=os.path.join(HERE, "calibration_ids.csv"))
    ap.add_argument("--thresholds", default=os.path.join(HERE, "thresholds_v2.json"))
    ap.add_argument("--fpr-a", type=float, default=0.01)
    ap.add_argument("--fpr-b", type=float, default=0.05)
    ap.add_argument("--write", action="store_true", help="update thresholds_v2.json in place")
    a = ap.parse_args()

    res = pd.read_csv(a.results, dtype={"id": str}).set_index("id")
    ids = pd.read_csv(a.ids, comment="#", dtype=str)
    neg = ids[ids["role"] == "negative"]["id"]
    pos = ids[ids["role"] == "cowls"]["id"]
    sn = res.reindex(neg)["S"].dropna()
    sp = res.reindex(pos)["S"].dropna()
    print(f"negatives scored: {len(sn)}/{len(neg)}; COWLS scored: {len(sp)}/{len(pos)}", flush=True)
    if len(sn) < 50:
        print("WARNING: fewer than 50 scored negatives - thresholds would be noise; not writing", flush=True)
        a.write = False
    t_A = threshold_at_fpr(sn, a.fpr_a)
    t_B = threshold_at_fpr(sn, a.fpr_b)
    out = {"t_A": t_A, "t_B": t_B, "n_neg": int(len(sn)), "n_pos": int(len(sp)),
           "fpr_A": float((sn >= t_A).mean()) if t_A is not None else None,
           "fpr_B": float((sn >= t_B).mean()) if t_B is not None else None,
           "recall_A": float((sp >= t_A).mean()) if t_A is not None and len(sp) else None,
           "recall_B": float((sp >= t_B).mean()) if t_B is not None and len(sp) else None}
    print(json.dumps(out, indent=1), flush=True)
    if a.write and t_A is not None and t_B is not None:
        thr = json.load(open(a.thresholds))
        prev = thr.get(a.model_key) or {}
        thr[a.model_key] = {"tau0": prev.get("tau0", thr["provisional"]["tau0"]), "t_A": t_A, "t_B": t_B,
                            "n_neg": out["n_neg"], "n_pos": out["n_pos"],
                            "calibrated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                            "results_sha16": hashlib.sha256(open(a.results, "rb").read()).hexdigest()[:16]}
        json.dump(thr, open(a.thresholds, "w"), indent=1)
        print(f"wrote {a.thresholds} [{a.model_key}] -> letters will carry letter_source={a.model_key}_calibrated", flush=True)


if __name__ == "__main__":
    main()
