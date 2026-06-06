#!/usr/bin/env python3
"""25_calibrate_members.py — isotonic-calibrate each member on the val split and
write a calibrated column `pc` back into each scores_member_<name>.parquet.

Puts every member on a common probability axis (so the average is meaningful and
ECE is honest) before combining. Calibrator is fit on val only.

    /home2/benson/.venvs/claudenet/bin/python 25_calibrate_members.py
"""
from __future__ import annotations

import json

import _clib as C
import _combine as CM
import _ensemble as E


def main():
    scores = CM.load_scores()
    report = {}
    for n, df in scores.items():
        val = df[df.split == "val"]
        cal = E.make_calibrator("isotonic").fit(val["p"].to_numpy(), val["label"].to_numpy())
        df = df.copy()
        df["pc"] = cal.transform(df["p"].to_numpy())
        pre = E.ece(val["label"].to_numpy(), val["p"].to_numpy())
        post = E.ece(val["label"].to_numpy(), cal.transform(val["p"].to_numpy()))
        report[n] = {"ece_val_pre": pre, "ece_val_post": post}
        df.to_parquet(C.DATA / f"scores_member_{n}.parquet", index=False)
        print(f"[cal] {n:12s} ECE val {pre:.4f} -> {post:.4f}")
    (C.DATA / "calibration.json").write_text(json.dumps(report, indent=2))
    print(f"[25] calibrated {len(scores)} members")


if __name__ == "__main__":
    main()
