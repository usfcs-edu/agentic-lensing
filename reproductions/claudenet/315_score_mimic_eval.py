#!/usr/bin/env python3
"""315_score_mimic_eval.py — ClaudeNet v3 A2: score members on the held-out mimic-eval.

Gathers the frozen held-out mimic-eval rows (312, source==dr10_sweep) straight from the
DR10 .npy shards (no FITS staging) and scores each listed member checkpoint on them with
112_score_pool's array path — the SAME math that produced the bank's member columns — then
applies the member's val-split isotonic calibrator (reproducing 121/25's pc). Writes one
scores_member_<art>_mimiceval.parquet [row_id,p,pc] per member, the negative side of the
recovery@matched-mimic-FPR metric for the v2-vs-v3 head-to-head (316).

Scorelist json entries: {"art","member","ckpt","val_scores"} (member -> variant via
members.json; val_scores -> the parquet whose split==val rows fit the isotonic).

    python 315_score_mimic_eval.py --scorelist data/v3/a2_scorelist.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C
import _ensemble as E

M112 = C._load("cn_315_112", C.ROOT / "112_score_pool.py")


def variant_for(member: str) -> str | None:
    for m in json.load(open(C.DATA / "members.json")):
        if m["name"] == member and m.get("variant"):
            return m["variant"]
    return getattr(M112, "VARIANT_FALLBACK", {}).get(member)


def gather_array(row_ids, parts_glob: str):
    """Gather cutouts for row_ids from the DR10 part shards into one (n,3,101,101) array,
    aligned to the returned id order (rows missing from every index are dropped)."""
    want = list(dict.fromkeys(str(r) for r in row_ids))
    want_set = set(want)
    parts = sorted(glob.glob(parts_glob))
    assert parts, f"no parts under {parts_glob}"
    found = {}
    t0 = time.time()
    for p in parts:
        idx = pd.read_parquet(Path(p) / "index.parquet",
                              columns=["row_id", "shard", "idx_in_shard", "ok"])
        idx["row_id"] = idx["row_id"].astype(str)
        sub = idx[idx["row_id"].isin(want_set) & idx["ok"]]
        if sub.empty:
            continue
        for k, g in sub.groupby("shard"):
            mm = np.load(Path(p) / f"cutouts_{int(k)}.npy", mmap_mode="r")
            order = np.argsort(g["idx_in_shard"].values)
            ids = g["row_id"].values[order]
            cubes = mm[g["idx_in_shard"].values[order]]
            for rid, cube in zip(ids, cubes):
                found[rid] = np.nan_to_num(np.asarray(cube, np.float32), nan=0.0)
            del mm
    ids = [r for r in want if r in found]
    x = np.stack([found[r] for r in ids]).astype(np.float32)
    print(f"[315] gathered {len(ids):,}/{len(want):,} eval cutouts {x.shape} "
          f"({time.time() - t0:.1f}s)")
    return ids, x


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scorelist", required=True)
    ap.add_argument("--bank-eval", default=str(C.DATA / "v3" / "mimic_bank_eval.parquet"))
    ap.add_argument("--parts-glob", default=None)
    ap.add_argument("--out-dir", default=str(C.DATA / "v3"))
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    parts_glob = args.parts_glob or str(
        Path(os.environ.get("SCRATCH", ".")) / "claudenet" / "cutouts" / "dr10" / "part*")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ev = pd.read_parquet(args.bank_eval)
    ev = ev[ev["source"] == "dr10_sweep"]
    ids, x = gather_array(ev["row_id"].tolist(), parts_glob)
    # persist the gathered id order once (so all members align + 316 can join)
    pd.DataFrame({"row_id": ids}).to_parquet(Path(args.out_dir) / "mimic_eval_ids.parquet",
                                             index=False)

    scorelist = json.load(open(args.scorelist))
    for e in scorelist:
        art, member, ckpt, valp = e["art"], e["member"], e["ckpt"], e["val_scores"]
        model, score_arch, mean, std = M112.load_member_checkpoint(
            Path(ckpt), device, variant_for(member))
        mean_t = torch.from_numpy(mean.astype(np.float32))
        std_t = torch.from_numpy(std.astype(np.float32))
        p = np.empty(len(x), np.float32)
        for s in range(0, len(x), args.batch):
            p[s:s + args.batch] = M112.score_array(
                model, score_arch, mean_t, std_t, x[s:s + args.batch], device)
        # isotonic from the member's val split (reproduces 121 score_and_write pc)
        v = pd.read_parquet(valp)
        v = v[v["split"] == "val"]
        cal = E.IsotonicCalibrator().fit(v["p"].to_numpy(float), v["label"].to_numpy(int))
        pc = cal.transform(p.astype(float))
        out = Path(args.out_dir) / f"scores_member_{art}_mimiceval.parquet"
        pd.DataFrame({"row_id": ids, "p": p, "pc": pc.astype(np.float32)}).to_parquet(out, index=False)
        print(f"[315] {art:22s} p[{p.min():.3f},{p.max():.3f}] "
              f"pc_mean={pc.mean():.3f} -> {out.name}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
