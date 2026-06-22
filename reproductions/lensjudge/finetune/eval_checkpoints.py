#!/usr/bin/env python3
"""finetune/eval_checkpoints.py — lever 2: select the LoRA checkpoint by DETECTION AUC (not token-acc).

v1's failure was selecting by eval_token_acc (JSON-structure tokens) -> shipped an inverted model.
This runs `swift infer` for each checkpoint on the 3-way-disjoint val-select detection set, parses the
student's p_lens, and computes detection AUC vs the TRUE labels. Prints a table; the best-AUC checkpoint
is the one to merge + gate on the 270-row eval.

  python -m lensjudge.finetune.eval_checkpoints \
    --ckpt-dir outputs/sft_v2/ckpt/v0-XXXX --valset outputs/sft_v2/valsel.jsonl \
    --labels outputs/sft_v2/valsel_labels.csv --gpu 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from lensjudge.common import parse  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402

SWIFT = os.environ.get("SWIFT", "/home2/benson/.venvs/ljtrain/bin/swift")


def _infer(ckpt: str, valset: str, result_path: str, gpu: int):
    cmd = [SWIFT, "infer", "--adapters", ckpt, "--val_dataset", valset, "--result_path", result_path,
           "--max_new_tokens", "512", "--temperature", "0", "--attn_impl", "sdpa",
           "--torch_dtype", "float16", "--quant_bits", "4", "--quant_method", "bnb"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "HF_HOME": os.environ.get("HF_HOME", "/home2/benson/.cache/huggingface"),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [infer FAILED] {r.stderr[-500:]}")
        return False
    return True


def _plens_from_result(result_path: str) -> list[float]:
    out = []
    for line in open(result_path):
        o = json.loads(line)
        resp = o.get("response") or o.get("infer_response") or ""
        if not resp and isinstance(o.get("messages"), list):
            resp = o["messages"][-1].get("content", "")
        g = parse.parse_model(resp, ImageGrade)
        out.append(float(g.p_lens) if g else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="dir containing checkpoint-* subdirs")
    ap.add_argument("--valset", required=True, help="valsel.jsonl (ms-swift messages format)")
    ap.add_argument("--labels", required=True, help="valsel_labels.csv (idx,label,name in valset order)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--result-dir", default=None)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels)
    y = (lab["label"] == "lens").astype(int).tolist()
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "checkpoint-*")),
                   key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    rdir = Path(args.result_dir or (Path(args.ckpt_dir) / "valsel_infer"))
    rdir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {len(ckpts)} checkpoints, valset n={len(y)} ({sum(y)} lens)")

    results = []
    for ck in ckpts:
        step = int(re.search(r"checkpoint-(\d+)", ck).group(1))
        rp = str(rdir / f"infer_{step}.jsonl")
        if not _infer(ck, args.valset, rp, args.gpu):
            continue
        p = _plens_from_result(rp)
        if len(p) != len(y):
            print(f"  step {step}: result rows {len(p)} != labels {len(y)} — skip"); continue
        try:
            auc = roc_auc_score(y, p)
        except Exception as e:
            print(f"  step {step}: AUC error {e}"); continue
        results.append((step, round(float(auc), 4)))
        print(f"  step {step:4d}: detection AUC = {auc:.4f}")

    if results:
        best = max(results, key=lambda t: t[1])
        print(f"\n[eval] BEST checkpoint: step {best[0]} (val detection AUC {best[1]})")
        print(f"[eval] -> merge {args.ckpt_dir}/checkpoint-{best[0]} and gate on the 270-row eval")
    else:
        print("\n[eval] no checkpoint produced a usable AUC")


if __name__ == "__main__":
    main()
