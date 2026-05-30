#!/usr/bin/env python3
"""
08_compare_models.py — Phase-5 controlled comparison (the headline table/figure).

Assemble the head-to-head AUCs of the two base models + the meta-learner +
the simple-average baseline, all trained/evaluated on the IDENTICAL split, and
compare to Inchausti+2025 Fig. 6 (val AUC: ResNet 0.9984, EfficientNet 0.9987,
meta-learner 0.9989 == average 0.9989). Also places the Huang+2021 60K shielded
and (optionally) the L18 ResNet for the params-vs-AUC view.

Reads (skips with a warning if absent):
  data/test_result_shielded194k.json
  data/test_result_efficientnet.json
  data/meta_metrics.json
  ../huang-2021/data/test_result_shielded_dr9.json   (60K shielded baseline)
  ../huang-2020/data/test_result.json                (L18 baseline)

Writes:
  data/model_comparison.csv
  papers/figures/ensemble_auc.png    (Fig-6-style bars + params-vs-AUC scatter)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
H2020 = HERE.parent / "huang-2020" / "data"
H2021 = HERE.parent / "huang-2021" / "data"
FIGDIR = HERE / "papers" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

PAPER = {"ResNet": 0.9984, "EfficientNet": 0.9987, "Meta-learner": 0.9989, "Average": 0.9989}


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    rows = []

    def add(model, params, val_auc, test_auc, note=""):
        rows.append({"model": model, "params": params, "val_auc": val_auc,
                     "test_auc": test_auc, "note": note})

    sh = jload(DATA / "test_result_shielded194k.json")
    if sh:
        add("ResNet (shielded 194K)", sh["n_params"], sh["best_val_auc"], sh["test_auc"],
            "Inchausti base model 1")
    eff = jload(DATA / "test_result_efficientnet.json")
    if eff:
        add("EfficientNetV2-S", eff["n_params"], eff["best_val_auc"], eff["test_auc"],
            "Inchausti base model 2")
    meta = jload(DATA / "meta_metrics.json")
    if meta:
        add("Meta-learner (FWLS)", 1201, meta["val"]["p_meta"], meta["test"]["p_meta"],
            "ensemble: 2->300->1 MLP")
        add("Simple average", 0, meta["val"]["p_avg"], meta["test"]["p_avg"],
            "parameterless baseline")
    sh60 = jload(H2021 / "test_result_shielded_dr9.json")
    if sh60:
        add("shielded 60K (Huang 2021)", sh60["n_params"], sh60["best_val_auc"],
            sh60["test_auc"], "Phase-4a baseline")
    l18 = jload(H2020 / "test_result.json")
    if l18:
        add("L18 ResNet (Huang 2020)", 3_508_833, l18.get("best_val_auc"),
            l18.get("test_auc"), "Phase-3a baseline")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no model results found — run 05/06/07 first")
    df.to_csv(DATA / "model_comparison.csv", index=False)

    print("\n[controlled comparison — identical split; the model is the only variable]")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(df[["model", "params", "val_auc", "test_auc", "note"]].to_string(index=False))

    if meta:
        print("\n[vs paper Fig. 6 — validation AUC]")
        ours = {"ResNet": meta["val"]["p_resnet"], "EfficientNet": meta["val"]["p_effnet"],
                "Meta-learner": meta["val"]["p_meta"], "Average": meta["val"]["p_avg"]}
        for k in ("ResNet", "EfficientNet", "Meta-learner", "Average"):
            print(f"   {k:<13s} ours {ours[k]:.4f}   paper {PAPER[k]:.4f}   Δ {ours[k]-PAPER[k]:+.4f}")
        d = meta["val"]["p_meta"] - meta["val"]["p_avg"]
        print(f"\n   meta − average (val) = {d:+.5f}  "
              f"(paper: equal — correlated bases mean stacking ≈ averaging)")

    # Figure: (A) Fig-6-style val-AUC bars; (B) params vs test AUC scatter.
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.3))
    if meta:
        order = ["ResNet", "EfficientNet", "Meta-learner", "Average"]
        ours = [meta["val"]["p_resnet"], meta["val"]["p_effnet"],
                meta["val"]["p_meta"], meta["val"]["p_avg"]]
        papv = [PAPER[k] for k in order]
        import numpy as np
        xs = np.arange(len(order)); w = 0.38
        axa.bar(xs - w / 2, ours, w, label="this reproduction", color="#2c7bb6")
        axa.bar(xs + w / 2, papv, w, label="Inchausti 2025", color="#fdae61")
        axa.set_xticks(xs); axa.set_xticklabels(order, rotation=15, ha="right")
        lo = min(min(ours), min(papv)) - 0.002
        axa.set_ylim(lo, 1.0005)
        axa.set_ylabel("validation AUC"); axa.set_title("Ensemble AUC vs paper Fig. 6")
        axa.legend(fontsize=8); axa.grid(True, axis="y", alpha=0.3)

    sc = df[df["params"] > 0]
    axb.scatter(sc["params"], sc["test_auc"], s=90, color="#2c7bb6",
                edgecolor="black", zorder=3)
    for _, r in sc.iterrows():
        axb.annotate(r["model"].split(" (")[0], (r["params"], r["test_auc"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)
    axb.set_xscale("log"); axb.set_xlabel("trainable parameters")
    axb.set_ylabel("held-out test AUC"); axb.set_title("Test AUC vs model size")
    axb.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "ensemble_auc.png", dpi=140)
    plt.close(fig)
    print(f"\n[done] wrote data/model_comparison.csv + {FIGDIR/'ensemble_auc.png'}")


if __name__ == "__main__":
    main()
