#!/usr/bin/env python3
"""
06_compare_architectures.py — Phase 4a headline.

Head-to-head comparison of the L18 (Huang+2020) ResNet vs the shielded
(Huang+2021) ResNet, trained on the IDENTICAL cutouts / positives / negatives /
seed / split, for both the DR9 and DR7 cutout sets. Reproduces the paper's
controlled claim (§3.3): the shielded net matches or beats the L18 validation
AUC (0.992 -> 0.997) with ~50x fewer parameters.

Reads:
  ../huang-2020/data/test_result.json       (L18, DR9)   via symlinked sibling
  ../huang-2020/data/test_result_dr7.json    (L18, DR7)
  data/test_result_shielded_dr9.json         (shielded, DR9)
  data/test_result_shielded_dr7.json         (shielded, DR7)

Writes:
  data/arch_comparison.csv
  papers/figures/arch_comparison.png   (param count vs test AUC, both DRs)
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
H2020 = HERE.parent.parent / "huang-2020" / "data"
FIGDIR = HERE / "papers" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

L18_PARAMS = 3_508_833
SHIELDED_PARAMS = 59_905

# (label, arch, dr, params, result_json)
RUNS = [
    ("L18 / DR9",      "L18",      "DR9", L18_PARAMS,      H2020 / "test_result.json"),
    ("L18 / DR7",      "L18",      "DR7", L18_PARAMS,      H2020 / "test_result_dr7.json"),
    ("shielded / DR9", "shielded", "DR9", SHIELDED_PARAMS, DATA / "test_result_shielded_dr9.json"),
    ("shielded / DR7", "shielded", "DR7", SHIELDED_PARAMS, DATA / "test_result_shielded_dr7.json"),
]


def main() -> None:
    rows = []
    for label, arch, dr, params, path in RUNS:
        if not path.exists():
            print(f"[warn] missing {path} — skipping {label}")
            continue
        r = json.loads(path.read_text())
        rows.append({
            "label": label, "arch": arch, "dr": dr, "params": params,
            "val_auc": r.get("best_val_auc"), "test_auc": r.get("test_auc"),
            "best_epoch": r.get("best_epoch"), "n_test": r.get("n_test"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no test_result JSONs found")
    df.to_csv(DATA / "arch_comparison.csv", index=False)

    print("\n[arch comparison]  (same data/seed/split; architecture is the only variable)")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(df[["label", "params", "val_auc", "test_auc", "best_epoch"]].to_string(index=False))

    # Paired DR deltas (shielded - L18).
    for dr in ("DR9", "DR7"):
        sub = df[df["dr"] == dr].set_index("arch")
        if {"L18", "shielded"} <= set(sub.index):
            d_val = sub.loc["shielded", "val_auc"] - sub.loc["L18", "val_auc"]
            d_test = sub.loc["shielded", "test_auc"] - sub.loc["L18", "test_auc"]
            ratio = L18_PARAMS / SHIELDED_PARAMS
            print(f"\n[{dr}] shielded vs L18:  Δval_auc={d_val:+.4f}  Δtest_auc={d_test:+.4f}  "
                  f"params {L18_PARAMS:,} -> {SHIELDED_PARAMS:,} ({ratio:.0f}x fewer)")

    # Figure: param count (log x) vs test AUC, marker per arch, colour per DR.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    markers = {"L18": "o", "shielded": "s"}
    colors = {"DR9": "#2c7bb6", "DR7": "#d7191c"}
    for _, r in df.iterrows():
        ax.scatter(r["params"], r["test_auc"], s=120,
                   marker=markers[r["arch"]], color=colors[r["dr"]],
                   edgecolor="black", zorder=3,
                   label=f"{r['arch']} / {r['dr']}")
        ax.annotate(f"{r['test_auc']:.4f}", (r["params"], r["test_auc"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("held-out test AUC")
    ax.set_title("Shielded vs L18 ResNet — same training data")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "arch_comparison.png", dpi=140)
    plt.close(fig)
    print(f"\n[done] wrote data/arch_comparison.csv + {FIGDIR/'arch_comparison.png'}")


if __name__ == "__main__":
    main()
