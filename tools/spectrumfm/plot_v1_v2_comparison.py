#!/usr/bin/env python
"""
Phase-14 V1-vs-V2 comparison figure + summary.

Left panel : TF redshift_acc trajectory for the three Approach-A arms
             (V1, V2+skips [collapsed], V2 no-skip) — note the z-bin caveat.
Right panel: binning-fair honest redshift metric (from eval_redshift_dz.py):
             fraction of spectra with |dz|/(1+z) below DESI tolerances.

Writes experiments/runs/_comparisons/v1_v2_l4x2_comparison.{png,md}.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = Path("/raid/benson/git/agentic-lensing/experiments/runs")
OUT = RUNS_DIR / "_comparisons"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = [
    ("V1 (256-lvl z)",          "*redshifty_approach_a_mix_l4x2_v1_*",       "#1f77b4"),
    ("V2 +skips (1024-lvl z)",  "*redshifty_approach_a_mix_l4x2_v2_*",       "#d62728"),
    ("V2 no-skip (1024-lvl z)", "*redshifty_approach_a_mix_l4x2_v2noskip_*", "#2ca02c"),
    ("codecs Mamba3+RFSQ (256-lvl z)", "*redshifty_approach_a_mix_l4x2_codecs*", "#9467bd"),
]

# Binning-fair metric from eval_redshift_dz.py (encoder-masked honest readout).
FAIR = {
    "tol": ["<0.0033\n(DESI)", "<0.01", "<0.05"],
    "V1 (256)":        [0.192, 0.264, 0.601],
    "V2 no-skip (1024)": [0.192, 0.262, 0.564],
}


def latest_run(pattern):
    dirs = sorted(RUNS_DIR.glob(pattern))
    return dirs[-1] if dirs else None


def load_traj(run_dir):
    steps, zacc = [], []
    mp = run_dir / "metrics.jsonl"
    for line in mp.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "val" and r.get("val_redshift_acc") is not None:
            steps.append(r["step"]); zacc.append(r["val_redshift_acc"])
    return steps, zacc


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

for label, pat, color in ARMS:
    rd = latest_run(pat)
    if rd is None:
        print(f"[warn] no run dir for {pat}")
        continue
    s, z = load_traj(rd)
    axL.plot(s, z, label=label, color=color, lw=2)
    print(f"{label:<26} {rd.name}  peak z_acc={max(z) if z else float('nan'):.3f}")

axL.set_xlabel("training step"); axL.set_ylabel("TF redshift_acc (exact-bin)")
axL.set_title("Approach-A redshift accuracy (2×L4, eff batch 64, bf16)\n"
              "V1 & codecs are 256-lvl z (directly comparable); V2 arms 1024-lvl")
axL.legend(loc="lower right", fontsize=8); axL.grid(alpha=0.3); axL.set_ylim(-0.02, 0.65)

x = range(len(FAIR["tol"])); w = 0.38
axR.bar([i - w/2 for i in x], FAIR["V1 (256)"], w, label="V1 (256-lvl z)", color="#1f77b4")
axR.bar([i + w/2 for i in x], FAIR["V2 no-skip (1024)"], w, label="V2 no-skip (1024-lvl z)", color="#2ca02c")
axR.set_xticks(list(x)); axR.set_xticklabels(FAIR["tol"])
axR.set_ylabel("fraction of spectra")
axR.set_title("Binning-FAIR redshift (encoder-masked, |dz|/(1+z))\n"
              "V1 vs V2-no-skip: statistically tied")
axR.legend(fontsize=9); axR.grid(alpha=0.3, axis="y")

axR.text(0.5, -0.16, "codecs uses 256-lvl z (same as V1) -> its raw z_acc is\n"
         "directly V1-comparable; no Δz correction needed",
         transform=axR.transAxes, ha="center", va="top", fontsize=8, color="#555")

fig.tight_layout()
png = OUT / "v1_v2_codecs_l4x2_comparison.png"
fig.savefig(png, dpi=130, bbox_inches="tight")
print(f"[saved] {png}")

md = OUT / "v1_v2_codecs_l4x2_comparison.md"
md.write_text(
    "# V1 vs V2 vs codecs tokenizer on 2×L4 (Approach-A, eff batch 64, bf16, 15k)\n\n"
    "| arm | tokenizer | z-bins | TF z_acc peak | AR z_acc peak | codebook | verdict |\n"
    "|---|---|---|---|---|---|---|\n"
    "| V1 | ConvNeXt+LFQ | 256 | 0.55 | 0.57 | 5.00 bits (163 codes) | baseline |\n"
    "| V2 +skips | +U-Net skips/cross-attn | 1024 | 0.00 | 0.00 | 0.00 bits (collapsed) | dead (skip bypass) |\n"
    "| V2 no-skip | +tophat/entropy, no skips | 1024 | 0.50 | 0.48 | 5.24 bits (113 codes) | ties V1 |\n"
    "| codecs | Mamba3+RFSQ (layer0) | 256 | 0.53 | 0.52 | layer0 6.27 bits (233/625) | ties V1 |\n\n"
    "**Findings.** (1) The 2×L4 rerun (bf16, eff batch 64) drives V1 to 55% TF / "
    "57% AR — 3.7× the single-A16 mix run (14.86%). (2) The full V2 tokenizer's "
    "U-Net skips route reconstruction around the quantizer; its discrete codebook "
    "collapses to a single code, so the transformer learns 0% redshift. (3) Skip-free "
    "V2 restores a healthy codebook and ignites, but only TIES V1 on the binning-fair "
    "|dz|/(1+z) metric. (4) The codecs Mamba3+RFSQ tokenizer (layer0, 256-lvl z so "
    "directly V1-comparable) starts ~2.7× slower but catches up by ~step 6k and "
    "plateaus at 0.53 TF / 0.52 AR — also a TIE with V1 (its spec_acc plateaus ~0.26, "
    "i.e. the layer0 codes resist AR prediction yet still carry the redshift signal).\n\n"
    "**Conclusion.** Three structurally different healthy tokenizers (ConvNeXt+LFQ, "
    "ConvNeXt+LFQ+tophat, Mamba3+RFSQ) all converge to ~0.50–0.55 redshift accuracy. "
    "The tokenizer architecture is NOT the bottleneck at this scale once the discrete "
    "codebook is healthy; the lever is data / model size / steps. Reconstruction "
    "quality and codebook entropy gate usability, but do not by themselves buy "
    "downstream accuracy. (codecs caveat: layer0-only — the 2 residual RFSQ layers "
    "were dropped to fit the 1024-code slot; a multi-token expansion could test them.)\n"
)
print(f"[saved] {md}")
