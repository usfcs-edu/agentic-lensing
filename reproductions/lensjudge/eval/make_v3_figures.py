#!/usr/bin/env python3
"""make_v3_figures.py — the two Part III (v3) figures for papers/.

  v3_hsc_flip.png       — the DESI 1.3" -> HSC PDR3 0.168" p_lens flip on the 25 SuGOHI-matched
        known lenses with HSC coverage (the Euclid-B1 analog at the coarser HSC rung): the
        over-skeptical DESI grader buries them, HSC tier-2 recovers 92% as A/B.
  v3_cascade_recall.png — does the cascade's rank-routing keep the lenses? lens recall vs the
        escalated fraction (pass_frac), with mimic/random escalation, on the 120-row labeled set.

Sources: outputs/hsc_validation.parquet, outputs/cascade_recall_stage1.parquet.
Run:  PYTHONPATH=reproductions python lensjudge/eval/make_v3_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# self-contained (runs under any python with matplotlib/pandas; no lensjudge import)
HERE = Path(__file__).resolve().parents[1]            # reproductions/lensjudge
OUT = HERE / "outputs"
FIG = HERE / "papers" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def top_frac_indices(scores, pass_frac):                # mirror of run_cascade.top_frac_indices
    n = len(scores)
    if n == 0:
        return set()
    n_pass = max(1, int(round(pass_frac * n)))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return set(order[:n_pass])
C_HSC, C_DESI, CHANCE = "#7b2cbf", "#0F4D92", "#c0392b"
C_LENS, C_MIM, C_RAND = "#2e7d32", "#b5651d", "#9aa0a6"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def fig_hsc_flip():
    p = OUT / "hsc_validation.parquet"
    if not p.exists():
        print("[hsc_flip] skip: hsc_validation.parquet missing"); return
    d = pd.read_parquet(p)
    e = d[d["escalated"] == True].copy()                                   # noqa: E712
    t1 = pd.to_numeric(e["p_lens_tier1"], errors="coerce")
    t2 = pd.to_numeric(e["p_lens_tier2"], errors="coerce")
    ok = t1.notna() & t2.notna()
    t1, t2, fg = t1[ok].to_numpy(), t2[ok].to_numpy(), e["final_grade"][ok].to_numpy()
    n = len(t1); m1, m2 = np.median(t1), np.median(t2)
    ab = np.isin(fg, ["A", "B"])

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.axhspan(0.5, 1.02, xmin=0, xmax=0.5 / 1.02, color="#eef0fa", zorder=0)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888", zorder=1)
    ax.axhline(0.5, ls=":", lw=0.8, color="#bbb"); ax.axvline(0.5, ls=":", lw=0.8, color="#bbb")
    ax.scatter(t1[ab], t2[ab], s=42, marker="o", facecolor=C_HSC, edgecolor="white",
               linewidth=0.5, label=f"recovered A/B at HSC  ({int(ab.sum())})", zorder=3)
    ax.scatter(t1[~ab], t2[~ab], s=42, marker="x", color=CHANCE, linewidth=1.6,
               label=f"stayed D  ({int((~ab).sum())})", zorder=3)
    ax.scatter([m1], [m2], s=240, marker="*", color="#d4af37", edgecolor="#222",
               linewidth=0.8, zorder=5)
    ax.annotate(f"median flip\n{m1:.2f} $\\rightarrow$ {m2:.2f}", xy=(m1, m2), xytext=(0.34, 0.32),
                fontsize=9.5, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#222", lw=1.1))
    ax.text(0.25, 0.965, "DESI-buried, HSC-recovered", fontsize=8.5, color=C_HSC,
            ha="center", style="italic")
    ax.set_xlabel(r"DESI tier-1 $p_{\mathrm{lens}}$ ($grz$, $1.3''$)")
    ax.set_ylabel(r"HSC tier-2 $p_{\mathrm{lens}}$ (PDR3, $0.168''$)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title(f"HSC PDR3 tier-2 recovers DESI-buried lenses\n"
                 f"({n} SuGOHI-matched known lenses; {int(ab.sum())}/{n} = {ab.mean():.0%} A/B)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.6)
    fig.tight_layout(); fig.savefig(FIG / "v3_hsc_flip.png"); plt.close(fig)
    print(f"[hsc_flip] wrote {FIG/'v3_hsc_flip.png'}  n={n} median {m1:.3f}->{m2:.3f} A/B={ab.mean():.0%}")


def fig_cascade_recall():
    p = OUT / "cascade_recall_stage1.parquet"
    if not p.exists():
        print("[recall] skip: cascade_recall_stage1.parquet missing"); return
    d = pd.read_parquet(p).reset_index(drop=True)
    is_lens = d["grade_truth"].isin(["A", "B", "C"]).to_numpy()
    mim = d["source"].eq("mimic_neg").to_numpy()
    ran = d["source"].eq("random_neg").to_numpy()
    scores = d["p_lens"].tolist()
    pfs = np.linspace(0.05, 1.0, 20)
    lens_rec, mim_esc, rand_esc = [], [], []
    for pf in pfs:
        idx = np.zeros(len(d), bool)
        idx[list(top_frac_indices(scores, float(pf)))] = True
        lens_rec.append((is_lens & idx).sum() / max(1, is_lens.sum()))
        mim_esc.append((mim & idx).sum() / max(1, mim.sum()))
        rand_esc.append((ran & idx).sum() / max(1, ran.sum()))

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#bbb", label="escalated = pass_frac (no signal)")
    ax.plot(pfs, lens_rec, "-o", ms=3, color=C_LENS, label="real lens recall (A/B/C)")
    ax.plot(pfs, mim_esc, "-s", ms=3, color=C_MIM, label="mimic escalation")
    ax.plot(pfs, rand_esc, "-^", ms=3, color=C_RAND, label="random-galaxy escalation")
    for pf, lr in [(0.3, None), (0.5, None)]:
        j = int(np.argmin(np.abs(pfs - pf)))
        ax.annotate(f"{lens_rec[j]:.0%}", xy=(pfs[j], lens_rec[j]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8, color=C_LENS)
        ax.axvline(pf, ls=":", lw=0.7, color="#ddd", zorder=0)
    ax.set_xlabel("pass_frac (top fraction escalated by the cheap stage-1 rank)")
    ax.set_ylabel("fraction escalated")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Cascade rank-routing keeps lenses, drops randoms\n"
                 "(lens recall above the diagonal; randoms below; mimics ~ baseline)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.4)
    fig.tight_layout(); fig.savefig(FIG / "v3_cascade_recall.png"); plt.close(fig)
    print(f"[recall] wrote {FIG/'v3_cascade_recall.png'}  "
          f"lens_recall@0.5={lens_rec[int(np.argmin(np.abs(pfs-0.5)))]:.0%}")


if __name__ == "__main__":
    fig_hsc_flip()
    fig_cascade_recall()
