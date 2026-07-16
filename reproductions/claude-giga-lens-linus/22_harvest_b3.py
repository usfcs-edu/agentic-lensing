#!/usr/bin/env python
"""22_harvest_b3.py -- B3 harvest (RESTART layout): figs FIRST, then frozen gates.

Supersedes 22_run_b3_harvest.py after the 2026-07-16 restart (see the dated entry
in research/checkpoints_b3.md): the A16s OOM the reference arm at every
pre-registered fallback size, so ALL FOUR cells now run on the L4 (GPU 8, devtag
l4-8), ref THEN smc sequentially -- same-device wall-clock pairing preserved per
cell. The A16-vs-L4 cross-device check is DROPPED (no A16 reference is possible;
recorded as a deviation, gates untouched).

Consumes data/b3_run_{ref,smc}_sys{i}_l4-8.json + .npz written by 22_run_b3.py.
Gate definitions are FROZEN in research/checkpoints_b3.md (written before runs):

  accuracy: worst-param |z| < 3,
            z_j = (mean_SMC,j - mean_REF,j) / sqrt(var_REF,j/ESS_REF,j
                                                   + var_SMC,j/ESS_SMC)
            ESS_REF,j = tfp per-param cross-chain ESS (stored by the runner);
            ESS_SMC   = final unique-particle count.
  cost    : grad-ratio RANGE [SMC/REF_hi, SMC/REF_lo] (HMC leapfrog bounds) and
            same-device wall ratio; bands WIN<0.7 / PARITY [0.7,2) / LOSS>=2;
            straddling range => PARITY-AMBIGUOUS.
  reference acceptance / SMC sanity: as pre-registered.

Usage:
  python 22_harvest_b3.py --figs-only   # draw figs (inspect BEFORE gates)
  python 22_harvest_b3.py               # figs + gates + data/b3_cells.json
CPU is fine (transforms only): CGL2_ALLOW_CPU=1 JAX_PLATFORMS=cpu.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import cgl2  # noqa: F401,E402
from cgl2 import paths  # noqa: E402

CELLS = [(i, "l4-8") for i in range(4)]   # restart layout: all cells on the L4
REF_C, SMC_C = "#1f77b4", "#ff7f0e"   # 2-series CVD-safe pair; fixed assignment
BANDS = ((0.7, "WIN"), (2.0, "PARITY"))


def _load(arm, sys_idx, tag):
    p = paths.DATA_DIR / f"b3_run_{arm}_sys{sys_idx}_{tag}.json"
    if not p.exists():
        return None, None
    row = json.load(open(p))
    npz_p = paths.DATA_DIR / f"b3_{arm}_sys{sys_idx}_{tag}.npz"
    npz = np.load(npz_p) if npz_p.exists() else None
    return row, npz


def _accepted_hmc(ref_row):
    return (ref_row["hmc_escalated"]
            if ref_row.get("accepted_stage") == "hmc_escalated"
            else ref_row["hmc"])


def _band(x):
    for hi, name in BANDS:
        if x < hi:
            return name
    return "LOSS"


def _grad_band(lo_ratio, hi_ratio):
    b1, b2 = _band(lo_ratio), _band(hi_ratio)
    return b1 if b1 == b2 else f"PARITY-AMBIGUOUS ({b1}..{b2})"


# --------------------------------------------------------------------------- #
# figs (BEFORE gate math)
# --------------------------------------------------------------------------- #
def draw_figs(pairs, names_by_sys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths.FIGS_DIR.mkdir(exist_ok=True)
    for (sys_idx, tag), (ref_row, ref_npz, smc_row, smc_npz) in pairs.items():
        if ref_npz is None or smc_npz is None:
            continue
        chains = np.asarray(ref_npz["chains"])          # (res, n_chains, dim)
        ref_flat = chains.reshape(-1, chains.shape[-1])
        parts = np.asarray(smc_npz["particles"])        # (512, dim)
        dim = parts.shape[1]
        names = names_by_sys.get(sys_idx) or [f"z{j}" for j in range(dim)]

        fig = plt.figure(figsize=(13, 7.5))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.42,
                              wspace=0.3)
        # top: z-space ladder, standardized to the REF posterior sd
        ax = fig.add_subplot(gs[0, :])
        mu_r, sd_r = ref_flat.mean(0), ref_flat.std(0) + 1e-300
        mu_s, sd_s = parts.mean(0), parts.std(0)
        x = np.arange(dim)
        ax.axhline(0, color="0.75", lw=0.8, zorder=0)
        ax.errorbar(x - 0.12, (mu_r - mu_r) / sd_r, yerr=np.ones(dim),
                    fmt="o", ms=4, lw=1.6, color=REF_C, capsize=2,
                    label="reference (MAP→SVI→HMC)")
        ax.errorbar(x + 0.12, (mu_s - mu_r) / sd_r, yerr=sd_s / sd_r,
                    fmt="o", ms=4, lw=1.6, color=SMC_C, capsize=2,
                    label="MC-SMC (MAMS, N=512, prior-seeded)")
        ax.set_xticks(x)
        ax.set_xticklabels([n.split("/")[-1] + "\n" + "/".join(n.split("/")[:3])
                            if "/" in n else n for n in names],
                           fontsize=5.5, rotation=90)
        ax.set_ylabel("(mean − ref mean) / ref sd\n(±1 posterior sd)")
        ax.set_title(f"B3 hs2_sys{sys_idx} [{tag}]: z-space posterior overlay "
                     "(marginal mean ± sd, standardized to reference)")
        ax.grid(alpha=0.15, axis="y")
        ax.legend(frameon=False, fontsize=8, loc="upper right")

        # bottom: 4 key physical/unconstrained marginals as histograms
        key_idx = _key_param_indices(names)
        for k, (j, label) in enumerate(key_idx):
            axh = fig.add_subplot(gs[1, k])
            lo = min(ref_flat[:, j].min(), parts[:, j].min())
            hi = max(ref_flat[:, j].max(), parts[:, j].max())
            bins = np.linspace(lo, hi, 40)
            axh.hist(ref_flat[:, j], bins=bins, density=True, color=REF_C,
                     alpha=0.55, label="ref")
            axh.hist(parts[:, j], bins=bins, density=True, color=SMC_C,
                     alpha=0.55, label="smc")
            zt = smc_row.get("z_true") or ref_row.get("z_true")
            if zt is not None:
                axh.axvline(zt[j], color="0.2", lw=1.0, ls="--")
            axh.set_title(label, fontsize=8)
            axh.grid(alpha=0.15)
            if k == 0:
                axh.legend(frameon=False, fontsize=7)
        fig.text(0.01, 0.01, "dashed: injected truth (sanity only, mock is ours)",
                 fontsize=7, color="0.4")
        out = paths.FIGS_DIR / f"b3_sys{sys_idx}_overlay_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[figs] wrote {out}")

    # cost figure (one, all cells): grads (ranges) + wall
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    labels, g_smc, g_lo, g_hi, w_smc, w_ref = [], [], [], [], [], []
    for (sys_idx, tag), (ref_row, _rn, smc_row, _sn) in pairs.items():
        if ref_row is None or smc_row is None:
            continue
        labels.append(f"sys{sys_idx}\n{tag}")
        g_smc.append(smc_row["grad_evals"]["total"])
        ga = ref_row["grad_evals_accepted"]
        g_lo.append(ga["total_lo"]); g_hi.append(ga["total_hi"])
        w_smc.append(smc_row["wall_s"]); w_ref.append(ref_row["wall_s"])
    x = np.arange(len(labels))
    a1.bar(x - 0.18, g_smc, 0.32, color=SMC_C, label="MC-SMC (exact count)")
    a1.bar(x + 0.18, g_hi, 0.32, color=REF_C, alpha=0.45,
           label="reference (billing range)")
    a1.bar(x + 0.18, g_lo, 0.32, color=REF_C)
    a1.set_yscale("log"); a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel("gradient evaluations"); a1.grid(alpha=0.15, axis="y")
    a1.set_title("cost: grad-evals (ref = [init_l, max_leapfrog] bounds)")
    a1.legend(frameon=False, fontsize=8)
    a2.bar(x - 0.18, np.asarray(w_smc) / 3600, 0.32, color=SMC_C, label="MC-SMC")
    a2.bar(x + 0.18, np.asarray(w_ref) / 3600, 0.32, color=REF_C,
           label="reference")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("wall-clock [h] (same device per pair)")
    a2.grid(alpha=0.15, axis="y"); a2.set_title("cost: wall-clock")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = paths.FIGS_DIR / "b3_cost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figs] wrote {out}")


def _key_param_indices(names):
    want = [("theta_E", "lens theta_E (z)"), ("mass/0/gamma", "lens gamma (z)"),
            ("mass/1/gamma1", "shear gamma1 (z)"),
            ("planes/1/light/0/R_sersic", "src R_sersic (z)")]
    out = []
    for pat, label in want:
        for j, n in enumerate(names):
            if pat in n:
                out.append((j, label))
                break
    while len(out) < 4 and names:
        j = len(out)
        out.append((j, names[j]))
    return out[:4]


# --------------------------------------------------------------------------- #
# gates (AFTER figs)
# --------------------------------------------------------------------------- #
def eval_cell(sys_idx, tag, ref_row, ref_npz, smc_row, smc_npz):
    cell = dict(sys=sys_idx, devtag=tag, target=f"hs2_sys{sys_idx}")
    if ref_row is None or smc_row is None:
        cell["status"] = "MISSING-RUN"
        return cell
    if ref_row.get("status") != "OK":
        cell["status"] = "BLOCKED-BY-REFERENCE (runner ERROR)"
        cell["ref_error"] = ref_row.get("error")
        return cell
    if not ref_row.get("reference_accepted", False):
        h = _accepted_hmc(ref_row)
        cell["status"] = "BLOCKED-BY-REFERENCE"
        cell["ref_diag"] = dict(rhat_worst=h["rhat_worst"], ess_min=h["ess_min"])
        return cell
    if smc_row.get("status") != "OK":
        cell["status"] = "SMC-ERROR"
        cell["smc_error"] = smc_row.get("error")
        return cell

    chains = np.asarray(ref_npz["chains"])
    ref_flat = chains.reshape(-1, chains.shape[-1])
    parts = np.asarray(smc_npz["particles"])
    h = _accepted_hmc(ref_row)
    ess_ref = np.asarray(h["ess"], dtype=np.float64)
    ess_smc = float(smc_row["n_unique_final"])

    mu_r, var_r = ref_flat.mean(0), ref_flat.var(0)
    mu_s, var_s = parts.mean(0), parts.var(0)
    se = np.sqrt(var_r / ess_ref + var_s / ess_smc)
    z = (mu_s - mu_r) / se
    worst = int(np.argmax(np.abs(z)))

    # SMC sanity (pre-registered)
    smc_sane = ess_smc >= 32
    cell["smc_sanity"] = dict(
        n_unique_final=int(ess_smc), min_unique=32, ok=bool(smc_sane),
        n_stages=smc_row["n_stages"],
        final_ess=smc_row["ess_trace"][-1],
        accept_final=smc_row["accept_trace"][-1])

    cell["accuracy"] = dict(
        worst_param_absz=float(np.max(np.abs(z))),
        worst_param_index=worst,
        z=z.tolist(),
        gate="worst-param |z| < 3",
        ess_smc=ess_smc, ess_ref_min=float(ess_ref.min()),
        PASS=bool(np.max(np.abs(z)) < 3.0))

    ga = ref_row["grad_evals_accepted"]
    g_smc = float(smc_row["grad_evals"]["total"])
    r_lo, r_hi = g_smc / ga["total_hi"], g_smc / ga["total_lo"]
    w_ratio = float(smc_row["wall_s"] / ref_row["wall_s"])
    cell["cost"] = dict(
        smc_grads=g_smc, smc_weight_logliks=smc_row["n_logp"],
        ref_grads_lo=ga["total_lo"], ref_grads_hi=ga["total_hi"],
        grad_ratio_range=[r_lo, r_hi],
        grad_verdict=_grad_band(r_lo, r_hi),
        smc_wall_s=smc_row["wall_s"], ref_wall_s=ref_row["wall_s"],
        ref_wall_incl_failed_s=ref_row.get("wall_s_incl_failed"),
        wall_ratio=w_ratio, wall_verdict=_band(w_ratio),
        bands="WIN<0.7 / PARITY[0.7,2) / LOSS>=2")
    cell["evidence"] = dict(
        logZ=smc_row["logZ"], logZ_boot_sigma=smc_row["logZ_boot_sigma"],
        note="SMC-only deliverable; the classic recipe produces no logZ")
    cell["ref_meta"] = dict(
        accepted_stage=ref_row["accepted_stage"],
        rhat_worst=h["rhat_worst"], ess_min=h["ess_min"],
        oom_fallbacks=ref_row.get("oom_fallbacks", []),
        map_n=ref_row["map"]["n_samples"], svi_n=ref_row["svi"]["n_vi"])
    cell["status"] = ("PASS" if (cell["accuracy"]["PASS"] and smc_sane)
                      else "FAIL")
    if not smc_sane:
        cell["status"] += " (SMC population SICK)"
    return cell


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figs-only", action="store_true")
    args = ap.parse_args()

    # z_param_names for labels (CPU build is fine; adapters are data-agnostic
    # for naming). Fall back to indices when a build is unavailable.
    names_by_sys = {}
    try:
        from cgl2 import zoo
        for i in range(4):
            names_by_sys[i] = zoo.build(f"hs2_sys{i}").meta.get("z_param_names")
    except Exception as e:  # noqa: BLE001
        print(f"[harvest] zoo names unavailable ({type(e).__name__}: {e}); "
              "using z-indices")

    pairs = {}
    for sys_idx, tag in CELLS:
        ref_row, ref_npz = _load("ref", sys_idx, tag)
        smc_row, smc_npz = _load("smc", sys_idx, tag)
        pairs[(sys_idx, tag)] = (ref_row, ref_npz, smc_row, smc_npz)

    draw_figs(pairs, names_by_sys)
    if args.figs_only:
        print("[harvest] figs only (inspect before gates -- house rule)")
        return

    cells = []
    for sys_idx, tag in CELLS:
        cells.append(eval_cell(sys_idx, tag, *pairs[(sys_idx, tag)]))

    gpu_h = {}
    for (sys_idx, tag), (ref_row, _a, smc_row, _b) in pairs.items():
        t = 0.0
        for r in (ref_row, smc_row):
            if r is not None:
                t += r.get("wall_total_s", 0.0)
        gpu_h[f"sys{sys_idx}_{tag}"] = round(t / 3600, 3)

    report = dict(
        script="22_harvest_b3.py",
        checkpoint="research/checkpoints_b3.md (pre-registered; 2026-07-16 "
                   "restart entry: all cells l4-8, cross-device check dropped)",
        cross_device_check=dict(
            status="DROPPED (documented deviation)",
            reason="A16 15.3GB OOMs the reference arm at every pre-registered "
                   "fallback size (MAP 500/256/128); no A16 reference exists, "
                   "so no A16-vs-L4 pair is possible"),
        cells=cells,
        gpu_hours_by_lane=gpu_h,
        gpu_hours_total=round(sum(gpu_h.values()), 2),
    )
    out = paths.DATA_DIR / "b3_cells.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"[harvest] wrote {out}")
    for c in cells:
        acc = c.get("accuracy", {})
        cost = c.get("cost", {})
        print(f"  sys{c['sys']} [{c['devtag']}] {c['status']:22s} "
              f"worst|z|={acc.get('worst_param_absz', float('nan')):.3f} "
              f"grad={cost.get('grad_verdict', '-'):18s} "
              f"wall={cost.get('wall_verdict', '-')} "
              f"({cost.get('wall_ratio', float('nan')):.2f}x)")
    print(f"  GPU-h total: {report['gpu_hours_total']}")


if __name__ == "__main__":
    main()
