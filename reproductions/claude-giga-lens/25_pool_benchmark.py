#!/usr/bin/env python
"""25_pool_benchmark.py -- pool all P2b benchmark cells into the ranked
tables + heatmap + data/bench_report.json. CPU-only (no jax).

Pools data/results/<sampler>/<target>/s<seed>_<track>.{json,npz}:
  * per-target Track-A table: method x [ESS/grad, ESS/s (device-tagged),
    R-hat(mass max), win?] with medians over seeds; win criterion (README
    section P2): median ESS/grad >= 2x S0 AND R-hat < 1.01 AND (multimodal)
    every reference mode recovered with occupancy within 10 pts;
  * T0-mixture mode table (occupancy vs truth, weight error, round trips)
    -- the S0 0.951/0.049 collapse is the reference point;
  * logZ table (bj_smc / nautilus_runner / glnt vs T0 analytic truth);
  * Track-B prefix-convergence analysis: smallest sample prefix reaching
    R-hat <= 1.01 AND mass-min ESS >= 1000, converted to gradient /
    wallclock cost (fixed phases full + sampling phase prorated);
  * dev-vs-eval ESS/grad deltas per method (overfitting-to-dev check;
    dev = gu2022_sys000-005 Track-A cells, eval = sys006-011).

Outputs: data/bench_report.json, figs/bench_matrix.png.
S5 pocoMC: SKIPPED (pre-approved drop; S4 covers preconditioned SMC) --
recorded in the report notes.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))

from cgl import metrics as M                                  # noqa: E402
from cgl.paths import DATA, FIGS, RESULTS                     # noqa: E402
from cgl.samplers import list_samplers                        # noqa: E402

BASELINE = "s0_baseline"
DEV_T1 = {f"gu2022_sys{i:03d}" for i in range(6)}
EVAL_T1 = {f"gu2022_sys{i:03d}" for i in range(6, 12)}
T0_SET = {"t0_mix2", "t0_mix2_f64", "t0_mix22", "t0_funnel10", "t0_illcond46"}
MIX_TARGETS = {"t0_mix2", "t0_mix2_f64", "t0_mix22"}
RHAT_WIN = 1.01
ESS_GRAD_FACTOR = 2.0
MODE_TOL = 0.10
TRACKB_RHAT = 1.01
TRACKB_ESS = 1000.0
NOTES = {
    "s5_pocomc": "SKIPPED: pre-approved drop (P2b brief); S4 adaptive "
                 "tempered SMC covers preconditioned-SMC scientifically.",
    "ess_semantics": "bj_smc/nautilus_runner efficiency ESS is importance-"
                     "weight based (ess_source in each cell); their R-hat "
                     "is a weak pseudo-chain diagnostic on exchangeable "
                     "draws and is NOT used to fail them on the R-hat "
                     "criterion; mode occupancy/logZ carry their weight.",
    "t0_init": "T0 cells: contenders use the zoo's analytic InitBundle "
               "(free); S0 pays its own MAP+SVI per its published recipe. "
               "T0 ESS/grad therefore favors contenders; T1 bills the init "
               "cache identically to every consumer.",
}


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_cells():
    cells = []
    for sampler_dir in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        if not sampler_dir.is_dir():
            continue
        for target_dir in sorted(sampler_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            for jp in sorted(target_dir.glob("s*_*.json")):
                try:
                    meta = json.loads(jp.read_text())
                except Exception as e:              # noqa: BLE001
                    print(f"WARN: unreadable {jp}: {e}")
                    continue
                meta["_json_path"] = str(jp)
                cells.append(meta)
    return cells


def eff_of(c):
    return c["metrics"]["efficiency"]


def rhat_mass_max(c):
    return c["metrics"]["diagnostics"]["summary"]["rhat_mass"]["max"]


def is_particle_method(c):
    return c["metrics"]["efficiency"].get("ess_source", "").startswith(
        ("smc", "nautilus"))


def device_of(c):
    tag = c["env"].get("device_tag")
    if tag:
        return tag
    cvd = str(c["env"].get("CUDA_VISIBLE_DEVICES", "")).split(",")[0]
    return ("NVIDIA A16" if cvd.isdigit() and int(cvd) <= 7
            else "NVIDIA L4" if cvd.isdigit() else "unknown")


# --------------------------------------------------------------------------- #
# per-target Track-A tables
# --------------------------------------------------------------------------- #
def track_a_tables(cells):
    by_tm = defaultdict(list)     # (target, sampler) -> [cell]
    for c in cells:
        if c["track"] == "A":
            by_tm[(c["target"], c["sampler"])].append(c)

    targets = sorted({t for (t, _) in by_tm})
    tables = {}
    for tgt in targets:
        rows = {}
        base = by_tm.get((tgt, BASELINE), [])
        base_eg = (float(np.median([eff_of(c)["ess_per_grad_mass"]
                                    for c in base])) if base else None)
        for smp in sorted({m for (t, m) in by_tm if t == tgt}):
            cs = by_tm[(tgt, smp)]
            eg = [eff_of(c)["ess_per_grad_mass"] for c in cs]
            es = [eff_of(c)["ess_per_sec_mass"] for c in cs]
            rh = [rhat_mass_max(c) for c in cs]
            essmin = [eff_of(c)["ess_mass_min"] for c in cs]
            grads = [eff_of(c)["n_grad"] for c in cs]
            devs = sorted({device_of(c) for c in cs})
            particle = any(is_particle_method(c) for c in cs)
            med_eg = float(np.median(eg))
            med_rh = float(np.median(rh))
            ratio = (med_eg / base_eg) if base_eg else None

            mode_err = None
            modes_ok = True
            occs = []
            for c in cs:
                mm = c["metrics"].get("modes")
                if mm:
                    occs.append(mm["occupancy"].get("occupancy"))
                    err = mm["occupancy"].get("max_abs_weight_error")
                    if err is not None:
                        mode_err = max(mode_err or 0.0, float(err))
                        if err > MODE_TOL or not all(
                                mm["occupancy"]["recovered"]):
                            modes_ok = False
            win = None
            if base_eg:
                win = bool(ratio is not None and ratio >= ESS_GRAD_FACTOR
                           and (med_rh < RHAT_WIN or particle) and modes_ok)
            rows[smp] = dict(
                n_seeds=len(cs),
                seeds=sorted(c["seed"] for c in cs),
                ess_per_grad_median=med_eg,
                ess_per_grad_all=eg,
                ess_per_grad_vs_s0=ratio,
                ess_per_sec_median=float(np.median(es)),
                devices=devs,
                rhat_mass_max_median=med_rh,
                rhat_mass_max_worst=float(np.max(rh)),
                ess_mass_min_median=float(np.median(essmin)),
                n_grad_median=float(np.median(grads)),
                particle_method=particle,
                mode_weight_error_worst=mode_err,
                modes_ok=modes_ok,
                win=win,
            )
        tables[tgt] = dict(baseline_ess_per_grad_median=base_eg, rows=rows)
    return tables


# --------------------------------------------------------------------------- #
# T0-mixture mode table
# --------------------------------------------------------------------------- #
def mixture_table(cells):
    out = defaultdict(dict)
    for c in cells:
        if c["target"] not in MIX_TARGETS or c["track"] != "A":
            continue
        mm = c["metrics"].get("modes")
        if not mm:
            continue
        occ = mm["occupancy"]
        rt = mm["round_trips"]
        key = f"seed{c['seed']}"
        out[c["target"]].setdefault(c["sampler"], {})[key] = dict(
            occupancy=occ["occupancy"], ref=occ.get("ref_weights"),
            max_abs_weight_error=occ.get("max_abs_weight_error"),
            round_trips=rt["total_round_trips"],
            migrating_chains=f"{rt['n_migrating_chains']}/{rt['n_chains']}",
        )
    summary = {}
    for tgt, per in out.items():
        summary[tgt] = {}
        for smp, seeds in per.items():
            errs = [v["max_abs_weight_error"] for v in seeds.values()
                    if v["max_abs_weight_error"] is not None]
            summary[tgt][smp] = dict(
                per_seed=seeds,
                weight_error_median=float(np.median(errs)) if errs else None,
                weight_error_worst=float(np.max(errs)) if errs else None,
                passes_recipe_bar=bool(errs and np.max(errs) < 0.05),
            )
    return summary


# --------------------------------------------------------------------------- #
# logZ table
# --------------------------------------------------------------------------- #
def logz_table(cells):
    rows = defaultdict(dict)
    for c in cells:
        if c["track"] != "A":
            continue
        ex = c["metrics"].get("extras", {})
        lz = ex.get("logZ", ex.get("log_z", ex.get("logZ_smc")))
        if lz is None:
            continue
        cmp_ = ex.get("logz_compare")
        rows[c["target"]].setdefault(c["sampler"], []).append(dict(
            seed=c["seed"], logZ=float(lz),
            logZ_ref=(cmp_ or {}).get("logz_ref"),
            abs_err=(cmp_ or {}).get("abs_diff")))
    table = {}
    for tgt, per in rows.items():
        table[tgt] = {}
        for smp, entries in per.items():
            vals = [e["logZ"] for e in entries]
            errs = [e["abs_err"] for e in entries
                    if e["abs_err"] is not None]
            table[tgt][smp] = dict(
                per_seed=entries, logZ_median=float(np.median(vals)),
                abs_err_median=float(np.median(errs)) if errs else None)
    return table


# --------------------------------------------------------------------------- #
# Track-B prefix convergence
# --------------------------------------------------------------------------- #
def trackb_convergence(cells):
    out = defaultdict(dict)
    for c in cells:
        if c["track"] != "B":
            continue
        jp = Path(c["_json_path"])
        npz = np.load(jp.with_suffix(".npz"))
        samples = np.asarray(npz["samples"], dtype=np.float64)  # (T, C, dim)
        labels = c["labels"]
        mass_idx = [labels.index(m) for m in c["mass_labels"]]
        T = samples.shape[0]
        fracs = np.linspace(0.125, 1.0, 8)
        found = None
        for f in fracs:
            k = max(8, int(T * f))
            sub = samples[:k][:, :, mass_idx]
            d = M.rank_diagnostics(sub, [labels[i] for i in mass_idx],
                                   [labels[i] for i in mass_idx])
            rh = d["summary"]["rhat_mass"]["max"]
            ess = d["summary"]["ess_bulk_mass"]["min"]
            if rh <= TRACKB_RHAT and ess >= TRACKB_ESS:
                found = dict(frac=float(f), draws=k, rhat=rh, ess=ess)
                break
        phases = c["budget"]["phases"]
        # sampling phase = the largest-grad phase; fixed = everything else
        samp_phase = max(phases, key=lambda p: phases[p]["n_grad"])
        fixed = sum(p["n_grad"] for n, p in phases.items() if n != samp_phase)
        samp = phases[samp_phase]["n_grad"]
        wall = c["timing"].get("total_s")
        entry = dict(converged=found is not None, at=found,
                     n_grad_total=c["budget"]["n_grad_total"],
                     wall_s=wall, device=device_of(c))
        if found:
            entry["n_grad_to_converge"] = int(fixed + samp * found["frac"])
            entry["wall_s_to_converge_est"] = (
                float(wall) * (fixed + samp * found["frac"])
                / max(fixed + samp, 1)) if wall else None
        out[c["target"]].setdefault(c["sampler"], {})[
            f"seed{c['seed']}"] = entry
    return dict(out)


# --------------------------------------------------------------------------- #
# dev-vs-eval overfitting check
# --------------------------------------------------------------------------- #
def dev_vs_eval(cells):
    agg = defaultdict(lambda: defaultdict(list))
    for c in cells:
        if c["track"] != "A":
            continue
        split = ("dev" if c["target"] in DEV_T1 else
                 "eval" if c["target"] in EVAL_T1 else None)
        if split:
            agg[c["sampler"]][split].append(
                eff_of(c)["ess_per_grad_mass"])
            agg[c["sampler"]][split + "_rhat"].append(rhat_mass_max(c))
    out = {}
    for smp, d in agg.items():
        dev = d.get("dev", [])
        ev = d.get("eval", [])
        out[smp] = dict(
            n_dev=len(dev), n_eval=len(ev),
            dev_ess_per_grad_median=float(np.median(dev)) if dev else None,
            eval_ess_per_grad_median=float(np.median(ev)) if ev else None,
            eval_over_dev_ratio=(float(np.median(ev) / np.median(dev))
                                 if dev and ev and np.median(dev) > 0
                                 else None),
            dev_rhat_median=float(np.median(d["dev_rhat"]))
            if d.get("dev_rhat") else None,
            eval_rhat_median=float(np.median(d["eval_rhat"]))
            if d.get("eval_rhat") else None,
        )
    return out


# --------------------------------------------------------------------------- #
# heatmap
# --------------------------------------------------------------------------- #
def heatmap(tables, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = [t for t in
               ["t0_mix2", "t0_mix2_f64", "t0_mix22", "t0_funnel10",
                "t0_illcond46"]
               + sorted(EVAL_T1) if t in tables]
    samplers = sorted({s for t in targets for s in tables[t]["rows"]})
    mat = np.full((len(samplers), len(targets)), np.nan)
    for j, t in enumerate(targets):
        for i, s in enumerate(samplers):
            r = tables[t]["rows"].get(s)
            if r and r["ess_per_grad_vs_s0"]:
                mat[i, j] = np.log2(r["ess_per_grad_vs_s0"])
    fig, ax = plt.subplots(figsize=(2 + 0.85 * len(targets),
                                    1.5 + 0.5 * len(samplers)))
    lim = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(samplers)))
    ax.set_yticklabels(samplers, fontsize=9)
    for i in range(len(samplers)):
        for j in range(len(targets)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{2**mat[i, j]:.2g}x", ha="center",
                        va="center", fontsize=7)
    ax.set_title("mass-min ESS/grad vs S0 (Track A, median over seeds; "
                 "log2 color)")
    fig.colorbar(im, ax=ax, label="log2(ESS/grad / S0)")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    cells = load_cells()
    print(f"pooled {len(cells)} cells from {RESULTS}")
    tables = track_a_tables(cells)
    mix = mixture_table(cells)
    logz = logz_table(cells)
    tb = trackb_convergence(cells)
    dve = dev_vs_eval(cells)

    # completion audit
    expected_a = {(s, t, seed) for s in list_samplers()
                  for t in sorted(T0_SET | EVAL_T1) for seed in (0, 1, 2)}
    have_a = {(c["sampler"], c["target"], c["seed"]) for c in cells
              if c["track"] == "A" and c["target"] in (T0_SET | EVAL_T1)}
    missing = sorted(expected_a - have_a)

    report = dict(
        generated_by="25_pool_benchmark.py",
        n_cells=len(cells),
        notes=NOTES,
        win_criteria=dict(rhat=RHAT_WIN, ess_grad_factor=ESS_GRAD_FACTOR,
                          mode_tol=MODE_TOL),
        track_a=tables,
        t0_mixture_modes=mix,
        logz=logz,
        track_b=tb,
        dev_vs_eval=dve,
        matrix_missing_track_a=[list(m) for m in missing],
    )
    out = DATA / "bench_report.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {out}")
    heatmap(tables, FIGS / "bench_matrix.png")
    print(f"wrote {FIGS / 'bench_matrix.png'}")

    # ---- console summary ------------------------------------------------------
    for tgt in sorted(tables):
        rows = tables[tgt]["rows"]
        print(f"\n== {tgt} (S0 ESS/grad "
              f"{tables[tgt]['baseline_ess_per_grad_median']}) ==")
        for smp in sorted(rows, key=lambda s: -(rows[s]["ess_per_grad_vs_s0"]
                                                or 0)):
            r = rows[smp]
            ratio = r["ess_per_grad_vs_s0"]
            print(f"  {smp:16s} ESS/grad {r['ess_per_grad_median']:.3e} "
                  f"({'%.2fx' % ratio if ratio else '  -- '}) "
                  f"Rhat {r['rhat_mass_max_median']:.3f} "
                  f"win={r['win']} seeds={r['n_seeds']}")
    if missing:
        print(f"\nMISSING Track-A cells: {len(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
