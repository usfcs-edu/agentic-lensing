#!/usr/bin/env python
"""40_modeler_summary_figs.py -- PI-requested modeler-summary figures (P6 follow-up).

Two NEW figures for the up-front "what lens modelers care about" summary:

  figs/rfig_critcurves.png  Critical curves (lens plane, on the v3b binned
      cutout) + caustics (source plane) for the three on-disk posterior-median
      mass models: corr-low (certified 1.1032, artifact-diagnosed), the
      gamma=1.293 diagonal-likelihood fit (real-space preference), and the
      evidence-disfavored steep basin (2.639). Conventions follow the team's
      GIGA-Lens ApJ paper (Gu et al. 2022): image-plane critical curves and
      source-plane caustics as curve overlays, source position marked with a
      star; per-model colors here because three models share each panel.

  figs/rfig_corner.png      Corner plot of the 6 mass params (theta_E, gamma,
      e1, e2, shear g1, g2) from the stored v3b-low correlated SMC particles,
      3 seed repeats overlaid (the SMC convergence visual) + the steep basin
      in grey; gamma zoom panel with the authoritative weighted medians.

Two-venv design (07/25a precedent):
  Stage A (--stage-a; auto-run via subprocess under the OLD validated venv
      /raid/benson/.venvs/cgl, CPU, bijector only, NEVER the likelihood):
      maps the stored unconstrained SMC particles/chains to physical space
      with the exact production transform cgl.e2.build_target('v3b').model
      and caches ROOT/data/rfig_modeler_params.npz.
  Main stage (THIS venv, /raid/benson/.venvs/cgl2): deflection-grid evals of
      the vendored gigalens-linus EPL(50)+Shear (light; GPU if visible, CPU
      allowed -- this is NOT the lensing likelihood), figure rendering,
      assertions, captions.

Sources of truth (medians asserted against these, never re-derived):
  low seeds 2/3/4 : OLD/data/results/e2_v3b_low_smc_canary_fix.{npz,json},
                    ROOT/data/results-perlmutter/e2_v3b_low_smc_seed{3,4}.*
  steep seeds     : OLD .../e2_v3b_steep_smc_canary_p96.*, ROOT seed{3,4}
  diag 1.293      : foundry-i/data/hmc_v13_v3b.npz low basin (gamma<1.7,
                    fermat-teaser convention), per-dim z74 median ->
                    e2.paper_z_to_marg_z (t04 head-to-head convention);
                    checked vs data/t04_realspace_headtohead.json
  cutout          : foundry-i/data/cutout_v3b.npz (130^2 @ 0.08"/px)

Run:  /raid/benson/.venvs/cgl2/bin/python 40_modeler_summary_figs.py
      [--refresh]   force Stage-A re-extraction
NOTE: l0g2 scene-leg draws (Perlmutter $PSCRATCH) are NOT local; the corner
uses the certified e2 old-stack particle sets (the seed-triplet of record).

E1 extension (2026-07-22 epilogue harvest): the v2d-native MCLMC diagonal fit
(data/mclmc_diag_v2d.{npz,json}, PASS E1-G1/G2) is added to BOTH figures via
the scene-native extraction branch — the stored draws are scene-z; physical
params come from the scene model's own bijector (mass draws were already
mapped at run time by 50_run_mclmc_diag._mass_of, KEYED via mass_names; the
critcurve median model re-maps the pooled per-dim scene-z median through
mv.model.bijector.forward, KEYED via z_param_names, PHYS_COLS order).
Corner medians are asserted against the run json's pooled gamma quantiles.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus")
OLD = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens")
FI_DATA = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i/data")
OLDR = OLD / "data" / "results"
NEW = ROOT / "data" / "results-perlmutter"
FIGS = ROOT / "figs"
CACHE = ROOT / "data" / "rfig_modeler_params.npz"
OLD_PY = "/raid/benson/.venvs/cgl/bin/python"

PHYS_COLS = ["theta_E", "gamma", "e1", "e2", "gamma1", "gamma2",
             "center_x", "center_y"]

RUNS = {  # (basin, seed) -> (stem, npz key, json basin key)
    ("low", 2): (OLDR / "e2_v3b_low_smc_canary_fix", "low_particles", "low"),
    ("low", 3): (NEW / "e2_v3b_low_smc_seed3", "low_particles", "low"),
    ("low", 4): (NEW / "e2_v3b_low_smc_seed4", "low_particles", "low"),
    ("steep", 2): (OLDR / "e2_v3b_steep_smc_canary_p96", "steep_particles", "steep"),
    ("steep", 3): (NEW / "e2_v3b_steep_smc_seed3", "steep_particles", "steep"),
    ("steep", 4): (NEW / "e2_v3b_steep_smc_seed4", "steep_particles", "steep"),
}

# report numbers of record (synthesis_tables.md T0.2 block; weighted medians)
REPORT = dict(low_s2=1.1032, low_s3=1.0967, low_s4=1.1005,
              sigma_seed=0.00325, steep_s2=2.6393, diag=1.29325,
              scene_low=1.1005, dlogZ_steep_minus_low=-28.9)
MCLMC_JSON = ROOT / "data" / "mclmc_diag_v2d.json"   # E1 run json (asserted vs)
MCLMC_NPZ = ROOT / "data" / "mclmc_diag_v2d.npz"

# categorical slots 1-3 (dataviz reference palette; all-pairs validated).
# Light steps for white panels, dark steps for the astronomical image panel.
# E1 4th series: slot-7 violet #4a3aa7 VALIDATED all-pairs vs the light triple
# (validator: CVD 9.2, normal 16.3, PASS). No 4th hue passes all-pairs against
# the dark triple (palette series cap of 3), so on the image panel the MCLMC
# curve wears NEUTRAL WHITE + its unique dotted style (relief rule; the black
# halo + legend dash pattern carry identity).
C_LIGHT = {"low": "#2a78d6", "diag": "#eb6834", "steep": "#1baf7a",
           "mclmc": "#4a3aa7"}
C_DARK = {"low": "#3987e5", "diag": "#d95926", "steep": "#199e70",
          "mclmc": "#ffffff"}
LSTYLE = {"low": "-", "diag": (0, (5, 2.4)), "steep": (0, (4, 1.6, 1, 1.6)),
          "mclmc": (0, (1, 1.5))}
SEED_C = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}   # corner: slots 1-3
C_MCLMC = "#4a3aa7"
GREY = "#8a8a86"
INK, MUT, GRID, SPINE = "#3a3a38", "#8a8a86", "#e8e7e3", "#d5d4d0"

MODEL_LABEL = {
    "low": r"corr-low $\gamma=1.103$ (certified fit; artifact-diagnosed)",
    "diag": r"diagonal-low $\gamma=1.293$ (real-space preference)",
    "steep": r"steep $\gamma=2.639$ (evidence-disfavored, $\Delta\log Z\approx-29$)",
    "mclmc": "v2d-native MCLMC $\\gamma=1.468$ (E1, diagonal; converged "
             "$\\hat{R}$ 1.003; white dotted on image panel)",
}


# =========================================================================== #
# Stage A: OLD-venv extraction (bijector only, CPU)
# =========================================================================== #
def stage_a():
    """Runs under /raid/benson/.venvs/cgl (JAX_PLATFORMS=cpu GIGALENS_X64=1).

    Writes CACHE with, for each (basin, seed): phys_{basin}_s{seed} (N, 8)
    in PHYS_COLS order (equal-weight resampled SMC particles, physical space);
    plus x46 per-dim-z-median constrained vectors for low/steep seed 2 and the
    diag 1.293 model (t04 head-to-head convention), with labels46."""
    assert os.environ.get("GIGALENS_X64") == "1"
    import jax.numpy as jnp

    from cgl import e2
    from cgl.likelihood import _flat_named46

    tgt = e2.build_target("v3b")            # bijector use only; no likelihood
    model = tgt.model
    out, meta = {}, {"phys_cols": PHYS_COLS, "inputs": {}}

    for (basin, seed), (stem, key, bkey) in RUNS.items():
        z = np.load(stem.with_suffix(".npz"), allow_pickle=True)
        parts = np.asarray(z[key], dtype=np.float64)
        phys = model.to_physical_mass(parts)
        out[f"phys_{basin}_s{seed}"] = np.stack(
            [np.asarray(phys[c], dtype=np.float64) for c in PHYS_COLS], axis=1)
        js = json.load(open(stem.with_suffix(".json")))["basins"][bkey]
        meta["inputs"][f"{basin}_s{seed}"] = dict(
            stem=str(stem), n=int(parts.shape[0]),
            gamma_median_weighted=js["gamma_median"], logZ=js["logZ"],
            ess_weight=js["ess_weight"], n_unique=js["n_unique"])
        if seed == 2:                        # t04 convention: per-dim z median
            zmed = np.median(parts, axis=0)
            x = model.bij.forward(list(jnp.asarray(zmed)[:, None]))
            named = dict(_flat_named46(x))
            out[f"x46_{basin}"] = np.array([float(named[k]) for k in
                                            model.index_labels], dtype=np.float64)

    # diag 1.293: hmc_v13_v3b low basin (gamma<1.7), z74 median -> marg z46
    hz = np.load(FI_DATA / "hmc_v13_v3b.npz", allow_pickle=True)
    sel = np.asarray(hz["mass_gamma"], dtype=np.float64).ravel() < 1.7
    z74 = np.asarray(hz["samples"], dtype=np.float64).reshape(-1, 74)[sel]
    z46d, rep = e2.paper_z_to_marg_z(np.median(z74, axis=0), model,
                                     ie_scale=float(tgt.conversion_factor))
    named = dict(_flat_named46(model.bij.forward(list(jnp.asarray(z46d)[:, None]))))
    out["x46_diag"] = np.array([float(named[k]) for k in model.index_labels],
                               dtype=np.float64)
    meta["diag"] = dict(n_low=int(sel.sum()), n_total=int(sel.size),
                        transform_report={k: float(v) for k, v in rep.items()})

    out["labels46"] = np.array(list(model.index_labels))
    out["meta"] = np.array(json.dumps(meta))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"[stage-a] wrote {CACHE}: " +
          ", ".join(f"{k}{v.shape}" for k, v in out.items() if k != "meta"))


def ensure_cache(refresh: bool):
    if CACHE.exists() and not refresh:
        return
    env = dict(os.environ, JAX_PLATFORMS="cpu", GIGALENS_X64="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    print(f"[main] running Stage A under {OLD_PY} (CPU, bijector only) ...",
          flush=True)
    subprocess.run([OLD_PY, str(ROOT / "40_modeler_summary_figs.py"),
                    "--stage-a"], env=env, cwd=str(OLD), check=True)


# =========================================================================== #
# helpers (main venv)
# =========================================================================== #
def mass_params_from_x46(x46, labels):
    d = dict(zip(labels, x46))
    return dict(theta_E=d["mass.theta_E"], gamma=d["mass.gamma"],
                e1=d["mass.e1"], e2=d["mass.e2"],
                cx=d["mass.center_x"], cy=d["mass.center_y"],
                g1=d["shear.gamma1"], g2=d["shear.gamma2"],
                src=(d["srcS.center_x"], d["srcS.center_y"]))


def load_mclmc(checks):
    """E1 v2d MCLMC pooled physical mass draws (N, 8) in PHYS_COLS order
    (scene-native: mapped at run time by 50_run_mclmc_diag._mass_of, KEYED),
    + the run-json pooled gamma quantiles they are asserted against."""
    z = np.load(MCLMC_NPZ, allow_pickle=True)
    mnames = [str(s) for s in z["mass_names"]]
    m = np.concatenate([np.asarray(z["mass_g0"], dtype=np.float64),
                        np.asarray(z["mass_g1"], dtype=np.float64)], axis=0)
    m = m.reshape(-1, len(mnames))
    phys = np.stack([m[:, mnames.index(c)] for c in PHYS_COLS], axis=1)
    jq = json.load(open(MCLMC_JSON))["gamma"]["pooled_quantiles"]
    gmed = float(np.median(phys[:, PHYS_COLS.index("gamma")]))
    checks["mclmc_corner_gamma"] = dict(
        eqw_draw_median=round(gmed, 6), run_json_q50=round(jq["0.5"], 6),
        n_draws=int(phys.shape[0]),
        ok=bool(abs(gmed - jq["0.5"]) < 1e-9))
    assert checks["mclmc_corner_gamma"]["ok"], (gmed, jq["0.5"])
    return phys, jq


def mclmc_median_model(checks):
    """E1 v2d MCLMC posterior-median mass model for the critcurve figure:
    pooled per-dim scene-z median -> mv.model.bijector.forward (KEYED via
    z_param_names; t04 per-dim-median convention, same as the other models)."""
    import jax.numpy as jnp

    from cgl2 import scene_build

    z = np.load(MCLMC_NPZ, allow_pickle=True)
    pooled = np.concatenate([np.asarray(z["pos_g0"], dtype=np.float64),
                             np.asarray(z["pos_g1"], dtype=np.float64)],
                            axis=0).reshape(-1, 46)
    zmed = np.median(pooled, axis=0)
    refs = np.load(ROOT / "data" / "parity_refs.npz", allow_pickle=True)
    prov = json.loads(str(refs["provenance"]))
    mv = scene_build.build_scene_model(view="marg", near_xy=prov["near_xy"])
    assert [str(s) for s in z["z_names"]] == list(mv.model.z_param_names), \
        "scene-z name/order mismatch vs stored chains"      # KEYED discipline
    u = mv.model.bijector.forward(jnp.asarray(zmed[None, :]))
    f = {k: float(np.asarray(v)[0]) for k, v in u.items()}
    p = dict(theta_E=f["planes/0/mass/0/theta_E"], gamma=f["planes/0/mass/0/gamma"],
             e1=f["planes/0/mass/0/e1"], e2=f["planes/0/mass/0/e2"],
             cx=f["planes/0/mass/0/center_x"], cy=f["planes/0/mass/0/center_y"],
             g1=f["planes/0/mass/1/gamma1"], g2=f["planes/0/mass/1/gamma2"],
             src=(f["planes/1/light/0/center_x"], f["planes/1/light/0/center_y"]))
    jq = json.load(open(MCLMC_JSON))["gamma"]["pooled_quantiles"]
    checks["mclmc_median_model"] = dict(
        gamma_zmed_forward=round(p["gamma"], 6),
        run_json_q50=round(jq["0.5"], 6),
        ok=bool(abs(p["gamma"] - jq["0.5"]) < 1e-3))
    assert checks["mclmc_median_model"]["ok"], p["gamma"]
    return p


def crit_and_caustics(p, n=1201, L=5.2):
    """detA=0 contours of EPL(50)+Shear at params p -> list of
    (crit_curve (M,2), caustic (M,2)); jax autodiff Jacobian on an n x n grid."""
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from gigalens.jax.profiles.mass import epl, shear
    EPL, SH = epl.EPL(50), shear.Shear()

    def alpha(xy):
        ax1, ay1 = EPL.deriv(xy[0], xy[1], p["theta_E"], p["gamma"],
                             p["e1"], p["e2"], p["cx"], p["cy"])
        ax2, ay2 = SH.deriv(xy[0], xy[1], p["g1"], p["g2"])
        return jnp.stack([ax1 + ax2, ay1 + ay2])

    xs = np.linspace(-L, L, n)
    X, Y = np.meshgrid(xs, xs)
    pts = jnp.asarray(np.stack([X.ravel(), Y.ravel()], 1))
    J = np.asarray(jax.vmap(jax.jacfwd(alpha))(pts))
    detA = ((1 - J[:, 0, 0]) * (1 - J[:, 1, 1])
            - J[:, 0, 1] * J[:, 1, 0]).reshape(n, n)
    figt, axt = plt.subplots()
    segs = [s for s in axt.contour(X, Y, detA, levels=[0.0]).allsegs[0]
            if len(s) > 20]
    plt.close(figt)
    out = []
    for s in segs:
        a = np.asarray(jax.vmap(alpha)(jnp.asarray(s)))
        out.append((s, s - a))                     # beta = theta - alpha
    return out


def asinh_img(img, a=0.02):
    v = np.arcsinh(np.clip(img, 0, None) / a)
    return v / np.percentile(v, 99.9)


def style_ax(ax):
    ax.tick_params(colors=MUT, labelsize=8)
    for s in ax.spines.values():
        s.set_color(SPINE)


# =========================================================================== #
# Figure 1: critical curves + caustics
# =========================================================================== #
def fig_critcurves(cache, checks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    from cgl2 import paths as c2paths
    c2paths.bootstrap_vendor()

    labels = list(cache["labels46"])
    models = {t: mass_params_from_x46(cache[f"x46_{t}"], labels)
              for t in ("low", "diag", "steep")}
    models["mclmc"] = mclmc_median_model(checks)         # E1 v2d-native fit
    curves = {t: crit_and_caustics(p) for t, p in models.items()}

    # sanity: outer critical curve radius ~ theta_E (theta_E ~ 2.6")
    for t, p in models.items():
        rmed = [float(np.median(np.hypot(c[:, 0] - p["cx"], c[:, 1] - p["cy"])))
                for c, _ in curves[t]]
        outer = max(rmed)
        checks[f"critcurve_{t}"] = dict(
            theta_E=round(p["theta_E"], 5), gamma=round(p["gamma"], 5),
            outer_r_median=round(outer, 4), n_curves=len(rmed),
            ok=bool(abs(outer - p["theta_E"]) / p["theta_E"] < 0.05))
        assert checks[f"critcurve_{t}"]["ok"], \
            f"{t}: outer critical curve r={outer:.3f} vs theta_E={p['theta_E']:.3f}"
    assert checks["critcurve_low"]["n_curves"] >= 2, \
        "low (gamma<2) must show an inner radial critical curve"

    prod = np.load(FI_DATA / "cutout_v3b.npz", allow_pickle=True)
    img = np.asarray(prod["img"], dtype=np.float64)
    L = 0.08 * img.shape[0] / 2.0                      # 5.2" half-field

    fig, (axL, axS) = plt.subplots(1, 2, figsize=(10.6, 5.15), dpi=200)
    axL.imshow(asinh_img(img), origin="lower", extent=[-L, L, -L, L],
               cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    halo = [pe.withStroke(linewidth=3.1, foreground="black", alpha=0.75)]
    for t in ("low", "diag", "steep", "mclmc"):
        for k, (crit, _) in enumerate(curves[t]):
            axL.plot(crit[:, 0], crit[:, 1], color=C_DARK[t], ls=LSTYLE[t],
                     lw=1.7, path_effects=halo,
                     label=MODEL_LABEL[t] if k == 0 else None)
    axL.plot([-4.7, -3.7], [-4.65, -4.65], color="white", lw=2)
    axL.text(-4.2, -4.45, '1"', color="white", ha="center", fontsize=8)
    axL.set_title('Image plane — critical curves on the v3b cutout '
                  '(130$^2$ @ 0.08"/px)', color=INK, fontsize=10)
    axL.set_xlabel('x ["]', color=INK, fontsize=9)
    axL.set_ylabel('y ["]', color=INK, fontsize=9)

    for t in ("low", "diag", "steep", "mclmc"):
        for k, (_, cau) in enumerate(curves[t]):
            axS.plot(cau[:, 0], cau[:, 1], color=C_LIGHT[t], ls=LSTYLE[t],
                     lw=1.6, label=MODEL_LABEL[t] if k == 0 else None)
        axS.plot(*models[t]["src"], marker="*", ms=13, mec=INK, mew=0.5,
                 color=C_LIGHT[t], ls="none")
    axS.set_title(r'Source plane — caustics ($\bigstar$ = median Sérsic-source '
                  'center)', color=INK, fontsize=10)
    axS.set_xlabel(r'$\beta_x$ ["]', color=INK, fontsize=9)
    axS.set_ylabel(r'$\beta_y$ ["]', color=INK, fontsize=9)
    axS.set_aspect("equal")
    axS.grid(True, color=GRID, lw=0.7)
    lim = 1.05 * max(np.abs(np.concatenate(
        [cau for t in curves for _, cau in curves[t]])).max(), 0.6)
    axS.set_xlim(-lim, lim), axS.set_ylim(-lim, lim)
    for ax in (axL, axS):
        style_ax(ax)
    h, l = axS.get_legend_handles_labels()
    fig.legend(h[:4], l[:4], loc="lower center", ncol=2, frameon=False,
               fontsize=8.2, labelcolor=INK, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Posterior-median mass models (v3b products + the E1 "
                 "v2d-native MCLMC fit) — critical curves & caustics"
                 r"  (outer curve radius $\simeq\theta_E\approx2.6$"
                 '"; inner radial structure because '
                 r"$\gamma_{\rm EPL}<2$ for the low/diagonal/MCLMC fits)",
                 color=INK, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    out = FIGS / "rfig_critcurves.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


# =========================================================================== #
# Figure 2: corner plot
# =========================================================================== #
def fig_corner(cache, checks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    meta = json.loads(str(cache["meta"]))
    cols = ["theta_E", "gamma", "e1", "e2", "gamma1", "gamma2"]
    idx = [PHYS_COLS.index(c) for c in cols]
    lab = [r'$\theta_E$ ["]', r"$\gamma$", r"$e_1$", r"$e_2$",
           r"$\gamma_{\rm ext,1}$", r"$\gamma_{\rm ext,2}$"]
    low = {s: cache[f"phys_low_s{s}"][:, idx] for s in (2, 3, 4)}
    steep = np.concatenate([cache[f"phys_steep_s{s}"][:, idx]
                            for s in (2, 3, 4)], axis=0)
    phys_mclmc, _ = load_mclmc(checks)               # E1 v2d-native, asserted
    mcl = phys_mclmc[:, idx]
    rng_m = np.random.default_rng(40)
    mcl_sc = mcl[rng_m.choice(mcl.shape[0], 2500, replace=False)]
    mcl_kde = mcl[rng_m.choice(mcl.shape[0], 20000, replace=False)]

    # ---- assertions: particle medians reproduce the numbers of record ------ #
    gmed = {s: float(np.median(low[s][:, 1])) for s in (2, 3, 4)}
    wmed = {s: meta["inputs"][f"low_s{s}"]["gamma_median_weighted"]
            for s in (2, 3, 4)}
    sig_seed = float(np.std(list(wmed.values()), ddof=1))
    checks["corner_gamma"] = dict(
        eqw_particle_median={s: round(g, 6) for s, g in gmed.items()},
        weighted_median_json={s: round(w, 6) for s, w in wmed.items()},
        report={2: REPORT["low_s2"], 3: REPORT["low_s3"], 4: REPORT["low_s4"]},
        sigma_seed_from_json=round(sig_seed, 6),
        steep_eqw_median=round(float(np.median(steep[:, 1])), 5))
    assert abs(gmed[2] - 1.1032) < 2e-4, gmed[2]          # money number
    for s in (2, 3, 4):
        assert abs(wmed[s] - REPORT[f"low_s{s}"]) < 5e-4, (s, wmed[s])
        assert abs(gmed[s] - wmed[s]) < 3.5e-3, (s, gmed[s], wmed[s])
    assert abs(sig_seed - REPORT["sigma_seed"]) < 1e-4, sig_seed
    assert abs(float(np.median(cache["phys_steep_s2"][:, 1]))
               - REPORT["steep_s2"]) < 2e-3

    n = len(cols)
    rng = []
    for j in range(n):
        allv = np.concatenate([low[s][:, j] for s in (2, 3, 4)]
                              + [steep[:, j], mcl_sc[:, j]])
        lo, hi = allv.min(), allv.max()
        pad = 0.09 * (hi - lo)
        rng.append((lo - pad, hi + pad))

    fig, axes = plt.subplots(n, n, figsize=(10.6, 10.6), dpi=200)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            style_ax(ax)
            ax.grid(True, color=GRID, lw=0.5)
            if i == j:
                xs = np.linspace(*rng[j], 400)
                kde_s = gaussian_kde(steep[:, j])(xs)
                ax.fill_between(xs, kde_s / kde_s.max() * 0.9, color=GREY,
                                alpha=0.30, lw=0)
                kde_m = gaussian_kde(mcl_kde[:, j])(xs)
                ax.plot(xs, kde_m / kde_m.max(), color=C_MCLMC, lw=1.5,
                        ls=(0, (4, 2)))
                for s in (2, 3, 4):
                    kd = gaussian_kde(low[s][:, j])(xs)
                    ax.plot(xs, kd / kd.max(), color=SEED_C[s], lw=1.5)
                ax.set_yticks([])
                ax.set_ylim(0, 1.12)
            else:
                ax.scatter(steep[:, j], steep[:, i], s=4, color=GREY,
                           alpha=0.35, lw=0)
                ax.scatter(mcl_sc[:, j], mcl_sc[:, i], s=4, color=C_MCLMC,
                           alpha=0.28, lw=0)
                for s in (2, 3, 4):
                    ax.scatter(low[s][:, j], low[s][:, i], s=5,
                               color=SEED_C[s], alpha=0.6, lw=0)
                ax.set_ylim(*rng[i])
            ax.set_xlim(*rng[j])
            if i == n - 1:
                ax.set_xlabel(lab[j], color=INK, fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(lab[i], color=INK, fontsize=10)
            else:
                ax.set_yticklabels([])

    # gamma zoom (upper-right): the SMC seed-repeat convergence visual
    axz = fig.add_axes([0.585, 0.72, 0.375, 0.225])
    style_ax(axz)
    axz.grid(True, color=GRID, lw=0.7)
    g_all = np.concatenate([low[s][:, 1] for s in (2, 3, 4)])
    xs = np.linspace(g_all.min() - 0.004, g_all.max() + 0.004, 500)
    for s in (2, 3, 4):
        axz.plot(xs, gaussian_kde(low[s][:, 1])(xs), color=SEED_C[s], lw=1.9,
                 label=f"seed {s}" + (" (production)" if s == 2 else "")
                       + rf":  $\tilde\gamma_w$ = {wmed[s]:.4f}")
        axz.axvline(wmed[s], color=SEED_C[s], lw=1.1, ls=(0, (4, 3)), alpha=0.8)
    axz.set_xlabel(r"$\gamma$ (v3b-low, correlated)", color=INK, fontsize=9)
    axz.set_yticks([])
    axz.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    axz.set_title(r"seed repeats (zoom): $\sigma_{\rm seed}=%.5f$"
                  % sig_seed, color=INK, fontsize=9)

    import matplotlib.lines as mlines
    handles = ([mlines.Line2D([], [], color=SEED_C[s], lw=2,
                              label=f"v3b-low SMC particles, seed {s}")
                for s in (2, 3, 4)]
               + [mlines.Line2D([], [], color=GREY, lw=6, alpha=0.4,
                                label=r"steep basin, seeds 2–4 pooled "
                                      r"($\gamma\approx2.64$; "
                                      r"$\Delta\log Z\approx-29$, disfavored)"),
                  mlines.Line2D([], [], color=C_MCLMC, lw=2, ls=(0, (4, 2)),
                                label="v2d-NATIVE MCLMC diagonal fit (E1: "
                                      r"$\gamma=1.468$ [1.434, 1.505], "
                                      r"$\hat{R}_{\rm worst}$ 1.003, "
                                      "min-ESS 30,334)")])
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.40, 0.70),
               frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Mass-parameter posteriors — v3b correlated SMC particles "
                 "(128/seed low, 96/seed steep) + the E1 v2d-native MCLMC "
                 "diagonal posterior (64 chains x 4000 draws)",
                 color=INK, fontsize=11, y=0.985)
    fig.subplots_adjust(hspace=0.06, wspace=0.06, top=0.955)
    out = FIGS / "rfig_corner.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


# =========================================================================== #
# captions
# =========================================================================== #
CAP_BEGIN = "<!-- BEGIN rfig_modeler_summary (40_modeler_summary_figs.py) -->"
CAP_END = "<!-- END rfig_modeler_summary -->"


def write_captions(checks):
    cap = f"""{CAP_BEGIN}

## Modeler-summary figures (PI request, 40_modeler_summary_figs.py)

| figure | placement | caption |
|---|---|---|
| figs/rfig_critcurves.png | up-front modeler summary | Critical curves (lens plane, over the v3b binned cutout, asinh stretch) and caustics (source plane; stars = median Sérsic-source centers) for the four posterior-median mass models on disk: corr-low γ=1.1032 (the certified but artifact-diagnosed correlated fit; per-dim z-median of the 128 production SMC particles, t04 convention), diagonal-low γ=1.29325 (hmc_v13_v3b low basin — the real-space preference), steep γ=2.639 (evidence-disfavored, ΔlogZ≈−29), and the E1 v2d-NATIVE MCLMC diagonal fit γ={checks.get('mclmc_median_model', {}).get('gamma_zmed_forward', '?')} (the PI-requested converged fit, R̂_worst 1.0031; pooled per-dim scene-z median through the scene bijector, same t04 convention; native 80²@0.13" product, curves overlaid on the binned cutout of the same sky). Curve overlays follow the team's GIGA-Lens (Gu et al. 2022) conventions; per-model colors + line styles because four models share each panel — the MCLMC curve is slot-7 violet on the white source panel (validated all-pairs vs the three existing hues) and neutral WHITE dotted on the image panel (no 4th hue passes the dark all-pairs floors; palette series cap, relief via unique dash + halo). Sanity: outer critical-curve median radius matches θ_E to <5% for all four (low {checks.get('critcurve_low', {}).get('outer_r_median', '?')}" vs θ_E {checks.get('critcurve_low', {}).get('theta_E', '?')}"; mclmc {checks.get('critcurve_mclmc', {}).get('outer_r_median', '?')}" vs {checks.get('critcurve_mclmc', {}).get('theta_E', '?')}"); the γ<2 models show the expected inner radial critical curve/caustic. Deflections: vendored gigalens-linus EPL(50)+Shear at the certified scene conventions; grid Jacobian via jax autodiff (1201², ±5.2"). |
| figs/rfig_corner.png | up-front modeler summary | Corner plot of the six mass parameters (θ_E, γ, e_1, e_2, γ_ext,1, γ_ext,2) from the stored equal-weight v3b-low correlated SMC particles, seeds 2/3/4 overlaid (this is the SMC convergence visual: SMC has no R̂; the seed-repeat spread σ_seed = 0.00325 plus per-stage ESS/acceptance are the analogues), steep basin (seeds 2–4 pooled) in grey, PLUS the E1 v2d-native MCLMC diagonal posterior in violet (256,000 pooled draws; scene-native physical extraction KEYED via mass_names; equal-weight draw median asserted equal to the run json's pooled γ q50 = {checks.get('mclmc_corner_gamma', {}).get('run_json_q50', '?')}). Weighted medians of record (from the run JSONs) γ = 1.1032/1.0967/1.1005; the equal-weight particle medians reproduce them to ≤0.0027 (≪ σ_stat 0.008; the resampled-particle convention). γ zoom panel shows the three correlated repeats with weighted medians (the MCLMC γ≈1.47 lies outside the zoom by construction). The cross-stack scene-API repeat (γ 1.1005, Δγ 0.0027, logZ to 0.11 nats) is reported in the text; its draws remain on Perlmutter $PSCRATCH (summary-JSON only locally). |

Derived input: data/rfig_modeler_params.npz — Stage-A (OLD cgl venv, CPU,
bijector-only) physical-space export of the six stored particle sets + the three
median mass models (25a/01a/07 precedent). Regenerate with --refresh.

{CAP_END}
"""
    p = FIGS / "rfig_captions.md"
    txt = p.read_text() if p.exists() else ""
    if CAP_BEGIN in txt:
        pre = txt[:txt.index(CAP_BEGIN)]
        post = txt[txt.index(CAP_END) + len(CAP_END):]
        txt = pre + cap.strip() + post
    else:
        txt = txt.rstrip() + "\n\n" + cap.strip() + "\n"
    p.write_text(txt)
    print(f"[captions] updated {p}")


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.stage_a:
        stage_a()
        return

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("CGL2_ALLOW_CPU", "1")   # deflection grids only --
    os.environ.setdefault("JAX_ENABLE_X64", "1")   # NEVER the likelihood
    os.environ.setdefault("GIGALENS_X64", "1")     # scene_build x64 guard
    ensure_cache(args.refresh)
    cache = np.load(CACHE, allow_pickle=True)

    checks = {}
    # diag transform check vs the t04 artifact of record
    t04 = json.load(open(ROOT / "data" / "t04_realspace_headtohead.json"))
    labels = list(cache["labels46"])
    diag_gamma = float(dict(zip(labels, cache["x46_diag"]))["mass.gamma"])
    t04_diag = t04["validation"]["diaglow_gamma"]["transformed"]
    checks["diag_gamma"] = dict(extracted=round(diag_gamma, 6),
                                t04_reference=round(t04_diag, 6),
                                ok=bool(abs(diag_gamma - t04_diag) < 1e-3))
    assert checks["diag_gamma"]["ok"]
    low_gamma = float(dict(zip(labels, cache["x46_low"]))["mass.gamma"])
    assert abs(low_gamma - 1.1032) < 2e-4, low_gamma
    checks["x46_low_gamma"] = round(low_gamma, 6)

    figs = [str(fig_critcurves(cache, checks)), str(fig_corner(cache, checks))]
    write_captions(checks)
    (ROOT / "data" / "rfig_modeler_checks.json").write_text(
        json.dumps(dict(generated_utc=__import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
            figures=figs, checks=checks), indent=2))
    print(json.dumps(dict(figures=figs, checks=checks), indent=1))


if __name__ == "__main__":
    main()
