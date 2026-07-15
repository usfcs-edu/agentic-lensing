"""Corner plot of the MCLMC posterior with a truth crosshair.

Reads the saved MCLMC samples (unconstrained), maps them to constrained (physical) space
via the model bijector, and draws a corner plot with the truth marked as a crosshair --
using the `corner` package directly the same way gigalens_research.plotting.corner does
(`corner.corner(..., truths=..., truth_color=..., show_titles=True, title_fmt='.3f')`).

Run (canonical Shifter container; no GPU needed -- pure plotting on saved samples):
    /usr/bin/python3 -m tests.multiplane_demo.cornerplot
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import corner

from .recover import build_recovery_model, constrained

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# readable short labels: plane index -> redshift, role M/L, param -> compact symbol
_Z = {0: "z1", 1: "z2", 2: "z3"}
_P = {"theta_E": r"\theta_E", "gamma": r"\gamma", "n_sersic": "n", "R_sersic": "R",
      "center_x": "c_x", "center_y": "c_y", "e1": "e_1", "e2": "e_2"}


def _label(key):
    p = key.split("/")                       # planes / i / role / j / param
    role = "M" if p[2] == "mass" else "L"
    return rf"${_Z[int(p[1])]}\,{role}\,{_P.get(p[4], p[4])}$"


def main(npz_name="demo_3plane_mclmc.npz", out_name="demo_3plane_corner.png"):
    d = np.load(os.path.join(ART, npz_name))
    draws = d["samples"]                      # (draws, chains, params), unconstrained
    keys = list(d["keys"]); z_true = jnp.asarray(d["z_true"])
    n_params = len(keys)

    model, _ = build_recovery_model()
    Zflat = jnp.asarray(draws.reshape(-1, n_params))
    cdict = model.bijector.forward(Zflat)            # constrained {key: (M,)}
    samples = np.column_stack([np.asarray(cdict[k]) for k in keys])
    ctru = constrained(model, z_true)
    truth_row = np.array([ctru[k] for k in keys])
    labels = [_label(k) for k in keys]

    print(f"corner: {samples.shape[0]} samples x {n_params} params from {npz_name}")
    fig = corner.corner(
        samples, labels=labels, truths=truth_row, truth_color="red",
        show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 7},
        label_kwargs={"fontsize": 8}, color="steelblue",
        hist_kwargs={"density": True, "color": "steelblue"},
        plot_datapoints=False, plot_density=True, fill_contours=True,
        levels=(0.68, 0.95), bins=30,
    )
    fig.suptitle("3-plane demo — MCLMC posterior (red = truth)", fontsize=14)
    out = os.path.join(ART, out_name)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
