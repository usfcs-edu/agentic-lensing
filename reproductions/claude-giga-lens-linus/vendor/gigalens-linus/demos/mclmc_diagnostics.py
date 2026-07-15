"""Standalone MCLMC tuning-diagnostic plots.

Drop-in, dependency-light (numpy + matplotlib only) replacement for the
GIGALens-Code diagnostic plotter, for people who run MCLMC but don't want to
pull in the full ``gigalens_research`` wrapper suite.

It consumes the *raw debug history* that the MCLMC sampler returns when run with
``debug_output=True`` (the ``Hist`` namedtuple from
``gigalens_research.inference.MCLMC_JIT`` / ``MCLMC``). All you need to keep is
that object (or its arrays). No gigalens import is required to make the plots.

Expected layout
----------------
Every field is **(num_chains, total_steps, ...)** -- chains first, then the scan
(step) axis. That is what ``MCLMC_JIT(..., debug_output=True)`` returns:

    hist.step_size            (num_chains, total_steps)
    hist.L                    (num_chains, total_steps)
    hist.xi                   (num_chains, total_steps)   energy-error ratio
    hist.nonan                (num_chains, total_steps)   1.0 = finite step, 0.0 = NaN/blow-up
    hist.inverse_mass_matrix  (num_chains, total_steps, dim, dim)
    hist.position             (num_chains, total_steps, dim)   (not plotted here)

``total_steps == num_burnin_steps + num_results``. The three tuning fractions
(``frac_tune1/2/3``) and ``num_burnin_steps`` are only needed to draw the
tuning-stage boundary lines; pass them if you have them.

Usage
-----
    from mclmc_diagnostics import plot_mclmc_diagnostics

    hist = MCLMC_JIT(..., debug_output=True)          # the raw Hist
    fig = plot_mclmc_diagnostics(
        hist,
        num_burnin_steps=2000,                        # for the stage boundaries
        frac_tune1=0.2, frac_tune2=0.6, frac_tune3=0.2,
        chain=0,                                       # which chain for the xi panel
    )
    fig.savefig("mclmc_diag.png", dpi=150)

You can also pass a plain dict of the arrays (same keys as the ``Hist`` fields)
instead of the namedtuple.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Fields the plots can use; pulled from either attributes or dict keys.
_FIELDS = ("step_size", "L", "xi", "nonan", "inverse_mass_matrix")


def _get(hist: Any, name: str) -> Optional[np.ndarray]:
    """Pull ``name`` from a namedtuple/object (getattr) or a mapping (key).

    Returns ``None`` if absent so the corresponding panel is simply left blank
    rather than raising -- a partially-captured history still plots.
    """
    if isinstance(hist, Mapping):
        val = hist.get(name, None)
    else:
        val = getattr(hist, name, None)
    if val is None:
        return None
    return np.asarray(val)


def _tuning_boundaries(
    num_burnin_steps: Optional[int],
    frac_tune1: float,
    frac_tune2: float,
    frac_tune3: float,
) -> Tuple[int, int, int]:
    """Step indices where MCLMC's three tuning stages end.

    Stage 1 = step-size find, stage 2 = mass-matrix, stage 3 = ``L``. With the
    default fractions (0.2/0.6/0.2, summing to 1) stage 3 ends exactly at the
    end of burn-in; anything after that is sampling. Returns ``(0, 0, 0)`` when
    ``num_burnin_steps`` is unknown, which suppresses the boundary lines.
    """
    if not num_burnin_steps:
        return 0, 0, 0
    nb = int(num_burnin_steps)
    f1, f2, f3 = float(frac_tune1), float(frac_tune2), float(frac_tune3)
    return int(f1 * nb), int((f1 + f2) * nb), int((f1 + f2 + f3) * nb)


def plot_mclmc_diagnostics(
    hist: Any,
    *,
    num_burnin_steps: Optional[int] = None,
    frac_tune1: float = 0.2,
    frac_tune2: float = 0.6,
    frac_tune3: float = 0.2,
    chain: int = 0,
    smooth: int = 30,
    figsize: Tuple[float, float] = (10, 9),
) -> Figure:
    """Five stacked panels of an MCLMC tuning run, vs. step:

    1. per-chain step size (log y),
    2. per-chain trajectory length ``L``,
    3. inverse-mass-matrix eigenvalue spread (min/mean/max, log y; chain 0),
    4. the per-step energy-error ratio ``xi`` for one chain (raw + smoothed),
    5. a success heatmap (green = finite step, red = NaN/blow-up).

    Vertical dashed lines mark the boundaries of MCLMC's three tuning stages
    (step size, mass matrix, ``L``); anything after the last line is sampling.
    They are drawn only when ``num_burnin_steps`` is supplied.

    Parameters
    ----------
    hist : Hist namedtuple, object, or dict
        Raw MCLMC debug history (``debug_output=True``). See module docstring
        for the expected fields and ``(num_chains, total_steps, ...)`` layout.
    num_burnin_steps, frac_tune1/2/3 : optional
        Used only to place the tuning-stage boundary lines.
    chain : int
        Which chain to show in the ``xi`` panel.
    smooth : int
        Boxcar window (in steps) for the smoothed ``xi`` overlay; <= 1 disables.
    """
    arr = {name: _get(hist, name) for name in _FIELDS}
    stage1, stage2, stage3 = _tuning_boundaries(
        num_burnin_steps, frac_tune1, frac_tune2, frac_tune3
    )

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=figsize)
    ax_ss, ax_L, ax_eig, ax_xi, ax_nan = axs

    # 1. step size (transpose -> step on x, one line per chain)
    if arr["step_size"] is not None:
        ax_ss.plot(arr["step_size"].T)
    ax_ss.set_title("Chain-wise step size")
    ax_ss.set_yscale("log")
    ax_ss.set_ylabel("step size")

    # 2. trajectory length L
    if arr["L"] is not None:
        ax_L.plot(arr["L"].T)
    ax_L.set_title("Chain-wise L")
    ax_L.set_ylabel("L")

    # 3. inverse-mass-matrix eigenvalues (chain 0; it's replicated across chains)
    if arr["inverse_mass_matrix"] is not None:
        imm = arr["inverse_mass_matrix"][0]  # (n_steps, dim, dim)
        # Symmetric PD covariance -> eigvalsh (real, ascending).
        eig = np.linalg.eigvalsh(imm)  # (n_steps, dim)
        ax_eig.plot(eig.min(axis=1), label="min", color="tab:blue")
        ax_eig.plot(eig.mean(axis=1), label="mean", color="black")
        ax_eig.plot(eig.max(axis=1), label="max", color="tab:red")
        ax_eig.set_yscale("log")
        ax_eig.legend(fontsize=8)
    ax_eig.set_title("Inverse mass-matrix eigenvalues")
    ax_eig.set_ylabel("eigenvalue")

    # 4. xi (energy-error ratio) for one chain, raw + smoothed
    if arr["xi"] is not None:
        xi = arr["xi"]
        c = chain % xi.shape[0]
        xi_c = xi[c]
        ax_xi.plot(xi_c, alpha=0.4, color="tab:blue")
        if smooth and smooth > 1 and xi_c.size >= smooth:
            kern = np.ones(smooth) / smooth
            ax_xi.plot(np.convolve(xi_c, kern, mode="same"), color="tab:blue")
        ax_xi.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax_xi.set_yscale("log")
        ax_xi.set_ylabel(f"xi (chain {c})")
    ax_xi.set_title("Energy-error ratio")

    # 5. success / NaN heatmap (chains x steps), restricted to the tuning span
    if arr["nonan"] is not None:
        nonan = arr["nonan"]
        upto = stage2 if stage2 > 0 else nonan.shape[1]
        ax_nan.imshow(
            nonan[:, :upto], aspect="auto", interpolation="none", cmap="RdYlGn",
            vmin=0, vmax=1,
        )
        ax_nan.set_ylabel("chain")
    ax_nan.set_title("Finite-step mask (green = ok, red = NaN)")
    ax_nan.set_xlabel("step")

    # Tuning-stage boundaries on every panel.
    for ax in axs:
        for x, color in (
            (stage1, "tab:red"),
            (stage2, "tab:blue"),
            (stage3, "tab:green"),
        ):
            if x > 0:
                ax.axvline(x, color=color, linestyle="--", linewidth=1)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Tiny self-test on synthetic data so you can eyeball the layout without a
    # real run: `python mclmc_diagnostics.py` writes mclmc_diag_demo.png.
    rng = np.random.default_rng(0)
    n_chains, n_steps, dim = 8, 2000, 12
    demo = {
        "step_size": np.abs(rng.normal(0.1, 0.02, (n_chains, n_steps))),
        "L": np.abs(rng.normal(5.0, 0.5, (n_chains, n_steps))),
        "xi": np.abs(rng.normal(1.0, 0.3, (n_chains, n_steps))),
        "nonan": (rng.random((n_chains, n_steps)) > 0.02).astype(float),
        "inverse_mass_matrix": np.broadcast_to(
            np.eye(dim), (n_chains, n_steps, dim, dim)
        ).copy(),
    }
    fig = plot_mclmc_diagnostics(demo, num_burnin_steps=int(0.5 * n_steps))
    fig.savefig("mclmc_diag_demo.png", dpi=150)
    print("wrote mclmc_diag_demo.png")
