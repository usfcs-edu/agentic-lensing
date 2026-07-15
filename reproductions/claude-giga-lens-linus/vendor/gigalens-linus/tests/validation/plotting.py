"""Diagnostic comparison plots for the validation suite.

method-discipline §5 ("plots before metrics — and the plot wins") and §4 ("name the
metric's blind spot"): a passing ``allclose`` is blind to *structured* residuals that
stay under tolerance. The numeric assertion remains the automated gate, but these
plots let a human LOOK at the full lenstronomy-vs-gigalens comparison and catch
structure the aggregate misses. A passing aggregate with a structured residual map is
an open finding, not a pass — reconcile toward investigating (§5).

Plots are *artifacts*, never the pass/fail decision. They are written when ``--plots``
is passed (so passing tests can be eyeballed too) and automatically on failure (for
debugging) — see ``conftest.py``. matplotlib is imported lazily so the suite collects
and runs without it when plotting is off.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class ComparisonPlotter:
    """Collects registered comparisons for one test; renders them on demand.

    A test calls ``plotter.image(...)`` / ``plotter.curve(...)`` to register what it
    compared; ``conftest`` calls ``render()`` afterwards iff plotting is enabled or the
    test failed. Registering is cheap (no matplotlib) so it is always safe to call.
    """

    def __init__(self, name: str, plot_dir: Path):
        self._name = name
        self._dir = Path(plot_dir)
        self._items: list = []

    def image(self, title, ref, model, *, rtol, atol=0.0, err=None,
              labels=("lenstronomy", "gigalens")):
        """Register a 2-D field comparison (e.g. final image, β_x, convergence).

        ``err``: optional per-pixel sigma. If given, the residual panel is the
        noise-normalized ``(model-ref)/err`` (should look like N(0,1) — see
        inference-diagnostics.md); else it is the relative residual ``(model-ref)/
        (|ref|+atol)``.
        """
        self._items.append(("image", dict(
            title=title, ref=np.asarray(ref), model=np.asarray(model),
            rtol=rtol, atol=atol, err=None if err is None else np.asarray(err),
            labels=labels)))

    def curve(self, title, x, ref, model, *, xlabel, ylabel,
              labels=("astropy", "gigalens")):
        """Register a 1-D overlay comparison (e.g. distance vs redshift)."""
        self._items.append(("curve", dict(
            title=title, x=np.asarray(x), ref=np.asarray(ref),
            model=np.asarray(model), xlabel=xlabel, ylabel=ylabel, labels=labels)))

    # -- rendering -----------------------------------------------------------------
    def render(self) -> Sequence[Path]:
        if not self._items:
            return []
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, (kind, p) in enumerate(self._items):
            fig = (_render_image(plt, p) if kind == "image" else _render_curve(plt, p))
            suffix = p["title"].replace("/", "_").replace(" ", "_")[:60]
            path = self._dir / f"{self._name}__{i:02d}_{suffix}.png"
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        return paths


def _squeeze2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim > 2:  # collapse any trailing/leading singletons defensively
        a = a.reshape(a.shape[-2], a.shape[-1]) if a.shape[-1] > 1 else a[..., 0]
    return a


def _render_image(plt, p):
    ref, model = _squeeze2d(p["ref"]), _squeeze2d(p["model"])
    diff = model - ref
    if p["err"] is not None:
        resid = diff / _squeeze2d(p["err"])
        resid_label = "(model − ref) / σ"
    else:
        resid = diff / (np.abs(ref) + p["atol"] + 1e-300)
        resid_label = "(model − ref) / (|ref| + atol)"

    max_abs = float(np.nanmax(np.abs(diff)))
    max_rel = float(np.nanmax(np.abs(diff) / (np.abs(ref) + p["atol"] + 1e-300)))

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
    vmax = float(np.nanmax(np.abs(ref))) or 1.0
    for ax, img, ttl in (
        (axes[0], ref, p["labels"][0]),
        (axes[1], model, p["labels"][1]),
    ):
        im = ax.imshow(img, origin="lower", vmin=-vmax, vmax=vmax, cmap="viridis")
        ax.set_title(ttl); fig.colorbar(im, ax=ax, fraction=0.046)
    dmax = float(np.nanmax(np.abs(diff))) or 1.0
    im = axes[2].imshow(diff, origin="lower", vmin=-dmax, vmax=dmax, cmap="RdBu_r")
    axes[2].set_title("difference"); fig.colorbar(im, ax=axes[2], fraction=0.046)
    rmax = float(np.nanmax(np.abs(resid))) or 1.0
    im = axes[3].imshow(resid, origin="lower", vmin=-rmax, vmax=rmax, cmap="RdBu_r")
    axes[3].set_title(resid_label); fig.colorbar(im, ax=axes[3], fraction=0.046)
    axes[4].hist(resid[np.isfinite(resid)].ravel(), bins=60, color="0.4")
    axes[4].set_title("residual histogram")

    if p["err"] is not None:
        # Noise-normalized residual. Report residual MAGNITUDE (always meaningful), NOT
        # a per-pixel rtol and NOT a "χ²": Σ(r/σ)²/N is only a calibrated goodness-of-fit
        # when the data carries matching noise (then ≈1 and the histogram ~N(0,1)); on
        # noiseless data a good fit drives it to ~0, so it is just a residual norm.
        maxr = float(np.nanmax(np.abs(resid)))
        peak = float(np.nanmax(np.abs(ref))) or 1.0
        fig.suptitle(
            f"{p['title']}  max|r|/σ={maxr:.2e}  max|Δ|={max_abs:.2e}  "
            f"max|Δ|/peak={max_abs / peak:.2e}  — noise-normalized residual "
            f"(structure ⇒ mismodeling, §5; histogram ~N(0,1) ⇔ good fit to NOISY data)",
            fontsize=10)
    else:
        status = "PASS" if np.allclose(model, ref, rtol=p["rtol"], atol=p["atol"]) else "FAIL"
        fig.suptitle(
            f"{p['title']}  [{status} @ rtol={p['rtol']:.0e}, atol={p['atol']:.0e}]  "
            f"max|Δ|={max_abs:.3e}  max relΔ={max_rel:.3e}  "
            f"— structured residual ⇒ open finding (method-discipline §5)",
            fontsize=10)
    return fig


def _render_curve(plt, p):
    x, ref, model = p["x"], p["ref"], p["model"]
    frac = (model - ref) / (np.abs(ref) + 1e-300)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
    a0.plot(x, ref, "o-", label=p["labels"][0])
    a0.plot(x, model, "x--", label=p["labels"][1])
    a0.set_ylabel(p["ylabel"]); a0.legend(); a0.set_title(p["title"])
    a1.axhline(0, color="0.7", lw=0.8)
    a1.plot(x, frac, "s-")
    a1.set_ylabel("frac. resid."); a1.set_xlabel(p["xlabel"])
    return fig
