"""Figure builders for the guide.

Each function is registered with @figure and called once per scheme. It returns
(fig, values); ``values`` are the numbers the figure establishes, recorded in
figures.json so the prose can quote them and the verify pass can check them.

House rule: compute, don't draw. A caustic here is the zero contour of a
numerically-differentiated Jacobian determinant, not an artist's curve. If the
reader sets q=0.3 and re-renders, the picture must still be true.
"""
from __future__ import annotations

import _style
import lensing as L
import numpy as np
from matplotlib import pyplot as plt
from registry import figure


def _mesh(half=2.5, n=400):
    g = np.linspace(-half, half, n)
    return np.meshgrid(g, g)


# --------------------------------------------------------------------------- #
# Part I — the mathematical spine
# --------------------------------------------------------------------------- #
@figure("ch04-det-j-area", chapter=4,
        caption_hint="det J is an area scaling factor — the whole of magnification")
def det_j_area(scheme):
    """A unit square pushed through a linear map; det J is the area ratio."""
    A = np.array([[0.6, 0.25], [0.1, 0.85]])
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    im = A @ sq
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, pts, t in ((axes[0], sq, "source plane  $d^2\\beta$"),
                       (axes[1], im, "image plane  $d^2\\theta$")):
        ax.fill(pts[0], pts[1], alpha=0.35, color=_style.GOOD[scheme])
        ax.plot(pts[0], pts[1], color=_style.GOOD[scheme])
        ax.set_aspect("equal")
        ax.set_xlim(-0.3, 1.5)
        ax.set_ylim(-0.3, 1.5)
        ax.set_title(t)
    det = float(np.linalg.det(A))
    axes[1].text(0.05, 1.32, f"area $\\times\\ |\\det A| = {det:.3f}$",
                 color=_style.ACCENT[scheme], fontsize=10)
    fig.suptitle("Magnification is a change of variables: $\\mu = 1/|\\det A|$",
                 fontsize=11)
    return fig, dict(det_A=det, mu=1.0 / abs(det))


@figure("ch05-kappa-gamma-eigen", chapter=5,
        caption_hint="Eigen-decomposing A splits convergence from shear")
def kappa_gamma_eigen(scheme):
    """A circle mapped by A = (1-k)I - Gamma, for pure kappa vs pure shear."""
    t = np.linspace(0, 2 * np.pi, 200)
    circ = np.vstack([np.cos(t), np.sin(t)])
    cases = [("$\\kappa=0.3$, $\\gamma=0$\n(isotropic squeeze)", 0.3, 0.0),
             ("$\\kappa=0$, $\\gamma=0.3$\n(pure stretch)", 0.0, 0.3),
             ("$\\kappa=0.3$, $\\gamma=0.3$", 0.3, 0.3)]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for ax, (title, k, g) in zip(axes, cases):
        A = np.array([[1 - k - g, 0], [0, 1 - k + g]])
        out = np.linalg.inv(A) @ circ  # source circle -> image ellipse
        ax.plot(circ[0], circ[1], ls="--", lw=1.0, color=_style.MUTED[scheme])
        ax.plot(out[0], out[1], color=_style.GOOD[scheme])
        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, dict(eig_tangential=1 - 0.3 - 0.3, eig_radial=1 - 0.3 + 0.3)


# --------------------------------------------------------------------------- #
# Part II — the observation
# --------------------------------------------------------------------------- #
@figure("ch10-sersic-profiles", chapter=10,
        caption_hint="Sersic profiles: n=1 exponential disc to n=4 de Vaucouleurs")
def sersic_profiles(scheme):
    R = np.linspace(0.05, 4, 400)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for n in (0.5, 1, 2, 4):
        ax.semilogy(R, L.sersic(R, 1.0, 1.0, n), label=f"$n={n}$")
    ax.axvline(1.0, color=_style.MUTED[scheme], ls=":", lw=1)
    ax.text(1.05, 4e-3, "$R_e$", color=_style.MUTED[scheme], fontsize=9)
    ax.set_xlabel("$R / R_e$")
    ax.set_ylabel("$I(R) / I_e$")
    ax.set_ylim(1e-3, 1e2)
    ax.legend(title="Sersic index")
    ax.set_title("All profiles pass through $I_e$ at $R_e$ — that is $b_n$'s job")
    return fig, dict(bn_n1=L.sersic_bn(1), bn_n4=L.sersic_bn(4))


@figure("ch11-drizzle-correlation", chapter=11,
        caption_hint="Why drizzling correlates noise: t(1)=(r-1)/(r-1/3)")
def drizzle_correlation(scheme):
    """The closed-form drizzle lag-1 correlation vs pixel-scale ratio."""
    r = np.linspace(1.001, 4.0, 500)
    t1 = L.drizzle_lag1(r)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(r, t1, color=_style.GOOD[scheme])
    for rr, tag in ((0.987, "native  v2d\n$0.128''$"), (3.2075, "fine  v3\n$0.04''$")):
        if rr > 1.0:
            ax.plot([rr], [L.drizzle_lag1(rr)], "o", color=_style.ACCENT[scheme])
            ax.annotate(f"{tag}\n$t(1)={L.drizzle_lag1(rr):.3f}$",
                        (rr, L.drizzle_lag1(rr)), textcoords="offset points",
                        xytext=(-70, -34), fontsize=9, color=_style.ACCENT[scheme])
    ax.set_xlabel("drizzle ratio $r$ = native / output pixel scale")
    ax.set_ylabel("lag-1 noise correlation  $t(1)$")
    ax.set_ylim(0, 1)
    ax.set_title("Upsample by $3.2\\times$ and neighbouring pixels are 77% correlated")
    return fig, dict(t1_fine=L.drizzle_lag1(3.2075), t1_r2=L.drizzle_lag1(2.0))


# --------------------------------------------------------------------------- #
# Part IV — lensing
# --------------------------------------------------------------------------- #
@figure("ch17-lens-equation", chapter=17,
        caption_hint="Solving beta = theta - alpha graphically for an SIS")
def lens_equation_1d(scheme):
    """The 1-D SIS lens equation: two intersections = two images."""
    th = np.linspace(-3, 3, 800)
    theta_E, beta0 = 1.0, 0.4
    lhs = th - theta_E * np.sign(th)  # SIS: alpha = theta_E * sign(theta)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(th, lhs, color=_style.GOOD[scheme], label=r"$\theta-\alpha(\theta)$")
    ax.axhline(beta0, color=_style.ACCENT[scheme], ls="--",
               label=rf"$\beta={beta0}$")
    roots = [beta0 + theta_E, beta0 - theta_E]
    for rt in roots:
        ax.plot([rt], [beta0], "o", color=_style.WARN[scheme], zorder=5)
    ax.set_xlabel(r"image position $\theta$ (arcsec)")
    ax.set_ylabel(r"source position $\beta$ (arcsec)")
    ax.legend(loc="upper left")
    ax.set_title("Two solutions, one source: that is a double")
    return fig, dict(image_1=roots[0], image_2=roots[1],
                     separation=abs(roots[0] - roots[1]))


@figure("ch18-sie-caustics", chapter=18,
        caption_hint="Critical curves and caustics for an SIE, from det A = 0")
def sie_caustics(scheme):
    """The classic astroid caustic and the cut — computed, not drawn.

    A *singular* isothermal ellipsoid has two branches of det A = 0, and they
    need different numerics, which is why this is not one contour call:

      * the TANGENTIAL critical curve, an oval at r ~ theta_E. A grid contour
        resolves it fine.
      * the RADIAL critical curve, which for s -> 0 collapses onto the central
        point. A cartesian grid cannot resolve it — a naive contour returns it
        as a ~3-pixel blob and maps it to a visibly polygonal artifact. Its
        image is nonetheless real and important: the CUT, the locus
        {-alpha(eps, phi)} traced as phi goes around the centre. Inside the cut
        a source has two images; outside it has one. So we build it by walking a
        small circle in phi with proper angular sampling, rather than pretending
        a grid can see it.

    The astroid's four cusps are where a source gains two more images: cross one
    and a double becomes a quad. That is the whole geometry of image multiplicity.
    """
    theta_E, q = 1.0, 0.7
    defl = lambda px, py: L.sie_deflection(px, py, theta_E, q)  # noqa: E731

    # --- tangential branch: grid contour, with the unresolved centre masked ---
    X, Y = _mesh(2.0, 600)
    D = L.det_a(defl, X, Y)
    D = np.where(np.hypot(X, Y) < 0.08, np.nan, D)  # exclude the radial branch

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))
    cs = axes[0].contour(X, Y, D, levels=[0.0], colors=[_style.GOOD[scheme]])

    # --- radial branch: a small circle, sampled in angle ---
    phi = np.linspace(0, 2 * np.pi, 720)
    eps = 1e-3
    cx, cy = eps * np.cos(phi), eps * np.sin(phi)
    axes[0].plot([0], [0], "o", ms=3, color=_style.WARN[scheme])

    # --- map both into the source plane ---
    n_seg = 0
    for path in cs.get_paths():
        v = path.vertices
        if len(v) < 2:
            continue
        ax_, ay_ = defl(v[:, 0], v[:, 1])
        axes[1].plot(v[:, 0] - ax_, v[:, 1] - ay_,
                     color=_style.ACCENT[scheme], lw=1.6,
                     label="astroid (tangential)" if n_seg == 0 else None)
        n_seg += 1
    acx, acy = defl(cx, cy)
    cut_x, cut_y = cx - acx, cy - acy
    axes[1].plot(cut_x, cut_y, color=_style.WARN[scheme], lw=1.4, ls="--",
                 label="cut (radial)")

    axes[0].set_title("image plane: critical curves\n$\\det A = 0$", fontsize=10)
    axes[1].set_title("source plane: caustics\n(the critical curves, mapped)",
                      fontsize=10)
    axes[1].legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xlabel("arcsec")
    return fig, dict(theta_E=theta_E, q=q, n_critical_segments=n_seg,
                     cut_radius_max=float(np.hypot(cut_x, cut_y).max()),
                     astroid_extent=float(np.abs(
                         np.array([v.vertices for v in cs.get_paths()][0]).max())))


@figure("ch19-mean-kappa", chapter=19,
        caption_hint="Mean convergence inside theta_E is exactly 1 — the definition")
def mean_kappa(scheme):
    th = np.linspace(0.05, 3.0, 300)
    kbar = np.array([L.mean_kappa_within(lambda t: 0.5 * 1.0 / t, x) for x in th])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(th, kbar, color=_style.GOOD[scheme], label=r"$\bar\kappa(<\theta)$, SIS")
    ax.axhline(1.0, color=_style.WARN[scheme], ls="--", lw=1.2,
               label=r"$\bar\kappa = 1$ (critical)")
    ax.axvline(1.0, color=_style.MUTED[scheme], ls=":", lw=1)
    ax.plot([1.0], [1.0], "o", color=_style.ACCENT[scheme], zorder=5)
    ax.annotate(r"$\theta_{\rm E}$", (1.0, 1.0), textcoords="offset points",
                xytext=(10, 10), color=_style.ACCENT[scheme])
    ax.set_xlabel(r"$\theta$ (units of $\theta_{\rm E}$)")
    ax.set_ylabel(r"$\bar\kappa$")
    ax.set_ylim(0, 3)
    ax.legend()
    ax.set_title("The Einstein radius is where the mean interior\nconvergence reaches critical")
    return fig, dict(kbar_at_theta_e=L.mean_kappa_within(lambda t: 0.5 / t, 1.0))


@figure("ch20-epl-slope", chapter=20,
        caption_hint="The EPL slope gamma, and where the money number sits")
def epl_slope(scheme):
    """kappa(R) for the gamma values this repo actually measured on one galaxy."""
    R = np.linspace(0.2, 3.0, 400)
    theta_E = 1.0
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    cases = [(1.103, "1.103  binned, correlated  (the money number)", _style.WARN[scheme]),
             (1.433, "1.433  native, diagonal  (the anchor)", _style.GOOD[scheme]),
             (2.000, "2.000  isothermal", _style.MUTED[scheme]),
             (2.585, "2.585  fine, diagonal  (the artifact)", _style.ACCENT[scheme])]
    for g, lab, col in cases:
        k = L.epl_kappa(R, np.zeros_like(R), theta_E, g, 1.0)
        ax.loglog(R, k, label=lab, color=col,
                  ls="--" if g in (2.0,) else "-")
    ax.axvline(1.0, color=_style.MUTED[scheme], ls=":", lw=1)
    ax.set_xlabel(r"$R / \theta_{\rm E}$")
    ax.set_ylabel(r"$\kappa(R)$")
    ax.legend(fontsize=8, title=r"$\gamma$  ($\rho \propto r^{-\gamma}$)",
              title_fontsize=8)
    ax.set_title("One galaxy, one parameter, four answers")
    return fig, dict(kappa_at_theta_e_g2=float(
        L.epl_kappa(np.array([1.0]), np.array([0.0]), 1.0, 2.0, 1.0)[0]))


# --------------------------------------------------------------------------- #
# Part V — the payoff
# --------------------------------------------------------------------------- #
@figure("ch25-gamma-bracket", chapter=25,
        caption_hint="Nine gamma values for one galaxy — the bracket, not a unification")
def gamma_bracket(scheme):
    """The guide's poster: every gamma this repo measured on DESI-165.4754-06.0423."""
    rows = [
        (1.0, None, "prior hard wall", "wall"),
        (1.103, 0.008, "binned $0.08''$, correlated, low  — the money number", "money"),
        (1.293, 0.012, "binned $0.08''$, diagonal, low", "diag"),
        (1.433, 0.034, "native $0.128''$, diagonal  — THE ANCHOR", "anchor"),
        (1.816, 0.114, "fine $0.04''$, correlated, steep", "corr"),
        (2.000, None, "prior mean / isothermal", "prior"),
        (2.353, 0.096, "native, correlated  — prior-pulled", "corr"),
        (2.423, 0.027, "binned, diagonal, steep", "diag"),
        (2.585, None, "fine $0.04''$, diagonal  — the artifact", "artifact"),
        (2.7, None, "prior hard wall", "wall"),
    ]
    col = {"wall": _style.MUTED[scheme], "money": _style.WARN[scheme],
           "anchor": _style.GOOD[scheme], "diag": _style.SERIES[scheme][2],
           "corr": _style.SERIES[scheme][4], "prior": _style.MUTED[scheme],
           "artifact": _style.ACCENT[scheme]}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, (g, s, lab, kind) in enumerate(rows):
        y = len(rows) - i
        if kind == "wall":
            ax.axvline(g, color=col[kind], ls="-", lw=2, alpha=0.5)
        if s:
            ax.errorbar([g], [y], xerr=[s], fmt="o", color=col[kind],
                        capsize=3, lw=1.6)
        else:
            ax.plot([g], [y], "|", ms=14, mew=2, color=col[kind])
        ax.text(2.78, y, lab, fontsize=8, va="center", color=col[kind])
    ax.axvline(1.433, color=_style.GOOD[scheme], ls="--", lw=1, alpha=0.6)
    ax.set_xlim(0.9, 2.8)
    ax.set_ylim(0, len(rows) + 1)
    ax.set_yticks([])
    ax.set_xlabel(r"$\gamma$   ($\rho \propto r^{-\gamma}$)")
    ax.set_title("The correlated likelihood brackets the anchor. It does not unify onto it.",
                 fontsize=10)
    ax.grid(axis="y", visible=False)
    return fig, dict(anchor=1.433, money=1.103, spread=2.585 - 1.103)


@figure("ch25-basin-flip", chapter=25,
        caption_hint="The 191-nat evidence flip: +162.2 -> -28.9")
def basin_flip(scheme):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [162.2, -28.9]
    labs = ["diagonal\nlikelihood", "correlated\nlikelihood"]
    cols = [_style.ACCENT[scheme], _style.GOOD[scheme]]
    bars = ax.bar(labs, vals, color=cols, width=0.5)
    ax.axhline(0, color=_style.INK[scheme], lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (8 if v > 0 else -18),
                f"{v:+.1f} nats\nfavours {'STEEP' if v > 0 else 'LOW'}",
                ha="center", fontsize=9,
                color=_style.ACCENT[scheme] if v > 0 else _style.GOOD[scheme])
    ax.annotate("", xy=(1, -28.9), xytext=(0, 162.2),
                arrowprops=dict(arrowstyle="->", lw=1.4,
                                color=_style.WARN[scheme], ls=":"))
    ax.text(0.5, 70, "191-nat swing\n$e^{191}\\approx10^{83}$", ha="center",
            fontsize=9, color=_style.WARN[scheme])
    ax.set_ylabel(r"$\Delta\log Z_{\rm steep-low}$  (nats)")
    ax.set_ylim(-70, 210)
    ax.set_title("The steep basin was a noise-covariance artifact")
    return fig, dict(swing=191.1, log10_bayes=82.99)


@figure("ch28-flat-purity", chapter=28,
        caption_hint="Purity is flat across human grades; the grade is ~chance vs truth")
def flat_purity(scheme):
    from scipy.stats import beta as beta_dist

    data = [("A", 79, 83), ("B", 23, 25), ("C", 20, 22)]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs, ps, los, his = [], [], [], []
    for i, (g, k, n) in enumerate(data):
        p = k / n
        lo = beta_dist.ppf(0.025, k + 0.5, n - k + 0.5)
        hi = beta_dist.ppf(0.975, k + 0.5, n - k + 0.5)
        xs.append(i); ps.append(p); los.append(p - lo); his.append(hi - p)
    ax.errorbar(xs, ps, yerr=[los, his], fmt="o", capsize=4,
                color=_style.GOOD[scheme], lw=1.6)
    ax.axhline(1.0, ls=":", color=_style.MUTED[scheme], lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g}\n{k}/{n}" for g, k, n in data])
    ax.set_xlabel("human grade")
    ax.set_ylabel("confirmed fraction (purity)")
    ax.set_ylim(0.5, 1.05)
    ax.set_xlim(-0.5, 2.5)
    ax.set_title("Flat, and not 100% at grade A\n(4 grade-A systems refuted by MUSE)",
                 fontsize=10)
    return fig, dict(purity_A=79 / 83, purity_B=23 / 25, purity_C=20 / 22)
