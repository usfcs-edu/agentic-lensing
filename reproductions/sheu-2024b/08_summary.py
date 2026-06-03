"""Summarize the Carousel Lens reproduction: forward-model validation numbers + the
independent PSO/emcee recovery, and make a corner plot + a recovered-vs-published table.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lenstronomy.Util.param_util import ellipticity2phi_q

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

setup = np.load(DATA / "model_setup.npz", allow_pickle=True)
chain = np.load(DATA / "emcee_chain.npy")
pso = np.load(DATA / "pso_bestfit.npy")

labels = ["theta_E_a", "gamma_a", "e1_a", "e2_a", "g1_ext", "g2_ext"]
med = np.median(chain, axis=0)
lo = np.percentile(chain, 16, axis=0)
hi = np.percentile(chain, 84, axis=0)

phi_a, q_a = ellipticity2phi_q(med[2], med[3])
g_ext = np.hypot(med[4], med[5]); phi_ext = 0.5 * np.rad2deg(np.arctan2(med[5], med[4]))

print("=" * 70)
print("CAROUSEL LENS (Sheu et al. 2024) — reproduction summary")
print("=" * 70)
print("\n[A] Forward-model validation (published best-fit -> derived quantities):")
print(f"  theta_E primary (tangential crit curve): {float(setup['theta_E_a_meas']):.2f}\" "
      f"(paper 13.03)")
print(f"  M(<theta_E) primary: {float(setup['mass_a']):.3e} Msun (paper 4.78e13, "
      f"ratio {float(setup['mass_a'])/4.78e13:.3f})")
print(f"  Sigma_crit (z_l=0.49, z_s=1.432): {float(setup['sigma_crit']):.3e} Msun/Mpc^2")

print("\n[B] Independent PSO + emcee recovery (closed-loop on conjugate image families):")
hdr = f"  {'param':10s} {'recovered':>22s}   {'published':>12s}"
print(hdr)
pub = {"theta_E_a": "13.03+/-0.02", "gamma_a": "1.67+/-0.01"}
print(f"  {'theta_E_a':10s} {med[0]:7.2f} (-{med[0]-lo[0]:.2f} +{hi[0]-med[0]:.2f})\"   "
      f"{'13.03+/-0.02':>12s}")
print(f"  {'gamma_a':10s} {med[1]:7.3f} (-{med[1]-lo[1]:.3f} +{hi[1]-med[1]:.3f})   "
      f"{'1.67+/-0.01':>12s}")
print(f"  {'q_a':10s} {q_a:7.3f}                  {'0.87+/-0.01':>12s}")
print(f"  {'PA_a [deg]':10s} {np.rad2deg(phi_a):7.1f}                  {'-45+/-1':>12s}")
print(f"  {'gamma_ext':10s} {g_ext:7.3f}                  {'0.11+/-0.01':>12s}")
print(f"  {'phi_ext':10s} {phi_ext:7.1f}                  {'9+/-1':>12s}")

# corner plot (simple)
try:
    import corner
    fig = corner.corner(chain, labels=labels,
                        truths=[13.03, 1.67, float(setup["e1_a"]), float(setup["e2_a"]),
                                float(setup["gamma1_ext"]), float(setup["gamma2_ext"])],
                        truth_color="red", show_titles=True, title_fmt=".3f")
    fig.suptitle("Carousel Lens emcee posterior (red = published best-fit)", y=1.02)
    fig.savefig(FIGS / "corner.png", dpi=110, bbox_inches="tight")
    print(f"\nWrote {FIGS / 'corner.png'}")
except ImportError:
    # fallback: pairwise scatter of the two key params
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(chain[:, 0], bins=50, color="steelblue")
    ax[0].axvline(13.03, color="r", ls="--", label="paper 13.03")
    ax[0].set_xlabel("theta_E_a [\"]"); ax[0].legend(); ax[0].set_title("theta_E posterior")
    ax[1].hist(chain[:, 1], bins=50, color="seagreen")
    ax[1].axvline(1.67, color="r", ls="--", label="paper 1.67")
    ax[1].set_xlabel("gamma_a"); ax[1].legend(); ax[1].set_title("gamma posterior")
    fig.tight_layout()
    fig.savefig(FIGS / "corner.png", dpi=120, bbox_inches="tight")
    print(f"\nWrote {FIGS / 'corner.png'} (histogram fallback)")
