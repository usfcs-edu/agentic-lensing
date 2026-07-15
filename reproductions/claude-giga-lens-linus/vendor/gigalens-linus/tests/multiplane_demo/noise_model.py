"""Add an explicit observational noise model to the 3-plane demo truth image and
emit a model card + a reviewable noisy-data figure.

This is the FIRST step of the recovery stage. It is deterministic (a fixed-seed noise
realization on the saved lenstronomy truth) -- it runs no fit. The expensive/consequential
runs (MAP, MCMC) are gated separately, after the noise + prior design is approved.

Noise model (gigalens ``ImageData`` convention; see scene_prob_model.py §3.4 -- there is NO
silent noise default, this must be stated explicitly):

    sigma_pixel = sqrt(background_rms**2 + clip(image, 0, inf) / exp_time)

i.e. a constant background term + a Poisson (shot-noise) term that grows with flux. The
data is the lenstronomy *truth* render (the oracle) + N(0, sigma). gigalens is the MODEL
that will be asked to recover the truth -- so even at the true parameters the gigalens
model differs from the data-mean by the fixed Sersic b_n-floor residual
(``resid = gimg - img``). The card reports that systematic's contribution to reduced chi^2
at truth, Sum(bn/sigma)^2 / N, which must stay << 1 (and << the chi^2 spread sqrt(2/N))
for the recovery to be noise- not systematic-limited.

Run (inside the canonical Shifter container; see GIGALens-Code/docs/env_setup.md):

    /usr/bin/python3 -m tests.multiplane_demo.noise_model        # from ~/gigalens
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from . import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# ---- noise level (deeper-than-single-orbit HST; user-selected 2026-06-24).
# Source-ring integrated SNR ~58, peak SNR ~41 -- gives the weakly-identified z1 its
# best shot. b_n systematic is negligible vs the noise at this level (see model card).
BACKGROUND_RMS = 0.005    # constant background sigma (flux units of the rendered image)
EXP_TIME = 2000.0         # effective exposure -> Poisson term sqrt(flux/EXP_TIME)
NOISE_SEED = 20260624     # recorded RNG seed for the noise realization (reproducibility)


def error_map(image, background_rms=BACKGROUND_RMS, exp_time=EXP_TIME):
    """Per-pixel noise sigma -- the same formula ImageData uses internally (§3.4)."""
    return np.sqrt(background_rms ** 2 + np.clip(image, 0.0, np.inf) / exp_time)


def make_noisy(image, background_rms=BACKGROUND_RMS, exp_time=EXP_TIME, seed=NOISE_SEED):
    """Return (noisy_image, sigma_map, noise_realization). Fixed seed = reproducible."""
    sig = error_map(image, background_rms, exp_time)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(image.shape) * sig
    return image + noise, sig, noise


def model_card(img, gimg, sig, background_rms=BACKGROUND_RMS, exp_time=EXP_TIME):
    """Print the effective forward model + noise diagnostics (no silent defaults)."""
    bn = gimg - img                                   # fixed b_n-floor model residual
    N = img.size
    spread = np.sqrt(2.0 / N)
    src = np.load(os.path.join(ART, "demo_3plane.npz"))["img_sz3"]
    m = src > 0.01 * src.max()                         # source-ring footprint
    peak_snr = img.max() / sig[np.unravel_index(img.argmax(), img.shape)]
    ring_snr = src[m].sum() / np.sqrt((sig[m] ** 2).sum())
    chi2_bn = float(np.sum((bn / sig) ** 2) / N)

    print("=" * 70)
    print("MODEL CARD -- 3-plane demo, recovery noise model (UNCERTIFIED, proposed)")
    print("=" * 70)
    print(f"  grid        : {C.NUM_PIX}x{C.NUM_PIX} @ {C.DELTA_PIX}\"/pix "
          f"(FOV {C.NUM_PIX*C.DELTA_PIX:.1f}\"), supersample {C.SUPERSAMPLE}")
    print(f"  PSF         : Gaussian FWHM {C.PSF_FWHM}\"  "
          f"(half-width {C.PSF_HALF_PIX} pix -> {2*C.PSF_HALF_PIX+1}x{2*C.PSF_HALF_PIX+1})")
    print(f"  precision   : float64 (likelihood_precision='float64')")
    print(f"  noise       : background_rms={background_rms}, exp_time={exp_time}, "
          f"seed={NOISE_SEED}")
    print(f"                sigma = sqrt(bg^2 + clip(img,0,inf)/exp_time)  [§3.4, explicit]")
    print(f"  sigma range : {sig.min():.4e} (bg-limited) .. {sig.max():.4e} (core, Poisson)")
    print(f"  SNR         : peak {peak_snr:.1f}  |  source-ring integrated {ring_snr:.1f}")
    print(f"  b_n floor   : reduced-chi2 bias at truth = Sum(bn/sig)^2/N = {chi2_bn:.2e}")
    print(f"                (chi2 spread sqrt(2/N) = {spread:.4f}; "
          f"bias is {chi2_bn/spread:.3f}x spread -> noise-limited, not systematic-limited)")
    print("=" * 70)


def make_figure(img, noisy, sig, noise, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    half = C.NUM_PIX * C.DELTA_PIX / 2
    ext = [-half, half, -half, half]
    vmax = img.max()
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))

    def show(a, title, norm, cmap="magma"):
        i = ax[k].imshow(a, origin="lower", extent=ext, norm=norm, cmap=cmap)
        ax[k].set_title(title); ax[k].set_xlabel("arcsec"); ax[k].set_ylabel("arcsec")
        plt.colorbar(i, ax=ax[k], fraction=0.046)

    k = 0; show(img, "truth (noiseless, sqrt)", PowerNorm(gamma=0.5, vmin=0, vmax=vmax))
    k = 1; show(noisy, "noisy data (sqrt)", PowerNorm(gamma=0.5, vmin=0, vmax=vmax))
    rr = float(np.max(np.abs(noise)))
    k = 2; i = ax[2].imshow(noise, origin="lower", extent=ext, cmap="RdBu_r",
                            vmin=-rr, vmax=rr)
    ax[2].set_title("noise realization (data - truth)")
    ax[2].set_xlabel("arcsec"); ax[2].set_ylabel("arcsec")
    plt.colorbar(i, ax=ax[2], fraction=0.046)
    k = 3; i = ax[3].imshow(sig, origin="lower", extent=ext, cmap="viridis")
    ax[3].set_title("per-pixel sigma (error map)")
    ax[3].set_xlabel("arcsec"); ax[3].set_ylabel("arcsec")
    plt.colorbar(i, ax=ax[3], fraction=0.046)

    plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    npz = np.load(os.path.join(ART, "demo_3plane.npz"))
    img, gimg = npz["img"], npz["gimg"]           # lenstronomy truth, gigalens-at-truth
    noisy, sig, noise = make_noisy(img)
    model_card(img, gimg, sig)
    fig_path = os.path.join(ART, "demo_3plane_noisy.png")
    make_figure(img, noisy, sig, noise, fig_path)
    np.savez(os.path.join(ART, "demo_3plane_noisy.npz"),
             noisy=noisy, sigma=sig, noise=noise, truth=img,
             background_rms=BACKGROUND_RMS, exp_time=EXP_TIME, seed=NOISE_SEED)
    print(f"saved: {fig_path}")
    print(f"saved: {os.path.join(ART, 'demo_3plane_noisy.npz')}")


if __name__ == "__main__":
    main()
