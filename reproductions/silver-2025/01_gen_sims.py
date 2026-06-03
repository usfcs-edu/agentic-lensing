#!/usr/bin/env python3
"""
01_gen_sims.py  --  Simulated HST images for Silver et al. 2025 "Model 1a" (HST-long).

Reproduces the lensed/unlensed image-simulation pipeline of Silver+2025 §3.1
for the CONVENTIONAL Einstein-radius regime (Model 1, 0.5" < theta_E < 1.5"),
which is the headline "find every conventional lens" result (validation AUC 0.9978).

Paper setup we follow (§3.1.1, §3.1.2, §3.1.5):
  - lenstronomy, pixel scale 0.031", Gaussian PSF truncated at 3 sigma.
  - LENS mass: SIE, theta_E ~ drawn so the lens sits in 0.5"-1.5".
  - For Model 1, the LENS is placed at the cutout center; the SOURCE is offset by
        X, Y ~ N(mu=0", sigma=0.25").
  - Source Sersic index n ~ U(2, 6) (paper's stated range to match Simard+2011).
  - Arc brightness scaled up by 10^U(0.5, 2.0)  (selection-bias replication).
  - Environmental galaxies: a few extra Sersic ellipses scaled by 10^U(0, 0.7).
  - LENSED  (class 1): lens light + lensed source + environmental galaxies.
  - UNLENSED(class 0): same central bright galaxy (lens light), NO source,
        same environmental galaxies (paper: "turn off source light and set theta_E=0").
  - NO noise added here. Noise is added as an augmentation layer in training
        (02_train_resnet.py): sigma_BKG ~ U(0,0.2), texp ~ 10^U(2,6).
  - Preprocessing AT SIM TIME for Model 1 (paper §3.1.5): subtract mean, divide std,
        clip pixels above the 99th percentile to the 99th-percentile value.

FIDELITY CAVEATS (documented next-steps, not blockers):
  * Sources here are analytic Sersic profiles, NOT the VELA hydrodynamical
    galaxy stamps the paper uses. VELA/CosmoDC2/JAGUAR are public but bulky; using
    Sersic sources is the explicit MVP proxy requested for this stand-up. This will
    under-represent the irregular/clumpy high-z morphologies that make the arcs
    visually rich (paper Fig. 6). Swapping in VELA stamps is the #1 fidelity upgrade.
  * Lens/source redshifts, M*, ellipticities come from simple priors here instead of
    being drawn jointly from CosmoDC2 + Shuntov+2022 M*-Mhalo relation. theta_E is
    sampled directly U(0.5,1.5)" rather than derived from a halo-mass cut.
  * Single band (F606W-like, 1 channel), matching the WFC3/F606W test set.

Output (gitignored data/):
  data/model1_images.npy   float32 (N, 1, NPIX, NPIX)  normalized, noiseless
  data/model1_labels.npy   uint8 (N,)   1 = lensed, 0 = unlensed
  data/model1_meta.csv     per-image parameters (theta_E, n_sersic, etc.)
  data/model1_preview.png  25-panel preview grid of lensed examples (matches Fig. 6 layout)

Run (gigalens-free; uses lenstronomy in the `lens` venv):
  /home/benson/.venvs/lens/bin/python 01_gen_sims.py --n_per_class 2000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Util import util

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# ---------- Instrument / simulation constants (paper §3.1.1) ----------------
PIXEL_SCALE = 0.031          # arcsec / pixel (paper)
NPIX = 64                    # cutout side in pixels (~2.0" FoV; paper doesn't state
                             # an exact value, 64 px is the conventional Huang-style
                             # cutout and comfortably contains a 1.5" Einstein ring).
PSF_FWHM = 0.08              # arcsec, Gaussian PSF (HST WFC3/UVIS F606W ~ 0.07-0.08")
THETA_E_MIN, THETA_E_MAX = 0.5, 1.5   # Model 1 range


def make_grid(npix: int, pixscale: float):
    """Lenstronomy coordinate grid (RA/Dec arrays in arcsec) for a square cutout."""
    x_grid, y_grid, ra0, dec0, _, _, Mpix2a, Ma2pix = util.make_grid_with_coordtransform(
        numPix=npix, deltapix=pixscale, subgrid_res=1, inverse=False
    )
    return x_grid, y_grid, ra0, dec0, Mpix2a, Ma2pix


def build_models():
    """Static lenstronomy model containers shared across all sims."""
    lens_model = LensModel(lens_model_list=["SIE"])
    # source: single Sersic ellipse; lens light: single Sersic ellipse;
    # environmental: up to N_ENV extra Sersic ellipses.
    source_model = LightModel(light_model_list=["SERSIC_ELLIPSE"])
    return lens_model, source_model


def normalize_model1(img: np.ndarray) -> np.ndarray:
    """Paper §3.1.5 preprocessing for Model 1 (done at sim time):
    subtract mean, divide by std, then clip above the 99th percentile."""
    x = img.astype(np.float64)
    x = x - x.mean()
    std = x.std()
    if std > 0:
        x = x / std
    p99 = np.percentile(x, 99.0)
    x = np.minimum(x, p99)
    return x.astype(np.float32)


def sample_sersic_ellipticity(rng):
    """Return (e1, e2) for a moderately flattened ellipse, |e| <~ 0.4."""
    q = rng.uniform(0.5, 0.95)          # axis ratio
    phi = rng.uniform(0, np.pi)
    eps = (1 - q) / (1 + q)
    e1 = eps * np.cos(2 * phi)
    e2 = eps * np.sin(2 * phi)
    return float(e1), float(e2)


def gen_one(rng, lens_model, source_model, x_grid, y_grid, npix, lensed: bool):
    """Generate one (image, meta) pair. lensed=True -> class 1, else class 0."""
    data_kw = {
        "image_data": np.zeros((npix, npix)),
        "transform_pix2angle": np.array([[PIXEL_SCALE, 0], [0, PIXEL_SCALE]]),
        "ra_at_xy_0": x_grid.reshape(npix, npix)[0, 0],
        "dec_at_xy_0": y_grid.reshape(npix, npix)[0, 0],
    }
    image_data = ImageData(**data_kw)
    psf = PSF(psf_type="GAUSSIAN", fwhm=PSF_FWHM, pixel_size=PIXEL_SCALE,
              truncation=3.0)

    # --- lens mass (SIE) ---
    theta_E = float(rng.uniform(THETA_E_MIN, THETA_E_MAX))
    le1, le2 = sample_sersic_ellipticity(rng)   # reuse for SIE mass ellipticity
    kwargs_lens = [{"theta_E": theta_E, "e1": le1, "e2": le2,
                    "center_x": 0.0, "center_y": 0.0}]

    # --- lens light (central bright Sersic) — present in BOTH classes ---
    lens_light_R = float(rng.uniform(0.4, 1.2))
    lens_light_n = float(rng.uniform(2.5, 5.0))   # de Vaucouleurs-ish elliptical
    ll_e1, ll_e2 = sample_sersic_ellipticity(rng)
    lens_light_amp = float(10 ** rng.uniform(1.0, 2.0))
    kwargs_lens_light = [{"amp": lens_light_amp, "R_sersic": lens_light_R,
                          "n_sersic": lens_light_n, "e1": ll_e1, "e2": ll_e2,
                          "center_x": 0.0, "center_y": 0.0}]

    # --- environmental galaxies (a few faint Sersic ellipses) — BOTH classes ---
    n_env = int(rng.integers(0, 4))
    env_light_list, kwargs_env = [], []
    for _ in range(n_env):
        ex = float(rng.uniform(-0.9, 0.9))
        ey = float(rng.uniform(-0.9, 0.9))
        e1, e2 = sample_sersic_ellipticity(rng)
        amp = float(10 ** rng.uniform(0.0, 0.7)) * float(rng.uniform(1.0, 5.0))
        env_light_list.append("SERSIC_ELLIPSE")
        kwargs_env.append({"amp": amp, "R_sersic": float(rng.uniform(0.1, 0.4)),
                           "n_sersic": float(rng.uniform(1.0, 4.0)),
                           "e1": e1, "e2": e2, "center_x": ex, "center_y": ey})

    # --- source (lensed for class 1; absent for class 0) ---
    src_n = float(rng.uniform(2.0, 6.0))               # paper: n ~ U(2,6)
    src_R = float(rng.uniform(0.05, 0.3))
    se1, se2 = sample_sersic_ellipticity(rng)
    src_x = float(rng.normal(0.0, 0.25))               # paper: X,Y ~ N(0, 0.25")
    src_y = float(rng.normal(0.0, 0.25))
    arc_boost = float(10 ** rng.uniform(0.5, 2.0))      # paper: arc x 10^U(0.5,2.0)
    src_amp = float(rng.uniform(2.0, 20.0)) * arc_boost

    # Build the combined light model (lens light + environmental).
    foreground_list = ["SERSIC_ELLIPSE"] + env_light_list
    foreground_kwargs = kwargs_lens_light + kwargs_env
    fg_model = LightModel(light_model_list=foreground_list)

    if lensed:
        src_kwargs = [{"amp": src_amp, "R_sersic": src_R, "n_sersic": src_n,
                       "e1": se1, "e2": se2, "center_x": src_x, "center_y": src_y}]
        im = ImageModel(image_data, psf, lens_model_class=lens_model,
                        source_model_class=source_model,
                        lens_light_model_class=fg_model)
        img = im.image(kwargs_lens=kwargs_lens, kwargs_source=src_kwargs,
                       kwargs_lens_light=foreground_kwargs)
        label = 1
    else:
        # Unlensed: lens light + environmental galaxies only (theta_E -> 0, no source).
        im = ImageModel(image_data, psf, lens_light_model_class=fg_model)
        img = im.image(kwargs_lens_light=foreground_kwargs)
        label = 0

    img = normalize_model1(np.asarray(img))
    meta = dict(label=label, theta_E=theta_E, src_n=src_n, src_R=src_R,
                src_x=src_x, src_y=src_y, arc_boost=arc_boost, n_env=n_env,
                lens_light_n=lens_light_n)
    return img, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=2000,
                    help="images per class (paper uses 10000; default 2000 starter set)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--npix", type=int, default=NPIX)
    ap.add_argument("--out_prefix", type=str, default="model1")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    npix = args.npix
    x_grid, y_grid, *_ = make_grid(npix, PIXEL_SCALE)
    lens_model, source_model = build_models()

    n_total = 2 * args.n_per_class
    images = np.zeros((n_total, 1, npix, npix), dtype=np.float32)
    labels = np.zeros((n_total,), dtype=np.uint8)
    meta_rows = []

    t0 = time.time()
    idx = 0
    for cls_lensed in (True, False):
        for _ in range(args.n_per_class):
            img, meta = gen_one(rng, lens_model, source_model, x_grid, y_grid,
                                npix, lensed=cls_lensed)
            images[idx, 0] = img
            labels[idx] = meta["label"]
            meta_rows.append(meta)
            idx += 1
            if idx % 500 == 0:
                dt = time.time() - t0
                print(f"  [{idx}/{n_total}]  {dt:.1f}s  ({idx/dt:.1f} img/s)", flush=True)

    # Shuffle so classes are interleaved (train script also splits/shuffles, but tidy).
    perm = rng.permutation(n_total)
    images, labels = images[perm], labels[perm]
    meta_rows = [meta_rows[i] for i in perm]

    np.save(DATA / f"{args.out_prefix}_images.npy", images)
    np.save(DATA / f"{args.out_prefix}_labels.npy", labels)

    # meta CSV
    import csv
    with open(DATA / f"{args.out_prefix}_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        w.writeheader()
        w.writerows(meta_rows)

    # preview grid of 25 lensed examples
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lensed_idx = np.where(labels == 1)[0][:25]
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for ax, li in zip(axes.ravel(), lensed_idx):
            ax.imshow(images[li, 0], origin="lower", cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Silver2025 Model 1a (HST-long) — simulated LENSED examples (Sersic-source MVP)")
        fig.tight_layout()
        fig.savefig(DATA / f"{args.out_prefix}_preview.png", dpi=90)
        print(f"  wrote preview grid -> data/{args.out_prefix}_preview.png")
    except Exception as e:
        print(f"  (preview skipped: {e})")

    dt = time.time() - t0
    n_pos = int(labels.sum())
    print(f"DONE: {n_total} images ({n_pos} lensed / {n_total-n_pos} unlensed) "
          f"in {dt:.1f}s  ({n_total/dt:.1f} img/s)")
    print(f"  images -> data/{args.out_prefix}_images.npy  shape={images.shape}")
    print(f"  labels -> data/{args.out_prefix}_labels.npy")
    print(f"  pixel value range: [{images.min():.3f}, {images.max():.3f}]")


if __name__ == "__main__":
    main()
