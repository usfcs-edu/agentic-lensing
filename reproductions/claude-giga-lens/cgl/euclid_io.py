"""Euclid Q1 VIS cutout loader -> the ``cgl.paths.load_product`` dict layout.

P3 loads a single galaxy-scale Euclid Q1 strong-lens cutout (the STRONG-LENS
DISCOVERY release: reproductions/euclid-q1/data, = cgl.paths.EUCLID_Q1_DATA)
into the SAME product dict (img / err_map / keep_mask / psf / meta) that
cgl.likelihood.build_marg_model consumes unchanged. VIS is the only band we
fit: it is NATIVE 0.1"/px (the VIS detector plate scale, NOT drizzle-resampled)
so its RMS map is diagonal/instrumental -> P3-Euclid is a DIAGONAL-likelihood
demonstration of the Pillar-2 sampler recipe (whiten_fn=None), independent of
the P1 correlated machinery (which only applies to the resampled NIR bands).

Per-cutout FITS (13 ext): PRIMARY + {VIS,NIR_Y,NIR_J,NIR_H} x {FLUX,PSF,RMS}.
VIS_FLUX/VIS_RMS are 300x300 @ 0.1"; VIS_PSF is 21x21 @ 0.1". RMS is per-pixel
sigma with a large sentinel (>=1e15, ~1e16) on bad pixels. We keep the pixels
inside the published circular mask (info.json mask_radius, default 4.0" = 40px)
that are finite (RMS < 1e15) and not flagged by mask_extra_galaxies.fits (the
neighbouring-galaxy mask). The cutout is cropped to a `crop_pix` window centred
on the cutout centre so the model coordinate origin (gigalens puts (0,0)" at
the grid centre) coincides with the lens (info.json mask_centre = [0,0]").

This module is deliberately jax-free (numpy + astropy only) so the CPU unit
tests can import and exercise the mask / sentinel / PSF-normalisation logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from cgl.paths import EUCLID_Q1_DATA

# VIS instrument constants (Q1 release).
VIS_DELTA_PIX = 0.1          # "/px, native VIS detector plate scale
VIS_PSF_PIXEL_SCALE = 0.1    # VIS_PSF is sampled at the VIS pixel scale
RMS_SENTINEL = 1e15          # bad-pixel RMS values are >= this (~1e16)
# Nominal VIS exposure metadata (INFORMATIONAL ONLY: the diagonal likelihood
# uses err_map=VIS_RMS directly, so exp_time never enters the log-posterior).
VIS_EXP_TIME = 566.0


def target_dir(id_str: str, subset: str = "lens") -> Path:
    """Directory of a Euclid Q1 target (subset in {lens, recenter, group, ...})."""
    return EUCLID_Q1_DATA / subset / id_str


def load_published_mass(id_str: str, subset: str = "lens") -> dict:
    """The PyAutoLens SIE+shear fit (result_lens_mass.json) as a plain dict.

    Keys of interest: einstein_radius_effective_* (area-equivalent, the
    convention-robust radius we anchor on), einstein_radius_* (major-axis SIE
    param), ell_comps_0/1, centre_0/1, shear_gamma_1/2. autolens conventions:
    (y, x) order on centres, arcsec, origin = cutout centre, z_lens=0.5 /
    z_src=1.0 fixed defaults (only the angular Einstein radius is meaningful).
    """
    p = target_dir(id_str, subset) / "result_lens_mass.json"
    return json.loads(p.read_text())


def _load_mask_extra(tdir: Path, shape) -> np.ndarray:
    """Neighbouring-galaxy mask (1 = flagged extra galaxy) or all-zeros."""
    f = tdir / "mask_extra_galaxies.fits"
    if not f.exists():
        return np.zeros(shape, dtype=bool)
    with fits.open(f) as h:
        data = None
        for hdu in h:
            if hdu.data is not None:
                data = np.asarray(hdu.data)
                break
    if data is None:
        return np.zeros(shape, dtype=bool)
    return data.astype(bool)


def load_euclid_vis(id_str: str, subset: str = "lens", *, crop_pix: int = 100,
                    supersample: int = 2, mask_radius_arcsec: float | None = None,
                    exp_time: float = VIS_EXP_TIME) -> dict:
    """Load one Euclid Q1 VIS cutout as a build_marg_model product dict.

    Args:
        id_str: target id (directory name), e.g. "102020061_NEG60708...".
        subset: data subset dir ("lens" default; "recenter" for the
            lens-light-recentred variants).
        crop_pix: side of the square crop centred on the cutout centre
            (default 100 = 10"; >= 2*(mask_radius + psf_halfwidth) so every
            kept pixel's PSF-convolution support is on the grid). Must be even.
        supersample: simulator supersampling (meta only; default 2).
        mask_radius_arcsec: circular-mask radius; None reads info.json
            mask_radius (default 4.0").
        exp_time: informational VIS exposure time (unused by the diagonal
            likelihood; err_map=VIS_RMS is the noise).

    Returns:
        dict(img, err_map, keep_mask, psf, meta) in the load_product layout.
        img=VIS_FLUX (cropped, float64); err_map=VIS_RMS (cropped, float64,
        sentinel preserved); keep_mask=(RMS<1e15)&circle(r=mask_radius/0.1)&
        ~mask_extra_galaxies; psf=VIS_PSF/psf.sum() (sum=1); meta declares
        delta_pix=0.1, supersample, psf_pixel_scale=0.1 (so the simulator PSF
        guard engages and passes), plus provenance and the published model.
    """
    if crop_pix % 2 != 0:
        raise ValueError(f"crop_pix must be even, got {crop_pix}")
    tdir = target_dir(id_str, subset)
    info = json.loads((tdir / "info.json").read_text())
    if mask_radius_arcsec is None:
        mask_radius_arcsec = float(info.get("mask_radius", 4.0))

    with fits.open(tdir / f"{id_str}.fits") as h:
        vis_flux = np.asarray(h["VIS_FLUX"].data, dtype=np.float64)
        vis_rms = np.asarray(h["VIS_RMS"].data, dtype=np.float64)
        vis_psf = np.asarray(h["VIS_PSF"].data, dtype=np.float64)
        prime_hdr = h["VIS_FLUX"].header
    mag_zero_point = float(prime_hdr.get("MAGZERO", np.nan))

    full_pix = vis_flux.shape[0]
    if vis_flux.shape != vis_rms.shape:
        raise ValueError("VIS_FLUX / VIS_RMS shape mismatch")
    mask_extra_full = _load_mask_extra(tdir, vis_flux.shape)

    # ---- crop centred on the cutout centre (== model (0,0)" origin) --------
    cc = full_pix // 2
    half = crop_pix // 2
    if cc - half < 0 or cc + half > full_pix:
        raise ValueError(f"crop_pix={crop_pix} exceeds cutout {full_pix}")
    sl = slice(cc - half, cc + half)
    img = np.ascontiguousarray(vis_flux[sl, sl])
    err_map = np.ascontiguousarray(vis_rms[sl, sl])
    mask_extra = np.ascontiguousarray(mask_extra_full[sl, sl])

    # ---- keep mask: finite RMS & inside the circular mask & not a neighbour -
    yy, xx = np.indices(img.shape)
    center = (crop_pix - 1) / 2.0            # gigalens origin = grid mean coord
    r = np.hypot(xx - center, yy - center)
    mask_radius_px = mask_radius_arcsec / VIS_DELTA_PIX
    finite = err_map < RMS_SENTINEL
    keep_mask = finite & (r <= mask_radius_px) & (~mask_extra)

    # ---- PSF normalised to sum = 1 (raw VIS_PSF sums to ~0.93-0.97) --------
    psf_sum = float(vis_psf.sum())
    psf = vis_psf / psf_sum

    n_sentinel = int((~finite).sum())
    n_extra = int(mask_extra.sum())
    meta = dict(
        delta_pix=VIS_DELTA_PIX, num_pix=int(crop_pix), supersample=int(supersample),
        psf_pixel_scale=VIS_PSF_PIXEL_SCALE, exp_time=float(exp_time),
        band="VIS", id_str=id_str, subset=subset,
        mag_zero_point=mag_zero_point,
        mask_radius_arcsec=float(mask_radius_arcsec),
        mask_radius_px=float(mask_radius_px),
        crop_center_full=int(cc), full_pix=int(full_pix),
        psf_raw_sum=psf_sum, n_keep=int(keep_mask.sum()),
        n_rms_sentinel_in_crop=n_sentinel, n_extra_galaxy_masked=n_extra,
        source="euclid-q1 VIS cutout (native 0.1\"/px, diagonal RMS) via "
               "cgl.euclid_io.load_euclid_vis; exp_time is informational only "
               "(diagonal likelihood uses err_map=VIS_RMS).",
    )
    return dict(img=img, err_map=err_map, keep_mask=keep_mask, psf=psf, meta=meta)
