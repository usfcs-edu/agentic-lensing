"""Stage A'': fix the v2 PSF sampling convention -> cutout_v2c.npz.

THE DEFECT: cutout_v2.npz carries psf = empirical_psf.npy (27x27 sampled at
0.065"/px, built by 17_build_empirical_psf.py), but its meta has no
'supersample' key, so _data_lib defaults supersample=2 with delta_pix=0.13 and
the simulator's subgrid_kernel(kernel, supersample) treats the kernel as
0.13-sampled. Every v2 fit therefore ran with an effectively 2x-broadened PSF.

THE FIX: provide the kernel at the OUTPUT pixel scale (15x15 @ 0.13"/px) with
supersample=2 explicit in meta. The kernel is FITTED (Gauss-Newton least
squares through the simulator's own subgrid_kernel operator, seeded by
lenstronomy degrade_kernel) so that subgrid_kernel(psf_13, 2, odd=True)
matches the 0.065" ePSF as closely as the operator allows.

MEASURED LIMIT OF THE ROUND TRIP: empirical_psf.npy carries strong
pixel-to-pixel (fine-grid Nyquist) checkerboard structure -- an EPSFBuilder
oversampling=2 artifact, also present in the natively built 0.04" ePSF --
which is outside the range of subgrid_kernel's bilinear interpolation. The
strict 0.065-grid flatten-cosine is therefore capped at ~0.943 for ANY
0.13-sampled kernel (least-squares optimum; plain degrade gives 0.938). The
output-relevant statistic is the BAND-LIMITED cosine (both kernels smoothed
with the separable [1,2,1]/4 filter, i.e. what the simulator's 2x2
average-pool to the 0.13" grid responds to): 0.999 for the fitted kernel vs
0.804 for the defective configuration. The hard gate here is band-limited
cosine >= 0.99 + centroid shift <= 0.005"; the strict cosine is reported.

img / err_map / keep_mask are passed through from cutout_v2.npz unmodified
(the error map is NOT touched -- this isolates the PSF effect).

Outputs: data/empirical_psf_13.npy, data/cutout_v2c.npz,
         data/v2_psf_check.json, figs/psf_13_fix.png

Run:  python 40d_fix_v2_psf.py                       # GPU (incl. render check)
      python 40d_fix_v2_psf.py --skip-render-check   # CPU-safe PSF stage only
"""
import argparse
import json
from pathlib import Path

import numpy as np

import _data_lib as D

REPRO = Path(__file__).resolve().parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

DELTA_IN = 0.065            # empirical_psf.npy sampling (arcsec/px)
DELTA_OUT = 0.13            # v2 output pixel scale (arcsec/px)
SPEC_FWHM_ARCSEC = 0.119    # phase-1 quoted ePSF FWHM (see note in checks:
                            # the profile-measured ePSF FWHM is ~0.157")


# --------------------------------------------------------------- small utils
def crop_center(a, n):
    i0 = (a.shape[0] - n) // 2
    j0 = (a.shape[1] - n) // 2
    return a[i0:i0 + n, j0:j0 + n]


def pad_center(a, n):
    out = np.zeros((n, n), dtype=np.float64)
    i0 = (n - a.shape[0]) // 2
    j0 = (n - a.shape[1]) // 2
    out[i0:i0 + a.shape[0], j0:j0 + a.shape[1]] = a
    return out


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def bandlimit(k):
    """Separable [1,2,1]/4 smoothing = the response kernel of the simulator's
    2x2 average-pool on the supersampled grid (output-relevant content)."""
    from scipy.ndimage import convolve
    w = np.array([0.5, 1.0, 0.5]) / 2.0
    k = convolve(np.asarray(k, dtype=np.float64), w[None, :], mode="constant")
    return convolve(k, w[:, None], mode="constant")


def centroid_px(k):
    """Flux centroid in px relative to the geometric kernel center."""
    k = np.asarray(k, dtype=np.float64)
    yy, xx = np.indices(k.shape)
    c = (k.shape[0] - 1) / 2.0
    s = k.sum()
    return float((k * xx).sum() / s - c), float((k * yy).sum() / s - c)


def fwhm_px(k, max_fine=1200):
    """FWHM in (input) pixels from the azimuthally averaged radial profile of
    a spline-zoomed copy (robust to the ePSF's Nyquist checkerboard, which
    inflates naive above-half-max area estimates)."""
    from scipy.ndimage import zoom
    k = np.asarray(k, dtype=np.float64)
    ov = max(9, min(33, max_fine // k.shape[0]))
    f = np.clip(zoom(k, ov, order=3), 0, None)
    py, px = np.unravel_index(int(np.argmax(f)), f.shape)
    yy, xx = np.indices(f.shape)
    ri = np.hypot(xx - px, yy - py).astype(int).ravel()
    prof = (np.bincount(ri, weights=f.ravel()) / np.bincount(ri))
    half = f.max() / 2.0
    below = np.where(prof < half)[0]
    if len(below) == 0 or below[0] == 0:
        return float("nan")
    i = int(below[0])
    r_half = (i - 1) + (prof[i - 1] - half) / (prof[i - 1] - prof[i])
    return float(2.0 * r_half / ov)


def r50_px(k):
    """50% encircled-energy radius in (input) pixels -- robust to the ePSF's
    Nyquist checkerboard and to interpolation peak-flattening, unlike the
    half-max FWHM."""
    k = np.asarray(k, dtype=np.float64)
    yy, xx = np.indices(k.shape)
    cy = (k.shape[0] - 1) / 2.0
    cx = (k.shape[1] - 1) / 2.0
    r = np.hypot(xx - cx, yy - cy).ravel()
    order = np.argsort(r)
    cum = np.cumsum(k.ravel()[order]) / k.sum()
    return float(r[order][np.searchsorted(cum, 0.5)])


def n_max_from_dim(dim):
    """Shapelet order from the unconstrained-z dimension (74 -> 6, 91 -> 8)."""
    n_amp = dim - 46                       # dim = 46 + (n_max+1)(n_max+2)/2
    n_max = int(round((-3.0 + np.sqrt(1.0 + 8.0 * n_amp)) / 2.0))
    assert (n_max + 1) * (n_max + 2) // 2 == n_amp, f"bad MAP dim {dim}"
    return n_max


# ----------------------------------------------------- stage 1: build psf_13
def build_psf13():
    """Fit the 0.13"-sampled kernel through subgrid_kernel + run the checks."""
    from lenstronomy.Util.kernel_util import degrade_kernel, subgrid_kernel

    psf27 = np.load(DATA / "empirical_psf.npy").astype(np.float64)
    psf27 /= psf27.sum()

    def fwd(p):
        # exactly what LensSimulator.__init__ applies to sim_config.kernel
        return subgrid_kernel(p, 2, odd=True)          # 15x15 -> 29x29

    # seed: flux-conserving 2x degrade (averaging_even convention, 27 -> 15
    # ODD px so the kernel stays centered)
    p = np.clip(degrade_kernel(psf27, 2), 0, None)
    p /= p.sum()
    n = p.shape[0]
    target = pad_center(psf27, 2 * n - 1)              # 29x29 common frame

    # Gauss-Newton least squares: subgrid_kernel is near-affine in the kernel,
    # so one numerical Jacobian + two correction solves converge.
    eps = 1e-4
    f0 = fwd(p)
    jac = np.empty((target.size, n * n))
    for j in range(n * n):
        dp = np.zeros(n * n)
        dp[j] = eps
        jac[:, j] = (fwd(p + dp.reshape(n, n)) - f0).ravel() / eps
    for _ in range(2):
        dp, *_ = np.linalg.lstsq(jac, (target - fwd(p)).ravel(), rcond=1e-8)
        p = p + dp.reshape(n, n)
    psf13 = np.clip(p, 0, None)
    psf13 /= psf13.sum()

    # (i) round trip vs the ePSF on the common 0.065" frame
    rt = fwd(psf13)
    cos_strict = cosine(rt, target)
    cos_band = cosine(bandlimit(rt), bandlimit(target))
    # defective configuration for contrast: 27x27 ePSF mis-treated as
    # 0.13-sampled -> 53x53 supersampled kernel, 2x physical stretch
    rt_old = subgrid_kernel(psf27, 2, odd=True)
    cos_defective = cosine(rt_old, pad_center(psf27, rt_old.shape[0]))

    # (ii) centroid shift in arcsec
    cx27, cy27 = centroid_px(psf27)
    cx13, cy13 = centroid_px(psf13)
    cen_shift = float(np.hypot(cx13 * DELTA_OUT - cx27 * DELTA_IN,
                               cy13 * DELTA_OUT - cy27 * DELTA_IN))

    # (iii) widths. Half-max FWHM is artifact-sensitive here (the ePSF's
    # sharp Nyquist peak sets the half level), so r50 is reported alongside.
    fwhm_epsf = fwhm_px(psf27)                       # in 0.065" px
    fwhm13_direct = fwhm_px(psf13)                   # in 0.13" px (pixelated)
    fwhm13_subgrid = fwhm_px(rt) / 2.0               # supersampled -> out px
    fwhm_broken = fwhm_px(rt_old) / 2.0              # defective, out px
    r50_epsf = r50_px(psf27)                         # in 0.065" px
    r50_fixed = r50_px(rt) / 2.0                     # out px
    r50_broken = r50_px(rt_old) / 2.0                # out px

    ok_band = cos_band >= 0.99
    ok_cen = cen_shift <= 0.005
    print(f"[psf] (i) round-trip cosine subgrid(psf_13,2) vs ePSF: "
          f"strict {cos_strict:.4f} | band-limited {cos_band:.4f} "
          f"(gate >= 0.99 on band-limited: {'PASS' if ok_band else 'FAIL'}) | "
          f"defective config {cos_defective:.4f}")
    print("[psf]     NOTE: strict 0.065-grid cosine is capped at ~0.943 for "
          "ANY 0.13-sampled kernel -- the ePSF's fine-Nyquist checkerboard "
          "(EPSFBuilder artifact) is outside subgrid_kernel's bilinear range; "
          "the band-limited cosine is what the 2x2 average-pooled output sees")
    print(f"[psf] (ii) centroid shift = {cen_shift * 1000:.4f} mas "
          f"(gate <= 5 mas: {'PASS' if ok_cen else 'FAIL'})")
    print(f"[psf] (iii) FWHM psf_13 = {fwhm13_direct:.3f} px direct, "
          f"{fwhm13_subgrid:.3f} out-px via subgrid kernel "
          f"({fwhm13_subgrid * DELTA_OUT:.4f}\"); spec expectation "
          f"{SPEC_FWHM_ARCSEC / DELTA_OUT:.2f} px from the quoted "
          f"{SPEC_FWHM_ARCSEC}\" -- the ePSF itself profile-measures "
          f"{fwhm_epsf * DELTA_IN:.4f}\" ({fwhm_epsf:.3f} px @ 0.065\"); "
          f"by r50 the fixed kernel matches the ePSF: "
          f"{r50_fixed * DELTA_OUT:.4f}\" vs {r50_epsf * DELTA_IN:.4f}\"")
    print(f"[psf] broadening: half-max {fwhm13_subgrid:.3f} vs "
          f"{fwhm_broken:.3f} out-px ({fwhm_broken / fwhm13_subgrid:.2f}x, "
          f"artifact-limited) | r50 {r50_fixed:.3f} vs {r50_broken:.3f} "
          f"out-px ({r50_broken / r50_fixed:.2f}x) | pure pixel-scale "
          f"stretch of the ePSF: {fwhm_epsf * DELTA_IN:.4f}\" -> "
          f"{fwhm_epsf * DELTA_OUT:.4f}\" (2.00x)")
    assert ok_band, f"band-limited round-trip cosine {cos_band:.4f} < 0.99"
    assert ok_cen, f"centroid shift {cen_shift:.6f} arcsec > 0.005"

    np.save(DATA / "empirical_psf_13.npy", psf13.astype(np.float32))

    # QA figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    axes[0].imshow(psf27, origin="lower", cmap="magma")
    axes[0].set_title(f"ePSF 27x27 @ {DELTA_IN}\"\n"
                      f"FWHM {fwhm_epsf * DELTA_IN:.4f}\"")
    axes[1].imshow(psf13, origin="lower", cmap="magma")
    axes[1].set_title(f"psf_13 {n}x{n} @ {DELTA_OUT}\" (GN fit)\n"
                      f"FWHM {fwhm13_subgrid:.2f} out-px")
    axes[2].imshow(rt, origin="lower", cmap="magma")
    axes[2].set_title(f"subgrid_kernel(psf_13, 2)\ncos {cos_strict:.4f} "
                      f"strict / {cos_band:.4f} band-lim")
    im3 = axes[3].imshow(rt / rt.max() - target / target.max(),
                         origin="lower", cmap="coolwarm", vmin=-0.2, vmax=0.2)
    axes[3].set_title("(round-trip - ePSF) / peak\n(checkerboard = ePSF "
                      "Nyquist artifact)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "psf_13_fix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    section = dict(
        in_psf="empirical_psf.npy", in_shape=list(psf27.shape),
        in_delta_pix=DELTA_IN,
        out_psf="empirical_psf_13.npy", out_shape=list(psf13.shape),
        out_delta_pix=DELTA_OUT,
        method=("Gauss-Newton least-squares fit of the 0.13\"-sampled kernel "
                "through lenstronomy subgrid_kernel(., 2, odd=True), seeded "
                "by degrade_kernel; clipped >= 0, renormalized to sum 1"),
        roundtrip_cosine_strict=cos_strict,
        roundtrip_strict_gate_0p99=bool(cos_strict >= 0.99),
        roundtrip_cosine_bandlimited=cos_band,
        roundtrip_bandlimited_pass=bool(ok_band),
        roundtrip_cosine_defective_config=cos_defective,
        roundtrip_note=(
            "strict 0.065-grid cosine is capped at ~0.943 for ANY "
            "0.13-sampled kernel (least-squares optimum through the "
            "simulator's bilinear subgrid_kernel; plain degrade gives 0.938): "
            "empirical_psf.npy carries fine-Nyquist checkerboard structure "
            "from EPSFBuilder oversampling=2 that no coarse kernel can "
            "represent. The band-limited cosine ([1,2,1]/4-smoothed, the "
            "content surviving the simulator's 2x2 average-pool) is the "
            "output-relevant gate: 0.999 fixed vs 0.804 defective."),
        centroid_shift_arcsec=cen_shift, centroid_pass=bool(ok_cen),
        fwhm_psf13_px_direct=fwhm13_direct,
        fwhm_psf13_out_px_via_subgrid=fwhm13_subgrid,
        fwhm_psf13_arcsec=fwhm13_subgrid * DELTA_OUT,
        fwhm_spec_expected_px=SPEC_FWHM_ARCSEC / DELTA_OUT,
        fwhm_epsf_measured_arcsec=fwhm_epsf * DELTA_IN,
        fwhm_note=(
            "the quoted 0.119\" ePSF FWHM is not reproduced by the ePSF "
            "itself: the radial-profile FWHM of empirical_psf.npy measures "
            f"{fwhm_epsf * DELTA_IN:.4f}\" and its r50 is "
            f"{r50_epsf * DELTA_IN:.4f}\"; by r50 the fixed kernel matches "
            "the ePSF and the defective config doubles it"),
        kernel_broadening=dict(
            fwhm_halfmax_fixed_out_px=fwhm13_subgrid,
            fwhm_halfmax_defective_out_px=fwhm_broken,
            fwhm_halfmax_ratio=fwhm_broken / fwhm13_subgrid,
            r50_fixed_out_px=r50_fixed,
            r50_defective_out_px=r50_broken,
            r50_ratio=r50_broken / r50_fixed,
            epsf_pixel_scale_stretch_fixed_fwhm_arcsec=fwhm_epsf * DELTA_IN,
            epsf_pixel_scale_stretch_defective_fwhm_arcsec=(
                fwhm_epsf * DELTA_OUT),
            epsf_pixel_scale_stretch_ratio=2.0,
            note=(
                "half-max FWHM under-displays the broadening because the "
                "defective kernel keeps the ePSF's artifact-sharp Nyquist "
                "peak that sets the half level; the 50% encircled-energy "
                "radius is robust and shows the ~2x stretch directly, and "
                "the pixel-scale mis-read itself is exactly 2x (same array, "
                "0.065\" vs 0.13\" per px)"),
        ),
    )
    return psf13, section


# --------------------------------------------------- stage 2: assemble v2c
def build_cutout_v2c(psf13, psf_section, out_path):
    """cutout_v2 arrays passed through untouched; only psf + meta change."""
    src = np.load(DATA / "cutout_v2.npz")
    meta = json.loads(str(src["meta"]))
    meta.update(
        supersample=2,
        parent="cutout_v2.npz",
        psf_file="empirical_psf_13.npy",
        psf_delta_pix=DELTA_OUT,
        psf_convention_fix=(
            "kernel refitted from empirical_psf.npy (27x27 @ 0.065\"/px) to "
            "the OUTPUT pixel scale (15x15 @ 0.13\"/px, Gauss-Newton through "
            "lenstronomy subgrid_kernel) with supersample=2 explicit, so the "
            "simulator's internal subgrid_kernel reconstructs the 0.065\" "
            "ePSF; cutout_v2.npz stored the 0.065\"-sampled kernel directly, "
            "which subgrid_kernel treated as 0.13-sampled -> all pre-fix v2 "
            "fits ran with an effectively 2x-broadened PSF"),
        psf_roundtrip_cosine_strict=psf_section["roundtrip_cosine_strict"],
        psf_roundtrip_cosine_bandlimited=(
            psf_section["roundtrip_cosine_bandlimited"]),
        psf_centroid_shift_arcsec=psf_section["centroid_shift_arcsec"],
        psf_fwhm_out_px=psf_section["fwhm_psf13_out_px_via_subgrid"],
    )
    out_path.parent.mkdir(exist_ok=True)
    np.savez(out_path,
             img=src["img"], err_map=src["err_map"], keep_mask=src["keep_mask"],
             psf=psf13.astype(np.float32), meta=json.dumps(meta))
    for k in ("img", "err_map", "keep_mask"):
        assert np.array_equal(np.load(out_path)[k], src[k]), f"{k} changed"
    print(f"[v2c] wrote {out_path} (img/err_map/keep_mask identical to "
          f"cutout_v2.npz; psf {psf13.shape[0]}x{psf13.shape[1]} @ "
          f"{DELTA_OUT}\", supersample=2)")


def build_cutout_v2d(out_path):
    """cutout_v2c + the 46_noise_audit honest noise: the v2 sky term was
    calibrated on img fluctuations that include diffuse lens-wing flux, so
    sigma is mildly over-stated; rebuild err with the model-subtracted
    variance factor (Poisson term unchanged). This is the product for the
    final native-scale posterior; v2c (noise untouched) stays the clean
    PSF-isolation comparison.
    """
    audit_file = DATA / "noise_audit.json"
    if not audit_file.exists():
        print("[v2d] data/noise_audit.json missing (run 46_noise_audit.py "
              "first) -- skipping v2d")
        return
    audit = json.loads(audit_file.read_text())
    f_var = float(audit["honest"]["cutout_v2"]["sky_term_variance_factor"])
    src = np.load(DATA / "cutout_v2c.npz")
    meta = json.loads(str(src["meta"]))
    img = src["img"].astype(np.float64)
    err = src["err_map"].astype(np.float64)
    exp_time = float(meta.get("exp_time", 1197.7))
    pois = np.clip(img, 0, None) / exp_time
    sky_var = np.clip(err ** 2 - pois, 0, None)
    err_d = np.sqrt(f_var * sky_var + pois)
    meta.update(
        parent="cutout_v2c.npz",
        rescale=float(meta["rescale"]) * float(np.sqrt(f_var)),
        rescale_pre_honest=meta["rescale"],
        sky_term_variance_factor=f_var,
        honest_noise=("sky term rescaled by the 46_noise_audit "
                      "model-subtracted factor (wing-flux contamination of "
                      "the img-based calibration); Poisson term unchanged. "
                      "Factor derived with the pre-fix-PSF nm8_d model; "
                      "verify post-re-MAP that sky residual chi2 ~ 1."),
    )
    np.savez(out_path, img=src["img"], err_map=err_d.astype(np.float32),
             keep_mask=src["keep_mask"], psf=src["psf"],
             meta=json.dumps(meta))
    print(f"[v2d] wrote {out_path} (v2c psf + honest noise, sky-term "
          f"variance factor {f_var:.4f})")


# ------------------------------------------------ stage 3: render check (GPU)
def pick_best_dim74():
    """Best v2 dim-74 MAP: lowest gate chi2 among non-v3 fits."""
    cands = []
    for f in sorted(DATA.glob("map_v11_*.npz")):
        gate = f.with_name(f.stem + "_gate.json")
        if not gate.exists():
            continue
        g = json.loads(gate.read_text())
        tag = str(g.get("tag", f.stem))
        if "v3" in tag or str(g.get("data", "")).startswith("cutout_v3"):
            continue                      # fit on the fine-skycell product
        if int(np.load(f)["best_z"].shape[0]) != 74:
            continue                      # n_max!=6 or companion3 variants
        cands.append((float(g["best_chi2"]), tag, f))
    assert cands, "no dim-74 v2 MAP found"
    cands.sort(key=lambda t: t[0])
    return cands[0]


def render_check(out_path):
    """Render each v2 MAP under the old and the fixed PSF convention.

    NEVER run this on CPU (XLA grouped-conv pathology) -- the orchestrator
    runs it on an L4; --skip-render-check skips it entirely.
    """
    D.bootstrap_vendor()
    import jax  # noqa: F401
    import jax.experimental.shard_map  # noqa: F401
    import jax.numpy as jnp
    from gigalens.jax.simulator import LensSimulator
    from gigalens.simulator import SimulatorConfig

    v2c = np.load(out_path)
    meta_c = json.loads(str(v2c["meta"]))

    chi74, tag74, f74 = pick_best_dim74()
    print(f"[render] dim-74 pick: {f74.name} (tag={tag74}, "
          f"gate chi2={chi74:.4f})")
    targets = [("nm8_d", DATA / "map_v11_nm8_d.npz"), (tag74, f74)]

    def render(prob, phys, cfg, z_best, d):
        sim = LensSimulator(phys, cfg, bs=2)
        zb = jnp.asarray(np.stack([z_best, z_best]))
        x = prob.bij.forward(list(zb.T))
        model = np.asarray(sim.simulate(x))[0].astype(np.float64)
        chi = (model - d["img"]) / d["err_map"]
        return model, float(np.mean(chi[d["keep_mask"]] ** 2))

    maps = {}
    for tag, f in targets:
        z_best = np.load(f)["best_z"].astype(np.float32)
        dim = int(z_best.shape[0])
        n_max = n_max_from_dim(dim)
        gate_chi2 = float(json.loads(
            f.with_name(f.stem + "_gate.json").read_text())["best_chi2"])

        d, _, phys, prob, cfg_a = D.build_all(n_max=n_max,
                                              data_file="cutout_v2.npz")
        cfg_b = SimulatorConfig(delta_pix=float(meta_c["delta_pix"]),
                                num_pix=int(v2c["img"].shape[0]),
                                supersample=int(meta_c["supersample"]),
                                kernel=v2c["psf"].astype(np.float32))
        model_a, chi2_a = render(prob, phys, cfg_a, z_best, d)
        model_b, chi2_b = render(prob, phys, cfg_b, z_best, d)
        rms = float(np.sqrt(np.mean(
            (((model_a - model_b) / d["err_map"]) ** 2)[d["keep_mask"]])))

        reproduced = bool(abs(chi2_a - gate_chi2) <= 0.02)
        print(f"[render] {tag} (dim {dim}, n_max {n_max}): "
              f"chi2 v2={chi2_a:.4f} (gate {gate_chi2:.4f}, "
              f"reproduced: {reproduced}) | chi2 v2c={chi2_b:.4f} "
              f"(upper bound -- params optimized under broadened PSF) | "
              f"model RMS diff = {rms:.3f} sigma")
        if tag == "nm8_d":
            assert reproduced, (f"nm8_d chi2 under cutout_v2 = {chi2_a:.4f} "
                                f"!= gate {gate_chi2:.4f} (tol 0.02)")
        maps[tag] = dict(file=f.name, dim=dim, n_max=n_max,
                         chi2_gate=gate_chi2, chi2_v2=chi2_a, chi2_v2c=chi2_b,
                         gate_reproduced_within_0p02=reproduced,
                         rms_model_diff_sigma=rms)
    return dict(skipped=False, maps=maps,
                note=("chi2_v2c is an UPPER BOUND: best_z was optimized under "
                      "the 2x-broadened PSF; the real test is the re-MAP on "
                      "cutout_v2c (Perlmutter)"))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=str, default="data/cutout_v2c.npz")
    ap.add_argument("--skip-render-check", action="store_true",
                    help="CPU-safe: skip the GPU forward renders")
    args = ap.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPRO / out_path

    psf13, psf_section = build_psf13()
    build_cutout_v2c(psf13, psf_section, out_path)
    build_cutout_v2d(DATA / "cutout_v2d.npz")

    check = dict(out=str(out_path), psf_construction=psf_section,
                 render_check=dict(skipped=True))
    if not args.skip_render_check:
        check["render_check"] = render_check(out_path)

    (DATA / "v2_psf_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))
    print(f"wrote {out_path}, {DATA / 'v2_psf_check.json'}, "
          f"{DATA / 'empirical_psf_13.npy'}, {FIGS / 'psf_13_fix.png'}")


if __name__ == "__main__":
    main()
