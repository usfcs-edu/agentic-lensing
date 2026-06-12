"""Stage A: build the v2 data product per the research-group feedback.

1. CROP   -- 80x80 px (10.4" at 0.13"/px) so the lens system fills the frame.
2. MASKS  -- interloping objects that do not contribute to the lensing:
             whole segments outside the arc annulus (the paper's faint
             galaxies), compact point-like interlopers ON the arcs (the
             paper's 'small object in arc A'), the nearby-companion core
             (modeled, core masked like the paper's 2.5-px drizzled mask),
             and the central 1.5-px lens core (as in all prior versions).
3. NOISE  -- per-pixel sigma from the drizzled WHT map, RESCALED so that
             normalized residuals over source-free sky have unit variance
             (this absorbs the drizzle pixel-correlation the WHT map ignores),
             plus the data-based Poisson term. One definition, used everywhere.

Gate (printed + saved): sky-region reduced chi^2 within 1.00 +/- 0.05.

Outputs: data/cutout_v2.npz, figs/cutout_v2_masks.png, data/cutout_v2_stats.json
"""
from pathlib import Path
import json

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, deblend_sources
from photutils.detection import DAOStarFinder
from scipy.ndimage import binary_dilation, gaussian_filter, median_filter

REPRO = Path(__file__).resolve().parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"

CROP = 80                 # pixels (10.4" at 0.13"/px)
DELTA_PIX = 0.13
EXP_TIME = 1197.7
CORE_MASK_PX = 1.5        # lens + companion core mask radius
ARC_ANNULUS = (1.2, 4.5)  # arcsec; the arc system lives here
INTERLOPER_PT_RAD = 2.5   # px mask radius for compact interlopers on arcs

# ---------------------------------------------------------------- load + sky
with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float64)
    wht = h["WHT"].data.astype(np.float64)

mean0, med0, std0 = sigma_clipped_stats(sci, sigma=3.0, maxiters=10)
sky = med0
img_full = sci - sky

n0 = img_full.shape[0]
c0 = n0 // 2
half = CROP // 2
sl = slice(c0 - half, c0 + half)
img = img_full[sl, sl].copy()
wht_c = wht[sl, sl].copy()

nb = np.load(DATA / "nearby_galaxy_loc.npz")
near_xy_arc = (float(nb["arcsec_x"]), float(nb["arcsec_y"]))

# pixel coordinate grids on the crop; arcsec offsets from image center
yy, xx = np.indices(img.shape)
cen = (CROP - 1) / 2.0
dx_arc = (xx - cen) * DELTA_PIX
dy_arc = (yy - cen) * DELTA_PIX
r_arc = np.hypot(dx_arc, dy_arc)
# (gigalens x is RA-like = columns, y = rows; nearby loc was derived on this grid)
near_px = (cen + near_xy_arc[0] / DELTA_PIX, cen + near_xy_arc[1] / DELTA_PIX)
r_near = np.hypot(xx - near_px[0], yy - near_px[1])

# ------------------------------------------------- per-pixel WHT noise + sky
with np.errstate(divide="ignore"):
    sig_wht = np.sqrt(1.0 / np.where(wht_c > 0, wht_c, np.nan))
sig_wht = np.where(np.isfinite(sig_wht), sig_wht, np.nanmax(sig_wht))

# ------------------------------------------------------- source segmentation
smoothed = gaussian_filter(img, 1.5)
seg_thresh = 1.5 * np.nanmedian(sig_wht)
seg = detect_sources(smoothed, seg_thresh, n_pixels=6)
if seg is not None:
    seg = deblend_sources(smoothed, seg, n_pixels=6, progress_bar=False)

mask_interloper = np.zeros(img.shape, dtype=bool)
n_faint_galaxies = 0
faint_locations = []
if seg is not None:
    for lab in seg.labels:
        m = seg.data == lab
        ys, xs_ = np.nonzero(m)
        cyx = (ys.mean(), xs_.mean())
        rc = np.hypot((cyx[1] - cen) * DELTA_PIX, (cyx[0] - cen) * DELTA_PIX)
        rnear = np.hypot(cyx[1] - near_px[0], cyx[0] - near_px[1])
        if rc < ARC_ANNULUS[1] or rnear * DELTA_PIX < 0.8:
            continue  # lens system / arcs / companion: modeled, not masked
        mask_interloper |= binary_dilation(m, iterations=2)
        n_faint_galaxies += 1
        faint_locations.append([float(cyx[1]), float(cyx[0]), float(rc)])

# ------------------------- compact object superposed on arc A (paper Fig. 7)
# DAOStarFinder on a high-passed image detects ALL compact ring features --
# which includes the four lensed images A-D (signal, must NOT be masked).
# The paper masks exactly ONE small object superposed on arc A, i.e. a compact
# source lying very close to a lensed image. Strategy: among ring detections,
# find the closest pair (image + contaminant) and mask only the more
# point-like (higher DAO sharpness) member of that pair.
highpass = img - median_filter(img, size=7)
_, hp_med, hp_std = sigma_clipped_stats(highpass, sigma=3.0, maxiters=10)
dao = DAOStarFinder(fwhm=2.5, threshold=5.0 * hp_std)
pts = dao(highpass - hp_med)
ring_dets = []
if pts is not None:
    for row in pts:
        px, py = float(row["x_centroid"]), float(row["y_centroid"])
        rc = np.hypot((px - cen) * DELTA_PIX, (py - cen) * DELTA_PIX)
        rnear = np.hypot(px - near_px[0], py - near_px[1]) * DELTA_PIX
        if not (ARC_ANNULUS[0] <= rc <= ARC_ANNULUS[1]) or rnear < 0.8:
            continue
        ring_dets.append(dict(x=px, y=py, r_arc=float(rc),
                              sharp=float(row["sharpness"]),
                              peak=float(row["peak"])))

n_arc_interlopers = 0
arc_pt_locations = []
arc_a_object = None
if len(ring_dets) >= 2:
    best_pair, best_sep = None, np.inf
    for i in range(len(ring_dets)):
        for j in range(i + 1, len(ring_dets)):
            sep = np.hypot(ring_dets[i]["x"] - ring_dets[j]["x"],
                           ring_dets[i]["y"] - ring_dets[j]["y"])
            if sep < best_sep:
                best_sep, best_pair = sep, (i, j)
    # only treat as image+contaminant if genuinely adjacent (< 1.3")
    if best_sep * DELTA_PIX < 1.3:
        i, j = best_pair
        cont = ring_dets[i] if ring_dets[i]["sharp"] >= ring_dets[j]["sharp"] else ring_dets[j]
        d2 = (xx - cont["x"]) ** 2 + (yy - cont["y"]) ** 2
        mask_interloper |= d2 <= INTERLOPER_PT_RAD ** 2
        n_arc_interlopers = 1
        arc_pt_locations.append([cont["x"], cont["y"], cont["r_arc"]])
        arc_a_object = cont

# ------------------------------------------------------------- core masks
mask_cores = (r_arc <= CORE_MASK_PX * DELTA_PIX) | (r_near <= CORE_MASK_PX)
keep_mask = ~(mask_interloper | mask_cores)

# ------------------------------------------- drizzle-correlation noise rescale
# Rescale so that the mean SQUARED normalized residual over source-free sky is
# exactly 1 -- this is the statistic the Gaussian likelihood actually sees.
# A mild 5-sigma clip removes genuine outliers (cosmic-ray remnants) without
# biasing the bulk variance the way a 3-sigma clipped std does.
source_free = (seg.data == 0 if seg is not None else np.ones(img.shape, bool))
sky_px = source_free & (r_arc > ARC_ANNULUS[1]) & keep_mask
norm_resid = img[sky_px] / sig_wht[sky_px]
_, _, robust_std = sigma_clipped_stats(norm_resid, sigma=5.0, maxiters=10)
clipped = norm_resid[np.abs(norm_resid) < 5.0 * robust_std]
rescale = float(np.sqrt(np.mean(clipped ** 2)))

err_map = np.sqrt((rescale * sig_wht) ** 2 +
                  np.clip(img, 0, np.inf) / EXP_TIME).astype(np.float32)

# ------------------------------------------------------------------- gate
chi2_sky = float(np.mean((img[sky_px] / err_map[sky_px]) ** 2))
chi2_sky_raw = float(np.mean((img[sky_px] / sig_wht[sky_px]) ** 2))
gate_ok = abs(chi2_sky - 1.0) <= 0.05

psf = np.load(DATA / "empirical_psf.npy").astype(np.float32)

stats = dict(
    crop=CROP, delta_pix=DELTA_PIX, exp_time=EXP_TIME, sky=float(sky),
    rescale=rescale, chi2_sky=chi2_sky, chi2_sky_wht_only=chi2_sky_raw,
    gate_sky_chi2_ok=bool(gate_ok),
    n_faint_galaxies_masked=n_faint_galaxies,
    n_arc_interlopers_masked=n_arc_interlopers,
    n_masked_px=int((~keep_mask).sum()), n_px=int(keep_mask.size),
    n_sky_px=int(sky_px.sum()),
    faint_locations=faint_locations, arc_pt_locations=arc_pt_locations,
    ring_detections=[[d["x"], d["y"], d["sharp"], d["peak"]] for d in ring_dets],
    arc_a_object=([arc_a_object["x"], arc_a_object["y"]] if arc_a_object else None),
    nearby_arcsec=list(near_xy_arc),
)
meta = json.dumps(stats)

np.savez(DATA / "cutout_v2.npz",
         img=img.astype(np.float32), err_map=err_map,
         keep_mask=keep_mask, psf=psf, meta=meta)
(DATA / "cutout_v2_stats.json").write_text(json.dumps(stats, indent=2))

# ------------------------------------------------------------- diagnostics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
stretch = np.arcsinh(img / (3 * np.nanmedian(err_map)))
axes[0].imshow(stretch, origin="lower", cmap="magma")
axes[0].set_title(f"cutout v2 ({CROP}x{CROP}, asinh)")
over = np.ma.masked_where(keep_mask, np.ones_like(img))
axes[1].imshow(stretch, origin="lower", cmap="gray")
axes[1].imshow(over, origin="lower", cmap="autumn", alpha=0.7, vmin=0, vmax=1)
for d in ring_dets:
    is_cont = arc_a_object is not None and d["x"] == arc_a_object["x"]
    axes[1].add_patch(plt.Circle((d["x"], d["y"]), INTERLOPER_PT_RAD, fill=False,
                                 color="red" if is_cont else "lime", lw=1.6))
axes[1].set_title(f"masks: {n_faint_galaxies} galaxies, "
                  f"{n_arc_interlopers} arc-A object (red; lime = lensed images, kept)")
im2 = axes[2].imshow(err_map, origin="lower", cmap="viridis")
axes[2].set_title(f"err map (rescale={rescale:.2f}, sky chi2={chi2_sky:.3f})")
plt.colorbar(im2, ax=axes[2], fraction=0.046)
for ax in axes:
    ax.plot(*near_px, "wx", ms=8)
fig.tight_layout()
FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / "cutout_v2_masks.png", dpi=130)

print(json.dumps({k: v for k, v in stats.items()
                  if k not in ("faint_locations", "arc_pt_locations")}, indent=2))
print(f"\nGATE sky chi2 = {chi2_sky:.4f} -> {'PASS' if gate_ok else 'FAIL'}")
print(f"wrote data/cutout_v2.npz, figs/cutout_v2_masks.png")
