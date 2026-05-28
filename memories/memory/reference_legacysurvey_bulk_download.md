---
name: reference-legacysurvey-bulk-download
description: How to bulk-process DECaLS cutouts — endpoint is rate-limited; use brick-level FITS instead
metadata:
  type: reference
---

The `legacysurvey.org/viewer/fits-cutout` endpoint generates cutouts
server-side on each request and caps at **~14s per request** regardless of
worker count (we tested with 8 and 16 parallel workers; both hit the same
~0.86 rows/sec ceiling). At that rate, a 5-6M-row DECaLS sweep would take
**~80+ days** — infeasible.

**For bulk processing (Phase 3b production sweep) use brick-level FITS
instead:**

  `https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr7/coadd/AAA/BBBB/legacysurvey-BBBB-image-{g,r,z}.fits.fz`

where `AAA` is the first 3 characters of `BBBB` (the BRICKNAME from sweep
catalogs). Each brick is **3600×3600 px float32, ~15 MB compressed per
band, ~45 MB for grz**. WCS is in HDU 1 (CompImageHDU). Use
`astropy.wcs.WCS(hdul[1].header).world_to_pixel_values(ra, dec)` to project
target positions to brick pixels, then slice 101×101 cutouts locally.

DECaLS DR7 has ~113K bricks. Brick-level processing achieves
**~1.16 bricks/sec/shard** (~100 galaxies/sec/shard scored) on a single
L4 GPU with 3 brick-download workers, network-bound. With 2 L4s in
parallel the full sweep finishes in **~13 hours** vs ~80 days
endpoint-based — a **~150× speedup**.

Implementation: `reproductions/huang-2020/11b_brick_inference_dr7.py`
(brick-driven) vs `11_stream_inference_dr7.py` (endpoint-driven; kept
for small-N use cases only).

The endpoint is fine for **smoke tests / small targeted lookups** (e.g.,
`08_smoketest_dr7.py` scoring 6 named candidates) but never for bulk.

See also: [[project-huang-2020-reproduction]].
