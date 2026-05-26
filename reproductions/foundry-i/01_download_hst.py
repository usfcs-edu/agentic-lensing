"""Download HST WFC3/F140W observations of DESI-165.4754-06.0423 (Foundry I demo system).

Reproduces the data ingestion step of Huang et al. 2025a (arXiv:2502.03455).
HST proposal: GO-15867 (PI: Huang). MAST HLSP DOI: 10.17909/hx0v-9260.
Total exposure 1197.7 s (3 x 399.23 s) in F140W; drizzled to 0.065"/px from native 0.13".
"""
from pathlib import Path
import sys

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations

OUT_DIR = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = SkyCoord(ra=165.4754 * u.deg, dec=-6.0423 * u.deg, frame="icrs")
SEARCH_RADIUS = 1.0 * u.arcmin
PROPOSAL_ID = "15867"

print(f"Query MAST near {TARGET.to_string('hmsdms')} (HST proposal {PROPOSAL_ID})")
obs = Observations.query_criteria(
    coordinates=TARGET,
    radius=SEARCH_RADIUS,
    obs_collection="HST",
    proposal_id=PROPOSAL_ID,
    instrument_name="WFC3/IR",
)
print(f"Matched {len(obs)} observations")
if len(obs) == 0:
    print("Falling back to proposal-only query (no spatial cut)")
    obs = Observations.query_criteria(obs_collection="HST", proposal_id=PROPOSAL_ID)
    # filter by separation
    import numpy as np
    coords = SkyCoord(ra=obs["s_ra"] * u.deg, dec=obs["s_dec"] * u.deg)
    sep = coords.separation(TARGET).arcmin
    obs = obs[np.argsort(sep)][:10]

cols_to_show = [c for c in ("obs_id", "filters", "target_name", "t_exptime", "s_ra", "s_dec", "dataproduct_type") if c in obs.colnames]
print(obs[cols_to_show])

if len(obs) == 0:
    sys.exit("No observations found — aborting download.")

print("\nResolving data products …")
products = Observations.get_product_list(obs)
print(f"Found {len(products)} products")

drz = Observations.filter_products(
    products,
    productSubGroupDescription=["DRZ", "DRC"],
    extension="fits",
)
print(f"After filter (DRZ/DRC FITS): {len(drz)}")
drz_cols = [c for c in ("obs_id", "productFilename", "size", "productType") if c in drz.colnames]
print(drz[drz_cols])

print("\nDownloading drizzled products …")
manifest = Observations.download_products(drz, download_dir=str(OUT_DIR))
print(manifest)
print(f"\nDone. Files under {OUT_DIR}")
