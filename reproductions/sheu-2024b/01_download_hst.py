"""Download HST WFC3 F140W + F200LP imaging of DESI-090.9854-35.9683 (the Carousel Lens).

Sheu et al. 2024 (arXiv:2408.10320). HST proposal #16773 (PI Glazebrook), 600 s each in
F140W and F200LP. Public data; MAST DOI 10.17909/zq07-4f53.
"""
from pathlib import Path
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = SkyCoord(ra=90.9854 * u.deg, dec=-35.9683 * u.deg, frame="icrs")
SEARCH_RADIUS = 1.5 * u.arcmin
PROPOSAL_ID = "16773"

print(f"Query MAST near {TARGET.to_string('hmsdms')} (HST proposal {PROPOSAL_ID})")
obs = Observations.query_criteria(
    coordinates=TARGET,
    radius=SEARCH_RADIUS,
    obs_collection="HST",
    proposal_id=PROPOSAL_ID,
)
print(f"Matched {len(obs)} observations")
if len(obs) == 0:
    print("Falling back to proposal-only query (no spatial cut)")
    obs = Observations.query_criteria(obs_collection="HST", proposal_id=PROPOSAL_ID)

cols = [c for c in ("obs_id", "filters", "target_name", "t_exptime", "s_ra", "s_dec",
                    "dataproduct_type", "calib_level") if c in obs.colnames]
print(obs[cols])
if len(obs) == 0:
    sys.exit("No observations found — aborting.")

print("\nResolving data products ...")
products = Observations.get_product_list(obs)
print(f"Found {len(products)} products")

# Drizzled science mosaics (DRZ for IR/F140W, DRC for UVIS/F200LP).
drz = Observations.filter_products(
    products,
    productSubGroupDescription=["DRZ", "DRC"],
    extension="fits",
)
print(f"After DRZ/DRC FITS filter: {len(drz)}")
dcols = [c for c in ("obs_id", "productFilename", "size", "productType") if c in drz.colnames]
print(drz[dcols])

print("\nDownloading drizzled science products ...")
manifest = Observations.download_products(drz, download_dir=str(OUT_DIR))
print(manifest)
print(f"\nDone. Files under {OUT_DIR}")
