"""
41 -- Fetch the SuGOHI strong-lens catalog + Legacy Survey cutouts (task 9).

Positives for the strong-lens retrieval task. SuGOHI master list
(Oguri/U-Tokyo) has ~3961 candidates graded A (definite) / B / C. We keep
grades A+B, fetch their g,r,i,z cutouts (the non-lens distractor corpus is the
already-downloaded PROVABGS galaxy images), and save the lens image array.

Outputs (data/raw/sugohi/): lens_radec.parquet, lens_image.npy (M,4,160,160),
lens_ok.npy.

Run: HF_HOME=... python 41_fetch_sugohi.py [--grades AB] [--workers 6]
"""

import argparse

import numpy as np
import pandas as pd

import _config as C
import _ls_cutout as LS

OUT = C.RAW / "sugohi"
OUT.mkdir(parents=True, exist_ok=True)
URL = ("https://www-utap.phys.s.u-tokyo.ac.jp/~oguri/sugohi/"
       "download_list.php?file=list_ra_asc_public.csv")
COLS = ["name", "ra", "dec", "zl_spec", "zs_spec", "zl_phot", "zs_phot",
        "theta_E", "mag_i_lens", "mag_i_src", "type", "method", "grade", "reference"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", default="AB", help="which grades to keep, e.g. AB or A")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    import requests
    import io
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), header=None, names=COLS)
    df["grade"] = df["grade"].astype(str).str.strip()
    keep = df[df["grade"].isin(list(args.grades))].reset_index(drop=True)
    keep.to_parquet(OUT / "lens_radec.parquet")
    print(f"SuGOHI: {len(df)} candidates; grades {args.grades} -> {len(keep)} lenses")
    print("grade counts:", df["grade"].value_counts().to_dict())

    coords = list(zip(keep["ra"].to_numpy(float), keep["dec"].to_numpy(float)))
    arrs, ok = LS.fetch_many(coords, layer="ls-dr10", size=160, workers=args.workers)
    imgs = np.stack([a for a, o in zip(arrs, ok) if o]).astype(np.float32)
    np.save(OUT / "lens_image.npy", imgs)
    np.save(OUT / "lens_ok.npy", ok)
    print(f"SUGOHI_OK saved {imgs.shape} for {ok.sum()}/{len(ok)} lenses "
          f"({100*ok.mean():.0f}% cutout success)")


if __name__ == "__main__":
    main()
