#!/usr/bin/env python3
"""
08_smoketest_dr8.py — Phase 4b decision gate.

Two checks, both must pass before the full DR8 sweep (11b):

  --mode endpoint  (runs anytime; needs both checkpoints)
      Fetch DR8 grz cutouts via the legacysurvey.org viewer at known
      Huang-catalog lens positions in BOTH the south (DECaLS) and north
      (BASS/MzLS) footprints; score with BOTH the L18 and shielded models.
      Confirms the trained nets score known DR8 lenses highly.

  --mode brick     (the routing gate; needs data/parent_dr8.parquet)
      Exercise the ACTUAL 11b brick path: match known lenses to parent rows,
      then download_brick() + load_brick() + extract_cutout() for at least one
      SOUTH and one NORTH (footprint,brick) unit, scoring both models. This is
      the real test that north/south coadd routing works — if north bricks 404,
      the footprint mapping in 10/11b is wrong.

Inputs:
  data/checkpoint_best.pt                  (L18, DR9-trained)
  data/checkpoint_best_shielded_dr9.pt     (shielded, from 05_train_shielded.py)
  data/neuralens_catalog.csv               (known lenses, with Region column)
  data/parent_dr8.parquet                  (brick mode only)
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402
_bi = import_module("11b_brick_inference_dr8")
load_model = _bi.load_model
download_brick = _bi.download_brick
load_brick = _bi.load_brick
extract_cutout = _bi.extract_cutout
BRICK_TMP = _bi.BRICK_TMP

DATA = HERE / "data"
VIEWER = "https://www.legacysurvey.org/viewer/fits-cutout"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_cube(models: dict, raw: np.ndarray) -> dict:
    """raw (3,101,101) -> {kind: prob}."""
    out = {}
    xt0 = raw[None]  # (1,3,101,101)
    for kind, (model, mean, std, _) in models.items():
        xs = np.clip((xt0 - mean) / std, -250.0, 250.0)
        with torch.no_grad():
            device = next(model.parameters()).device
            lo = model(torch.from_numpy(xs.astype(np.float32)).to(device)).cpu().numpy()
        out[kind] = float(sigmoid(lo)[0])
    return out


def fetch_dr8_cutout(ra: float, dec: float, size: int = 101, pixscale: float = 0.262) -> np.ndarray:
    url = (f"{VIEWER}?ra={ra:.6f}&dec={dec:.6f}&size={size}&layer=ls-dr8"
           f"&pixscale={pixscale}&bands=grz")
    for attempt in range(1, 6):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            if len(r.content) < 256:
                raise RuntimeError(f"cutout too small ({len(r.content)} bytes)")
            with fits.open(io.BytesIO(r.content), memmap=False) as hdul:
                data = hdul[0].data
            if data is None or data.ndim != 3 or data.shape[0] != 3:
                raise ValueError(f"unexpected FITS shape: "
                                 f"{None if data is None else data.shape}")
            return data.astype(np.float32)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(4 * attempt)
    raise RuntimeError("unreachable")


def load_models(device):
    ck_l18 = DATA / "checkpoint_best.pt"
    ck_sh = DATA / "checkpoint_best_shielded_dr9.pt"
    models = {"l18": load_model("l18", ck_l18, device)}
    if ck_sh.exists():
        models["shielded"] = load_model("shielded", ck_sh, device)
    else:
        print(f"[warn] {ck_sh.name} not found — scoring L18 only "
              f"(re-run after 05_train_shielded.py --dr dr9)")
    for kind, (_, _, _, va) in models.items():
        print(f"[init] {kind} model loaded  val_auc={va:.4f}")
    return models


def known_lenses(n_each: int = 6) -> pd.DataFrame:
    """Sample known lenses from neuralens_catalog.csv, tagged south/north by Region."""
    cat = pd.read_csv(DATA / "neuralens_catalog.csv")
    cat.columns = [c.strip() for c in cat.columns]
    # Parse RA/DEC from the DESI-RA±DEC Name.
    name_col = [c for c in cat.columns if c.lower() == "name"][0]
    region_col = [c for c in cat.columns if "region" in c.lower()][0]
    rows = []
    for _, r in cat.iterrows():
        m = pd.Series([r[name_col]]).str.extract(r"DESI-(\d{3}\.\d{4})([+\-]\d{2}\.\d{4})")
        if m.isnull().values.any():
            continue
        ra = float(m.iloc[0, 0]); dec = float(m.iloc[0, 1])
        region = str(r[region_col]).strip().upper()
        fp = "north" if "MZLS" in region or "BASS" in region else "south"
        rows.append({"name": r[name_col], "RA": ra, "DEC": dec, "footprint": fp})
    df = pd.DataFrame(rows)
    out = []
    for fp in ("south", "north"):
        out.append(df[df["footprint"] == fp].head(n_each))
    return pd.concat(out).reset_index(drop=True)


def mode_endpoint(models) -> bool:
    lenses = known_lenses()
    print(f"\n[endpoint] scoring {len(lenses)} known lenses via DR8 viewer\n")
    print(f"{'name':24s} {'fp':>5s} {'RA':>9s} {'Dec':>8s}  {'L18':>6s} {'shield':>6s}")
    print("-" * 66)
    scores = []
    for _, L in lenses.iterrows():
        try:
            cube = fetch_dr8_cutout(L["RA"], L["DEC"])
            s = score_cube(models, cube)
            scores.append(max(s.values()))
            print(f"{L['name']:24s} {L['footprint']:>5s} {L['RA']:9.4f} {L['DEC']:8.4f}  "
                  f"{s.get('l18', float('nan')):6.3f} {s.get('shielded', float('nan')):6.3f}")
        except Exception as e:
            print(f"{L['name']:24s} {L['footprint']:>5s} {L['RA']:9.4f} {L['DEC']:8.4f}  "
                  f"FAILED: {str(e)[:30]}")
    arr = np.array(scores)
    if len(arr) == 0:
        print("[endpoint] all failed"); return False
    pr = float((arr >= 0.5).mean())
    print(f"\n[endpoint] n={len(arr)} mean(max-score)={arr.mean():.3f} pass(>=0.5)={pr:.0%}")
    return pr >= 0.5


def mode_brick(models) -> bool:
    parent_path = DATA / "parent_dr8.parquet"
    if not parent_path.exists():
        print(f"[brick] {parent_path.name} not found — run 10_select_parent_sample_dr8.py first")
        return False
    parent = pd.read_parquet(parent_path, columns=["RA", "DEC", "BRICKNAME", "BRICKID",
                                                    "OBJID", "footprint"])
    psky = SkyCoord(ra=parent["RA"].values * u.deg, dec=parent["DEC"].values * u.deg)
    lenses = known_lenses(n_each=20)
    lsky = SkyCoord(ra=lenses["RA"].values * u.deg, dec=lenses["DEC"].values * u.deg)
    idx, sep, _ = lsky.match_to_catalog_sky(psky)
    lenses = lenses.assign(sep=sep.to(u.arcsec).value, pidx=idx)
    matched = lenses[lenses["sep"] < 3.0].copy()
    matched["BRICKNAME"] = parent["BRICKNAME"].values[matched["pidx"]]
    matched["p_footprint"] = parent["footprint"].values[matched["pidx"]]

    # Take up to 2 south + 2 north matched lenses to exercise both coadd dirs.
    tests = pd.concat([matched[matched["p_footprint"] == "south"].head(2),
                       matched[matched["p_footprint"] == "north"].head(2)])
    if tests.empty:
        print("[brick] no known lenses matched the parent sample within 3\"")
        return False

    print(f"\n[brick] testing brick path for {len(tests)} known lenses\n")
    print(f"{'name':24s} {'fp':>5s} {'brick':>10s}  {'L18':>6s} {'shield':>6s}  status")
    print("-" * 72)
    fps_ok = set()
    n_scored_high = 0
    for _, L in tests.iterrows():
        fp, brick = L["p_footprint"], L["BRICKNAME"]
        tmp = BRICK_TMP / f"smoke_{fp}_{brick}"
        tmp.mkdir(parents=True, exist_ok=True)
        paths, err = download_brick(fp, brick, tmp)
        if err:
            print(f"{L['name']:24s} {fp:>5s} {brick:>10s}  {'':6s} {'':6s}  FAIL {err[:30]}")
            continue
        cube, wcs = load_brick(paths)
        ct = extract_cutout(cube, wcs, float(L["RA"]), float(L["DEC"]))
        for p in paths.values():
            p.unlink(missing_ok=True)
        tmp.rmdir()
        if ct is None:
            print(f"{L['name']:24s} {fp:>5s} {brick:>10s}  edge-clipped")
            continue
        s = score_cube(models, ct)
        fps_ok.add(fp)
        if max(s.values()) >= 0.5:
            n_scored_high += 1
        print(f"{L['name']:24s} {fp:>5s} {brick:>10s}  "
              f"{s.get('l18', float('nan')):6.3f} {s.get('shielded', float('nan')):6.3f}  ok")

    north_ok = "north" in fps_ok
    south_ok = "south" in fps_ok
    print(f"\n[brick] south routing={'OK' if south_ok else 'MISSING'}  "
          f"north routing={'OK' if north_ok else 'MISSING'}  "
          f"high-scoring={n_scored_high}/{len(tests)}")
    return south_ok and north_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("endpoint", "brick", "both"), default="both")
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    ok = True
    if args.mode in ("endpoint", "both"):
        ok = mode_endpoint(models) and ok
    if args.mode in ("brick", "both"):
        ok = mode_brick(models) and ok

    print()
    if ok:
        print("[gate] PASS — proceed to full 11b DR8 inference.")
    else:
        print("[gate] CHECK — see above; resolve before full 11b run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
