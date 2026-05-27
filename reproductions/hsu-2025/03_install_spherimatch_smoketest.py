#!/usr/bin/env python3
"""
03_install_spherimatch_smoketest.py

Validate the spherimatch FoF API by:
  1. Asserting parity with astropy.coordinates.search_around_sky on a 10k synthetic catalog.
  2. Timing fof at N = 1e3, 1e4, 1e5, 1e6 to extrapolate the full-DR1 (~28M) wall-clock cost.

Hsu et al. 2025 (arXiv:2509.16033) uses a 3 arcsec linking length on (RA, Dec).
spherimatch.fof takes tolerance in DEGREES, so we pass 3.0/3600.0.

Outputs:
  - stdout / logged to nuts smoketest_run.log
  - data/smoketest_timings.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from spherimatch import fof
from spherimatch.catalog import Catalog


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

LINK_DEG = 3.0 / 3600.0          # 3 arcsec, Hsu 2025
LINK_ARCSEC = 3.0
RNG = np.random.default_rng(20260526)


def random_radec(n: int, dense_patch_deg: float | None = None) -> np.ndarray:
    """Sample (N, 2) RA[deg], Dec[deg].

    If dense_patch_deg is None: uniform on the sphere.
    Else: uniform inside a square of side dense_patch_deg around (RA=180, Dec=0),
    used to force enough close pairs that 3″-FoF actually produces groups.
    """
    if dense_patch_deg is None:
        u1 = RNG.random(n)
        u2 = RNG.random(n)
        ra = 360.0 * u1
        dec = np.degrees(np.arcsin(2.0 * u2 - 1.0))
    else:
        half = dense_patch_deg / 2.0
        ra = 180.0 + (RNG.random(n) - 0.5) * dense_patch_deg
        dec = (RNG.random(n) - 0.5) * dense_patch_deg
    return np.column_stack([ra, dec])


def fof_groups_spherimatch(radec: np.ndarray) -> set[frozenset[int]]:
    """Return spherimatch FoF groups (size >= 2) as sets of original-catalog indices.

    spherimatch returns a DataFrame indexed by (Group, Object). The Object index is
    the ORIGINAL row index in the input catalog (we verify this below by matching
    coordinates back), so we use it directly as the catalog index.
    """
    res = fof(radec, LINK_DEG)
    df = res.get_group_dataframe()
    if df is None or len(df) == 0:
        return set()
    out: set[frozenset[int]] = set()
    for grp_id, sub in df.groupby(level="Group"):
        if len(sub) < 2:
            continue
        idx = sub.index.get_level_values("Object").to_numpy()
        # Map (Ra, Dec) back to original catalog row index (Object is internal id,
        # not necessarily the original row index — verify by exact coord match).
        orig = []
        for ra_v, dec_v in zip(sub["Ra"].to_numpy(), sub["Dec"].to_numpy()):
            hits = np.where((radec[:, 0] == ra_v) & (radec[:, 1] == dec_v))[0]
            if len(hits) == 0:
                # Fall back to Object id if coords don't match exactly
                orig.append(int(idx[len(orig)]))
            else:
                orig.append(int(hits[0]))
        out.add(frozenset(orig))
    return out


def fof_groups_astropy(radec: np.ndarray) -> set[frozenset[int]]:
    """Brute-force reference: union-find over astropy search_around_sky neighbors."""
    sc = SkyCoord(ra=radec[:, 0] * u.deg, dec=radec[:, 1] * u.deg)
    i1, i2, _, _ = search_around_sky(sc, sc, LINK_ARCSEC * u.arcsec)
    mask = i1 < i2
    i1, i2 = i1[mask], i2[mask]
    parent = list(range(len(radec)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(i1, i2):
        ra_root, rb_root = find(int(a)), find(int(b))
        if ra_root != rb_root:
            parent[ra_root] = rb_root

    groups: dict[int, list[int]] = {}
    for i in range(len(radec)):
        groups.setdefault(find(i), []).append(i)
    return {frozenset(v) for v in groups.values() if len(v) >= 2}


def parity_check(n: int = 10_000) -> None:
    # Use a 1° x 1° patch so that 3″-link FoF actually finds groups; on the full
    # sphere at N=10k, the expected pair count is <<1.
    patch_deg = 1.0
    print(
        f"[parity] N={n} in {patch_deg}°×{patch_deg}° patch, "
        f"tolerance={LINK_ARCSEC}″ ({LINK_DEG:.6e} deg)"
    )
    radec = random_radec(n, dense_patch_deg=patch_deg)
    g_sm = fof_groups_spherimatch(radec)
    g_ap = fof_groups_astropy(radec)
    print(f"[parity] spherimatch groups (size>=2): {len(g_sm)}")
    print(f"[parity] astropy   groups (size>=2): {len(g_ap)}")
    only_sm = g_sm - g_ap
    only_ap = g_ap - g_sm
    if only_sm or only_ap:
        print(f"[parity] DISAGREEMENT — sm-only: {len(only_sm)}, ap-only: {len(only_ap)}")
        for g in list(only_sm)[:3]:
            print(f"  sm-only example: {sorted(g)}")
        for g in list(only_ap)[:3]:
            print(f"  ap-only example: {sorted(g)}")
        raise AssertionError("FoF parity check failed")
    print("[parity] OK — spherimatch matches astropy reference")


def timing_curve() -> dict[int, float]:
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    timings: dict[int, float] = {}
    for n in sizes:
        radec = random_radec(n)
        t0 = time.perf_counter()
        _ = fof(radec, LINK_DEG)
        dt = time.perf_counter() - t0
        timings[n] = dt
        print(f"[timing] N={n:>9,d}  fof={dt:8.2f} s")
    return timings


def extrapolate(timings: dict[int, float], target_n: int = 28_000_000) -> dict:
    ns = np.array(sorted(timings))
    ts = np.array([timings[n] for n in ns])
    log_ns, log_ts = np.log(ns), np.log(ts)
    slope, intercept = np.polyfit(log_ns, log_ts, 1)
    extrap = float(np.exp(intercept + slope * np.log(target_n)))
    print(
        f"[extrapolate] power-law fit  t ~ N^{slope:.2f}\n"
        f"[extrapolate] full-DR1 (N={target_n:,}) predicted: "
        f"{extrap:.0f} s  ({extrap / 60:.0f} min, {extrap / 3600:.1f} h)"
    )
    return dict(slope=float(slope), intercept=float(intercept), extrap_seconds=extrap)


def main() -> None:
    parity_check(10_000)
    timings = timing_curve()
    extrap = extrapolate(timings)
    out = DATA / "smoketest_timings.json"
    payload = {
        "tolerance_deg": LINK_DEG,
        "tolerance_arcsec": LINK_ARCSEC,
        "timings_seconds": {str(n): t for n, t in timings.items()},
        "extrapolation": extrap,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
