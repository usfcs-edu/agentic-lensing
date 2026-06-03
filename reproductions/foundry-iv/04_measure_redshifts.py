"""04 - Extract 1D spectra from MUSE cubes and AUTO-measure lens & source redshifts.

For each downloaded cube (matched to a confirmed Foundry-IV system):
  1. open with mpdaf (Cube), get the white-light image
  2. locate the LENS at the catalog RA/Dec (the brightest continuum source there)
  3. extract a small-aperture 1D spectrum at the lens -> auto find_z_absorption
  4. find a candidate SOURCE position: the cube spaxel that, after lens-continuum
     removal, shows the strongest emission OR an offset secondary continuum knot;
     extract there -> auto find_z_emission and find_z_absorption(UV), keep the better.
  5. compare measured z to the published z; write results + per-system figures.

This is a fully automated analog of the paper's MANUAL line-ID. We do NOT hand-place
apertures: lens aperture is at the catalog coordinate; source aperture is found by an
automated emission-peak / residual search within the MUSE FoV.

Run:
    python 04_measure_redshifts.py                  # all downloaded cubes
    python 04_measure_redshifts.py --target Lens16  # one

Outputs:
    data/measured_redshifts.csv
    figs/04_<target>_spectra.png
"""
from pathlib import Path
import argparse
import csv
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

import _zfinder as zf

warnings.filterwarnings("ignore")

REPRO = Path(__file__).parent
DATA = REPRO / "data"
CUBES = DATA / "cubes"
FIGS = REPRO / "figs"
FIGS.mkdir(exist_ok=True)

LENS_APER_ARCSEC = 2.0   # radius of lens extraction aperture (higher S/N on the galaxy)
SRC_APER_ARCSEC = 0.8
SRC_ANNULUS_ARCSEC = (1.2, 7.0)  # search the lensed arc in this annulus around the lens


def load_match_table():
    rows = list(csv.DictReader((DATA / "cube_match_table.csv").open()))
    return [r for r in rows if r["match_name"]]


def downloaded_cubes():
    return {p.stem: p for p in CUBES.glob("*.fits") if p.stat().st_size > 100_000_000}


def pick_cube_for_target(rows, target):
    cands = [r for r in rows if r["cube_target"] == target]
    return max(cands, key=lambda x: float(x["exptime"])) if cands else None


def extract_aperture_spectrum(cube, ra, dec, radius_arcsec):
    """mpdaf aperture extraction at (ra,dec). Returns (wave_A, flux, var)."""
    from mpdaf.obj import Cube  # noqa
    # mpdaf aperture takes (dec, ra) in degrees and radius in arcsec
    sp = cube.aperture((dec, ra), radius_arcsec, unit_center=u.deg, unit_radius=u.arcsec)
    wave = sp.wave.coord()          # Angstrom
    flux = sp.data.filled(np.nan) if hasattr(sp.data, "filled") else np.asarray(sp.data)
    var = np.asarray(sp.var, float) if sp.var is not None else None
    return np.asarray(wave, float), np.asarray(flux, float), var


def measure_system(cube_path, target, name, ra, dec, z_lens_pub, z_src_pub,
                   src_features, qz_src, lens_only=False):
    from mpdaf.obj import Cube
    print(f"\n=== {target}  ({name})  pub: zL={z_lens_pub} zS={z_src_pub} ===")
    cube = Cube(str(cube_path))
    print(f"    cube shape {cube.shape}, lambda {cube.wave.coord()[0]:.0f}-{cube.wave.coord()[-1]:.0f} A")

    # --- LENS ---
    wl, fl, vl = extract_aperture_spectrum(cube, ra, dec, LENS_APER_ARCSEC)
    zL, scoreL, zgL, detL = zf.find_z_absorption(wl, fl, var=vl, zmin=0.05, zmax=1.2)
    print(f"    LENS   auto z={zL:.4f}  (pub {z_lens_pub})  dz={zL-z_lens_pub:+.4f}")

    if lens_only:
        return dict(
            target=target, name=name, ra=ra, dec=dec,
            z_lens_pub=z_lens_pub, z_lens_auto=round(float(zL), 4),
            dz_lens=round(float(zL - z_lens_pub), 4), lens_snr=round(float(scoreL.max()), 1),
            z_src_pub=z_src_pub, z_src_auto=float("nan"), dz_src=float("nan"),
            src_engine="(skipped)", src_snr=float("nan"), src_sep_arcsec=float("nan"),
            src_features_pub=src_features, qz_src=qz_src,
        )

    # --- SOURCE: joint spatial+spectral emission scan to find arc position & z ---
    data = cube.data.filled(np.nan) if hasattr(cube.data, "filled") else np.asarray(cube.data)
    cvar = cube.var.filled(np.nan) if (cube.var is not None and hasattr(cube.var, "filled")) \
        else (np.asarray(cube.var) if cube.var is not None else None)
    yc, xc = cube.wcs.sky2pix([[dec, ra]], unit=u.deg)[0]
    pixscale = abs(cube.wcs.get_step(unit=u.arcsec)[0])
    jb = zf.find_emission_source_in_cube(data, wl, (yc, xc), pixscale, var=cvar,
                                         zmin=0.2, zmax=1.6)
    sky = cube.wcs.pix2sky([[jb["iy"], jb["ix"]]], unit=u.deg)[0]
    src_dec, src_ra, sep = float(sky[0]), float(sky[1]), jb["sep_arcsec"]
    print(f"    SOURCE arc auto-located {sep:.1f}\" from lens (joint [OII]/[OIII] scan), "
          f"z_guess={jb['z']:.4f}")
    ws, fs, vs = extract_aperture_spectrum(cube, src_ra, src_dec, SRC_APER_ARCSEC)
    # refine z with the 1D engines: emission (low-z) vs UV-absorption (high-z LBG).
    # Seed the emission search in a tight window around the joint-scan guess; also run
    # the global emission + UV-absorption engines as cross-checks.
    zS_em, scoreS_em, zgS_em, detS_em = zf.find_z_emission(
        ws, fs, var=vs, zmin=max(0.1, jb["z"] - 0.05), zmax=jb["z"] + 0.05)
    zS_uv, scoreS_uv, zgS_uv, detS_uv = zf.find_z_absorption(
        ws, fs, var=vs, zmin=1.5, zmax=3.5, line_list=zf.ABS_LINES_UV)
    if scoreS_em.max() >= scoreS_uv.max():
        zS, engine, scoreS, zgS, detS = zS_em, "emission", scoreS_em, zgS_em, detS_em
        src_snr = float(scoreS_em.max())
    else:
        zS, engine, scoreS, zgS, detS = zS_uv, "uv_abs", scoreS_uv, zgS_uv, detS_uv
        src_snr = float(scoreS_uv.max())
    print(f"    SOURCE auto z={zS:.4f} via {engine} (SNR={src_snr:.1f}) "
          f"(pub {z_src_pub})  dz={zS-z_src_pub:+.4f}")

    # --- figure ---
    _plot_system(target, name, wl, fl, zL, z_lens_pub, scoreL, zgL,
                 ws, fs, zS, z_src_pub, scoreS, zgS, engine)

    return dict(
        target=target, name=name, ra=ra, dec=dec,
        z_lens_pub=z_lens_pub, z_lens_auto=round(float(zL), 4), dz_lens=round(float(zL - z_lens_pub), 4),
        lens_snr=round(float(scoreL.max()), 1),
        z_src_pub=z_src_pub, z_src_auto=round(float(zS), 4), dz_src=round(float(zS - z_src_pub), 4),
        src_engine=engine, src_snr=round(src_snr, 1), src_sep_arcsec=round(sep, 1),
        src_features_pub=src_features, qz_src=qz_src,
    )


def _plot_system(target, name, wl, fl, zL, zLpub, scoreL, zgL,
                 ws, fs, zS, zSpub, scoreS, zgS, engine):
    fig, ax = plt.subplots(2, 2, figsize=(14, 7))
    ax[0, 0].plot(wl, fl, lw=0.5, color="k")
    ax[0, 0].set_title(f"{name}  LENS spectrum  (auto z={zL:.4f}, pub {zLpub})")
    for nm, lam0 in zf.ABS_LINES_GAL.items():
        lam = lam0 * (1 + zL)
        if wl[0] < lam < wl[-1]:
            ax[0, 0].axvline(lam, color="C3", ls=":", lw=0.7)
    ax[0, 0].set_xlabel("obs wavelength [A]")
    ax[0, 1].plot(zgL, scoreL, lw=0.8)
    ax[0, 1].axvline(zLpub, color="g", ls="--", label=f"pub {zLpub}")
    ax[0, 1].axvline(zL, color="r", ls=":", label=f"auto {zL:.4f}")
    ax[0, 1].set_title("lens z-score"); ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(ws, fs, lw=0.5, color="k")
    ax[1, 0].set_title(f"SOURCE spectrum [{engine}] (auto z={zS:.4f}, pub {zSpub})")
    ll = zf.EMIS_LINES if engine == "emission" else zf.ABS_LINES_UV
    for nm, lam0 in ll.items():
        lam = lam0 * (1 + zS)
        if ws[0] < lam < ws[-1]:
            ax[1, 0].axvline(lam, color="C0", ls=":", lw=0.7)
    ax[1, 0].set_xlabel("obs wavelength [A]")
    ax[1, 1].plot(zgS, scoreS, lw=0.8)
    ax[1, 1].axvline(zSpub, color="g", ls="--", label=f"pub {zSpub}")
    ax[1, 1].axvline(zS, color="r", ls=":", label=f"auto {zS:.4f}")
    ax[1, 1].set_title("source z-score"); ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    out = FIGS / f"04_{target}_spectra.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"    saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--lens-only", action="store_true",
                    help="skip the slow joint source scan; measure only z_lens")
    args = ap.parse_args()

    rows = load_match_table()
    have = downloaded_cubes()
    print(f"Downloaded cubes: {sorted(have)}")

    cat = {r["name"]: r for r in csv.DictReader((DATA / "confirmed_catalog.csv").open())}

    results = []
    targets = [args.target] if args.target else None
    seen = set()
    for r in rows:
        t = r["cube_target"]
        if targets and t not in targets:
            continue
        if t in seen:
            continue
        best = pick_cube_for_target(rows, t)
        if best["dp_id"] not in have:
            continue
        seen.add(t)
        cinfo = cat[r["match_name"]]
        res = measure_system(
            have[best["dp_id"]], t, r["match_name"],
            float(cinfo["ra_deg"]), float(cinfo["dec_deg"]),
            float(cinfo["z_lens"]), float(cinfo["z_source"]),
            cinfo["source_features"], cinfo["qz_source"], lens_only=args.lens_only)
        results.append(res)

    if not results:
        print("\nNo downloaded cubes ready to measure yet.")
        return

    out = DATA / "measured_redshifts.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\nWrote {out}")
    print("\nSUMMARY (auto vs published):")
    print(f"  {'target':10} {'system':26} {'zL_auto':>8} {'zL_pub':>7} {'dzL':>7}  "
          f"{'zS_auto':>8} {'zS_pub':>7} {'dzS':>7}")
    for r in results:
        print(f"  {r['target']:10} {r['name']:26} {r['z_lens_auto']:8.4f} {r['z_lens_pub']:7.3f} "
              f"{r['dz_lens']:+7.4f}  {r['z_src_auto']:8.4f} {r['z_src_pub']:7.3f} {r['dz_src']:+7.4f}")


if __name__ == "__main__":
    main()
