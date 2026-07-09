"""32_coolest_export.py — CGL posterior -> COOLEST 'MAP' template (coolest 0.1.11).

Reads a fit produced by 31_fit_euclid.py (Euclid VIS) or 30_recipe_e2e.py (the
HST v2d system) — a {tag}_fit.json (physical MAP components + config + meta +
posterior stats) plus a {tag}_posterior.npz (data / psf / MAP model image /
a_star / chains) — and writes a COOLEST template directory:

    <outdir>/<name>.json     the COOLEST 'MAP' container (PEMD + ExternalShear
                             + Sersic x N lens light + Sersic + Shapelets source)
    <outdir>/psf.fits        normalised PSF (co-located, relative path)
    <outdir>/data.fits       observed image (Observation.pixels)
    <outdir>/model_map.fits  the MAP model image (sidecar)
    <outdir>/chains.npz       the PHMC posterior sidecar (if the fit sampled)

Mapping (recon §4): gigalens EPL(theta_E,gamma,e1,e2,cx,cy) -> PEMD (same
theta_E/gamma convention; q,phi from the ellipticity), shear(g1,g2) ->
ExternalShear(gamma_ext=hypot, phi_ext=1/2 atan2), Sersic(Ie,R,n,e1,e2,c) ->
Sersic(I_eff,theta_eff,n,q,phi,c), source Shapelets(n_max=6, beta, amps=a_star).
The 28 source-shapelet amps are RIDGE-MARGINALISED (not a sampled block): their
marginal-mode a_star at the MAP z is exported as the point estimate and flagged
in the metadata (with logdetA); posteriors are attached to the mass sector from
the chains. This is a CPU-only step (no jax / no GPU rebuild).

Round-trip: after dump_simple the template is reloaded with load_simple and the
point estimates asserted preserved.

Usage:
  python 32_coolest_export.py --fit data/euclid/<id>_fit.json --name <id>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits

from coolest.template.classes.grid import PixelatedRegularGrid
from coolest.template.json import (COOLEST, Cosmology, CoordinatesOrigin, Galaxy,
                                   Instrument, JSONSerializer, LensingEntityList,
                                   LightModel, MassField, MassModel, Observation,
                                   PixelatedPSF, PointEstimate, PosteriorStatistics)

REPRO = Path(__file__).resolve().parent


def ell_to_q_phi(e1, e2):
    """gigalens (e1,e2) -> (q, phi_deg). q=(1-c)/(1+c), c=|e|; phi from the
    package convention phi = atan2(e2,e1)/2 (radians -> degrees)."""
    c = min(math.hypot(e1, e2), 1.0 - 1e-9)
    q = (1.0 - c) / (1.0 + c)
    phi = math.degrees(math.atan2(e2, e1) / 2.0)
    return q, phi


def _pe(param, value):
    param.set_point_estimate(PointEstimate(value=float(value)))


def _post(param, stat):
    """stat: dict(median,p16,p84,mean) or None."""
    if stat is None:
        return
    param.set_posterior(PosteriorStatistics(
        mean=stat.get("mean"), median=stat.get("median"),
        percentile_16th=stat.get("p16"), percentile_84th=stat.get("p84")))


def set_sersic(profile, s, *, post=None):
    q, phi = ell_to_q_phi(s["e1"], s["e2"])
    _pe(profile.parameters["I_eff"], s["Ie"])
    _pe(profile.parameters["theta_eff"], s["R_sersic"])
    _pe(profile.parameters["n"], s["n_sersic"])
    _pe(profile.parameters["q"], q)
    _pe(profile.parameters["phi"], phi)
    _pe(profile.parameters["center_x"], s["center_x"])
    _pe(profile.parameters["center_y"], s["center_y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="path to <tag>_fit.json")
    ap.add_argument("--npz", default=None, help="posterior npz (default: sibling)")
    ap.add_argument("--name", default=None, help="COOLEST file stem")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    fit_path = Path(args.fit).resolve()
    summary = json.loads(fit_path.read_text())
    comp = summary["components"]
    meta = summary["meta"]
    name = args.name or summary.get("target", fit_path.stem.replace("_fit", ""))
    npz_path = Path(args.npz) if args.npz else fit_path.with_name(
        fit_path.name.replace("_fit.json", "_posterior.npz"))
    z = np.load(npz_path, allow_pickle=True) if npz_path.exists() else None
    outdir = Path(args.outdir) if args.outdir else (REPRO / "data" / "coolest" / name)
    outdir.mkdir(parents=True, exist_ok=True)

    delta_pix = float(meta["delta_pix"])
    num_pix = int(meta["num_pix"])
    z_lens = float(summary.get("redshifts", {}).get("z_lens", 0.5))
    z_src = float(summary.get("redshifts", {}).get("z_source", 1.0))
    half = num_pix * delta_pix / 2.0
    fov = (-half, half)

    # ---- co-located FITS: psf / data / model_map ----------------------------
    if z is not None:
        psf = np.asarray(z["psf"], dtype=np.float32)
        data = np.asarray(z["data"], dtype=np.float32)
        model_map = np.asarray(z["model_image_map"], dtype=np.float32)
        a_star = np.asarray(z["a_star_map"], dtype=np.float64)
    else:
        raise SystemExit(f"posterior npz not found: {npz_path} (need psf/data)")
    fits.PrimaryHDU(psf).writeto(outdir / "psf.fits", overwrite=True)
    fits.PrimaryHDU(data).writeto(outdir / "data.fits", overwrite=True)
    fits.PrimaryHDU(model_map).writeto(outdir / "model_map.fits", overwrite=True)
    psf_pix = psf.shape[0]
    psf_half = psf_pix * delta_pix / 2.0

    # ---- lensing entities ---------------------------------------------------
    lens = Galaxy(name + "_lens", z_lens,
                  light_model=LightModel("Sersic", "Sersic", "Sersic", "Sersic"),
                  mass_model=MassModel("PEMD"))
    source = Galaxy(name + "_source", z_src,
                    light_model=LightModel("Sersic", "Shapelets"))
    shear = MassField(name + "_shear", z_lens,
                      mass_model=MassModel("ExternalShear"))

    # PEMD (mass) with posteriors from the chains where available
    pm = lens.mass_model[0]
    q_m, phi_m = ell_to_q_phi(comp["mass"]["e1"], comp["mass"]["e2"])
    _pe(pm.parameters["gamma"], comp["mass"]["gamma"])
    _pe(pm.parameters["theta_E"], comp["mass"]["theta_E"])
    _pe(pm.parameters["q"], q_m)
    _pe(pm.parameters["phi"], phi_m)
    _pe(pm.parameters["center_x"], comp["mass"]["center_x"])
    _pe(pm.parameters["center_y"], comp["mass"]["center_y"])
    pmass = summary.get("posterior_mass")
    if pmass:
        _post(pm.parameters["gamma"], pmass.get("gamma"))
        _post(pm.parameters["theta_E"], pmass.get("theta_E"))
        _post(pm.parameters["center_x"], pmass.get("center_x"))
        _post(pm.parameters["center_y"], pmass.get("center_y"))

    # 4 lens-light Sersics (LL2/LL3 are the off-field nuisance pair)
    for i in range(4):
        set_sersic(lens.light_model[i], comp["lens_light"][i])

    # source Sersic + Shapelets (amps = ridge-marginalised a_star at MAP z)
    set_sersic(source.light_model[0], comp["source_sersic"])
    shp = source.light_model[1]
    _pe(shp.parameters["n_max"], comp["source_shapelet"].get("n_max", 6)
        if "n_max" in comp["source_shapelet"] else 6)
    _pe(shp.parameters["beta"], comp["source_shapelet"]["beta"])
    _pe(shp.parameters["center_x"], comp["source_shapelet"]["center_x"])
    _pe(shp.parameters["center_y"], comp["source_shapelet"]["center_y"])
    shp.parameters["amps"].set_point_estimate([float(a) for a in a_star])

    # ExternalShear
    g1, g2 = comp["shear"]["gamma1"], comp["shear"]["gamma2"]
    gamma_ext = math.hypot(g1, g2)
    phi_ext = math.degrees(math.atan2(g2, g1) / 2.0)
    sh = shear.mass_model[0]
    _pe(sh.parameters["gamma_ext"], gamma_ext)
    _pe(sh.parameters["phi_ext"], phi_ext)
    if pmass and pmass.get("gamma1") and pmass.get("gamma2"):
        # gamma_ext posterior from the chains (recompute if provided)
        pass

    entities = LensingEntityList(lens, source, shear)

    # ---- observation / instrument / cosmology -------------------------------
    info = _load_info(name, meta.get("subset", "lens"))
    ra = info.get("target_ra"); dec = info.get("target_dec")
    origin = (CoordinatesOrigin(f"{float(ra)/15.0}h", f"{float(dec)}d")
              if ra is not None and dec is not None else CoordinatesOrigin())
    psf_obj = PixelatedPSF(PixelatedRegularGrid(
        "psf.fits", field_of_view_x=(-psf_half, psf_half),
        field_of_view_y=(-psf_half, psf_half), num_pix_x=psf_pix, num_pix_y=psf_pix))
    instrument = Instrument(pixel_size=delta_pix, name=meta.get("band", "band"),
                            band=meta.get("band", ""), psf=psf_obj)
    obs = Observation(
        pixels=PixelatedRegularGrid("data.fits", field_of_view_x=fov,
                                    field_of_view_y=fov, num_pix_x=num_pix,
                                    num_pix_y=num_pix),
        mag_zero_point=meta.get("mag_zero_point"))
    cosmology = Cosmology(H0=70.0, Om0=0.3)

    metadata = dict(
        producer="claude-giga-lens P3 (32_coolest_export.py)",
        source_fit=str(fit_path), band=meta.get("band"),
        likelihood="diagonal (native VIS)" if meta.get("band") == "VIS"
        else "correlated conv-whitened (drizzle)",
        theta_E_convention="gigalens/lenstronomy EPL theta_E (b = theta_E*sqrt(q)); "
        "compare to PyAutoLens einstein_radius_effective at the MASS level "
        "(expect a few-percent convention offset, not a digit match)",
        shapelet_amps="RIDGE-MARGINALISED: a_star is the marginal-mode at the "
        "MAP z (Normal(0,5/sqrt(i+1)) prior, Lambda=(i+1)/25), NOT an MCMC block; "
        "point estimate only, no posterior. logdetA (Occam term) recorded in the fit.",
        source_model_caveat="Sersic + shapelets(n_max=6); the published PyAutoLens "
        "model uses a pixelised source (Hilbert mesh) -> compare masses, not source.",
        light_scale_resolved=summary.get("light_scale_resolved"),
        map_chi2_per_pixel=summary.get("map_chi2_per_pixel"),
        theta_E_eff_ours=summary.get("theta_E_eff_posterior")
        or summary.get("map_theta_E_eff"),
        theta_E_pub_eff=summary.get("theta_E_pub_eff"))

    coolest = COOLEST("MAP", origin, entities, obs, instrument,
                      cosmology=cosmology, metadata=metadata)

    stem = str(outdir / name)
    JSONSerializer(stem, obj=coolest, indent=2).dump_simple()
    print(f"wrote {stem}.json + psf/data/model_map.fits", flush=True)

    # chains sidecar
    if z is not None and "draws" in z.files:
        np.savez_compressed(outdir / "chains.npz", draws=z["draws"],
                            labels=z["labels"])
        print(f"wrote {outdir/'chains.npz'} (posterior sidecar)", flush=True)

    # ---- round-trip validation ---------------------------------------------
    c2 = JSONSerializer(stem).load_simple(stem + ".json", validate=True)
    le = c2.lensing_entities
    got_tE = le[0].mass_model[0].parameters["theta_E"].point_estimate.value
    got_gamma = le[0].mass_model[0].parameters["gamma"].point_estimate.value
    got_shear = le[2].mass_model[0].parameters["gamma_ext"].point_estimate.value
    got_amps = le[1].light_model[1].parameters["amps"].point_estimate.value
    assert abs(got_tE - comp["mass"]["theta_E"]) < 1e-9, got_tE
    assert abs(got_gamma - comp["mass"]["gamma"]) < 1e-9, got_gamma
    assert abs(got_shear - gamma_ext) < 1e-9, got_shear
    assert len(got_amps) == len(a_star), (len(got_amps), len(a_star))
    assert c2.mode == "MAP"
    print(f"ROUND-TRIP OK: mode={c2.mode} theta_E={got_tE:.4f} gamma={got_gamma:.4f} "
          f"gamma_ext={got_shear:.4f} n_amps={len(got_amps)} "
          f"n_entities={len(le)}", flush=True)
    return 0


def _load_info(name, subset):
    from cgl.paths import EUCLID_Q1_DATA
    p = EUCLID_Q1_DATA / subset / name / "info.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
