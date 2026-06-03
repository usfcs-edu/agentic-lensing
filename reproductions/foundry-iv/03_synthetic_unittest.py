"""03 - Unit-test the automated redshift finder (_zfinder) on SYNTHETIC spectra.

We build noisy fake MUSE spectra at known redshifts and confirm the finder recovers
them to MUSE's spectral-sampling precision (~1.25 A/pix -> dz ~ a few x 1e-4). This
validates the engine independently of the ESO download, and documents the expected
precision floor.

Cases:
  A) passive lens galaxy, z=0.431 (Ca H&K, G-band, Hbeta absorption) -> find_z_absorption
  B) [OII]-emission source, z=0.821 ([OII] doublet + Hg + [OIII]) -> find_z_emission
  C) Lyman-break source, z=2.45 (UV metal absorption + [CIII]1909) -> find_z_absorption(UV)

Run:  python 03_synthetic_unittest.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import _zfinder as zf

REPRO = Path(__file__).parent
FIGS = REPRO / "figs"
FIGS.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

WAVE = np.arange(zf.MUSE_MIN, zf.MUSE_MAX, 1.25)  # MUSE WFM sampling


def gaussian(w, c, amp, sig):
    return amp * np.exp(-0.5 * ((w - c) / sig) ** 2)


def make_absorption_spectrum(z, line_list, depth=0.4, cont_level=100.0, snr=30.0, sig=2.5):
    flux = np.full_like(WAVE, cont_level)
    flux *= 1.0 + 0.05 * np.sin(WAVE / 800.0)  # gentle continuum slope
    for lam0 in line_list.values():
        lam = lam0 * (1 + z)
        if WAVE[0] < lam < WAVE[-1]:
            flux -= gaussian(WAVE, lam, depth * cont_level, sig)
    flux += RNG.normal(0, cont_level / snr, size=WAVE.shape)
    return flux


def make_emission_spectrum(z, line_list, amp=40.0, cont_level=20.0, snr=20.0, sig=2.0):
    flux = np.full_like(WAVE, cont_level)
    for name, lam0 in line_list.items():
        lam = lam0 * (1 + z)
        if WAVE[0] < lam < WAVE[-1]:
            a = amp * (1.4 if "[OII]" in name or "5007" in name else 0.6)
            flux += gaussian(WAVE, lam, a, sig)
    flux += RNG.normal(0, cont_level / snr * cont_level / cont_level, size=WAVE.shape)
    flux += RNG.normal(0, 2.0, size=WAVE.shape)
    return flux


def run_case(label, wave, flux, finder, z_true, **kw):
    z_best, score, zgrid, detail = finder(wave, flux, **kw)
    err = z_best - z_true
    ok = abs(err) < 0.001
    print(f"  [{ 'PASS' if ok else 'FAIL'}] {label:28} z_true={z_true:.4f} "
          f"z_found={z_best:.4f}  dz={err:+.4f}")
    return z_best, score, zgrid, detail, ok


def main():
    print("Synthetic unit tests for _zfinder:")
    results = []

    # A) passive lens at z=0.431
    zt = 0.431
    fluxA = make_absorption_spectrum(zt, zf.ABS_LINES_GAL)
    rA = run_case("lens absorption (gal)", WAVE, fluxA, zf.find_z_absorption,
                  zt, zmin=0.0, zmax=1.2)
    results.append(("A_lens_abs", zt, rA))

    # B) emission source at z=0.821
    zt = 0.821
    fluxB = make_emission_spectrum(zt, {k: v for k, v in zf.EMIS_LINES.items()
                                        if "[OII]" in k or k in ("Hgamma", "[OIII]_4959", "[OIII]_5007")})
    rB = run_case("source emission [OII]+[OIII]", WAVE, fluxB, zf.find_z_emission,
                  zt, zmin=0.0, zmax=1.5)
    results.append(("B_src_emis", zt, rB))

    # C) Lyman-break UV-absorption source at z=2.45
    zt = 2.45
    fluxC = make_absorption_spectrum(zt, zf.ABS_LINES_UV, depth=0.35, snr=25.0)
    rC = run_case("source UV absorption (LBG)", WAVE, fluxC, zf.find_z_absorption,
                  zt, zmin=1.5, zmax=3.5, line_list=zf.ABS_LINES_UV)
    results.append(("C_src_uv_abs", zt, rC))

    n_pass = sum(r[2][4] for r in results)
    print(f"\n{n_pass}/{len(results)} synthetic cases passed (dz < 0.001).")

    # figure: spectra + score curves
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    specs = [("A: lens z=0.431", WAVE, fluxA, rA), ("B: src z=0.821", WAVE, fluxB, rB),
             ("C: src z=2.45", WAVE, fluxC, rC)]
    for i, (title, w, fx, r) in enumerate(specs):
        z_best, score, zgrid, detail, ok = r
        zt = results[i][1]
        axes[i, 0].plot(w, fx, lw=0.5, color="k")
        axes[i, 0].set_title(f"{title}  (found {z_best:.4f})")
        axes[i, 0].set_xlabel("observed wavelength [A]")
        axes[i, 1].plot(zgrid, score, lw=0.8)
        axes[i, 1].axvline(zt, color="g", ls="--", label=f"truth {zt}")
        axes[i, 1].axvline(z_best, color="r", ls=":", label=f"found {z_best:.4f}")
        axes[i, 1].set_xlabel("trial z")
        axes[i, 1].set_ylabel("score")
        axes[i, 1].legend(fontsize=8)
    fig.tight_layout()
    out = FIGS / "03_synthetic_unittest.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")
    assert n_pass == len(results), "synthetic unit tests failed"


if __name__ == "__main__":
    main()
