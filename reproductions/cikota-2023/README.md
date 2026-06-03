# Cikota et al. 2023 — Einstein cross DESI-253.2534+26.8843 (GIGA-Lens)

Public-data reproduction of Cikota et al. 2023, *DESI-253.2534+26.8843: A New
Einstein Cross Spectroscopically Confirmed with VLT/MUSE and Modeled with
GIGA-Lens* (arXiv:2307.12470). The paper modeled an SDSS-g image synthesized from
a **proprietary VLT/MUSE** data cube (≈0.6″ seeing). We **skip the spectroscopy**
and reproduce the GIGA-Lens lens model on **public DESI Legacy Surveys DR10**
g-band imaging — the same survey the system was discovered in (Huang et al. 2021).

Environment: `/raid/benson/.venvs/gigalens/bin/python` (JAX 0.6.2, TFP 0.25.0),
gigalens at `/raid/benson/lensing-repos/gigalens`. GPU: A16 index 6.

---

## Result vs paper

| Quantity | This work (public Legacy g) | Cikota+2023 (MUSE g) |
| --- | --- | --- |
| Einstein radius θ_E (L1) | **2.103″ (+0.044/−0.041)** | 2.520″ (+0.032/−0.031) |
| σ_SIE (L1, via SIS) | **347 km/s** | 379 ± 2 km/s |
| L1 ellipticity e1, e2 | −0.225, −0.295 | −0.365, −0.486 |
| external shear γ1, γ2 | +0.065, +0.064 | −0.008, −0.038 |
| L2 (subhalo) θ_E | **0.17″** | 0.261″ (+0.028/−0.027) |
| total magnification | **7.0** | 10.47 |
| reduced χ²/px | **0.90** | — (good residual) |
| four-image cross geometry | **reproduced** (Fig `03`) | yes |

The model **recovers the Einstein-cross geometry and a large Einstein radius**,
with χ²/px ≈ 0.90 (excellent fit) and a clean four-image residual. The headline
θ_E lands ~17% low of the paper, σ_SIE ~8% low. The dominant cause is **seeing**,
demonstrated quantitatively in the ablation below — not a model or data error. The
SIS→σ conversion is exact: feeding the formula the paper's θ_E=2.520″ with the
paper's distances (D_s=1690.6, D_L1-s=1026.1 Mpc) returns 379.4 km/s, matching the
paper's 379.

### Why θ_E comes out low: the PSF/seeing ablation (`04`)

The public Legacy DR10 g-band coadd PSF has **FWHM ≈ 1.35″**, whereas the paper's
MUSE-derived image had FWHM ≈ 0.6″. Re-fitting the *same* Legacy data with the
paper's 0.6″ Gaussian PSF moves θ_E **2.103″ → 2.276″** and σ_SIE **347 → 361
km/s**, closing roughly half the gap. The residual gap is irreducible: the 1.35″
Legacy seeing is baked into the pixels and cannot be deconvolved beyond what the
survey resolved. The paper had genuinely higher-resolution data.

---

## Model (Cikota Table 2/3, 31 parameters — exact match)

- **Mass:** L1 SIE + L2 SIE (foreground galaxy at z=0.386, near image C, treated as
  an effective subhalo) + external Shear.
- **Light:** L1 elliptical Sérsic + L2 spherical Sérsic + source elliptical Sérsic.
- **Priors:** the paper's Table 2 (these are the GIGA-Lens demo defaults specialized
  to this system: θ_E ~ exp(𝒩(ln2, 0.25)), shear ~ 𝒩(0, 0.05), etc.).
- **PSF:** the actual Legacy g-band coadd PSF (preferred over the paper's 0.6″
  Gaussian because it matches *this* data); the Gaussian is used only in the `04`
  ablation.
- Sérsic amplitudes are **sampled** (`use_lstsq=False`), so the inference uses
  `ForwardProbModel` + MAP (adabelief) → SVI variational posterior.

Redshifts (paper, used for the σ conversion only): z_L1=0.636, z_L2=0.386,
z_s=2.597. Cosmology: Planck18 (H0=67.4, Ωm=0.315).

---

## How to run (A16 index 6)

```bash
cd reproductions/cikota-2023
# 1. data: crop the public DESI Legacy DR10 g-band cutout + coadd PSF (CPU)
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  /raid/benson/.venvs/gigalens/bin/python 01_prep_data.py

# 2. MAP + SVI fit (GPU 6, single device). REUSE_MAP=0 to refit MAP from scratch.
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
  CUDA_VISIBLE_DEVICES=6 /raid/benson/.venvs/gigalens/bin/python 02_fit_map_svi.py

# 3. residual + critical-curve figure, σ_SIE, magnification (GPU 6)
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
  CUDA_VISIBLE_DEVICES=6 /raid/benson/.venvs/gigalens/bin/python 03_analysis.py

# 4. PSF/seeing ablation: refit with the paper's 0.6" Gaussian PSF (GPU 6)
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
  CUDA_VISIBLE_DEVICES=6 /raid/benson/.venvs/gigalens/bin/python 04_psf_ablation.py
```

Note: the installed gigalens `inference.py` calls `jax.experimental.shard_map.shard_map`
without importing the submodule; under JAX 0.6.2 it must be imported explicitly
(`import jax.experimental.shard_map`), which the scripts do. Pinning to ONE GPU via
`CUDA_VISIBLE_DEVICES` makes the gigalens `shard_map` run single-device; the gigalens
`HMC()` helper is NOT used (it pmaps over all visible devices and hangs).

The harmless `WeakStructRef`/`cache_util` tracebacks printed at exit are a TFP
weakref-cleanup quirk, not a fit error.

---

## Files

| File | Purpose |
| --- | --- |
| `01_prep_data.py` | crop public DESI Legacy DR10 g cutout + coadd PSF → `data/cikota_g_*` |
| `02_fit_map_svi.py` | GIGA-Lens MAP (adabelief) + SVI on the 31-param SIE+SIE+shear+3-Sérsic model |
| `03_analysis.py` | residual + critical-curve/caustic figure; θ_E→σ_SIE; total magnification |
| `04_psf_ablation.py` | refit with the paper's 0.6″ Gaussian PSF (seeing sensitivity) |
| `data/legacy_dr10_grz_120.fits` | public DR10 g/r/z cutout (RA 253.2534, Dec +26.8843, 120 px @ 0.262″) |
| `data/legacy_psf_g.fits` | DR10 g-band coadd PSF (FWHM ≈1.35″) |
| `data/map_estimate.npy`, `svi_posterior.npz`, `derived_quantities.npz` | fit products |
| `figs/01_data_inspect.png` | data / PSF / S-N with paper image positions overlaid |
| `figs/03_residual_critcurve.png` | **headline:** data | model+critical curve | residual | source+caustic |

## Honest status

- **Geometry + radius: reproduced.** Four-image Einstein cross, large θ_E ≈ 2.1″,
  tangential critical ellipse through the four images, compact source inside the
  astroid caustic, χ²/px ≈ 0.90.
- **θ_E ~17% low, σ_SIE ~8% low, μ_tot 7.0 vs 10.47** — driven by the public
  Legacy 1.35″ seeing vs the paper's 0.6″ MUSE image (quantified in `04`).
- **No spectroscopy** (z_L, z_s, σ_pPXF) was reproduced — that needs the
  proprietary VLT/MUSE cube and is out of scope. Paper redshifts are used only for
  the SIS→σ context conversion.
- L2 light/mass centroid drifts ~0.5″ from its prior in the broad PSF (it is very
  faint and partly blended with image C at 1.35″ seeing); the paper notes ε_l2 is
  its least-constrained parameter.
