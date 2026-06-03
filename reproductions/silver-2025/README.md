# Silver et al. 2025 — ML-Driven Strong Lens Discoveries (reproduction)

Paper: `papers/Silver_2025_ML_driven_discoveries.pdf`
("ML-Driven Strong Lens Discoveries down to theta_E ~ 0.03 arcsec", Silver, Wang,
Huang, Bolton, Storfer, Banka.)

This reproduction stands up the **simulated-image ResNet classification pipeline**
and trains **Model 1a (HST-long)**, the conventional-lens regime
(0.5" < theta_E < 1.5"), whose published validation AUC is **0.9978** (paper §3.2.1.1,
reached by ~epoch 150).

## Scope of this stand-up
- DONE: Model 1a simulation pipeline + 360-epoch training launched on an L4.
- FOLLOW-ON (not done here): Model 2 (JWST 0.15-0.50"), Model 3 (JWST 0.02-0.15"),
  Model 1b (HST-short), and the U-Net pixel-level localizer for small-theta_E lenses.

## Scripts
- `01_gen_sims.py` — lenstronomy simulation of HST-regime images.
  - SIE lens (theta_E ~ U(0.5,1.5)"), lens at center; Sersic source offset N(0,0.25").
  - Source Sersic index n ~ U(2,6); arc brightness x 10^U(0.5,2.0) (selection bias).
  - Lens light (Sersic) + a few environmental Sersic galaxies (x 10^U(0,0.7)).
  - Lensed (class 1) = lens light + lensed source + env; Unlensed (class 0) = lens
    light + env, source off / theta_E=0 (paper Fig. 5 / §3.1.2).
  - Pixel scale 0.031", Gaussian PSF (FWHM 0.08", truncated at 3 sigma), 64x64 cutouts.
  - Preprocessing at sim time (paper §3.1.5, Model 1): subtract mean, divide std,
    clip above 99th percentile. Images saved NOISELESS.
- `02_train_resnet.py` — trains the Huang+2021 **shielded ResNet**
  (`../huang-2021/01b_shielded_resnet.py`, 32-channel shields = paper's Model 1
  setting, ~57K params) with `in_channels=1`.
  - **Noise is an in-training augmentation layer** (paper §3.1.5): per image per
    iteration, Poisson noise with texp ~ 10^U(2,6) s + Gaussian background
    sigma_BKG ~ U(0,0.2). Varies every iteration.
  - Hyperparameters (paper §3.1.6, Model 1): lr0 = 1.25e-3, decay 5x every 80 epochs,
    360 epochs, batch 64, Adam, BCEWithLogits, 80/20 stratified split, rot/flip aug.

## How to run
```bash
# 1. generate sims (4000-image starter set; paper uses 20000)
CUDA_VISIBLE_DEVICES="" /home/benson/.venvs/lens/bin/python 01_gen_sims.py \
    --n_per_class 2000 --out_prefix model1
# (for the full paper-scale set: --n_per_class 10000)

# 2. train Model 1a on ONE L4 (index 8 or 9)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 \
  /home/benson/.venvs/lens/bin/python 02_train_resnet.py \
    --images data/model1_images.npy --labels data/model1_labels.npy \
    --epochs 360 --batch 64 --tag model1a
```

### Check the running training
```bash
tail -f reproductions/silver-2025/train_model1a.log
# best AUC + full history:
python -c "import json;h=json.load(open('reproductions/silver-2025/data/training_history_model1a.json'));print('best AUC',h['best_auc']);print(h['history'][-1])"
```

## Outputs (data/, gitignored)
- `model1_images.npy` (N,1,64,64) float32 normalized noiseless; `model1_labels.npy`;
  `model1_meta.csv`; `model1_preview.png` (25 lensed examples, cf. paper Fig. 6).
- `checkpoint_best_model1a.pt`, `training_history_model1a.json`.

## Fidelity caveats / next steps
1. **VELA sources.** The paper's sources are VELA hydrodynamical galaxy stamps
   (34 galaxies, many redshifts/angles). Here they are analytic Sersic profiles —
   the explicit MVP proxy. Swapping in VELA stamps is the #1 upgrade and will add
   the clumpy/irregular high-z morphology that gives the arcs their realism.
2. **CosmoDC2 priors.** Lens z_l, z_s, M*, ellipticity are simple priors here, not
   drawn jointly from CosmoDC2 + the Shuntov+2022 M*-Mhalo relation; theta_E is sampled
   directly rather than derived from a halo-mass cut.
3. **Starter scale.** 4000 images (2000+2000) vs the paper's 20000. Sim throughput is
   ~440 img/s, so the full 20000 set takes <1 min — trivial to scale `--n_per_class`.
4. **Models 2/3 (JWST)** need the JWST noise model (absolute noise, W18 background,
   texp 1000/10000 s), source-centered cutouts, and Type-1 overlapping-galaxy
   non-lenses (paper §3.1.2-§3.1.3). Architecture changes: 16-channel shields,
   batch 16, 500 epochs, lr no decay.
5. **U-Net localizer** (paper §4) does pixel-level lens *localization* for
   small-theta_E candidates — a follow-on after the JWST classifiers.
