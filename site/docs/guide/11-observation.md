# From photons to pixels: PSF, noise, and drizzle

Every number this book eventually fits — a slope, a shear, an Einstein radius —
comes from an array of floating-point numbers that started as photons hitting
silicon. This chapter is the chain between the two: how a CCD turns light into
counts, why the resulting image is never quite what the sky actually looked
like (the point-spread function), why each pixel's error bar has the specific
algebraic form it does, and why *HST*'s habit of combining several dithered
exposures onto one output grid — drizzling — quietly correlates the noise
between neighbouring pixels. That last fact is not a footnote. It is the first
link in what this book calls **Spine 1**: drizzle correlates noise, a
diagonal (independent-pixel) likelihood assumes it does not, that mismatch
biases a fitted slope, and undoing it is what produces the 191-nat evidence
swing you will compute yourself in [Ch. 25](25-money-number.md#the-evidence-flip).
Everything in this chapter is arithmetic you can check on a calculator, and
every check matches a real number this campaign published or retracted.

!!! abstract "What you can skip"
    You already know that a Poisson-distributed count has variance equal to
    its mean, and you already know what a convolution is (Ch. 7 built it from
    scratch either way). What is genuinely new here is astronomy-specific:
    what a point-spread function *is* physically, the specific two-term shape
    of a CCD's noise model, and the drizzle resampling scheme — none of which
    has a direct ML analogue, so this chapter does not assume you have seen
    them.

## CCDs and the point-spread function { #ccds-and-psf }

A CCD (or the HgCdTe array HST's infrared channel actually uses — the physics
is the same for this chapter's purposes) is a grid of light-sensitive wells.
Over an exposure of duration $t_{\mathrm{exp}}$, each well accumulates
photo-electrons: a photon that reaches the detector frees an electron with
some wavelength-dependent probability (the quantum efficiency), and at
readout the well's accumulated charge is reported as a count. Two facts about
this process matter for everything that follows. First, it is a **counting**
process — the number of photo-electrons in a well over a fixed exposure is a
Poisson random variable, and for a Poisson variable with mean $c$, the
variance is also $c$. Second, nothing about this process cares what the true
sky brightness distribution looked like at scales finer than the telescope
can resolve — which is the point-spread function's whole story.

No real telescope images a point source as a point. Diffraction at a finite
aperture, small imperfections in the optics, and (for ground-based
instruments) the atmosphere all spread a true point of light into a smooth,
extended blob before it reaches the detector — the **point-spread function**
(PSF). The image that lands on the array is not the sky; it is the sky
*convolved* with the PSF. This is the exact convolution of
[Ch. 7](07-fourier.md#convolution-vs-correlation), not an analogy: the forward
model this repo fits ([Ch. 22](22-inference.md#the-forward-model)) literally
renders a noiseless model image on a fine grid and convolves it with an
empirical PSF kernel before comparing it, pixel by pixel, to the data. Get the
kernel wrong and every downstream number — every $\chi^2$, every posterior —
inherits the error.

!!! tip "You already know this"
    A point-spread function is a fixed, non-learned convolution kernel —
    exactly a single frozen conv layer with one channel. "The kernel must be
    sampled at the data's own pixel scale" is exactly the ML bug family of
    feeding a fixed-resolution operator an input at the wrong resolution:
    nothing raises an exception, the array shapes still broadcast, and the
    output is silently wrong by a scale factor.

That last sentence is not hypothetical; it is the single largest defect this
campaign's predecessor reproduction actually shipped for a while. `gigalens`'s
simulator hands the PSF kernel to `lenstronomy`'s `subgrid_kernel`, which
**assumes the kernel it receives is already sampled at the data's own pixel
scale**, `delta_pix`, and upsamples it internally by the rendering
`supersample` factor. `foundry-i` instead fed it an empirical PSF built
directly from stars at its own, finer native sampling ($0.065''$) while
telling the simulator `delta_pix=0.13`, `supersample=2` — so the function
upsampled a kernel that was already oversampled, double-applying the
refinement and broadening the *effective* PSF actually used in every
native-scale fit by roughly $2\times$.

The tell was not a crash; it was a $\chi^2_\nu$ **floor**. No matter how much
extra flexibility the model was given — additional Sérsic light components,
more shapelet terms in the source — the reduced chi-squared refused to drop
below about
<!-- check: ch11.psf_chi2_broadened = 3.4 ± 0.15 -->
$\chi^2_\nu \approx 3.4$.
That distinction matters: underfitting is a problem more model flexibility
relieves; a floor that flexibility cannot touch is the signature of a
*systematic* — something structurally wrong that no amount of extra freedom
in the light or source model can absorb, because the mismatch is in the
instrument model, not the sky model. Fixing *only* the PSF sampling
convention — same noise model, same data, same everything else — dropped the
floor to
<!-- check: ch11.psf_chi2_fixed = 1.051 ± 0.01 -->
$\chi^2_\nu = 1.051$,
and simultaneously resolved what had looked like a sampler pathology: the
slope parameter's effective sample size, which had been stuck near 260 after
thousands of draws under the broadened kernel, jumped past 5,700 once the PSF
was corrected. The sampler had not been struggling; the model had been
structurally wrong in a way that also happened to flatten one posterior
direction. `cgl/guards.py:74` encodes this incident as a hard check,
`assert_psf_sampling`, so it cannot recur silently:

```python
if abs(psf_pixel_scale - delta_pix) > atol:
    raise GuardError(
        f"PSF kernel sampled at {psf_pixel_scale}\" but delta_pix={delta_pix}\". "
        "subgrid_kernel upsamples internally; passing a supersampled kernel "
        "double-applies the refinement..."
    )
```

## The noise model: background plus Poisson { #the-noise-model }

Every pixel's error bar in this repo's likelihood comes from two independent
noise sources added in quadrature. The first is a roughly constant background
term — sky brightness, detector read noise, dark current — with RMS
$\sigma_{\mathrm{bkg}}$, approximately the same in every pixel of an exposure.
The second is the source's own photon-counting (Poisson) noise, and it is
where the CCD physics above becomes an equation.

Let $f$ be the model's predicted flux *rate* in a pixel (counts per unit
time). Over an exposure of length $t_{\mathrm{exp}}$, the expected photon
count is $c = f\, t_{\mathrm{exp}}$, and because counting is Poisson,
$\mathrm{Var}[\text{count}] = c = f\, t_{\mathrm{exp}}$. Converting back to a
flux-rate estimate $\hat f = \text{count}/t_{\mathrm{exp}}$ rescales the
variance by $1/t_{\mathrm{exp}}^2$:

$$
\mathrm{Var}[\hat f] \;=\; \frac{\mathrm{Var}[\text{count}]}{t_{\mathrm{exp}}^2}
\;=\; \frac{f\, t_{\mathrm{exp}}}{t_{\mathrm{exp}}^2} \;=\; \frac{f}{t_{\mathrm{exp}}}.
$$

Two independent noise sources add in variance, not in RMS, so the total
per-pixel error is

$$
\mathrm{err} \;=\; \sqrt{\sigma_{\mathrm{bkg}}^2 \;+\; \frac{f}{t_{\mathrm{exp}}}}.
$$

That is the exact line the repo runs — `foundry-i/_hmc_lib.py:135` writes
`err_map = jnp.sqrt(self.background_rms ** 2 + im_sim / self.exp_time)`, and
`cgl/mocks.py:227`, the mock generator this book's own examples trace back to,
writes the identical formula with its own constants: `SIGMA_BKG = 0.2`,
`EXP_TIME = 100.0` (`cgl/mocks.py:65-66`). At a pixel where the model predicts
flux $f=1.0$ in those same units,
<!-- check: ch11.err_example = 0.2236 ± 0.0001 -->

$$
\mathrm{err} = \sqrt{0.2^2 + 1.0/100} = \sqrt{0.05} = 0.2236,
$$

a number you can check on a calculator against the repo's own constants
before you ever look at a data file.

Notice what this buys, and what it costs. Because $\mathrm{err}$ depends on
the *model* flux $f$ and not on the data, brighter model pixels automatically
get a larger error bar and therefore less weight — the likelihood is a
**weighted** least squares with weight $\sqrt{W} = 1/\mathrm{err}$
(`cgl/likelihood.py:326`), and that weight is heteroscedastic by construction,
not fit separately. The cost is subtler: the error map is a function of the
*parameters being fit*, recomputed at every forward pass, not a fixed
quantity decided once from the data alone.

!!! tip "You already know this"
    An error map that is itself a function of the model is exactly what an
    aleatoric-uncertainty network predicts alongside its mean output — the
    "noise" you divide by is recomputed every forward pass, not looked up.
    Down-weighting noisy pixels by their own predicted variance
    ($\sqrt W = 1/\mathrm{err}$) is the identical move to uncertainty-weighted
    regression loss.

Every data product the repo loads carries the same five-object shape — image,
error map, keep-mask, PSF, metadata (`cgl/paths.py:81`, `load_product`) — but
the error map's *origin* differs by survey: for the *HST* products above it
is rebuilt from $\sigma_{\mathrm{bkg}}$ and $t_{\mathrm{exp}}$; for the Euclid
Q1 targets of [Ch. 22](22-inference.md#the-gigalens-recipe), the survey
delivers a per-pixel RMS map directly (`VIS_RMS`), so `err_map` is used as-is
and $t_{\mathrm{exp}}$ never enters the log-posterior at all
(`cgl/euclid_io.py:34-39`) — same interface, different physical origin.

A calibration this formula depends on is where the campaign's second real
incident lives. Early in the reproduction, the fine-scale product's MAP fit
reported
<!-- check: ch11.sky_chi2_artifact = 0.451 ± 0.001 -->
$\chi^2_\nu = 0.451$ — comfortably, almost suspiciously, under the campaign's
own $<1.1$ quality bar. A $\chi^2_\nu$ well below 1 is itself a red flag, not
a free win: with correctly calibrated Gaussian noise, a good fit clusters
$\chi^2_\nu$ *near* 1, not far below it, because a $\chi^2_\nu \ll 1$ means the
error bars you divided by were too large — the noise model overstated the
noise. An audit found exactly that: $\sigma_{\mathrm{bkg}}$ had been
calibrated on raw-image fluctuations at large radius that were actually about
70% diffuse light from the lens galaxy's own outer wings — starlight the
segmentation had missed, not noise at all — inflating every downstream error
bar. Recalibrating the identical kernel on the **model-subtracted** residual
(where the lens light is already removed) gave the honest
<!-- check: ch11.sky_chi2_honest = 0.92 ± 0.01 -->
$\chi^2_\nu = 0.92$: still comfortably under the bar, but no longer
suspicious. `cgl/guards.py:129`, `assert_model_subtracted_sky`, now refuses
any noise-calibration artifact that is not explicitly tagged
`model_subtracted=True`. Put the two incidents side by side: a $\chi^2$ that
refuses to *improve* and a $\chi^2$ that looks *too good* are equally
diagnostic, in opposite directions, of the same kind of mistake — a wrong
noise model, not a wrong sky model.

## Drizzle { #drizzle }

*HST* rarely takes one exposure of a target; it takes several, each offset by
a fraction of a pixel (a *dither*), and combines them onto a common output
grid — often finer than any single native exposure — with an algorithm called
**drizzle** (Fruchter & Hook 2002). Each input pixel's flux is distributed
over the output-pixel footprint its "drop" overlaps, scaled by a parameter
called `pixfrac`. This campaign's target system, Foundry I, exists in three
such drizzle products at different output scales
([main.tex, Data and Targets](../current/claude-giga-lens/index.md#sec:data)):
a fine $0.04''$/px MAST HAP mosaic (`v3`), a $2\times2$-binned $0.08''$/px
rebuild of it (`v3b`), and the pipeline's own native-scale product (`v2d`) at
close to the true WFC3/IR detector pixel.

Think of an output pixel's noise value as a **weighted average of the native
pixels whose drop footprint overlaps it**. When the output grid is finer than
native, two *adjacent* output pixels typically share at least one of those
native parents — part of their noise comes from literally the same random
draw, so they are not independent. This is the same mechanism
[Ch. 7](07-fourier.md#psd-and-autocorrelation) covers in general: a
moving-average filter turns white noise into a correlated process, with
correlation extending over the filter's width. The extreme case makes the
mechanism plain: at a *fixed* sub-pixel phase with `pixfrac=1`, each output
pixel would simply *copy* the one native pixel it falls inside — any two
sharing that native parent perfectly correlated, any two that do not exactly
independent (native pixels are drawn iid). Real dithers do not sit at one
fixed phase; successive exposures land the output grid at different sub-pixel
offsets, and averaging the covariance and the variance *separately* over all
such unknown registrations (`cgl/noise.py:166`, `drizzle_acf`) is what turns
that simple same-phase argument into the campaign's closed-form anchor for
the along-axis, nearest-neighbour correlation:

$$
t(1) \;=\; \frac{r - 1}{r - 1/3}, \qquad
r \;=\; \frac{\text{native detector pixel}}{\text{output pixel}}.
$$

This is the one line that opens Spine 1. `site/guide_src/lensing.py:194` is
the guide's own copy of it. The true WFC3/IR detector pixel is a header fact,
<!-- check: ch11.native_pix_wfc3ir = 0.1283 ± 0.0001 -->
$0.1283''$ (`02_fit_noise_kernels.py:79`); for the fine skycell's output scale
<!-- check: ch11.px_fine = 0.04 ± 0.0001 -->
of $0.04''$/px, that gives
<!-- check: ch11.r_fine = 3.2075 ± 0.001 -->

$$
r_{\mathrm{fine}} = 0.1283/0.04 = 3.2075,
$$

and the closed form evaluates to
<!-- check: ch11.drizzle_t1_fine = 0.76805 ± 0.00001 -->

$$
t(1) = \frac{3.2075 - 1}{3.2075 - 1/3} = 0.76805 —
$$

matched to three decimal places — $0.76805 - 0.76799 = 6\times10^{-5}$ — by
numerically enumerating the actual drop-overlap operator on a test patch
(`cgl/noise.py:166`; the campaign's own report quotes $0.76799$ for that
enumeration,
[Methods I, covariance model and drizzle anchor](../current/claude-giga-lens/index.md#sec:methods1)).
That agreement between a closed-form phase-average and a full numerical
enumeration of the real operator is what makes the number trustworthy rather
than merely plausible — you can reproduce both sides yourself with
`worked_examples.py --show ch11`.

<figure markdown="span">
  ![The closed-form drizzle lag-1 noise correlation t(1) as a function of the pixel-scale ratio r](figures/ch11-drizzle-correlation-light.svg#only-light){ width="90%" }
  ![The closed-form drizzle lag-1 noise correlation t(1) as a function of the pixel-scale ratio r](figures/ch11-drizzle-correlation-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 11.1.** $t(1) = (r-1)/(r-1/3)$ against the
  pixel-scale ratio $r$, from just above 1 (almost no resampling) to 4
  (aggressive oversampling). The campaign's fine skycell sits at
  $r=3.2075$ — $0.04''$ output pixels built from a $0.1283''$ native detector
  pixel — where the curve gives $t(1)=0.768$: over three-quarters of one
  output pixel's noise is shared with its neighbour. Nothing here is fit to
  real data; the curve is the closed form above, evaluated on a grid.</figcaption>
</figure>

The closed form is not universal, and treating it as though it were is its
own small lesson. It is derived for well-oversampled resampling and is only
valid for $r \geq 2$ (`02_fit_noise_kernels.py:225-227`). The native product
sits at $r = 0.1283/0.13 \approx 0.987$ — *below* that domain — and plugging
it in anyway returns
<!-- check: ch11.drizzle_t1_native = -0.020 ± 0.001 -->
$t(1) \approx -0.02$: a negative "correlation" from a construction (a
weighted average with non-negative weights) that can never produce one in the
regime the formula was actually derived for. The formula does not raise an
exception; it just stops meaning anything. That is exactly why $r\approx1$ at
native scale is not a defect to fix but the reason this product is called the
campaign's *diagonal-trustworthy anchor*: almost no resampling happened
there, so whatever residual correlation exists must be measured directly
(the numerical enumeration), never read off this closed form.

## Why drizzle correlates noise, and what it breaks { #why-drizzle-correlates-noise }

The closed form above is the pure drizzle-kernel mechanism; the *full*
measured correlation on real data includes it plus other real structure. On
the model-subtracted residual, the campaign measures an along-axis lag-1
correlation of
<!-- check: ch11.rho1_fine_measured = 0.815 ± 0.001 -->
$\rho(1) = 0.815$ on the fine product,
<!-- check: ch11.rho1_binned_measured = 0.615 ± 0.001 -->
$0.615$ binned, and
<!-- check: ch11.rho1_native_measured = 0.305 ± 0.001 -->
$0.305$ native
([main.tex, Table 1](../current/claude-giga-lens/index.md#sec:data)) — every
one of them larger than the pure drizzle anchor at that scale, because real
pixels also carry PSF and detector structure the idealized kernel does not
model. The consequence is quantified as an **effective independent-pixel
fraction**: on the fine product,
<!-- check: ch11.neff_over_n_fine = 0.017 ± 0.001 -->
$N_{\mathrm{eff}}/N \approx 0.017$ — a $2\times2$ block of fine pixels is
about 78% internally correlated, not four separate measurements.

This is the mismatch that makes a *diagonal* Gaussian likelihood
mis-specified on drizzled data. A diagonal likelihood implicitly assumes it
has $N$ independent pixels; on the fine product it really has $N_{\mathrm{eff}}
\approx 0.017N$. Posterior widths shrink as the inverse square root of the
effective sample size (the same central-limit scaling behind every standard
error you have ever quoted), so believing $N$ when the truth is $N_{\mathrm{eff}}$
makes the reported error bars too tight by a factor
<!-- check: ch11.diagonal_overconfidence_factor = 7.67 ± 0.01 -->

$$
\sqrt{\frac{N}{N_{\mathrm{eff}}}} = \sqrt{\frac{1}{0.017}} \approx 7.67:
$$

almost eight times too confident, on this product alone. This is not a
rounding correction; it is the wrong calibration entirely, and it is why a
diagonal likelihood cannot simply be "corrected" with a fudge factor on its
error bars — the correlation structure itself has to enter the likelihood.
[Ch. 24](24-correlated-noise.md#the-covariance-model) builds exactly that: a
correlated-noise likelihood $C = D^{1/2}KD^{1/2}$ with $K$ a stationary kernel
anchored on the closed form of this chapter, and
[convolutional whitening](24-correlated-noise.md#convolutional-whitening) to
make it tractable at these pixel counts. Once you correct this mismatch on
the real system, the answer moves — by how much, and whether the correction
is itself trustworthy, is the 191-nat evidence swing you derive in
[Ch. 25, The chain](25-money-number.md#the-chain).

!!! note "γ Ledger"
    This chapter never touches $\gamma$ — nothing here constrains the EPL
    slope. What it establishes is the *reason a constraint on $\gamma$ can be
    wrong at all*: the diagonal likelihood every fast lens-modeling code uses
    by default is misspecified on drizzled data by a measured factor of
    $\sim\!7.7\times$ in its own error bars on the fine product. Every
    $\gamma$ number that follows in this book inherits that fact until
    [Ch. 24](24-correlated-noise.md#the-covariance-model) fixes it.

## Connect to the repo { #connect }

- `cgl/guards.py:74-91` (`assert_psf_sampling`) and `cgl/guards.py:129-141`
  (`assert_model_subtracted_sky`) — the two incidents this chapter teaches,
  hard-coded as guards so they cannot recur silently.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/guards.py#L74)
- `cgl/mocks.py:65-66,225-227` (`add_noise_native`) — the noise model,
  $\mathrm{err}=\sqrt{\sigma_{\mathrm{bkg}}^2 + f/t_{\mathrm{exp}}}$,
  instantiated with its own constants.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/mocks.py#L225)
- `foundry-i/_hmc_lib.py:135` — the identical formula in the earlier
  reproduction (`background_rms`, `exp_time`).
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/foundry-i/_hmc_lib.py#L135)
- `cgl/paths.py:81-90` (`load_product`) and `cgl/likelihood.py:262-289,326` —
  the five-object product dict every driver script loads (`img`, `err_map`,
  `keep_mask`, `psf`, `meta`), and where $\sqrt W = 1/\mathrm{err}$ becomes the
  likelihood's per-pixel weight.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/likelihood.py#L326)
- `cgl/euclid_io.py:34-39` — the Euclid contrast: the survey delivers
  `err_map=VIS_RMS` directly, so `t_exp` never enters the log-posterior.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/euclid_io.py#L34)
- `site/guide_src/lensing.py:194-205` (`drizzle_lag1`) and
  `cgl/noise.py:117-213` (`drizzle_overlap_matrix_1d`, `drizzle_acf`) — the
  closed form and the numerical enumeration that validates it.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/noise.py#L166)
- `02_fit_noise_kernels.py:79-83,202-227` — where $r_{\mathrm{fine}}=3.2075$
  is defined from a header fact, and the "closed form only valid for $r\geq2$"
  note that flags the native product as out of domain.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/02_fit_noise_kernels.py#L202)
- main.tex: [Data and Targets, the real system and its three drizzle
  products](../current/claude-giga-lens/index.md#sec:data) and
  [Methods I, covariance model and drizzle
  anchor](../current/claude-giga-lens/index.md#sec:methods1).
- The foundry-i final report: [The sampling-convention
  rule](../reproductions/foundry-i/index.md#sec:psf-convention) and [Why
  earlier attempts disagreed: the defect
  chain](../reproductions/foundry-i/index.md#sec:defects).

## Exercises { #exercises }

??? question "Exercise 11.1 — the noise model, from counting statistics to a calculator"
    Starting from $\mathrm{Var}[\text{count}] = c$ for a Poisson-distributed
    photon count with mean $c = f\,t_{\mathrm{exp}}$, derive
    $\mathrm{Var}[\hat f]$ for the flux-rate estimate $\hat f =
    \text{count}/t_{\mathrm{exp}}$, and show it reduces to $f/t_{\mathrm{exp}}$.
    Then, using the repo's own mock constants $\sigma_{\mathrm{bkg}}=0.2$,
    $t_{\mathrm{exp}}=100$ (`cgl/mocks.py:65-66`), compute $\mathrm{err}$ at a
    pixel where the model predicts $f=1.0$.

    ??? success "Solution"
        Rescaling a random variable by a constant $1/t_{\mathrm{exp}}$
        rescales its variance by the square of that constant:

        $$
        \mathrm{Var}[\hat f] = \mathrm{Var}\!\left[\frac{\text{count}}{t_{\mathrm{exp}}}\right]
        = \frac{\mathrm{Var}[\text{count}]}{t_{\mathrm{exp}}^2}
        = \frac{f\, t_{\mathrm{exp}}}{t_{\mathrm{exp}}^2} = \frac{f}{t_{\mathrm{exp}}}.
        $$

        With $\sigma_{\mathrm{bkg}}=0.2$, $t_{\mathrm{exp}}=100$, $f=1.0$:

        $$
        \mathrm{err} = \sqrt{0.2^2 + 1.0/100} = \sqrt{0.04+0.01} = \sqrt{0.05} \approx 0.2236
        $$

        <!-- check: ch11.err_example = 0.2236 ± 0.0001 -->
        — matching `worked_examples.py --show ch11`'s `err_example` exactly,
        because it is the same formula run on the same two numbers.

??? question "Exercise 11.2 — a formula outside its own domain"
    The closed form $t(1)=(r-1)/(r-1/3)$ is only valid for $r\geq2$
    (`02_fit_noise_kernels.py:225-227`). Evaluate it at $r=0.987$ (the native
    product) and interpret the sign. Given that a drizzled output pixel is a
    weighted average with non-negative weights, could a *true* correlation
    computed that way ever come out negative? What should you conclude when a
    formula that "should" only ever return values between 0 and 1 hands you a
    negative number without complaining?

    ??? success "Solution"

        $$
        t(1) = \frac{0.987-1}{0.987 - 1/3} = \frac{-0.013}{0.6537} \approx -0.020
        $$

        <!-- check: ch11.drizzle_t1_native = -0.020 ± 0.001 -->
        A weighted average of non-negative weights cannot produce a negative
        correlation between two such averages that share any positively
        weighted term, so $-0.020$ cannot be a real correlation. The formula
        does not check its own domain — it is a closed form derived under an
        approximation ($r\geq2$, well-oversampled resampling) that simply does
        not hold here, and it returns a syntactically valid, semantically
        meaningless number rather than an error. The lesson generalizes past
        drizzle: any closed form you did not derive yourself is only as
        trustworthy as your knowledge of the assumptions that produced it. At
        native scale the campaign does not extrapolate the closed form at
        all; it measures the correlation directly by enumerating the real
        drop-overlap operator (`cgl/noise.py:166`), which is non-negative by
        construction.

??? question "Exercise 11.3 — how overconfident is a diagonal likelihood, exactly?"
    The fine product's measured $N_{\mathrm{eff}}/N \approx 0.017$. If a
    diagonal likelihood believes it has $N$ independent pixels when only
    $N_{\mathrm{eff}}$ behave independently, and posterior widths shrink as
    the inverse square root of the effective sample size, by what factor does
    a diagonal fit on this product understate its own posterior width?

    ??? success "Solution"

        $$
        \frac{\sigma_{\mathrm{diag,\, believed}}}{\sigma_{\mathrm{true}}}
        = \sqrt{\frac{N_{\mathrm{eff}}}{N}} \;\;\Rightarrow\;\;
        \text{understatement factor} = \sqrt{\frac{N}{N_{\mathrm{eff}}}}
        = \sqrt{\frac{1}{0.017}} \approx 7.67
        $$

        <!-- check: ch11.diagonal_overconfidence_factor = 7.67 ± 0.01 -->
        A diagonal fit on the fine product reports error bars roughly $7.7$
        times too tight — not a rounding issue but a fundamentally wrong
        calibration. This is precisely the miscalibration
        [Ch. 24](24-correlated-noise.md#the-covariance-model)'s
        correlated-noise likelihood exists to remove.

??? question "Exercise 11.4 — two chi-squared incidents, one lesson"
    Section [The noise model](#the-noise-model) described a $\chi^2_\nu$
    floor that extra model flexibility could not move, and a $\chi^2_\nu$
    that looked *too good* until the sky calibration was corrected. Both were
    eventually caught, but neither would have been caught by simply "fitting
    harder." What single diagnostic habit would have caught both, and why
    does adding model flexibility fail to substitute for it?

    ??? success "Solution"
        Both incidents are visible in the *shape* of the residual, not its
        magnitude: a PSF-broadening defect leaves structured, PSF-scale
        residual power no source-model flexibility can absorb (the mismatch
        is in the instrument model, not the sky model), and a
        wing-contaminated sky calibration leaves a model-subtracted residual
        whose variance does not match the raw image's. Adding model
        flexibility only ever changes what is being fit; it cannot diagnose
        whether the *noise model being divided by* is correct, since a
        chi-squared statistic is a ratio of the two. The habit that catches
        both: before adding model complexity, inspect the model-subtracted
        residual itself — its autocorrelation, its variance by region —
        rather than only watching whether $\chi^2_\nu$ moves. This is
        exactly what `assert_model_subtracted_sky` now enforces mechanically.
