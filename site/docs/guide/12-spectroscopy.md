# Redshifts and what a spectrum tells you

Every chapter so far has treated an image as the whole of the data: a grid of
pixels, a point-spread function, a noise model. A spectrum measures the same
object a different way — flux broken out by wavelength instead of by
position — and it buys you two numbers no image can give you on its own: a
redshift $z$, which tells you how far away something is, and a velocity
dispersion $\sigma_v$, which tells you how much mass is moving around inside
it. This chapter derives both from first principles and then reproduces them
on a real DESI system. By the end you will have run, by hand, the same
redshift-ratio test that DESI Strong Lens Foundry's discovery pipeline
(Hsu et al. 2025) uses to flag lens candidates out of 28 million fiber
spectra — before anyone looks at a picture.

!!! abstract "What you can skip"
    You do not need atomic physics: take it as given that an electron dropping
    between two fixed energy levels emits or absorbs light at one specific,
    reproducible wavelength, and move on. You do not need spectrograph
    engineering (fibers, gratings, detectors) — for this chapter's purposes a
    spectrograph is a very high-resolution, wavelength-binned camera. You do
    already own the one piece of signal-processing math this chapter leans on:
    matching a template against data by sliding it and scoring the overlap is
    a cross-correlation, and you met it formally in Ch. 7.

## What a spectrum is, and why it has lines { #spectral-lines }

A galaxy's spectrum is a plot of flux against wavelength. Most of it is a
smooth continuum — the light of a few hundred billion stellar photospheres,
each close to a blackbody. Sitting on that continuum are sharp features at
specific wavelengths: **emission lines**, where hot ionized gas (star-forming
regions, an active nucleus) radiates at the exact energy of an electron
transition, and **absorption lines**, where cooler gas or a stellar
photosphere in front of the continuum removes light at that same energy.
Because the transition energy is fixed by quantum mechanics and not by the
local environment, every hydrogen atom in the universe emits Balmer-alpha
(Hα, a name — not the deflection-angle $\alpha$ of later chapters) at the
same rest-frame wavelength, every once-ionized oxygen ion
produces the same [OII] doublet near 3727 Å, and every calcium ion in an old
star's atmosphere absorbs at the same Ca II H & K pair near 3934/3969 Å. The
rest wavelength is a physical constant of the transition — not a measurement
of the object in front of you.

What *does* vary from object to object is where that fixed pattern shows up in
the observed spectrum. If every line — from the reddest Balmer line to the
bluest oxygen line — sits at the *same* multiplicative factor away from its
rest wavelength, that object has a redshift. Measuring that one factor is the
whole content of the next section.

A chapter-local notation note, made explicit because this book is careful
about it elsewhere: this chapter writes wavelength as $\lambda$, the standard
symbol in every spectroscopy text. That has nothing to do with the SMC
tempering parameter of the same letter you will meet in
[Ch. 23](23-samplers.md#tempering-and-smc) — the two never appear in the same
equation, and $\lambda$ means wavelength everywhere in this chapter and
nowhere else in the book.

## Measuring a redshift { #measuring-redshift }

Define the redshift operationally as

$$
1 + z \;\equiv\; \frac{\lambda_{\mathrm{obs}}}{\lambda_{\mathrm{emit}}},
\label{eq:zdef}
$$

where $\lambda_{\mathrm{emit}}$ is a line's known rest-frame wavelength and
$\lambda_{\mathrm{obs}}$ is where you find it in the measured spectrum. This
is a definition, not yet a physical claim — whether the underlying cause is
the expansion of space or a literal recession velocity is exactly the
question [Ch. 13](13-expansion.md#redshift-is-expansion) answers. For this
chapter, treat $z$ as a wavelength ratio you read off a spectrum, full stop.

In practice nobody eyeballs one line. A real pipeline (DESI's `redrock`, the
tool behind every redshift in this section) carries a library of template
spectra — quiescent galaxy, star-forming galaxy, quasar, star — and for each
template scores, on a fine grid of trial redshifts, how well the redshifted
template matches the observed spectrum. The redshift reported is the trial
value that maximizes that score.

!!! tip "You already know this"
    Scoring a redshifted template against data and taking the arg-max over
    trial redshifts is the cross-correlation operation of
    [Ch. 7](07-fourier.md#convolution-vs-correlation), with the redshift
    playing the role of the lag. Matching the *whole* pattern of lines, not
    one feature, is what keeps the estimate honest — the spectroscopic analog
    of a matched filter rejecting a lone spurious peak — and it is why
    redshift catalogs carry a warning flag (`ZWARN`) for fits the pipeline
    itself does not trust.

**A real pair, measured.** Here is a real system recovered by exactly this kind of fit, matched to
grade-A candidate DESI-004.5374+01.0382 in Hsu et al. (2025)'s Table 2. Two
DESI fiber spectra, separated by under an arcsecond on the sky, carry measured
redshifts

$$
z_{\mathrm{lens}} = 0.3266, \qquad z_{\mathrm{src}} = 0.7497.
$$

<!-- check: ch12.z_lens = 0.3266 ± 0.0001 -->
<!-- check: ch12.z_src = 0.7497 ± 0.0001 -->

Feed the lens redshift into $\eqref{eq:zdef}$ with the Ca II K absorption line
(rest wavelength 3933.66 Å — its depth traces old stars, so it shows up in a
quiescent lens galaxy, not a star-forming background source):

$$
\lambda_{\mathrm{obs}} = 3933.66\ \text{Å} \times (1+0.3266) = 5218.47\ \text{Å}.
$$

<!-- check: ch12.lambda_obs_lens_caK = 5218.47 ± 0.01 -->

Do the same for the source with the [OII] doublet (rest wavelength 3727 Å, an
emission feature of ionized gas — present in a star-forming galaxy, essentially
absent in an old elliptical) at $z_{\mathrm{src}} = 0.7497$:

$$
\lambda_{\mathrm{obs}} = 3727.42\ \text{Å} \times (1+0.7497) = 6521.90\ \text{Å}.
$$

<!-- check: ch12.lambda_obs_src_oii = 6521.90 ± 0.01 -->

Both land inside an optical spectrograph's range with room to spare, which is
why one exposure of one patch of sky yields both redshifts, from two entirely
different atomic transitions. Try the line most people reach for first,
Balmer-alpha (rest 6564.61 Å), on the *source* instead:

$$
\lambda_{\mathrm{obs}} = 6564.61\ \text{Å} \times 1.7497 = 11486.15\ \text{Å},
$$

<!-- check: ch12.lambda_rest_halpha = 6564.61 ± 0.01 -->
<!-- check: ch12.lambda_obs_src_halpha_if_used = 11486.15 ± 0.01 -->

deep in the near-infrared, off the red end of a ground-based optical
spectrograph entirely. That is precisely why a redshift pipeline searches a
whole template library rather than a hard-coded line list: which lines are
even observable depends on $z$ itself.

**The ratio that starts a discovery pipeline.** Divide the two redshifts:

$$
\frac{z_{\mathrm{src}}}{z_{\mathrm{lens}}} = \frac{0.7497}{0.3266} = 2.2954.
$$

<!-- check: ch12.z_ratio_desi004 = 2.2954 ± 0.001 -->

That ratio, and the threshold it clears, are not incidental to this example —
they are the literal discovery criterion of Hsu et al. (2025)'s pairwise
spectroscopic search
(`reproductions/hsu-2025/05_run_full_fof.py:40`,
`reproductions/hsu-2025/05_run_full_fof.py:115`). Group every DESI fiber
spectrum by sky position (a friends-of-friends match at 3″), then keep any
group whose maximum-to-minimum redshift ratio is at least
$1.3$<!-- check: ch12.z_ratio_threshold = 1.3 ± 0.0 -->. Two spectra that
close together on the sky but that far apart in redshift are not one blended
object — they are a foreground galaxy and a background galaxy along one line
of sight, the geometric definition of a lens candidate. Run on the full DR1
catalog, the cut takes 28,425,963<!-- check: ch12.n_raw_fibers = 28425963 ± 0 -->
raw fiber spectra down to 15,786,243<!-- check: ch12.n_after_prefilter = 15786243 ± 0 -->
after basic quality cuts, and finally to
13,530<!-- check: ch12.n_groups_after_ratio_cut = 13530 ± 0 --> candidate
groups (27,334<!-- check: ch12.n_spectra_after_ratio_cut = 27334 ± 0 -->
spectra) — a purely spectroscopic discovery channel that runs *before* any
imaging classifier ([Ch. 27](27-discovery.md#the-finders)) has seen a pixel.

Not every published "redshift" is this kind of measurement. A single-author
paper on NISP spectroscopy of Euclid Q1 lenses (arXiv:2604.02726) was
withdrawn at its sixth revision after it emerged that 385 of its 440 tabulated
"deflector redshifts" — 87.5%<!-- check: ch12.nisp_photz_fraction = 0.875 ± 0.001 -->
of them — were not spectroscopic at all but *photometric* redshifts:
estimates from broad-band colors, not from a resolved emission or absorption
line. The paper's own validation against real (blind) spectroscopy recovered
only about 35%<!-- check: ch12.nisp_blind_recovery_frac = 0.35 ± 0.01 --> of
them correctly (`reproductions/lensjudge/parity/FINDINGS.md:91`). Calling a
color fit a redshift measurement is exactly the substitution this section
warns against: a photometric estimate is a guess dressed in the units of a
measurement, and this one did not survive review.

## $\sigma_v$ from line width { #sigma-v-from-lines }

A redshift measures where the *center* of a line sits: the bulk line-of-sight
velocity of the whole galaxy. A velocity dispersion measures how *wide* the
line is, and it comes from a different effect entirely: the stars inside the
galaxy are not all moving at the galaxy's bulk velocity. Some orbit toward
you, some away, with a spread of line-of-sight velocities $\sigma_v$ around
the mean. Each star's own light picks up its own small Doppler shift; sum the
light of the whole population and a symmetric spread of shifts smears the
line rather than displacing it.

For $v \ll c$ — every $\sigma_v$ in this repo is a few hundred km/s, three
orders of magnitude below $c$ — the Doppler shift linearizes to

$$
\frac{\Delta\lambda}{\lambda} = \frac{v}{c},
\label{eq:doppler}
$$

the same small-parameter linearization as Ch. 2's Taylor argument
([Ch. 2](02-derivatives.md#taylor)), applied here to a relativistic formula
instead of a lens equation. A line's intrinsic profile, convolved with a
Gaussian kernel of standard deviation

$$
\sigma_\lambda = \lambda_{\mathrm{rest}}\,\frac{\sigma_v}{c}
$$

from $\eqref{eq:doppler}$, and further convolved with the spectrograph's own
instrumental resolution, is what a pipeline actually fits. Not read off a
plot, for the same reason a redshift isn't: recovering a width against a
comparable instrumental width needs a template and a model, not a ruler.

**The scale of the effect, and why it needs a fit.** Take the median velocity dispersion FastSpecFit recovers for the lens galaxies
in Hsu et al. (2025)'s sample:
$\sigma_v = 217.09\ \text{km/s}$<!-- check: ch12.sigma_v_median_kms = 217.09 ± 0.01 -->
(the middle 68% of the measured population spans roughly 152.12–292.27 km/s
<!-- check: ch12.sigma_v_p16_kms = 152.12 ± 0.01 -->
<!-- check: ch12.sigma_v_p84_kms = 292.27 ± 0.01 -->). Relative to $c$,

$$
\frac{\sigma_v}{c} = \frac{217.09}{299792.458} = 7.24\times10^{-4},
$$

<!-- check: ch12.sigma_v_over_c_median = 7.241e-4 ± 1e-6 -->

and on the same Ca II K line used above it broadens the line by only

$$
\Delta\lambda = \lambda_{\mathrm{rest}}\,\frac{\sigma_v}{c}
= 3933.66\ \text{Å} \times 7.24\times10^{-4} = 2.848\ \text{Å}.
$$

<!-- check: ch12.delta_lambda_caK_at_median_sigma_v = 2.848 ± 0.001 -->

Compare that to the *shift* computed in the last section for the same line:
$5218.47 - 3933.66 = 1284.81$ Å, about 450 times larger. That ratio is the
whole reason a redshift is visible by eye on a plotted spectrum and a velocity
dispersion is not: the shift moves a line across a large fraction of the
visible band, while the width perturbs it by a few Ångstroms — comparable to
the spectrograph's own resolution element. Recovering $\sigma_v$ means fitting
the line's shape against an instrumental-resolution template, which is
exactly what a stellar-population code like FastSpecFit does.

**A fitter's own trap.** That fit does not always succeed, and how it fails is worth internalizing.
Only 4,238 of the 13,530 candidate pairs — 31.3%
<!-- check: ch12.frac_with_reliable_sigma_v = 0.3132 ± 0.0005 -->
<!-- check: ch12.n_with_reliable_sigma_v = 4238 ± 0 -->
— carry a $\sigma_v$ that FastSpecFit itself trusts
(`reproductions/hsu-2025/07_classify_einstein_dimple.py:145-153`): a positive
inverse variance, `VDISP_IVAR > 0`. The remaining
9,292<!-- check: ch12.n_without_reliable_sigma_v = 9292 ± 0 --> either have no
usable fit or return the fitter's own failure default of exactly
250.0 km/s<!-- check: ch12.failed_fit_cap_kms = 250.0 ± 0.0 --> with
`VDISP_IVAR = 0` — a number that looks completely plausible for a massive
galaxy and is, in fact, a placeholder the code emits when it gives up. There
is no way to tell a real 250 km/s from a fake one except by checking the
diagnostic that comes with it: exactly the discipline this guide's own
numbers are held to, applied here to someone else's fitting code.

**The payoff.** Feed a trustworthy $\sigma_v$ into the SIS Einstein-radius formula of
[Ch. 19](19-einstein-radius.md#theta-e-from-sigma-v) — the physical argument
for *why* that formula works comes from galaxy dynamics, derived in
[Ch. 10](10-galaxies.md#velocity-dispersion) — and a spectrum alone gives you
a mass estimate. For the 4,238 Hsu et al. pairs with a reliable $\sigma_v$,
the median predicted Einstein radius is
$\theta_{\mathrm{E}} = 0.6815''$<!-- check: ch12.theta_e_median_arcsec = 0.6815 ± 0.001 -->,
computed straight from catalog numbers, no image involved.

That is the payoff this chapter has been building toward. A spectroscopic
redshift tells you two objects sit at different distances — not a blend, not
a chance alignment, a genuine foreground-background pair. A spectroscopic
velocity dispersion turns that pair into a mass estimate. An image without
either number is a ring of light with an unknown cause: it could be a lens,
a tidal tail, a face-on spiral, or two unrelated galaxies at the same
redshift. That is the content of this chapter's destination sentence — a lens
without redshifts is a picture, not a mass — and it is why an entire
discovery channel (Hsu et al. 2025, and the DESI Foundry modeling papers
downstream of it) exists purely to mine a spectroscopic catalog that DESI
built for cosmology, not for lensing.

## Connect to the repo { #connect }

- `reproductions/hsu-2025/05_run_full_fof.py:40` sets `Z_RATIO_MIN = 1.3`; line
  `:115` computes `z_stats["z_ratio"] = z_stats["zmax"] / z_stats["zmin"]` and
  applies the cut. Run at full DR1 scale this is the 28.4M-fiber → 13,530-group
  funnel quoted above
  ([GitHub](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/hsu-2025/05_run_full_fof.py#L40)).
- `reproductions/hsu-2025/07_classify_einstein_dimple.py:103` implements
  `theta_e_sis`, the same formula Ch. 19 derives; lines `:145-153` are the
  comment documenting FastSpecFit's `VDISP = 250.0` failure-mode cap, and the
  `VDISP_IVAR > 0` guard that catches it
  ([GitHub](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/hsu-2025/07_classify_einstein_dimple.py#L103)).
- `reproductions/hsu-2025/data/classified_stats.json`, `dr1_stats.json`, and
  `xmatch_table2.json` carry every count and percentile quoted in this
  chapter's worked examples.
- `reproductions/lensjudge/tools/spectrum.py:32` (`sis_theta_e`) is an agentic
  grading tool computing this exact arithmetic live — a "does this pair make
  physical sense" check LensJudge runs during its own grading, not just a
  textbook formula
  ([GitHub](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/tools/spectrum.py#L32)).
- `reproductions/lensjudge/parity/FINDINGS.md:91` documents the withdrawn NISP
  paper discussed above
  ([GitHub](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/parity/FINDINGS.md#L91)).
- `reproductions/cikota-2023/README.md:118` is an honest converse example: that
  reproduction states plainly that it did **not** reproduce any spectroscopy,
  and uses the discovery paper's published redshifts only for a unit
  conversion, rather than quietly presenting them as its own measurement
  ([GitHub](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/cikota-2023/README.md#L118)).

## Exercises { #exercises }

??? question "Exercise 12.1 — The ratio, by hand"
    Using $z_{\mathrm{lens}} = 0.3266$ and $z_{\mathrm{src}} = 0.7497$, compute
    $z_{\mathrm{src}}/z_{\mathrm{lens}}$ and state whether this DESI pair would
    survive Hsu et al. (2025)'s friends-of-friends discovery cut.

    ??? success "Solution"
        $0.7497 / 0.3266 = 2.2954$<!-- check: ch12.z_ratio_desi004 = 2.2954 ± 0.001 -->,
        comfortably above the threshold of
        $1.3$<!-- check: ch12.z_ratio_threshold = 1.3 ± 0.0 -->. The pair survives —
        it is, in fact, one of the 13,530 groups the cut keeps at full DR1
        scale.

??? question "Exercise 12.2 — Why Balmer-alpha doesn't work here"
    Compute where Balmer-alpha (rest wavelength 6564.61 Å) would land if you
    searched for it in the *source* spectrum at $z_{\mathrm{src}} = 0.7497$.
    Then explain why a redshift pipeline does not simply fail when one
    expected line falls outside the observable band.

    ??? success "Solution"
        $6564.61 \times 1.7497 = 11486.15$ Å
        <!-- check: ch12.lambda_obs_src_halpha_if_used = 11486.15 ± 0.01 -->,
        deep in the near-infrared — well outside where a ground-based optical
        spectrograph is sensitive, compared to the 5218–6522 Å range where the
        Ca II K and [OII] lines in this chapter's worked example actually
        landed. The pipeline does not fail, because it never depended on any
        one line: it cross-correlates a whole template (Ch. 7's operation)
        against the data, and at any given $z$ some *other* line in the
        template — with a different rest wavelength — falls inside the band
        instead.

??? question "Exercise 12.3 — Shift vs. width, by the numbers"
    Using $\sigma_v = 217.09$ km/s on the Ca II K line, verify
    $\Delta\lambda_{\text{width}} \approx 2.85$ Å. Compare it to the *shift*
    computed for the same line, $5218.47 - 3933.66$ Å. What is the ratio, and
    what does it tell you about why a redshift is visible by eye but a
    velocity dispersion is not?

    ??? success "Solution"
        $\Delta\lambda_{\text{width}} = 3933.66 \times (217.09/299792.458) = 2.848$ Å
        <!-- check: ch12.delta_lambda_caK_at_median_sigma_v = 2.848 ± 0.001 -->.
        The shift is $5218.47 - 3933.66 = 1284.81$ Å. The ratio is about 450:
        the shift displaces a line across more than a thousand Ångstroms,
        while the width perturbs it by less than three — a scale comparable to
        the spectrograph's own resolution element. That is why $z$ is legible
        on a plotted spectrum and $\sigma_v$ requires fitting the line profile
        against an instrumental-resolution model.

??? question "Exercise 12.4 — The fitter's cap"
    9,292 of Hsu et al.'s 13,530 candidate pairs lack a trustworthy $\sigma_v$.
    What fraction of the *whole* sample is that? If an analysis used
    FastSpecFit's `VDISP` column without checking `VDISP_IVAR > 0`, why can't
    you simply look at the number 250.0 km/s and know it's a placeholder
    rather than a real measurement?

    ??? success "Solution"
        $9292 / 13530 = 0.687$, about 68.7%
        <!-- check: ch12.n_without_reliable_sigma_v = 9292 ± 0 -->
        <!-- check: ch12.n_groups_after_ratio_cut = 13530 ± 0 -->
        of the sample. You can't tell a real 250 km/s from the fitter's
        failure cap by inspecting the value alone — 250 km/s is a perfectly
        ordinary velocity dispersion for a massive elliptical
        (`reproductions/hsu-2025/07_classify_einstein_dimple.py:145-153`). The
        only way to distinguish them is the diagnostic that ships alongside
        the number, `VDISP_IVAR`: zero means the fit gave up and returned its
        template floor, not a measurement.
