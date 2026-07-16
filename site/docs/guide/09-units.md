# Arcseconds, magnitudes, and the units of the sky

Every number this guide computes from here on is quoted in one of four units:
an angle in arcsec, a brightness in magnitudes, a surface brightness in
magnitudes-per-square-arcsec (or counts per pixel), and a pixel scale in
arcsec/pixel. None of the physics in Part IV needs anything harder than
arithmetic once you have these pinned down — the hard part is that all four
are unfamiliar conventions wearing familiar-looking symbols. This chapter buys
you the ability to read a figure caption, a FITS header, or a line of this
repo's code (`pixscale=0.262`, `m_z < 20`, `delta_pix=0.04`) and know exactly
what physical statement it is making. Part III (cosmology) adds one more unit,
the megaparsec, once distances stop being purely angular.

!!! abstract "What you can skip"
    You already know what a logarithmic unit is — decibels, pH, the bits and
    nats you already use for cross-entropy. Skip any explanation of "why take
    a log of a wide-dynamic-range quantity." What is new here is not the
    concept but the specific, historically-frozen conventions: which base,
    which sign, which zero point, and why gravitational lensing cares about
    the difference between a flux and a surface brightness. If you are
    comfortable with the small-angle approximation from
    [Ch. 2](02-derivatives.md#taylor) (that $\sin\theta \approx \theta$ to
    first order), skip straight to the parsec derivation.

## Angles on the sky { #angles-on-the-sky }

Nothing astronomical comes with a ruler. Every telescope measures a direction,
not a distance, so the natural unit for anything on the sky is an angle:
degrees, and for anything as small as a galaxy or a lensed arc, the much finer
**arcsecond** ($1^\circ = 60' = 3600''$).

The reason arcseconds show up multiplied by $206265$ everywhere is nothing
more than the Taylor expansion you already derived in
[Ch. 2](02-derivatives.md#taylor). One radian is, by definition, the angle
whose arc length equals its radius, so converting radians to arcseconds is
a matter of unit bookkeeping:

$$
1\ \text{rad} = \frac{180}{\pi} \times 3600'' \approx 206264.8''.
$$

<!-- check: ch09.arcsec_per_rad = 206264.806 ± 0.001 -->

This exact constant lives in the repo's own numerics as `ARCSEC_PER_RAD` —
see `site/guide_src/lensing.py:25`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lensing.py#L25)) —
because every deflection angle, Einstein radius, and image position in this
repo's lens equation is carried in arcsec, never radians, and the conversion
factor has to come from somewhere concrete.

The reason $206265$ is worth deriving rather than memorizing: it is also a
license to stop worrying about the small-angle approximation everywhere else
in this guide. At $\theta = 1''$ (already large by lensing standards — most
Einstein radii are $1$–$2''$), the gap between $\sin\theta$ and $\theta$
itself is

$$
\frac{|\sin\theta - \theta|}{\theta} \approx \frac{\theta^2}{6} \approx 3.9\times10^{-12},
$$

<!-- check: ch09.small_angle_rel_error_1as = 3.917e-12 ± 0.01e-12 -->

twelve orders of magnitude smaller than the angle itself. This is why the
lens equation in [Ch. 17](17-lens-equation.md#the-lens-equation),
$\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta)$,
is written as ordinary vector subtraction in a flat 2-D plane with no
trigonometry in sight: at arcsecond scales, the curved sky and the flat
tangent plane are the same surface to twelve decimal places.

The same approximation pins down the **parsec**, the distance unit every
paper you will ever read (including this repo's own cosmology chapters, once
you convert into physical Mpc) actually uses. A parsec is *defined* as the
distance at which one astronomical unit — Earth's orbital radius, roughly
$1.496\times10^8$ km — subtends an angle of exactly $1''$. Since $1''$ is
minuscule, $\tan\theta \approx \theta$ applies immediately and the definition
becomes pure division:

$$
1\,\text{pc} = \frac{1\,\text{AU}}{\theta(1'')} = \frac{1.496\times10^8\,\text{km}}{4.848\times10^{-6}} \approx 3.086\times10^{13}\,\text{km} \approx 3.26\,\text{light-years}.
$$

<!-- check: ch09.pc_from_small_angle_km = 3.0856775814913664e13 ± 1e6 -->
<!-- check: ch09.pc_in_ly = 3.2616 ± 0.001 -->

Deriving it this way — from one triangle and the same first-order Taylor term
already in hand — reproduces astropy's own built-in parsec constant exactly,
to the last bit of double-precision floating point.

<!-- check: ch09.pc_definition_match_km = 0.0 ± 1e-6 -->

Notice what this repo does *not* do with parsecs: the lens-modeling
likelihood at the center of Parts IV–V never leaves arcsec (see
`site/guide_src/cosmo.py:1`–`20`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/cosmo.py#L1))
— cosmology enters in exactly three places, and the fitting itself is
"arcsec in, arcsec out"). Mpc reappears only once you convert an Einstein
radius into a physical mass, which is [Ch. 15](15-distances.md#the-carousel)'s
job. You need the parsec to read an abstract, not to run this repo's own
fits.

## Magnitudes { #magnitudes }

!!! tip "You already know this"
    A magnitude is a log-likelihood with the sign and the base changed. Sky
    surveys record fluxes spanning ten or more orders of magnitude in a
    single image — a nearby star and a faint background galaxy on the same
    CCD — exactly the dynamic-range problem that makes you compute a
    log-likelihood instead of a likelihood. Astronomy solved the same problem
    a century earlier, kept the sign backwards for historical reasons, and
    called the result a magnitude.

Flux is what a detector actually integrates: energy per unit time per unit
collecting area. The **magnitude** system converts flux into a logarithmic,
and famously *backwards*, scale: brighter objects get *smaller* (even
negative) magnitude numbers. The definition, for two fluxes $F_1, F_2$:

$$
m_1 - m_2 = -2.5\log_{10}\!\left(\frac{F_1}{F_2}\right).
$$

The $-2.5$ is Norman Pogson's 1856 fix to make the scale consistent with
Hipparchus's ancient by-eye ranking of stars from 1 (brightest) to 6 (faintest):
Pogson set five magnitudes to mean exactly a factor of 100 in flux, which is
where $2.5 = 5/\log_{10}(100)$ comes from. The minus sign is the price of
keeping "1st magnitude" attached to the brightest stars.

Modern digital surveys drop the "compare to a reference star" step and fix an
absolute zero point instead. This repo's own imaging — the DESI Legacy
Survey cutouts fetched throughout `reproductions/aion-1/` — uses the
"nanomaggie" flux unit with zero point $22.5$, so a single flux measurement
converts to a magnitude with no reference object at all:

$$
m = 22.5 - 2.5\log_{10}(f_{\text{nmgy}}).
$$

This is exactly the formula at
`reproductions/aion-1/03_fetch_provabgs.py:65`–`67`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/aion-1/03_fetch_provabgs.py#L65)):
`flux = 10.0 ** ((22.5 - mags) / 2.5)`. You can invert this yourself on a
number this repo actually uses. The DR11-south discovery sweep's own
faint-end quality cut is $m_z < 20$ in the $z$-band
(`reproductions/dr11-campaign/papers/main.tex:94`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/dr11-campaign/papers/main.tex#L94))).
Plugging $m = 20$ into the zero-point formula:

$$
f = 10^{(22.5-20)/2.5} = 10^{1} = 10\ \text{nanomaggies exactly}.
$$

<!-- check: ch09.flux_nmgy_at_mz20 = 10.0 ± 1e-9 -->

That the cut lands on a clean power of ten is not a coincidence you need to
explain — it is Pogson's $2.5$ doing exactly what it was built to do: two
magnitudes of difference from the zero point is one clean order of magnitude
in flux. (For the same reason, a source at $100$ nanomaggies sits at
$m = 22.5 - 2.5\log_{10}(100) = 17.5$.)

<!-- check: ch09.mag_at_f100nmgy = 17.5 ± 1e-9 -->

## Surface brightness { #surface-brightness }

Flux is a *total*: integrate a light distribution over the solid angle an
object occupies and you get a flux. **Surface brightness** $I$ is the
integrand — flux per unit solid angle, e.g. magnitudes per square arcsec, or
(as this repo's rendering code actually stores it) linear counts per pixel.
The distinction matters here for one reason: gravitational lensing conserves
surface brightness and only ever moves flux around by changing solid angle.

This is a restatement of [Ch. 18](18-magnification.md#magnification-is-a-jacobian)'s
magnification in a new unit. Lensing bends light rays; it does not create or
destroy photons, so the specific intensity carried along any single ray is
unchanged: $I(\boldsymbol\beta) = I(\boldsymbol\theta)$, point for point, source
to image. What magnification $\mu = 1/\det A$ changes is how much *solid
angle* on the sky that unchanged intensity ends up covering — a lensed image
looks brighter in total flux only because it is bigger, never because each
patch of it got more photons per unit area than its unlensed source patch
had. The repo's forward model renders every source (the Sersic profiles of
[Ch. 10](10-galaxies.md#the-sersic-profile), `site/guide_src/lensing.py:162`–`164`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lensing.py#L162)))
as exactly this kind of surface-brightness field, ray-traces it through the
lens equation, and only afterward multiplies by pixel area to get the counts
a detector would actually record.

That multiplication is worth naming explicitly, because it is another
det-J moment for the [Log-Det Ledger](04-multivariable.md#the-log-det-ledger):
converting a continuous surface brightness (flux per solid angle) into flux
per discrete pixel is a change of variables, and its Jacobian determinant is
the pixel's own area. At the campaign's finest pixel scale,
$\texttt{delta\_pix} = 0.04''$, that area is

$$
\det(\texttt{delta\_pix} \cdot I) = (0.04'')^2 = 0.0016\ \text{arcsec}^2,
$$

<!-- check: ch09.pixel_area_fine_arcsec2 = 0.0016 ± 1e-9 -->

which is the literal line `conversion_factor = float(cfg["delta_pix"]) ** 2  #
det(delta_pix*I)` at `reproductions/claude-giga-lens/cgl/e2.py:455`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L455)).
The comment in the repo's own source names it a determinant because it is
one — the same idea Ch. 4 introduces for magnification and Ch. 23 closes out
for the flow and Occam log-dets.

The repo also tests the conservation directly rather than trusting it: the
regression `test_mock_pipeline_matches_exact_average_and_units` in
`reproductions/claude-giga-lens/tests/test_drizzle.py:198`–`201`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/tests/test_drizzle.py#L198))
feeds the drizzle resampling operator a spatially uniform mock scene and
asserts the output stays uniform at the same value — a resampling step that
is not allowed to manufacture or destroy surface brightness, checked the way
you would check that a change-of-variables in a normalizing flow leaves total
probability mass at 1.

## Pixel scale { #pixel-scale }

**Pixel scale** is how many arcseconds one detector pixel spans — a fixed
property of the telescope's optics, set once at design time. It is a
different number from **seeing**, the width of the point-spread function
(PSF) imposed by the atmosphere (or, for a space telescope, by diffraction),
which varies exposure to exposure. Confusing the two is the single most
common unit mistake in reading a lensing figure.

!!! tip "You already know this"
    The relationship between pixel scale and seeing is the Nyquist–Shannon
    sampling theorem, applied to a spatial signal instead of a time series.
    The PSF is the highest-frequency structure in the image; to resolve it
    without aliasing you need at least two samples (pixels) across its width.
    Fewer, and you are undersampled — real structure gets folded onto itself.
    Many more, and you are oversampled: extra pixels that do not carry extra
    independent information, only extra correlated noise. Ch. 11 turns that
    exact fact into the reason a naive diagonal likelihood is wrong on
    resampled data.

The DESI Legacy Survey — the discovery half of this repo's science — has a
native pixel scale of $0.262''$/pixel (the `pixscale=0.262` default at
`reproductions/aion-1/_ls_cutout.py:36`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/aion-1/_ls_cutout.py#L36))).
Its own $g$-band coadd PSF, measured directly rather than assumed, has FWHM
$\approx 1.35''$ (`reproductions/cikota-2023/papers/main.tex:270`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/cikota-2023/papers/main.tex#L270))).
That is

$$
\frac{1.35''}{0.262''/\text{px}} \approx 5.15\ \text{pixels across the FWHM},
$$

<!-- check: ch09.fwhm_in_native_px = 5.153 ± 0.01 -->

comfortably above the Nyquist floor of two: the survey is not undersampled by
its own pixel grid. But $5.15$ pixels of blur is not the same statement as
"$5.15$ pixels of resolution" — the seeing disk itself, $1.35''$, is
comparable to or larger than a typical Einstein radius. Sampling the blur
finely does not un-blur it: `reproductions/cikota-2023`'s own PSF ablation
shows that swapping in the source paper's sharper $0.6''$ PSF only closes
about half the Einstein-radius discrepancy, because "the $1.35''$ Legacy
seeing is baked into the pixels and cannot be deconvolved beyond what the
survey resolved" (`reproductions/cikota-2023/README.md:41`–`43`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/cikota-2023/README.md#L41))).
[Ch. 27](27-discovery.md#deriving-the-wall) turns this exact seeing-vs-Einstein-radius
comparison into the resolution wall that bounds automated lens discovery, and
[Ch. 28](28-the-label.md#same-wall) shows it is the same wall that makes
human labels noisy.

When this repo fits a lens model, it resamples the survey's own pixels onto
finer synthetic grids so the model has room to render sub-pixel structure.
The P1c money-number chain ([Ch. 25](25-money-number.md#the-chain)) uses
three such products, defined at `reproductions/claude-giga-lens/cgl/e2.py:55`–`59`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L55)):

| product | pixel scale | grid | oversampling vs. the $0.262''$ survey pixel |
|---|---|---|---|
| fine   | $0.04''$ | $260^2$ | $6.55\times$ |
| binned | $0.08''$ | $130^2$ | $3.28\times$ |
| native | $0.13''$ | $80^2$  | $2.02\times$ |

<!-- check: ch09.oversample_fine_vs_survey = 6.55 ± 0.01 -->
<!-- check: ch09.oversample_binned_vs_survey = 3.275 ± 0.01 -->
<!-- check: ch09.oversample_native_vs_survey = 2.015 ± 0.01 -->

Finer pixels here are not finer *information* — they are the same photons,
interpolated onto a denser grid, which is precisely why the fine product
turns out to be the most internally correlated of the three. That
consequence, quantified, is [Ch. 11](11-observation.md#drizzle)'s job; the
point to take from this chapter is narrower: pixel scale is a unit
conversion (arcsec per pixel), seeing is a physical measurement (arcsec
FWHM), and no amount of resampling moves information between them.

## Connect to the repo { #connect }

- `site/guide_src/lensing.py:25`
  ([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lensing.py#L25)) —
  `ARCSEC_PER_RAD`, the constant this chapter derives from a Taylor expansion.
- `site/guide_src/cosmo.py:1`–`20`
  ([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/cosmo.py#L1)) —
  why this repo's lens-modeling likelihood never uses a parsec or an Mpc.
- `reproductions/aion-1/03_fetch_provabgs.py:65`–`67`
  ([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/aion-1/03_fetch_provabgs.py#L65)) —
  the Pogson magnitude-to-flux conversion this repo actually runs.
- `reproductions/dr11-campaign/papers/main.tex:94`
  ([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/dr11-campaign/papers/main.tex#L94)) —
  the $m_z < 20$ cut used to build the 53.8M-galaxy discovery parent sample.
- `reproductions/claude-giga-lens/tests/test_drizzle.py:198`–`201`
  ([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/tests/test_drizzle.py#L198)) —
  the regression test that pins surface-brightness conservation under resampling.
- `reproductions/claude-giga-lens/cgl/e2.py:55`–`59` and `:455`
  ([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L55)) —
  the three drizzle pixel scales behind the money number, and the pixel-area
  Jacobian that converts surface brightness into counts.
- `reproductions/cikota-2023/papers/main.tex:270` and `README.md:38`–`43`
  ([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/cikota-2023/papers/main.tex#L270)) —
  the measured $1.35''$ Legacy seeing and the PSF ablation showing it cannot
  be deconvolved away.
- Every number above reproduces with
  `~/.venvs/lensjudge/bin/python site/guide_src/worked_examples.py --show ch09`.

## Exercises { #exercises }

??? question "Exercise 9.1 — the parsec, from one triangle"
    Derive $1\,\text{pc}$ in kilometers using only $1\,\text{AU} \approx
    1.496\times10^8\,\text{km}$ and the small-angle approximation — no other
    astronomical constant allowed. Then explain in one sentence why the
    answer does not depend on which galaxy, star, or lens you are observing.

    ??? success "Solution"
        By definition, a parsec is the distance $d$ at which $1\,\text{AU}$
        subtends $1''$. Geometrically, $\tan(1'') = \text{AU}/d$; because
        $1''$ in radians is $\theta = 1/206264.8 \approx 4.848\times10^{-6}$,
        which is tiny, $\tan\theta \approx \theta$ to twelve decimal places
        (Ch. 2's Taylor expansion, quantified in this chapter). So

        $$
        d = \frac{\text{AU}}{\theta} = \frac{1.496\times10^8\,\text{km}}{4.848\times10^{-6}} \approx 3.086\times10^{13}\,\text{km},
        $$

        <!-- check: ch09.pc_from_small_angle_km = 3.0856775814913664e13 ± 1e6 -->
        which matches astropy's own `u.pc` exactly.
        <!-- check: ch09.pc_definition_match_km = 0.0 ± 1e-6 -->
        It is purely a statement about the geometry of one right triangle (an
        AU and a $1''$ angle) — nothing about the object being observed
        enters, which is exactly why it is usable as a universal distance
        unit rather than an object-specific one.

??? question "Exercise 9.2 — inverting Pogson's law"
    The DESI Legacy Survey's zero point is $22.5$ (flux in nanomaggies). What
    magnitude corresponds to $100$ nanomaggies? What flux corresponds to the
    DR11-south sweep's own faint cut, $m_z < 20$? Why does the second answer
    come out to a clean number without any special pleading?

    ??? success "Solution"
        Inverting $m = 22.5 - 2.5\log_{10}f$ for $f = 100$:
        $m = 22.5 - 2.5\log_{10}(100) = 22.5 - 5 = 17.5$.
        <!-- check: ch09.mag_at_f100nmgy = 17.5 ± 1e-9 -->
        For $m = 20$: $f = 10^{(22.5-20)/2.5} = 10^{1} = 10$ nanomaggies
        exactly.
        <!-- check: ch09.flux_nmgy_at_mz20 = 10.0 ± 1e-9 -->
        The cleanness is not a coincidence to explain away — Pogson fixed
        $2.5$ so that every $2.5$ magnitudes is exactly one power of ten in
        flux, by construction. Any cut that lands a whole multiple of $2.5$
        magnitudes from the zero point will always be a round number in
        flux.

??? question "Exercise 9.3 — is the survey oversampled or under-resolved?"
    DESI Legacy's native pixel scale is $0.262''$ and its measured $g$-band
    seeing is $1.35''$ FWHM. Compute pixels-per-FWHM and say whether the
    survey is Nyquist-sampled. Then explain why that answer alone does not
    tell you whether a $1.2''$-Einstein-radius lens will be *resolved*.

    ??? success "Solution"
        Pixels across the FWHM: $1.35/0.262 \approx 5.15$.
        <!-- check: ch09.fwhm_in_native_px = 5.153 ± 0.01 -->
        Nyquist only requires $\geq 2$ samples across the highest-frequency
        feature, so the survey's own pixel grid comfortably resolves its own
        PSF — it is not pixel-undersampled. But Nyquist sampling of the PSF
        is a statement about not losing information *the PSF already
        contains*; it says nothing about whether the PSF itself is narrow
        enough to separate a $1.2''$ ring from its $1.35''$-wide blur disk.
        Those are two different comparisons — pixel scale vs. PSF width, and
        PSF width vs. Einstein radius — and only the second one is the wall
        [Ch. 27](27-discovery.md#deriving-the-wall) derives.

??? question "Exercise 9.4 — the pixel-area Jacobian"
    The campaign's "fine" drizzle product has `delta_pix = 0.04`. Compute the
    pixel area in square arcsec, and explain in one sentence why
    `reproductions/claude-giga-lens/cgl/e2.py:455` labels this quantity a
    determinant rather than only an area.

    ??? success "Solution"
        Area $= (0.04'')^2 = 0.0016\ \text{arcsec}^2$.
        <!-- check: ch09.pixel_area_fine_arcsec2 = 0.0016 ± 1e-9 -->
        Converting a surface brightness (flux per unit solid angle) into a
        per-pixel flux is a change of variables from continuous arcsec$^2$ to
        discrete pixels, and the scaling factor of any change of variables is
        the Jacobian determinant of the map between them — here
        $\text{delta\_pix} \cdot I$ acting on a 2-D patch, whose determinant
        is $\text{delta\_pix}^2$. It is the same object as the magnification
        $1/\det A$ in [Ch. 18](18-magnification.md#magnification-is-a-jacobian)
        and the flow log-det in [Ch. 23](23-samplers.md#closing-the-log-det-ledger) —
        an area-scaling factor, computed the same way every time.
