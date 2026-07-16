# 6. Telescopes, surveys, and the three observatories

Three instruments carry this program's entire evidence base. The DESI Legacy
Imaging Surveys find the lens candidates; the *Hubble Space Telescope* takes the
sharp pictures the money number is fitted to; *Euclid* is the independent check.
The main guide names all three across nine chapters and introduces none of them.
This chapter is that introduction, and it reduces to two numbers per instrument:
how finely it samples the sky, and how badly the light was blurred before it
arrived. The second number decides things, and for the survey that finds the
lenses it is bad enough to put a floor under what can be found at all.

!!! abstract "What you can skip"
    You already own the sampling theory: pixel scale versus blur width is
    Nyquist–Shannon on a spatial signal, and the main guide says so rather than
    making you infer it. You do not need optical design, detector electronics or
    the CCD noise model — [Ch. 11](../guide/11-observation.md#the-noise-model)
    builds that from scratch. What is new is narrow: which physical effect limits
    which instrument, and what DESI, HST and Euclid refer to.

## What a telescope collects { #what-a-telescope-collects }

A telescope does two separate jobs, and keeping them apart matters, because they
scale differently.

The first is **collecting**. Light arrives thinned out by the inverse-square law
([Ch. 5](05-light.md#the-inverse-square-law)), so how faint an object you can
detect is set by how many photons you gather — the mirror's *area*, growing as
the square of its diameter. Here a telescope really is a bucket in the rain: a
wider bucket catches proportionally more. The second is **resolving**: telling
apart two directions on the sky that are nearly the same. This is where the
bucket analogy breaks completely — a wider bucket tells you nothing more about
*where* the rain fell; a wider mirror does, for reasons unrelated to how many
photons it caught. Both jobs improve with diameter, through unrelated physics,
and only one of them survives the atmosphere.

Angles on the sky are measured in **arcseconds**: one arcsecond is $1/3600$ of a
degree. Every number in this chapter is in arcseconds; the small-angle
arithmetic converting one into a radian and then a physical size is
[Ch. 9's](../guide/09-units.md#angles-on-the-sky) job.

Two arcsecond quantities describe any camera, and they mean different things.
**Pixel scale**, in arcsec per pixel, is how much sky one detector pixel covers:
a unit conversion, fixed at design time, exact. **Point-spread function** width,
quoted as FWHM in arcsec, is how wide a true point of light lands after
everything between the source and the silicon has smeared it: a *measurement*,
and it changes. Confusing the two, the main guide says, is "the single most
common unit mistake in reading a lensing figure"
([Ch. 9](../guide/09-units.md#pixel-scale)). It then gives both numbers for DESI
and never says which instruments sit where.

## Ground versus space { #ground-versus-space }

There are two ways to blur a point of light, and which one dominates is the
entire difference between the three observatories.

**Diffraction** is unavoidable. Light is a wave
([Ch. 5](05-light.md#the-spectrum)), and a wave passing through a finite
aperture spreads. The angular scale of the spread goes as $\lambda/D$ —
wavelength over aperture diameter — so bigger mirrors are genuinely sharper, and
redder light blurrier than bluer through the same optics. For HST's $2.4$ m
mirror at optical and near-infrared wavelengths this lands around a tenth of an
arcsecond. No engineering removes it.

**Seeing** is the atmosphere. Pockets of air at slightly different temperatures
have slightly different refractive indices, so each acts as a weak, wandering
lens, and over an exposure of any real length a point source is smeared into a
disk roughly an arcsecond across at a good site. Here is the fact that is not
obvious until someone states it: **seeing does not care how big your telescope
is.** An $8$ m and a $4$ m telescope on the same mountain on the same night have
the *same* resolution. The larger sees fainter — twice the diameter, four times
the photons — but not sharper. Aperture buys depth on the ground, not sharpness.
That is why a $2.4$ m telescope in orbit out-resolves every seeing-limited
telescope ever built, however large its mirror. It is not a better mirror. It is
above the air.

The main guide compresses all of this into one parenthesis, defining seeing as
"the width of the point-spread function (PSF) imposed by the atmosphere (or, for
a space telescope, by diffraction)"
([Ch. 9](../guide/09-units.md#pixel-scale)). Five bracketed words carry the
whole HST-versus-DESI split, and if you did not already know that ground
telescopes are seeing-limited and space telescopes diffraction-limited, the
parenthesis tells you nothing.

DESI's imaging is taken through about $1.3''$ of atmosphere:

<!-- check: pch06.desi_seeing_arcsec = 1.3 ± 0.001 -->

$$
\mathrm{FWHM}_{\mathrm{DESI}} \approx 1.3''.
$$

Treat that as a round number, not a constant. The main guide quotes the DESI
Legacy Survey's *measured* $g$-band coadd PSF at $1.35''$, careful to say it was
measured rather than assumed. The gap between the two figures is not sloppiness;
it is the point. Seeing varies by site, by night, by band, and by how many
exposures got stacked. Diffraction does not.

!!! note "Where this simplification stops"
    Adaptive optics — sensing the distortion in real time and bending a small
    mirror to cancel it — genuinely works, and modern ground-based systems reach
    space-like sharpness with it. It is not a loophole here: it works best in
    the infrared, over a field far too small to survey the sky with, and needs a
    bright reference star nearby. Nothing in this program's discovery half uses
    it. "Ground means blurry" is true of *wide-field survey imaging* — the only
    ground-based imaging this book cares about — not of ground-based astronomy
    in general.

## Bands and filters { #bands-and-filters }

An imaging camera does not record a spectrum. It records one number per pixel:
total photons, summed over whatever wavelengths the filter in front of it lets
through. That filter's transmission window is a **band**, and imaging in several
bands gives a handful of coarse samples of each object's spectrum instead of the
thousands a spectrograph would give.

The DESI Legacy Surveys use three optical bands, named $g$, $r$ and $z$ — "hence
*grz*", as [Ch. 27](../guide/27-discovery.md#the-survey) puts it in passing.
Roughly: $g$ is blue-green, near $480$ nm; $r$ is red, near $640$ nm; $z$ is the
far red running into the near-infrared, near $900$ nm — past the red edge of
what a human eye responds to ([Ch. 5](05-light.md#the-spectrum)). Every finder
in this program's lineage consumes a three-channel cutout of exactly these
bands, stacked like the channels of an RGB image and emphatically not one.

!!! tip "You already know this"
    A band is a lossy encoder: broadband imaging projects a spectrum of
    thousands of resolvable samples onto three basis functions — the filter
    transmission curves — and keeps the coefficients. Everything downstream is
    inference from a three-dimensional feature vector. A *photometric* redshift
    is regression from those three numbers; a *spectroscopic* redshift
    ([Ch. 7](07-spectra.md#continuum-and-lines)) reads a line position off the
    uncompressed signal. The accuracy gap is a compression-ratio story.

Which band an object is brightest in is therefore a crude thermometer: an old,
red stellar population ([Ch. 2](02-stars.md#why-massive-means-short)) puts more
of its light in $z$ than in $g$, and the difference of two bands' magnitudes —
a **colour** — is astronomy's cheapest classifier.

Now the collision, which this corpus never flags.

**The $z$-band is a filter. Redshift $z$ is a dimensionless ratio.** They are
different things, spelled the same, and both appear constantly in the main
guide — sometimes on the same page. [Ch. 9](../guide/09-units.md#magnitudes),
mid-derivation of the magnitude system, quotes the discovery sweep's quality cut
as "$m_z < 20$ in the $z$-band"; that $z$ is a piece of glass.
[Ch. 12](../guide/12-spectroscopy.md#measuring-redshift) and
[Ch. 13](../guide/13-expansion.md#redshift-is-expansion) write $z$ for redshift
on nearly every line. The rule that separates them is mechanical: a $z$ attached
to a brightness ($m_z$, "the $z$-band flux") is the filter; a $z$ standing
alone, or subscripted by an object ($z_l$ for the lens, $z_s$ for the source),
is redshift.

The collision is not quite a coincidence, which makes it stickier. Stretching
every wavelength by $1+z$ ([Ch. 8](08-redshift.md#what-redshift-measures)) drags
a distant galaxy's light out of the blue bands into the red ones, so a catalog
of distant red galaxies — what lens hunters want — needs a filter out at $900$
nm to catch them. The band is named $z$ and it selects on $z$.

## What a survey is { #what-a-survey-is }

Classical observing is *pointed*: you have a target, you win telescope time, you
observe it. A **survey** inverts that. You image a defined footprint to a defined
depth in defined bands, with no target in mind, reduce it once with one pipeline,
and publish catalogs that other people mine for science years later. The DESI
Legacy Surveys' data releases, DR7 through DR11 across this program's finder
lineage, are exactly this: public, uniform, pullable by anyone with a URL.

Surveys come in two kinds, and this program uses both. **Imaging surveys**
photograph everything in the footprint: every object gets a picture and a few
band magnitudes — cheap per object, weak per object. **Spectroscopic surveys**
put a fiber optic on a pre-chosen target and disperse its light into a full
spectrum: thousands of wavelength samples, one object at a time, and you must
know where to aim *before* you start. So the imaging survey comes first, and its
catalog tells the spectroscopic survey where to aim — which sets up the name
collision below.

Why image the whole sky? Because a strong lens is an accident of alignment
([Ch. 16](16-what-is-a-strong-lens.md#why-it-is-rare)), and you cannot point a
telescope at an accident you have not found yet. The only way to find something
that rare is to photograph everything and filter afterwards — which is why
[Ch. 27](../guide/27-discovery.md#the-survey) opens with tens of millions of
galaxies and warns that "a survey-scale finder is not a bigger version of a
Kaggle classifier: the operating point has to be chosen against tens of millions
of negatives, not a held-out test split."

## The three observatories { #the-three-observatories }

**The DESI Legacy Imaging Surveys.** Ground-based, imaging, public. Three bands
($grz$), about $1.3''$ of seeing, and a pixel scale of

<!-- check: pch06.desi_arcsec_per_px = 0.262 ± 0.0001 -->

$$
0.262''/\mathrm{px}.
$$

This is the discovery half.

The name is a trap the main guide walks into without comment. **DESI proper is a
spectrograph** — a $5000$-fiber instrument on a $4$ m telescope, built to
measure redshifts. **The DESI Legacy Imaging Surveys are imaging**, taken with
*different telescopes* (DECam in the south, 90Prime and Mosaic-3 in the north),
to choose which objects DESI's fibers would later point at. When
[Ch. 27](../guide/27-discovery.md#the-survey) says the surveys "photographed
most of the extragalactic sky in three colors," that is the cameras; when
[Ch. 12](../guide/12-spectroscopy.md#measuring-redshift) says "two DESI fiber
spectra, separated by under an arcsecond," that is the spectrograph.

**The Hubble Space Telescope.** A $2.4$ m mirror in low Earth orbit since 1990.
Diffraction-limited, so about a tenth of an arcsecond, and its near-infrared
detector's own pixel is

<!-- check: pch06.hst_arcsec_per_px = 0.128 ± 0.0001 -->

$$
0.128''/\mathrm{px}.
$$

Compare those two numbers. The PSF is roughly *one pixel* wide — and Nyquist
asks for at least two samples across it. HST's detector undersamples HST's own
optics: a single exposure discards resolution the mirror already delivered. This
is a deliberate trade of sampling for field of view, fixed in software. Take
several exposures, each offset by a fraction of a pixel — a **dither** — combine
them onto one finer output grid, and you recover the sampling the optics were
always capable of. That step is **drizzle**, and it is why
[Ch. 11](../guide/11-observation.md#drizzle) can state as bare habit that "*HST*
rarely takes one exposure of a target; it takes several, each offset by a
fraction of a pixel." It is not a habit; it is the only way to get HST's
resolution out of HST's pixels.

**Euclid.** A $1.2$ m telescope launched in 2023 to L2, with a visible camera at

<!-- check: pch06.euclid_arcsec_per_px = 0.1 ± 0.0001 -->

$$
0.1''/\mathrm{px}.
$$

Euclid is the combination the other two do not offer: space-based sharpness over
a survey-sized area of sky. HST is sharp and narrow; DESI is wide and blurry;
Euclid is sharp and wide. That is why its first data release is what the main
guide's Chapters 27 and 28 re-grade their DESI candidates against.

<figure markdown="span">
  ![One 1.2 arcsecond Einstein ring rendered four times: unblurred truth, and blurred to Euclid, HST and DESI resolution](figures/p06-resolution-ladder-light.svg#only-light){ width="90%" }
  ![One 1.2 arcsecond Einstein ring rendered four times: unblurred truth, and blurred to Euclid, HST and DESI resolution](figures/p06-resolution-ladder-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 6.1.** The same simulated $1.2''$
  Einstein ring, convolved with each instrument's characteristic resolution
  scale; the leftmost panel is the unconvolved truth. Euclid's and HST's panels
  are barely distinguishable from it. DESI's is a fat, soft annulus — and note
  what it is *not*: the hole is still open. This ring clears the resolution wall
  by about a factor of two, and the panel shows what clearing it by only that
  much buys you — a ring you could find, not a ring you could measure. The
  panels blur but do not pixelate: what is drawn is the PSF alone, with no
  detector grid, noise or sky background — all of which make the real DESI panel
  worse.</figcaption>
</figure>

Now put the survey's two numbers against a real lens. A typical galaxy-scale
Einstein ring has $\theta_{\mathrm{E}} \approx 1.2''$: the value Figure 6.1
uses, and close to the fiducial elliptical the main guide builds in
[Ch. 19](../guide/19-einstein-radius.md#theta-e-from-sigma-v). Opposite sides of
that ring sit a full diameter apart, $2\theta_{\mathrm{E}}$, which in DESI
pixels is

<!-- check: pch06.ring_diameter_px_desi = 9.16 ± 0.01 -->

$$
\frac{2 \times 1.2''}{0.262''/\mathrm{px}} \approx 9.16\ \mathrm{px}.
$$

Nine pixels sounds workable. Now the blur:

<!-- check: pch06.seeing_px_desi = 4.96 ± 0.01 -->

$$
\frac{1.3''}{0.262''/\mathrm{px}} \approx 4.96\ \mathrm{px}.
$$

The blur's full width at half maximum is more than half the ring's entire
diameter. Each side of the ring is smeared across a distance comparable to its
separation from the other — not enough to merge them, as Figure 6.1's DESI panel
shows, but enough that what survives the atmosphere is the ring's position and
almost none of its shape. That panel is a detection, not a picture. A ring much
smaller than this one would not even be that.

This is the wall. [Ch. 27](../guide/27-discovery.md#deriving-the-wall) turns the
same comparison into a closed-form floor on the Einstein radius a ground-based
survey can resolve at all, using a second-derivative sign test to decide when
two blurred points stop being two. Then it confirms that floor from the other
direction: when Euclid re-observed a small sample of DESI grade-C candidates,
several jumped straight to grade A. The objects did not change; the wall moved.

How far? Against DESI's blur, Euclid's resolution element is smaller by

<!-- check: pch06.euclid_gain = 13 ± 0.5 -->

$$
\frac{1.3''}{0.1''} = 13,
$$

a comparison needing one caveat, unpacked in Exercise 6.3: it divides a PSF
width by a pixel scale, legitimate only because in space those two numbers
converge by design.

One last calibration, because $1.3''$ sounds small and is not. A galaxy the size
of the Milky Way ([Ch. 1](01-scale-ladder.md#our-galaxy)), at the distance of
the main guide's standard lens, covers only a few arcseconds of sky — the
angle-to-size conversion is
[Ch. 15's](../guide/15-distances.md#three-distances) job. DESI's blur is not a
blemish on a picture of a galaxy. It is a good fraction of the galaxy.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 9 — Arcseconds, magnitudes, and the units of the sky](../guide/09-units.md#pixel-scale)**
      defines seeing as "the width of the point-spread function (PSF) imposed by
      the atmosphere (or, for a space telescope, by diffraction)" — hanging the
      entire ground-versus-space distinction on that parenthesis. You now have
      what it omits: two unrelated blur mechanisms, only one improving with
      aperture, and which observatory is limited by which. It also drops
      "$m_z < 20$ in the $z$-band" ([Ch. 9](../guide/09-units.md#magnitudes))
      into a book where $z$ means redshift on almost every other page, and never
      flags it.
    - **[Ch. 11 — From photons to pixels: PSF, noise, and drizzle](../guide/11-observation.md#drizzle)**
      states as bare fact that "*HST* rarely takes one exposure of a target; it
      takes several, each offset by a fraction of a pixel (a *dither*)" — and
      never says why. You now have the why: HST's detector pixel undersamples
      HST's own diffraction-limited PSF. That makes drizzle, and the correlated
      noise it produces — the first link in the guide's Spine 1 — a consequence
      rather than a convention.
    - **[Ch. 27 — Finding lenses: the survey, the nets, and the resolution wall](../guide/27-discovery.md#the-survey)**
      opens on "the DESI Legacy Imaging Surveys photographed most of the
      extragalactic sky in three colors" and asks "how good does a picture have
      to be before a ring is even visible in it." You now have the survey, the
      three colours, and the ring-versus-blur comparison in its own pixels;
      [Ch. 27](../guide/27-discovery.md#deriving-the-wall) turns that into a
      closed form, and [Ch. 28](../guide/28-the-label.md#same-wall) shows the
      same wall making human graders unreliable.

## Exercises { #exercises }

??? question "Exercise 6.1 — The wall, in DESI pixels"
    A ring has $\theta_{\mathrm{E}} = 1.2''$; DESI's pixel scale is $0.262''$/px
    and its seeing about $1.3''$ FWHM. Compute (a) the ring's diameter and (b)
    the seeing FWHM, both in DESI pixels. Then: is DESI *undersampled* by its
    pixel grid, and is that the same question as whether it can see the ring?

    ??? success "Solution"
        (a) The diameter is $2\theta_{\mathrm{E}}$, so $2 \times 1.2/0.262
        \approx 9.16$ pixels
        <!-- check: pch06.ring_diameter_px_desi = 9.16 ± 0.01 -->.
        (b) $1.3/0.262 \approx 4.96$ pixels
        <!-- check: pch06.seeing_px_desi = 4.96 ± 0.01 -->.

        DESI is *not* undersampled: (b) sits comfortably above the Nyquist floor
        of two pixels per FWHM. But that is a statement about the pixel grid,
        not about the sky — the pixels sample the blur faithfully; they cannot
        undo it. What decides whether the ring is visible is (b) against (a),
        and no finer pixel grid changes it. Sampling the blur and resolving the
        ring are different questions, which is why
        [Ch. 9](../guide/09-units.md#pixel-scale) insists pixel scale and seeing
        are not the same quantity.

??? question "Exercise 6.2 — Which $z$ is which?"
    Classify each $z$ as the filter or the redshift, without looking anything up.

    1. "the faint-end quality cut is $m_z < 20$"
    2. "a lens at $z_l = 0.5$ and a source at $z_s = 2.0$"
    3. "$1 + z = \lambda_{\mathrm{obs}}/\lambda_{\mathrm{emit}}$"

    ??? success "Solution"
        1. Filter — a subscript on a magnitude $m$, naming the band.
        2. Redshift — subscripted *itself*, $l$ for lens and $s$ for source,
           which only makes sense for a per-object quantity.
        3. Redshift — bare, inside a dimensionless ratio of wavelengths. A
           filter has no arithmetic.

??? question "Exercise 6.3 — What Euclid's factor of 13 does and does not say"
    DESI's seeing is about $1.3''$; Euclid's visible pixel scale is $0.1''$.
    Compute the ratio, then state what is wrong with that calculation, and why
    the answer survives anyway.

    ??? success "Solution"
        $1.3/0.1 = 13$
        <!-- check: pch06.euclid_gain = 13 ± 0.5 -->.

        What is wrong: it divides a PSF width (a measurement, in arcsec FWHM) by
        a pixel scale (a unit conversion, in arcsec per pixel) — the two
        quantities [Ch. 9](../guide/09-units.md#pixel-scale) warns are most often
        confused.

        Why it survives: for Euclid the two coincide near $0.1''$, and not by
        accident. In space the only blur is diffraction, so a camera's pixels are
        sized to sample the diffraction limit and the two converge by
        construction — which is why the main guide uses $0.1''$ as Euclid's FWHM.
        The ratio is really blur against blur, the comparison that matters. The
        same shortcut on DESI would be nonsense: its pixel scale and seeing
        differ by the factor from Exercise
        6.1<!-- check: pch06.seeing_px_desi = 4.96 ± 0.01 -->, because on the
        ground nothing forces them together.
