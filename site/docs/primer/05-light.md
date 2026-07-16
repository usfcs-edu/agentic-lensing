# 5. Light is the only messenger

Every number in the rest of this book — a distance, an age, a mass — is going
to be extracted from light and nothing else. No probe has ever been sent to
another galaxy; no sample has ever been returned from one. What follows is
the physical vocabulary that makes that extraction possible: what light *is*
across its full range, why an object's temperature announces itself as a
color, why brightness alone never tells you how far away something is, and
the plain fact that sits underneath every redshift claim in this program —
that light takes time to travel, so an image is never a report of *now*.
Chapter 1 gave you a ruler for space; this chapter gives you the other
half, a ruler for time, built out of the same unit.

!!! abstract "What you can skip"
    You do not need Maxwell's equations, quantum electrodynamics, or anything
    about *why* accelerating charges radiate. Take electromagnetic radiation
    as given, the way you'd take "matrix multiplication computes a linear
    map" as given without re-deriving it. You also do not need to derive the
    inverse-square law from a solid-angle integral; the one-line argument
    below is the whole content, and you already reason this way every time
    you think about how a sensor's signal-to-noise falls off with range.

## The electromagnetic spectrum { #the-spectrum }

Light is an oscillating electric and magnetic field, and like any wave it
has a wavelength $\lambda$ and a frequency $\nu$, related by

$$
c = \lambda \nu,
$$

where $c$ is the speed of light, $299{,}792.458$ km/s exactly (it is a
defined constant, not a measured one — the meter is defined *from* it).
"Light" in the everyday sense — what your eyes detect — is one narrow slice
of a much wider **electromagnetic spectrum** that runs, by wavelength, from
kilometers-long radio waves down to sub-picometer gamma rays, with
microwaves, infrared, the visible band, ultraviolet, and X-rays in between.
Shorter wavelength means higher frequency (fixed $c$), and light also comes
in discrete energy packets called **photons**, whose energy per photon rises
as wavelength falls — which is the whole reason gamma rays are dangerous and
radio waves are not. Nothing later in this book needs that quantization
explicitly; treat light as a wave characterized by $\lambda$ and reach for
"photon" only when someone hands you an energy in electron-volts instead of
a wavelength.

The **visible band** — the sliver human eyes evolved to catch — runs
roughly $380$–$750$ nm, violet to red. Everything else in the spectrum is
invisible to you but not to a telescope built to catch it: infrared cameras,
radio dishes, and X-ray satellites are all reading the same electromagnetic
field, just at wavelengths your retina cannot. One unit convention is worth
fixing now because you will see it constantly: spectroscopy is conventionally
quoted in **Ångströms** ($1$ Å $= 0.1$ nm $= 10^{-10}$ m), a unit chosen when
atomic spacings, not visible-light wavelengths, were the natural scale.
[Ch. 12 of the main guide](../guide/12-spectroscopy.md#spectral-lines) will
hand you rest wavelengths like "$3933.66$ Å" and "$6564.61$ Å" without ever
saying which part of the spectrum those numbers are *in*. You can check now:
$3933.66$ Å $= 393.366$ nm sits just outside the visible band, in the near
ultraviolet; $6564.61$ Å $= 656.461$ nm sits at the red edge of visible,
which is why that particular hydrogen line (Balmer-alpha, called H$\alpha$)
is the one every optical spectrograph is built to catch.

!!! tip "You already know this"
    Splitting light into a spectrum — passing it through a prism or a
    diffraction grating — is not merely *analogous* to a Fourier transform;
    it is a physical implementation of one. A grating imposes a
    path-length difference that varies with the sine of the exit angle, and
    the intensity pattern it produces is the squared magnitude of the
    Fourier transform of the incident wave, exactly the machinery of
    [Ch. 7](../guide/07-fourier.md). A spectrograph is a Fourier transform
    built out of glass.

## Why hot means blue { #why-hot-means-blue }

A dense, opaque body in thermal equilibrium — a stellar photosphere is the
working example throughout this book — radiates a continuous spectrum whose
*shape* depends on exactly one number, its temperature $T$. This is a
**blackbody spectrum**, and its single most useful feature is **Wien's
displacement law**: the wavelength at which the emission peaks is inversely
proportional to temperature,

$$
\lambda_{\text{peak}} \approx \frac{2.898\times10^{6}\ \text{nm·K}}{T}.
$$

Hotter objects peak at *shorter* wavelengths — bluer light — and cooler
objects peak at longer wavelengths — redder light. This is the entire
physical content behind every "old and red" / "young and blue" statement
this program makes about stars and galaxies, stated here in one formula
instead of being taken on faith.

Put the Sun's own surface temperature in:

$$
\lambda_{\text{peak}} = \frac{2.898\times10^{6}}{5772} \approx 502.0\ \text{nm},
$$

<!-- check: pch05.sun_peak_nm = 502.039 ± 0.01 -->

which lands squarely in the visible band — unsurprising, since human color
vision evolved under exactly this light source. That peak wavelength
corresponds to a frequency of

<!-- check: pch05.sun_peak_freq_hz = 5.9715e14 ± 0.001e14 -->

$$
\nu = \frac{c}{\lambda} \approx 5.97\times10^{14}\ \text{Hz},
$$

roughly half a million times faster than a modern CPU's clock — a useful
sense of scale for how fast an electromagnetic oscillation actually is.

A much hotter star, $10{,}000$ K, gives $\lambda_{\text{peak}} \approx
289.8$ nm<!-- check: pch05.star10000k_peak_nm = 289.777 ± 0.01 -->
— already past the blue end of the visible band, into the ultraviolet. This
is a real trap worth naming: the star's *peak* isn't inside the visible
window at all, so "the peak wavelength is the color you see" is not quite
right. What you see is the shape of the curve *restricted* to $380$–$750$
nm, and a $10{,}000$ K blackbody's visible tail still strongly favors blue
over red relative to a cooler star's — so the star looks blue-white, for a
subtler reason than "the peak is blue." A cool star at $3000$ K peaks at
$\lambda_{\text{peak}} \approx 965.9$ nm
<!-- check: pch05.cool_star_peak_nm = 965.924 ± 0.01 -->,
deep in the near-infrared, and the sliver of that curve inside the visible
band is skewed red — hence "red dwarf." An even hotter object, the afterglow
of the Big Bang itself at $2.725$ K, peaks at
<!-- check: pch05.cmb_peak_mm = 1.0634 ± 0.001 -->$1.06$ mm — not visible
light at all, but microwave. [Ch. 11](11-big-bang.md#the-cmb) is where that
number stops being a curiosity and becomes the single most informative
picture ever taken.

<figure markdown="span">
  ![Planck curves for a 3000 K red star, the 5772 K Sun, and a 10000 K blue star, with Wien's law peak marked and the visible band shaded](figures/p05-blackbody-light.svg#only-light){ width="90%" }
  ![Planck curves for a 3000 K red star, the 5772 K Sun, and a 10000 K blue star, with Wien's law peak marked and the visible band shaded](figures/p05-blackbody-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 5.1.** Blackbody curves at three temperatures, each
  normalized to its own peak, with the visible band ($380$–$750$ nm) shaded
  and each curve's Wien-law peak marked by a dotted line. The $10{,}000$ K
  curve's peak falls to the *left* of the shaded band entirely — its light
  is still bluer than the Sun's where the two curves overlap in the visible,
  but the actual maximum of the curve is ultraviolet light your eye cannot
  see at all.</figcaption>
</figure>

## The inverse-square law { #the-inverse-square-law }

A source radiating total power (luminosity) $L$ spreads that energy over the
surface of an ever-expanding sphere as it travels outward. At distance $d$,
that sphere has area $4\pi d^2$, so the power crossing a unit area — the
**flux** $F$ you actually measure — is

$$
F = \frac{L}{4\pi d^2}.
$$

Nothing about the source changed; the same total light is simply divided
over a larger sphere the farther out you go. Double the distance and the
sphere's area quadruples, so the flux drops to a quarter:

<!-- check: pch05.flux_ratio_at_2x = 0.25 ± 0.001 -->

$$
\frac{F(2d)}{F(d)} = \left(\frac{d}{2d}\right)^2 = \frac{1}{4}.
$$

Push it to ten times the distance and flux falls to one part in a hundred:

<!-- check: pch05.flux_ratio_at_10x = 0.01 ± 0.001 -->

$$
\frac{F(10d)}{F(d)} = \frac{1}{100}.
$$

This is the reason apparent brightness, by itself, is ambiguous: a dim
source nearby and a luminous source far away can produce the identical
flux. [Ch. 9's magnitude system](../guide/09-units.md#magnitudes) is a
logarithmic re-encoding of exactly this quantity, chosen because real
surveys must represent fluxes spanning ten or more orders of magnitude in
one image — the inverse-square law is *why* that dynamic range is so wide
in the first place, not an incidental fact about detectors. Breaking the
distance/luminosity ambiguity is precisely the job
[Chapter 9](09-distance-ladder.md) of this book takes on with **standard
candles**: objects whose luminosity $L$ is known independently, so a
measured flux $F$ can be inverted for $d$ instead of read as an
unresolvable mix of the two.

## Looking back in time { #looking-back-in-time }

Light travels fast, but not infinitely fast, and every distance this book
has quoted so far converts directly into a *delay*. The Sun's light takes

<!-- check: pch05.sun_light_travel_min = 8.317 ± 0.01 -->

$$
\frac{1\ \text{AU}}{c} \approx 8.3\ \text{minutes}
$$

to reach Earth, so the Sun you see is never the Sun *now* — it is the Sun as
it was $8.3$ minutes ago. That gap only grows with distance, because a
light-year is, by definition, the distance light covers in a year: the
travel time in years and the distance in light-years are numerically the
same number. Andromeda, the nearest large galaxy beyond our own, sits about
$2.5$ million light-years away
<!-- check: pch05.andromeda_ly = 2500000 ± 1 -->,
so its light took<!-- check: pch05.andromeda_travel_myr = 2.5 ± 0.001 -->
$2.5$ million years to reach the telescope pointed at it tonight — the
photons left before *Homo sapiens* existed, and what a picture of Andromeda
shows is that departure, not any "now" on the other end.

There is no way around this, and no sense in which a more sensitive
telescope could someday see Andromeda "as it is today": simultaneity across
that distance isn't a measurement problem, it's what "distance" *means* once
you accept that nothing — light included — carries information faster than
$c$. Every image is old news, and the farther out you look, the older the
news gets. For the systems this program actually studies — a lens galaxy at
redshift $z=0.5$, a source galaxy at $z=2$ — that delay is not minutes or
millions of years but billions: [Chapter 8](08-redshift.md) turns this exact
idea into a number for each one. This chapter's job was only to establish
that the delay is real and unavoidable; everything about *how big* it gets,
and what redshift has to do with it, is what the rest of Part II builds
toward.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 9 — Arcseconds, magnitudes, and the units of the sky](../guide/09-units.md#magnitudes)**
      defines flux as "what a detector actually integrates" and builds an
      entire logarithmic magnitude system to compress it, but never explains
      *why* flux needs ten-plus orders of magnitude of dynamic range in the
      first place. You now have the answer: the inverse-square law, applied
      to real objects sitting at wildly different distances and
      luminosities, and the standard-candle idea (this book's own
      [Ch. 9](09-distance-ladder.md#standard-candles)) that inverts it back
      into a distance.
    - **[Ch. 12 — Redshifts and what a spectrum tells you](../guide/12-spectroscopy.md#spectral-lines)**
      calls a galaxy's continuum "the light of a few hundred billion stellar
      photospheres, each close to a blackbody" and moves on without ever
      defining a blackbody. It also hands you rest wavelengths in Ångströms
      — $3933.66$ Å, $6564.61$ Å — and later notes that one redshifted line
      lands "deep in the near-infrared, off the red end of a ground-based
      optical spectrograph entirely," a claim you can now check yourself
      because you know where the visible band's edges sit in nanometers.

## Exercises { #exercises }

??? question "Exercise 5.1 — Wien's law by hand"
    Using $\lambda_{\text{peak}} T \approx 2.898\times10^{6}\ \text{nm·K}$,
    compute the peak wavelength for a $3000$ K star and for a $10{,}000$ K
    star. For each, say which part of the electromagnetic spectrum
    (infrared, visible, ultraviolet) the peak falls in, and explain why
    neither answer is the whole story about what color the star looks like
    to a human eye.

    ??? success "Solution"
        $3000$ K: $\lambda_{\text{peak}} = 2.898\times10^6/3000 \approx
        965.9$ nm<!-- check: pch05.cool_star_peak_nm = 965.924 ± 0.01 -->,
        in the near-infrared — beyond the red edge of the visible band
        entirely. $10{,}000$ K: $\lambda_{\text{peak}} =
        2.898\times10^6/10{,}000 \approx 289.8$ nm
        <!-- check: pch05.star10000k_peak_nm = 289.777 ± 0.01 -->, in the
        ultraviolet — beyond the blue edge. Neither peak is inside the
        $380$–$750$ nm band the eye responds to, so "peak wavelength" is not
        "perceived color": what the eye sees is the *shape* of the whole
        curve restricted to the visible window, and a curve whose true peak
        lies just past either edge still tilts that visible sliver toward
        red (cool case) or blue (hot case) relative to a curve peaking
        inside the band.

??? question "Exercise 5.2 — Inverse square, both directions"
    Two galaxies have identical luminosity $L$. Galaxy B is $10\times$
    farther away than Galaxy A. What is the ratio of their fluxes? Now
    suppose instead that A and B are at the *same* distance but you measure
    B's flux to be $1/100$ of A's — can you conclude B is $100\times$ less
    luminous? What is the one piece of information missing that would let
    you conclude that?

    ??? success "Solution"
        Same luminosity, $10\times$ the distance:
        $F_B/F_A = (d_A/d_B)^2 = 1/100$
        <!-- check: pch05.flux_ratio_at_10x = 0.01 ± 0.001 -->. But flux
        alone never separates "farther" from "dimmer" — a source $100\times$
        less luminous at the *same* distance produces an identical flux
        ratio to a source of identical luminosity $10\times$ farther away.
        Concluding B is $100\times$ less luminous requires knowing the
        *distance* independently, which is exactly what a standard candle
        supplies and a bare flux measurement never does on its own.

??? question "Exercise 5.3 — How old is the news?"
    Using $d = ct$, compute how many minutes ago the light left the Sun (use
    $1$ AU $/c$), and how many years ago the light left Andromeda (use its
    distance in light-years directly). Then answer: is there any telescope,
    however sensitive, that could show you Andromeda "as it is right now"?

    ??? success "Solution"
        Sun: $1\ \text{AU}/c \approx 8.3$ minutes
        <!-- check: pch05.sun_light_travel_min = 8.317 ± 0.01 -->.
        Andromeda: because a light-year is defined as the distance light
        travels in a year, its distance in light-years, $2.5$ million
        <!-- check: pch05.andromeda_ly = 2500000 ± 1 -->, *is* its light
        travel time in years
        <!-- check: pch05.andromeda_travel_myr = 2.5 ± 0.001 -->. No
        telescope can show Andromeda "as it is now": there is no signal that
        outruns light to tell you what is happening there today, so "now, at
        Andromeda" is not a measurement anyone could ever make — it isn't
        that our instruments fall short, it's that the question presupposes
        a simultaneity that a finite light speed does not allow.

??? question "Exercise 5.4 — Reading an Ångström"
    The main guide's Ch. 12 quotes a rest wavelength of $3933.66$ Å for the
    Ca II K absorption line. Convert this to nanometers and say whether it
    is inside the visible band. Do the same for $6564.61$ Å
    (Balmer-alpha/H$\alpha$).

    ??? success "Solution"
        $3933.66$ Å $= 393.366$ nm — just below the $380$–$750$ nm visible
        band's blue edge is $380$ nm, so this line sits right at the
        boundary, in the near ultraviolet/violet. $6564.61$ Å $= 656.461$
        nm — comfortably inside the visible band, near its red edge, which
        is why H$\alpha$ is a workhorse line for ground-based optical
        spectrographs: it stays observable out to larger redshifts before
        being stretched past $750$ nm and out of the visible window
        entirely.
