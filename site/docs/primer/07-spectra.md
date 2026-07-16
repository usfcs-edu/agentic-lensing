# 7. Spectra — the fingerprint of everything

[Ch. 5](05-light.md#the-spectrum) treated light as a messenger carrying two
numbers: how much (flux) and what color (temperature, via Wien's law). A
spectrograph — the instrument [Ch. 6](06-telescopes.md#what-a-telescope-collects)
described as a wavelength-binned camera rather than an ordinary one — throws
away that single color and replaces it with hundreds or thousands of numbers:
flux measured in narrow bins across the whole spectrum, not one broadband
average. What comes back is not smooth. Sitting on top of a smooth background
are sharp spikes and dips at exact, repeatable wavelengths — and that pattern
is the reason [Ch. 12](../guide/12-spectroscopy.md#spectral-lines) of the main
guide can open with the sentence "you do not need atomic physics" and mean it
literally. This chapter is what that sentence is quietly resting on.

!!! abstract "What you can skip"
    You do not need the quantum mechanics that fixes an atom's energy levels —
    accept it as a postulate, exactly as Ch. 12 asks you to. You do not need
    spectrograph engineering (fibers, gratings, detectors); Ch. 6 already gave
    you the instrument, and this chapter only needs its output. You do not need
    a line list for every element in the periodic table — one worked example
    (hydrogen's Balmer series) carries the whole argument, and two more lines
    are quoted only because Ch. 12 uses them by name without saying where they
    come from.

## Continuum and lines { #continuum-and-lines }

A star's photosphere radiates close to a blackbody curve set by its
temperature ([Ch. 5](05-light.md#why-hot-means-blue)) — smooth, one broad hump
peaking wherever Wien's law puts it. A galaxy is a few hundred billion of
these photospheres summed together, at a spread of temperatures, so its
overall spectrum is smooth for the same reason a histogram of many overlapping
distributions is smooth: no single star's signature survives the sum. That
smooth hump is the **continuum**.

Real spectra are not just a hump, though. Superimposed on the continuum are
narrow features — flux spikes above it or dips below it — sitting at specific
wavelengths and nowhere else. These are **lines**, and unlike the continuum
they are not a property of the mixture. They are a property of exactly which
chemical elements and ions are present in the gas doing the radiating or the
absorbing, which is what makes them worth the name "fingerprint."

<figure markdown="span">
  ![A galaxy spectrum: a smooth continuum with sharp absorption dips marking Ca II K, Ca II H, H-beta, Mg b and H-alpha](figures/p07-spectrum-lines-light.svg#only-light){ width="90%" }
  ![A galaxy spectrum: a smooth continuum with sharp absorption dips marking Ca II K, Ca II H, H-beta, Mg b and H-alpha](figures/p07-spectrum-lines-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 7.1.** An elliptical galaxy's spectrum: a
  smooth continuum (Ch. 5's summed blackbodies) with five absorption dips at
  fixed wavelengths — Ca II K, Ca II H, H$\beta$, Mg b, H$\alpha$ — and no
  emission anywhere. Old stars, no ongoing star formation: exactly the
  population [Ch. 3](03-galaxies.md#which-ones-lens) already singled out as
  this program's lenses.</figcaption>
</figure>

Two questions follow immediately. Why do the dips sit at those five
wavelengths and nowhere else? And why dips, rather than spikes? The next two
sections answer them in turn.

## Why elements have fingerprints { #why-elements-have-fingerprints }

An atom's electrons occupy a fixed ladder of allowed energies. When an
electron falls from a higher rung to a lower one, it emits a photon carrying
exactly the energy difference between those two rungs, no more and no less.
Because the rungs are fixed by quantum mechanics and not by the local
environment, that energy — and therefore the wavelength — is identical
wherever in the universe the same atom makes the same jump. Hydrogen's version
of this ladder is well characterized enough to fit in one formula, the
Rydberg formula for the Balmer series (the family of jumps that land on
hydrogen's second rung):

Two notation flags before writing it, because this corpus does not let a
Greek letter drift. Wavelength is written $\lambda$ here — a chapter-local
use, with nothing to do with the SMC tempering parameter of the same letter
used elsewhere in this corpus (flagged explicitly in
[Ch. 12](../guide/12-spectroscopy.md#spectral-lines) too, the one place both
books actually name a spectrum). And the lines themselves carry names fixed by
nineteenth-century spectroscopists — H$\alpha$, H$\beta$, H$\gamma$ for the
reddest, second-reddest and third-reddest lines in hydrogen's series — which
share no meaning with the deflection angle $\alpha$, the source position
$\beta$, or the density slope $\gamma$ this corpus otherwise reserves for
lensing. Every bare $\alpha$, $\beta$, $\gamma$ without an "H" in front of it
means the lensing quantity, everywhere else in both books.

$$
\frac{1}{\lambda} = R\left(\frac{1}{4} - \frac{1}{n^2}\right), \qquad n = 3, 4, 5, \ldots,
$$

with $R = 1.0967758\times10^{7}\,\mathrm{m}^{-1}$ (the Rydberg constant) and
$n$ the rung the electron falls from. Feed in $n = 3$, hydrogen's reddest and
brightest visible line:

$$
\lambda_{\mathrm{H\alpha}} = \frac{1}{R\left(\frac14 - \frac19\right)} = 656.470\ \mathrm{nm}.
$$

<!-- check: pch07.halpha_nm = 656.470 ± 0.01 -->

One formula, one constant, an exact prediction (Exercise 7.1 gets two more).
That reproducibility — not "hydrogen glows red," but "hydrogen glows at
*this* wavelength, computed from one number, everywhere" — is the whole
content of "fingerprint." Compare it to the continuum: [Ch. 5](05-light.md#why-hot-means-blue)'s
Sun peaks near 502 nm<!-- check: pch05.sun_peak_nm = 502.044 ± 0.01 -->, a
wavelength set entirely by its 5772 K surface temperature — heat the
photosphere and the peak moves. H$\alpha$'s 656.470 nm does not move. It is
fixed by hydrogen's internal structure, true whether the gas producing it
sits in the Sun, a star-forming galaxy near the edge of the observable
universe, or a lab on Earth.

That fixed number, though, hides one convention most line lists don't bother
to state. The formula above gives H$\alpha$'s wavelength *in vacuum*.
Precision spectroscopy is historically done through air, and older line
lists — some still in circulation — quote instead the air wavelength,
656.288 nm<!-- check: pch07.halpha_air_nm = 656.288 ± 0.01 -->, shorter because
air's refractive index slows the light down and compresses the wave. The two
conventions disagree by only
0.0277%<!-- check: pch07.halpha_air_vacuum_gap_percent = 0.0277 ± 0.001 --> of
H$\alpha$'s own wavelength (Exercise 7.2 works the arithmetic), and that gap
is exactly air's refractive index. It sounds negligible until you remember
that [Ch. 8](08-redshift.md#what-redshift-measures) defines a redshift as a
wavelength *ratio*: divide by the wrong rest wavelength and every redshift
measured against that line is wrong in its fourth digit, silently, with no
flag raised anywhere in the pipeline.

Hydrogen is not the only fingerprint this program leans on. A singly-ionized
calcium atom absorbs at 393.366 nm<!-- check: pch07.ca_ii_k_nm = 393.366 ± 0.001 -->
(called Ca II K — the element, its ionization state, and a spectroscopist's
historical letter for this particular line; the main guide writes the same
number as 3933.66 Å, since 1 nm is exactly 10 Å), and singly-ionized oxygen
emits a close doublet just past 372 nm<!-- check: pch07.oii_nm = 372.7 ± 0.001 -->
(written [OII], the brackets marking a transition that only happens in gas
thin enough that nothing collides an excited ion back down before it can
radiate). Both numbers are exact for the same reason H$\alpha$ is: a fixed
quantum ladder, unique to that ion, reproducible anywhere in the universe.
[Ch. 12](../guide/12-spectroscopy.md#measuring-redshift) uses this exact pair,
unexplained, to pull two redshifts out of one exposure. This section is
where those two numbers come from.

## Emission versus absorption { #emission-versus-absorption }

The fixed energy that names a fingerprint says nothing about whether it shows
up as a spike above the continuum or a dip below it. That depends on which
direction the electron jumps, and what is sitting where along the line of
sight.

An **emission line** appears when an electron falls from a higher rung to a
lower one and radiates a photon the instrument would not otherwise have
collected: it *adds* light at that exact wavelength, on top of whatever
continuum is already there. That needs a supply of atoms already sitting in
the excited (higher) state, which needs something heating or ionizing the
gas — hot young stars flooding a star-forming region with ultraviolet light,
or an active galactic nucleus. The [OII] doublet
(372.7 nm<!-- check: pch07.oii_nm = 372.7 ± 0.001 -->) is the classic marker
of exactly this: gas ionized by ongoing star formation, radiating.

An **absorption line** appears when the opposite jump happens *in front of* a
continuum source rather than beside it: a photon from the continuum, on its
way to the observer, carries exactly the energy an electron in some cooler
foreground gas needs to jump up a rung, and that electron takes it — removing
one photon from the beam at that wavelength and leaving a dip. That needs
cooler gas, or a stellar atmosphere, sitting between the continuum's source
and the observer, with nothing re-ionizing it before the jump can happen.
Ca II K (393.366 nm<!-- check: pch07.ca_ii_k_nm = 393.366 ± 0.001 -->) is the
classic marker of the opposite population: old stars, whose own cooler outer
atmospheres absorb calcium's signature out of their own light, with no
ongoing star formation anywhere in the galaxy to turn it into emission
instead.

That is exactly the distinction [Ch. 2](02-stars.md#why-massive-means-short)
and [Ch. 3](03-galaxies.md#which-ones-lens) already drew without any
spectroscopy: a spiral galaxy, still forming stars, carries ionized gas and
shows emission; an elliptical, whose star formation stopped long ago
([Ch. 3](03-galaxies.md#why-star-formation-stops)), has none left to ionize
and shows absorption only. Figure 7.1 is a real instance of the second case —
five absorption dips and not one emission spike, because there is no ionized
gas left in this galaxy to radiate anything. That absence is not a detail; it
is the entire reason [Ch. 3](03-galaxies.md#which-ones-lens) can single out
ellipticals as this program's lenses on sight, before a redshift is ever
measured.

!!! tip "You already know this"
    A redshift pipeline does not search for one line; it slides a whole
    expected *pattern* of lines against the data and scores how well the
    pattern lines up at each trial redshift — matching a fixed multi-feature
    template against noisy data by sliding it and scoring the overlap. That is
    a cross-correlation, the identical operation whether the "template" is a
    galaxy spectrum or a convolutional kernel: the pattern, not any single
    feature, is what survives noise. [Ch. 12](../guide/12-spectroscopy.md#measuring-redshift)
    runs exactly this, over a library of galaxy, quasar and star templates, at
    full DESI scale.

## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 12 — Redshifts and what a spectrum tells you](../guide/12-spectroscopy.md#spectral-lines)**
      opens with a skip box that hands you this chapter's entire content in
      one sentence: "take it as given that an electron dropping between two
      fixed energy levels emits or absorbs light at one specific, reproducible
      wavelength." You now have the mechanism behind that sentence, why it can
      be a spike or a dip, and — concretely — where Ch. 12's own two working
      numbers, Ca II K at 3933.66 Å for the lens and [OII] at 3727 Å for the
      source, come from and why those two lines, specifically, are the right
      diagnostic for an old galaxy and a star-forming one respectively.

## Exercises { #exercises }

??? question "Exercise 7.1 — The rest of the Balmer series, by hand"
    Using $\dfrac{1}{\lambda} = R\left(\dfrac14 - \dfrac{1}{n^2}\right)$ with
    $R = 1.0967758\times10^{7}\,\mathrm{m}^{-1}$, compute the wavelengths of
    H$\beta$ ($n=4$) and H$\gamma$ ($n=5$), and check them against the values
    below.

    ??? success "Solution"
        For H$\beta$: $\frac14 - \frac1{16} = 0.1875$, so
        $\lambda = 1/(R \times 0.1875) = 486.274$ nm
        <!-- check: pch07.hbeta_nm = 486.274 ± 0.01 -->. For H$\gamma$:
        $\frac14 - \frac1{25} = 0.21$, so $\lambda = 1/(R \times 0.21) = 434.173$ nm
        <!-- check: pch07.hgamma_nm = 434.173 ± 0.01 -->. Same formula, same
        constant, two more exact predictions — nothing about hydrogen changes
        between lines, only $n$ does.

??? question "Exercise 7.2 — Quantifying the air/vacuum trap"
    Using the vacuum value 656.470 nm and the air value 656.288 nm, compute the
    fractional difference as a percentage. Then say, in one sentence, why a
    redshift pipeline needs to know which convention its line list uses.

    ??? success "Solution"
        $(656.470 - 656.288)/656.470 = 2.77\times10^{-4}$, i.e.
        $0.0277\%$<!-- check: pch07.halpha_air_vacuum_gap_percent = 0.0277 ± 0.001 -->
        of the wavelength itself — small, but a redshift is defined as a
        wavelength *ratio* ([Ch. 8](08-redshift.md#what-redshift-measures)), so
        a 0.03% error in the rest wavelength you divide by becomes a 0.03%
        error in every redshift measured against that line, in the fourth
        significant digit, with no warning flag raised anywhere downstream.

??? question "Exercise 7.3 — Reading Figure 7.1"
    Figure 7.1 shows five absorption dips and no emission. If you were instead
    handed the spectrum of a star-forming spiral galaxy over the same
    wavelength range, which single line would you most expect to flip from
    absorption to emission, and which line, absent here entirely, would you
    expect to appear for the first time? Why do Ca II K and Mg b not flip the
    same way?

    ??? success "Solution"
        H$\alpha$ is the one most likely to flip: hot ionized gas around young
        stars emits strongly there, often overwhelming the much weaker
        absorption from the same transition in the galaxy's own stellar
        atmospheres. The line that would appear for the first time is [OII]
        at 372.7 nm<!-- check: pch07.oii_nm = 372.7 ± 0.001 -->, since a
        quiescent elliptical has no ionized gas anywhere to produce it. Ca II K
        and Mg b do not flip, because they trace absorption in the light of
        the whole stellar population, old and young alike — star formation
        elsewhere in the galaxy adds an emission channel without removing that
        underlying absorption.
