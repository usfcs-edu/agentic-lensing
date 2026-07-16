# Fourier, power spectra, and whitening

This chapter has no lensing in it. It is entirely about signals: what a
convolution is, what its cross-correlation cousin is and why the difference
bites, and what a power spectrum has to do with a covariance matrix. None of
that appears in a standard general-relativity or lensing course, and none of
it is needed until Part V — but without it, [Methods I: the correlated-noise
likelihood](../current/claude-giga-lens/index.md#sec:methods1) of the
campaign's final report is unreadable notation: $C=D^{1/2}KD^{1/2}$, a
"stationary correlation kernel," an "operator-norm whiteness gate
$e_{\mathrm{op}}\le0.02$," a "positive-semidefinite-by-construction" kernel
family. Every one of those phrases is Fourier analysis wearing an astronomy
costume. By the end of this chapter you can read
`reproductions/claude-giga-lens/cgl/whiten.py` start to finish, and
[Ch. 24](24-correlated-noise.md) can spend its entire budget on the physics
instead of re-teaching this.

!!! abstract "What you can skip"
    The Fast Fourier Transform itself — the $O(N\log N)$ divide-and-conquer
    algorithm — is a standard algorithms-course result and is not re-derived
    here; every `np.fft.fft2` in this chapter (and in the repo) is used as a
    black box, the same way you already use it. If $\hat x[k]=\sum_n
    x[n]e^{-2\pi ikn/N}$ is already loaded, skim past its reintroduction
    below. What is *not* assumed: any prior exposure to power spectra,
    autocorrelation, or whitening as *statistical* objects rather than
    signal-processing ones, or the specific fact that the "convolution" every
    deep-learning framework applies is not the flip-and-slide convolution of a
    signal-processing textbook.

## Convolution vs. cross-correlation { #convolution-vs-correlation }

[Ch. 6](06-vector-calculus.md#greens-function) handed you a convolution
without dwelling on the word: the lensing potential is $\psi = 2(\kappa * G)$
— the factor of 2 is $\nabla^2\psi=2\kappa$'s, not a new one — the convergence
smeared out by the Green's function of the Laplacian. For
finite sequences $f,g$ of length $N$ (indices taken mod $N$ — "circular," to
keep the algebra finite; the repo zero-pads when it needs a genuinely *linear*
convolution instead, and says so explicitly where it matters),

$$
(f * g)[n] \;=\; \sum_{m=0}^{N-1} f[m]\,g[(n-m) \bmod N]
\qquad\text{convolution.}
$$

**Cross-correlation** looks almost identical, but does not flip the second
sequence before sliding it:

$$
(f \star g)[n] \;=\; \sum_{m=0}^{N-1} f[m]\,g[(m+n) \bmod N]
\qquad\text{cross-correlation.}
$$

Define the time-reversal $\tilde g[m] = g[(-m)\bmod N]$. Substituting shows
$(f\star g)[n] = (f * \tilde g)[n]$ exactly: correlation *is* convolution,
against a flipped kernel. The two operations coincide, for every $f$,
precisely when $g$ is symmetric under negation ($g[-m]=g[m]$, an *even*
kernel) — then $\tilde g = g$ and the flip changes nothing.

This is not a pedantic distinction. Every deep-learning framework's
`Conv2d` is, by this definition, cross-correlation: no kernel is flipped
before the sliding dot-product. This repo's own conv-whitening code hits the
identical fact head-on and documents it in its own docstring
(`reproductions/claude-giga-lens/cgl/whiten.py:25`–`29`
([source](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/whiten.py#L25))):

> Convolution semantics: `jax.lax.conv_general_dilated` with `'SAME'` padding
> is CROSS-CORRELATION (no kernel flip)... Symmetric whitening kernels make
> the distinction moot; asymmetric callers must pass h in correlation
> orientation.

`make_conv_whitener` (`reproductions/claude-giga-lens/cgl/whiten.py:39`) is
therefore locked against exactly this hazard:
`reproductions/claude-giga-lens/tests/test_marg.py:83`–`96` builds a
*deliberately asymmetric* random $3\times3$ kernel and checks the JAX
operator against `scipy.ndimage.correlate` (not `scipy.ndimage.convolve`) to
machine precision. The whitening kernel that `build_whitener` actually
constructs sidesteps the whole question by construction — it is forced
symmetric, `h = 0.5 * (h + h[::-1, ::-1])`
(`reproductions/claude-giga-lens/cgl/whiten.py:145`) — but the generic
operator underneath it is correlation, not convolution, and the test exists
because a future asymmetric kernel would silently get the wrong filter
applied if that fact were ever forgotten.

!!! tip "You already know this"
    "Convolution" in a CNN is cross-correlation, full stop — you have almost
    certainly used this fact without naming it. What you may not have used
    before is the flip side: correlation is the tool for asking "how much
    does this pattern resemble itself, offset by $\Delta$?" — which is a
    *statistical* question, not a filtering one. That statistical question is
    this chapter's real subject, and it is where the next section starts.

## Power spectra and autocorrelation: the Wiener–Khinchin theorem { #psd-and-autocorrelation }

A sequence is **(weakly) stationary** if its statistics don't depend on
*where* you look: the correlation between two positions depends only on their
separation, never on their location. Write $R[\Delta] = \mathrm{Cov}(x[n],
x[n+\Delta])$ — by stationarity this does not depend on $n$ — and
$\rho[\Delta] = R[\Delta]/R[0]$ for its normalized version ($\rho[0]=1$). (This
$\rho$ is a *correlation*, not a density — [Ch. 3](03-integrals.md#the-abel-projection)'s
$\rho \sim r^{-\gamma}$ is a different, unrelated use of the same letter.
Neither this guide nor the repo's own papers ever let the two contexts
overlap.)

!!! tip "You already know this"
    Weight-sharing in a convolutional layer *is* an assumption of
    stationarity: the same kernel applies at every spatial location because
    the network is told, by construction, that image statistics don't
    privilege one pixel over another. This chapter's "stationary correlation
    kernel $\rho(\Delta)$" is the identical assumption, written down as a
    noise model instead of a weight tensor.

The **discrete Fourier transform** of a length-$N$ real sequence is
$\hat x[k] = \sum_{n=0}^{N-1} x[n]\,e^{-2\pi ikn/N}$. You don't need complex
analysis for what follows — $e^{i\phi}=\cos\phi+i\sin\phi$ is bookkeeping, and
its conjugate simply flips the sign of $\phi$, $\overline{e^{i\phi}} =
e^{-i\phi}$. Every quantity that survives to matter in this chapter (a power
spectrum, a correlation, a variance) is real.

**The convolution theorem**, derived, not quoted: substitute $n' = n-m$ inside
the double sum,

$$
\widehat{f*g}[k] = \sum_n\sum_m f[m]\,g[(n-m)]\,e^{-2\pi ikn/N}
= \Big(\sum_m f[m]e^{-2\pi ikm/N}\Big)\Big(\sum_{n'} g[n']e^{-2\pi ikn'/N}\Big)
= \hat f[k]\,\hat g[k].
\label{eq:conv-thm}
$$

Convolution in the index domain is pointwise multiplication in frequency.
[Exercise 7.2](#exercises) asks you to run the identical substitution on
cross-correlation; here, take $g=f$ directly — the case that matters — and
substitute $p = m+n$ into $R[n] = (x\star x)[n] = \sum_m x[m]\,x[m+n]$:

$$
\hat R[k] = \sum_n\sum_m x[m]x[m+n]\,e^{-2\pi ikn/N}
= \Big(\sum_m x[m]e^{2\pi ikm/N}\Big)\Big(\sum_p x[p]e^{-2\pi ikp/N}\Big)
= \overline{\hat x[k]}\,\hat x[k] = |\hat x[k]|^2 =: S[k].
\label{eq:wk}
$$

(The first factor picked up a $+$ sign in the exponent from re-indexing;
because $x$ is real, that sum is exactly $\overline{\hat x[k]}$.) Equation
$\eqref{eq:wk}$ is the **Wiener–Khinchin theorem**: the power spectral density
(PSD) is the Fourier transform of the autocorrelation, and it is
*automatically non-negative*, because it is manifestly a squared magnitude.
(The general, measure-theoretic statement of the same fact — a function is a
valid autocorrelation if and only if its Fourier transform is everywhere
non-negative — is Bochner's theorem; what you just derived is its four-line
finite special case, and it is all this chapter needs.)

Computing $R[\Delta]$ at every lag directly costs $O(N^2)$ — every pair of
positions, once. Equation $\eqref{eq:wk}$ turns that into two $O(N\log N)$
FFTs and one $O(N)$ pointwise multiply — the same zero-padding trick you may
know from FFT-based long-integer or polynomial multiplication, here keeping
a *linear* autocorrelation from wrapping around a *circular* grid. This is
not a textbook aside; it is the literal content of
`reproductions/foundry-i/46_noise_audit.py:89`,

```python
acf = np.real(np.fft.ifft2(np.abs(np.fft.fft2(pad_n)) ** 2))
```

read right to left: FFT the (zero-padded) residual image, take the squared
magnitude — the PSD, by $\eqref{eq:wk}$ — inverse-FFT it back, and every lag
of the autocorrelation falls out at once. An all-pairs statistic, computed
without ever forming a pair.

**Worked example.** `reproductions/claude-giga-lens/data/noise_kernel_report.json`
stores exactly this calculation's output: the measured, model-subtracted,
along-axis autocorrelation of the fine (`v3`, $0.04''$) drizzle product.

| lag $\Delta$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $\rho(\Delta)$, fine product | $1.000$ | $0.815$ | $0.547$ | $0.357$ | $0.259$ | $0.229$ |

<!-- check: ch07.rho1_v3_fine = 0.8147 ± 0.001 -->

and it keeps decaying only slowly out to lag $11$, the report's cutoff. A
crude estimator of the **integrated autocorrelation time**,
$\tau_{\mathrm{int}} = 1 + 2\sum_{k=1}^{K}\rho(k)$, truncated at the $K=11$
lags actually measured, gives

<!-- check: ch07.tau_int_v3_11lag = 7.453 ± 0.01 -->

$\tau_{\mathrm{int}} \approx 7.45$. For a 1-D stationary sequence,
$N_{\mathrm{eff}}/N = 1/\tau_{\mathrm{int}} \approx$
<!-- check: ch07.n_eff_over_n_1d_v3 = 0.1342 ± 0.001 --> $0.134$: along one
axis, only about $13\%$ of pixels carry independent information. Pixel noise
correlates along *both* image axes; if they were independent of each other,
a 2-D pixel's independent fraction would be the square,
<!-- check: ch07.n_eff_over_n_2d_estimate = 0.0180 ± 0.001 --> $\approx 0.018$
— within about $6\%$ of the campaign's own headline figure,
$N_{\mathrm{eff}}/N\approx$ <!-- check: ch07.n_eff_over_n_reported = 0.017 ± 0.0005 -->
$0.017$ (`reproductions/claude-giga-lens/papers/main.tex:213`), from a
fuller, radially-averaged 2-D audit this guide does not re-run. The gap is
what truncating at $11$ lags of an undecayed tail, plus approximate
separability, should cost — four lines of arithmetic on a published table,
landing within a few percent of a number from a completely different, more
careful route.

$\tau_{\mathrm{int}}$ has another name you will meet in
[Ch. 23](23-samplers.md#why-gradients): for a Markov chain instead of a pixel
grid, the identical sum over the chain's own autocorrelation is the
*integrated autocorrelation time*, and $N/\tau_{\mathrm{int}}$ is the
*effective sample size* every $\hat R$/ESS diagnostic in this repo's sampler
benchmark reports. Same formula, same reason it exists: correlated draws —
in time, or in space — carry less information than their raw count suggests.

## Whitening { #whitening }

A **whitening** transform $u = Wx$ makes $\mathrm{Cov}(u) = I$: every
whitened coordinate unit-variance, zero cross-correlation with every other.

!!! tip "You already know this"
    ZCA/PCA whitening does this by eigendecomposing a covariance $\Sigma =
    V\Lambda V^\top$ and setting $W = V\Lambda^{-1/2}V^\top$ — divide by the
    square root of each eigenvalue, along its own eigenvector. What follows
    is the identical operation; the only thing that changes is which
    eigenbasis is in play.

That eigenbasis is not a coincidence you have to look up — it falls straight
out of $\eqref{eq:conv-thm}$. Let $e_k[n] = e^{2\pi ikn/N}$ be the $k$-th
Fourier mode, and consider "convolve by a fixed kernel $\rho$" as a linear
operator. Direct substitution shows

$$
(\rho * e_k)[n] = \sum_m \rho[m]\,e_k[n-m]
= e_k[n]\sum_m \rho[m]\,e^{-2\pi ikm/N} = \hat\rho[k]\,e_k[n].
$$

Every Fourier mode is an eigenvector of "convolve by $\rho$," with eigenvalue
$\hat\rho[k] = S[k]$ — the same power spectrum $\eqref{eq:wk}$ built. A
stationary covariance matrix $K_{ij} = \rho[i-j]$ (this is exactly what
"stationary" means as a matrix statement) is therefore diagonalized by the
Fourier modes, with eigenvalues $S[k]$: [Ch. 5](05-linear-algebra.md#symmetric-2x2)'s
eigendecomposition of a symmetric matrix, generalized from $2\times2$ to
$N\times N$, with the eigenvectors handed to you for free instead of solved
for numerically. So "divide by $\sqrt{\text{eigenvalue}}$ along each
eigenvector" becomes: divide by $\sqrt{S[k]}$ at each frequency, then
transform back —

$$
\hat h[k] = \frac{1}{\sqrt{S[k]}}, \qquad h = \mathrm{IFFT}(\hat h).
\label{eq:whitener}
$$

which is `reproductions/claude-giga-lens/cgl/whiten.py:141` verbatim
(`h_full = np.real(np.fft.ifft2(1.0 / np.sqrt(S)))`), after $S$ is floored —
the next section's subject.

An *exact* global divide-by-$\sqrt{S}$ is not what the production code
applies, though: the likelihood needs whitening to be a local,
differentiable, mask-aware JAX operation (`make_conv_whitener`,
`reproductions/claude-giga-lens/cgl/whiten.py:39`–`72`), and a single
monolithic FFT over a masked, irregular image is not that. `build_whitener`
instead truncates $h$ to a small $(2M+1)^2$ stencil — a short local filter,
refined by Gauss–Newton and Lawson–IRLS rounds to push its worst-frequency
error down — so it can no longer whiten *every* frequency exactly, and the
construction is accepted only if its worst-case departure from perfect
whitening clears a gate (writing the spectrum as $S(\omega)$, a function of
continuous frequency $\omega=2\pi k/N$, rather than $S[k]$'s bin index —
what a filter's response actually has to cover):

$$
e_{\mathrm{op}} \;=\; \max_\omega \big|\,S(\omega)\,|\hat h_M(\omega)|^2 - 1\,\big|
\;\le\; 0.02.
\label{eq:eop-gate}
$$

$\hat h_M^2 = 1/S$ exactly would give $S\,\hat h_M^2 \equiv 1$ at every
frequency; $e_{\mathrm{op}}$ measures the worst single frequency's failure of
that identity, and
`reproductions/claude-giga-lens/cgl/whiten.py:147`–`149`'s `e_op_of` computes
exactly $\eqref{eq:eop-gate}$.

**Worked example.** For the binned product (`v3b`, $0.08''$ — the "money"
product the [Ch. 25](25-money-number.md#the-chain) chain samples), the
accepted whitener uses a stencil half-width
<!-- check: ch07.m_v3b = 10 ± 0 --> $M=10$ and achieves
<!-- check: ch07.eop_v3b = 0.0124 ± 0.0005 --> $e_{\mathrm{op}} = 0.0124$,
comfortably inside the $0.02$ gate — at the cost of eroding
<!-- check: ch07.n_eroded_v3b = 9273 ± 0 --> $9{,}273$ of the product's
pixels (any pixel whose $(2M+1)^2$ stencil would touch a mask or the border
cannot be whitened cleanly and is dropped, not down-weighted), leaving
<!-- check: ch07.n_keep_v3b = 16653 ± 0 --> $16{,}653$ kept.

## Positive semi-definiteness { #positive-semidefiniteness }

A symmetric matrix $K$ is **positive semi-definite** (PSD) if $x^\top K x \ge
0$ for every vector $x$, equivalently if every eigenvalue is $\ge 0$. This is
not an optional nicety for a covariance or correlation matrix — it is the
entire content of what makes it *valid*: a negative eigenvalue would mean
some linear combination of pixels has negative variance, which is not a
statement about noise, it is nonsense. Combined with the last section's fact
— a stationary $K$'s eigenvalues *are* the samples $S[k]$ of its power
spectrum — positive-semi-definiteness translates exactly into $S(\omega)\ge0$
at every frequency: [Ch. 5](05-linear-algebra.md#definiteness-and-saddles)'s
linear algebra and this chapter's Fourier analysis stating the identical
requirement in two languages.

$\eqref{eq:wk}$ already showed that the *true* PSD of a genuine
autocorrelation is automatically non-negative — a squared magnitude by
construction. The practical difficulty is that the repo's $\rho(\Delta)$
isn't handed down as one; it is *fit* to noisy, model-subtracted pixel
residuals by least squares, and a generic multi-parameter fit carries no
such guarantee. The pre-registered plan tried the simplest such family
first — a single Gaussian — and failed a more basic bar before PSD ever came
up: it could not even match the measured correlation closely enough (worst
residual
<!-- check: ch24.kernel_single_family_v3 = 0.311 ± 0.001 --> $0.311$ against
a $0.05$ gate, on the fine product;
`reproductions/claude-giga-lens/papers/main.tex:383`–`384`). Its
replacement — the kernel family at
`reproductions/claude-giga-lens/papers/main.tex:389`–`394`, rendered in full
in [Methods I](../current/claude-giga-lens/index.md#sec:methods1) — is built
the other way around: a weighted sum of a delta (flat spectrum, $S\equiv1$,
trivially non-negative), a drizzle-anchored kernel convolved with a
bivariate Gaussian (a Gaussian's Fourier transform is itself a positive
Gaussian, so this term's spectrum is a *product* of two non-negative
spectra), and a second Gaussian pedestal — three pieces, each individually
guaranteed PSD, combined with non-negative weights. The paper's own name for
this is exact: "the minimal positive-semidefinite-by-construction
extension."

Guaranteed-PSD-in-principle is not the end of the story, because the
*fitted*, finite kernel's numerically estimated spectrum can still come
arbitrarily close to zero at some frequency — a real near-degeneracy, not a
fitting artifact: an oversampled grid genuinely carries almost no independent
information at some spatial frequencies. $\eqref{eq:whitener}$ divides by
$\sqrt{S}$, so a near-zero value there blows the whitening filter up. The
fix is a floor:

$$
S \;\leftarrow\; \max\!\big(S_{\mathrm{raw}},\ s_{\mathrm{floor}}\cdot\overline{S_{\mathrm{raw}}}\big),
\label{eq:floor}
$$

`reproductions/claude-giga-lens/cgl/whiten.py:137`–`138`, with a module default
<!-- check: ch07.s_floor_default = 0.05 ± 0.001 --> $s_{\mathrm{floor}}=0.05$
— clip any frequency's power to at least $5\%$ of the spectrum's own mean.

**Worked example.** The fine product's raw spectral minimum is only
<!-- check: ch07.s_raw_min_over_mean_v3 = 0.0527 ± 0.001 --> $5.27\%$ of its
mean — barely inside that default floor. The binned product's is
<!-- check: ch07.s_raw_min_over_mean_v3b = 0.0241 ± 0.001 --> $2.41\%$ —
already *below* it. Clamping `v3b` to the fixed $0.05$ floor anyway, rather
than adapting it downward, is exactly the mistake the campaign made and
retracted: a flat floor biased the Monte-Carlo whitened variance to
<!-- check: ch07.var_u_hard_floor_biased = 0.981 ± 0.001 --> $\mathrm{Var}(u)
= 0.981$ (`reproductions/claude-giga-lens/papers/main.tex:428`,
`reproductions/claude-giga-lens/CAMPAIGN.md:622`–`623`) and failed the
whiteness gate outright. Letting the floor *adapt* down to
<!-- check: ch07.s_floor_v3b = 0.02 ± 0.001 --> $s_{\mathrm{floor}}=0.02$
for this product is what the adopted whitener actually uses, landing at
<!-- check: ch07.var_u_dense_v3b = 1.00003 ± 0.0005 --> $\mathrm{Var}(u)
\approx 1.00003$ — inside the gate. This is the same move
[Ch. 8](08-probability.md#ridge-is-a-prior) will name explicitly — ridge
regression is a Gaussian prior is a refusal to trust a small estimated
eigenvalue — applied here to a spectrum instead of a design matrix's Gram
matrix, at the identical price: a small, deliberate bias, bought in exchange
for not dividing by a number that measurement noise alone might have made
zero.

One more quantity from `build_whitener` is worth banking now and cashing
later: its `logdet_per_pix` output is $\overline{\log S}$, the
frequency-average of the log-spectrum
(<!-- check: ch07.logdet_per_pix_v3b = -0.973 ± 0.005 --> $-0.973$ for
`v3b`). By the Szegő limit theorem this average equals $\lim_{n\to\infty}
\log\det(K)/n$ — so this one FFT-derived number *is*, per pixel, the
$\log\det$ that [Ch. 22](22-inference.md#the-occam-term)'s Occam term
(`reproductions/claude-giga-lens/cgl/marg.py:19`) needs for a covariance
built this way. This guide has been
running a ledger of $\log|\det(\cdot)|$ showing up in unrelated costumes
since [Ch. 4](04-multivariable.md#the-log-det-ledger); here it shows up as
the *average of a Fourier transform*.

## Connect to the repo { #connect }

- [`reproductions/claude-giga-lens/cgl/whiten.py`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/whiten.py)
  is this entire chapter, executable: `make_conv_whitener` (line `39`) is the
  local, differentiable convolution operator; `build_whitener` (line `95`) is
  the spectral construction $\eqref{eq:whitener}$, the adaptive floor
  $\eqref{eq:floor}$ (lines `136`–`138`), and the $e_{\mathrm{op}}$ gate
  $\eqref{eq:eop-gate}$ (`e_op_of`, lines `147`–`149`).
- [`reproductions/claude-giga-lens/tests/test_marg.py:83`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/tests/test_marg.py#L83)
  locks the convolution-vs-correlation semantics against `scipy.ndimage.correlate`
  with a deliberately asymmetric kernel.
- [`reproductions/foundry-i/46_noise_audit.py:89`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/foundry-i/46_noise_audit.py#L89)
  is the Wiener–Khinchin theorem, one line, computing an all-pairs
  autocorrelation without forming a pair.
- `reproductions/claude-giga-lens/data/noise_kernel_report.json` and
  `reproductions/claude-giga-lens/data/whitener_report.json` hold this
  chapter's worked numbers — the same files
  `site/guide_src/worked_examples.py`'s `ch07_fourier_whitening` reads.
- [Methods I](../current/claude-giga-lens/index.md#sec:methods1) and its
  [Convolutional whitening](../current/claude-giga-lens/index.md#sec:whiten)
  subsection are the report prose this chapter exists to make readable.
- [Ch. 11](11-observation.md#why-drizzle-correlates-noise) explains *where*
  the correlation this chapter treats as given data comes from (the drizzle
  kernel); [Ch. 24](24-correlated-noise.md#the-covariance-model) is where
  $C=D^{1/2}KD^{1/2}$ becomes a full pixel likelihood.

## Exercises { #exercises }

??? question "Exercise 7.1 — Why the whitening kernel is forced symmetric"
    Using the time-reversal identity $(f\star g)[n]=(f*\tilde g)[n]$ from
    this chapter, prove that convolution and cross-correlation by a kernel
    $g$ agree for *every* input $f$ if and only if $g[-m]=g[m]$ (an even
    kernel). Then explain, in one sentence, why `build_whitener` bothers to
    symmetrize its taps
    (`reproductions/claude-giga-lens/cgl/whiten.py:145`) even though the
    operator that ultimately applies them (`make_conv_whitener`) implements
    correlation, not convolution.

    ??? success "Solution"
        $(f\star g)[n] = (f*\tilde g)[n]$ for every $f$ iff $\tilde g = g$ as
        sequences (convolving against two different kernels agrees on every
        input iff the kernels are identical — take $f$ to be a unit impulse
        to see this directly). $\tilde g = g$ means $g[(-m)\bmod N] = g[m]$
        for all $m$: exactly the evenness condition. Symmetrizing the taps
        means the distinction this chapter opened with stops mattering for
        *this* kernel — correlating and convolving by it produce the same
        result — so the code no longer has to reason about which orientation
        it is implicitly using; the test at
        `reproductions/claude-giga-lens/tests/test_marg.py:83` still
        exists to catch anyone who ever calls `make_conv_whitener` with an
        asymmetric kernel and forgets that the underlying operator is
        correlation.

??? question "Exercise 7.2 — The general correlation theorem"
    Section [Power spectra and autocorrelation](#psd-and-autocorrelation)
    derived $\widehat{x\star x}[k] = |\hat x[k]|^2$ by substituting $p=m+n$
    into the autocorrelation's own definition. Repeat that substitution for
    two *different* sequences, $(f\star g)[n]=\sum_m f[m]g[m+n]$, and show
    that $\widehat{f\star g}[k] = \overline{\hat f[k]}\,\hat g[k]$.

    ??? success "Solution"
        $\widehat{f\star g}[k] = \sum_n\sum_m f[m]g[m+n]\,e^{-2\pi ikn/N}$.
        Substitute $p=m+n$, so $n=p-m$:
        $\sum_m\sum_p f[m]g[p]\,e^{-2\pi ik(p-m)/N} = \big(\sum_m
        f[m]e^{2\pi ikm/N}\big)\big(\sum_p g[p]e^{-2\pi ikp/N}\big) =
        \overline{\hat f[k]}\,\hat g[k]$, using that $f$ is real so
        conjugating its transform flips the exponent's sign. Setting $g=f$
        recovers $\eqref{eq:wk}$ exactly, since $\overline{\hat f}\hat f =
        |\hat f|^2$.

??? question "Exercise 7.3 — A toy spectrum, and why the fine product needs such an aggressive floor"
    Model the fine product's along-axis correlation as a first-order AR
    process, $\rho(\Delta) = r^{|\Delta|}$, with $r=0.8147$ the measured
    lag-1 value <!-- check: ch07.rho1_v3_fine = 0.8147 ± 0.001 -->. Its power
    spectrum is the geometric-series closed form $S(\omega) =
    (1-r^2)/(1-2r\cos\omega+r^2)$. Evaluate $S(0)$ (DC) and $S(\pi)$
    (Nyquist), and compute their ratio.

    ??? success "Solution"
        At $\omega=0$: $S(0)=(1-r^2)/(1-r)^2=(1+r)/(1-r)$
        <!-- check: ch07.ar1_s0_v3 = 9.792 ± 0.01 --> $\approx 9.79$. At
        $\omega=\pi$: $S(\pi)=(1-r^2)/(1+r)^2=(1-r)/(1+r)$
        <!-- check: ch07.ar1_spi_v3 = 0.1021 ± 0.001 --> $\approx 0.102$. The
        ratio is
        <!-- check: ch07.ar1_s0_over_spi_v3 = 95.87 ± 0.05 --> $\approx 96$:
        the DC power outweighs the Nyquist power by nearly two orders of
        magnitude. A flat $5\%$-of-mean floor sits far more aggressively
        relative to the *low* end of a spectrum this lopsided than it would
        for a milder correlation — which is exactly why the fine product
        (the most correlated of the three, [Ch. 11](11-observation.md#drizzle))
        is also the one whose whitener needed the largest stencil,
        <!-- check: ch07.m_v3 = 20 ± 0 --> $M=20$, to hold $e_{\mathrm{op}}$
        under gate.

??? question "Exercise 7.4 — Redo the $N_{\mathrm{eff}}$ estimate for the binned product"
    `reproductions/claude-giga-lens/data/noise_kernel_report.json`'s binned (`v3b`) measured along-axis
    autocorrelation, out to its own $7$-lag cutoff, is $\rho(\Delta) =
    1.000, 0.615, 0.268, 0.193, 0.172, 0.143, 0.107, 0.080$ for
    $\Delta=0,\dots,7$. Compute $\tau_{\mathrm{int}}$ and $N_{\mathrm{eff}}/N$
    (1-D, along one axis) exactly as this chapter did for the fine product,
    and explain in one sentence why the binned number should come out so
    much less extreme than the fine one.

    ??? success "Solution"
        $\tau_{\mathrm{int}} = 1+2(0.615+0.268+0.193+0.172+0.143+0.107+0.080)$
        <!-- check: ch07.tau_int_v3b_7lag = 4.156 ± 0.01 --> $\approx 4.16$,
        so $N_{\mathrm{eff}}/N \approx$
        <!-- check: ch07.n_eff_over_n_1d_v3b = 0.2406 ± 0.001 --> $0.241$ —
        nearly twice the fine product's $0.134$. A $2\times2$ rebinning of
        the fine mosaic averages together exactly the pixels that were most
        correlated with each other in the first place, so the correlation
        length measured in *binned* pixels shrinks even as the correlation
        length measured in *sky* angle does not: binning does not create
        information, but it does stop double-counting the information the
        fine grid never had.
