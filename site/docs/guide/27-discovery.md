# Finding lenses: the survey, the nets, and the resolution wall

Every chapter in Parts IV and V started from a lens already on someone's desk:
a cutout, a redshift pair, a PSF file. This chapter is about how it got there.
The DESI Legacy Imaging Surveys photographed most of the extragalactic sky in
three colors, and this repo's own finder lineage has scored tens of millions
of those galaxies by machine, at a scale no human team could inspect by eye.
Two questions decide everything that follows from a candidate list: how good
does a picture have to be before a ring is even visible in it, and once a
network has ranked 53.8 million galaxies by lens-likeness, what number do you
actually read off that ranking to decide who gets looked at next? The first
question has a closed-form answer, derivable from nothing but the survey's
own point-spread function. The second is answered, expensively, by this
repo's own deployment history — where a $105\times$ parameter increase bought
less than a thousandth of an AUC point, and a change to how five scores get
combined, at *fixed* AUC, moved recovered candidates by 34 percentage points.

!!! abstract "What you can skip"
    You already own AUC, ROC curves, true/false positive rate, threshold
    selection, why accuracy is the wrong metric under class imbalance,
    ResNets, EfficientNets, ensembling and stacking, and what "training-set
    leakage" means — standard ML practice, and this chapter never re-explains
    it. What is new: what a ground-based survey's pixel scale and seeing
    actually do to a candidate *together*, at the scale a real deployment
    operates at — and why "same architecture family, 100x the parameters"
    behaves nothing like it does on a typical vision benchmark.

## The survey { #the-survey }

The **DESI Legacy Imaging Surveys** photograph the sky in three optical
bands — $g$, $r$, $z$ (hence "grz") — combining exposures from DECam in the
south and the 90Prime/Mosaic-3 cameras (BASS/MzLS) in the north, reduced by
the Tractor pipeline into public per-object catalogs and coadded images,
released in numbered Data Releases (DR7 through DR11 across this repo's own
finder lineage). [Ch. 9](09-units.md#pixel-scale) already fixed the two
numbers that decide everything in this chapter: a native pixel scale of
$0.262''$/px
<!-- check: ch27.pixel_scale_arcsec = 0.262 ± 0.0001 -->
and a $g$-band coadd point-spread function measured, not assumed, at
$\mathrm{FWHM}\approx1.35''$
(`reproductions/cikota-2023/README.md:38`).

What every one of this repo's finders actually consumes is a small array
built from that imaging. `tools/desi-dr11-cookbook/dr11_collect.py` writes
one $(101,101,3)$ cutout per surviving object — $g$, $r$, $z$ stacked as
channels, raw nanomaggies, no normalization applied at storage time — and
`huang-2020`'s finder pipeline fetches the identical shape directly from the
public cutout service
(`reproductions/huang-2020/papers/main.tex:138`). At the survey's own pixel
scale that is a

$$
101 \times 0.262'' = 26.462''
$$

<!-- check: ch27.cutout_side_arcsec = 26.462 ± 0.001 -->
field of view on a side — every finder in this book's lineage sees the same
patch of sky, at the same resolution, every time. "Surviving" means five
filters, applied in this exact order in the cookbook's own reimplementation
of the release pipeline: primary detection, positive flux in all three
bands, at least three exposures per band, $z$-band magnitude under 20, and a
Tractor light-profile type in $\{\text{SER},\text{EXP},\text{DEV},\text{REX}\}$
(`tools/desi-dr11-cookbook/README.md:52-57`). (Even the array layout has a
real gotcha: each stored plane is transposed relative to the sky image — a
quirk of the original pipeline the cookbook reproduces bit-for-bit rather
than silently "fixing," precisely so newly generated bricks stay drop-in
compatible with the existing dataset;
`tools/desi-dr11-cookbook/README.md:41-50`.)

Applied to the full DR11-south footprint, those five filters leave a parent
sample of

$$
53{,}809{,}040
$$

<!-- check: ch27.n_parent_galaxies = 53809040 ± 0 -->
galaxies (`reproductions/dr11-campaign/papers/main.tex:93`) — the same
published cuts Huang 2020/2021 and Inchausti 2025 used at smaller data
releases, now run against the deepest DECam reduction to date. That number,
rounded to $53.8$M, is the denominator behind every false-positive-rate
arithmetic later in this chapter, and it is the reason a survey-scale finder
is not a bigger version of a Kaggle classifier: the operating point has to be
chosen against tens of millions of negatives, not a held-out test split.

## Deriving the wall { #deriving-the-wall }

!!! tip "You already know this"
    The test below — does a critical point sit at a maximum or a minimum? —
    is exactly [Ch. 5](05-linear-algebra.md#definiteness-and-saddles)'s
    definiteness test, run on a single scalar function of one variable
    instead of a Hessian. A second derivative changing sign is a saddle test
    in miniature, and it is what "resolved" and "unresolved" formally mean.

[Ch. 11](11-observation.md#ccds-and-psf) established that no image shows you
the sky; it shows you the sky convolved with a point-spread function, and for
ground-based imaging that function's width is the **seeing**. Ch. 9 flagged,
without proof, that DESI's $1.35''$ seeing disk is "comparable to or larger
than a typical Einstein radius." Here is the derivation that makes that
precise.

Model the two brightest images of a lensed arc — two points on opposite sides
of an Einstein ring, say — as two equal point sources separated by a distance
$d$, each blurred by the same seeing disk. Approximate that disk as a
Gaussian of standard deviation $\sigma$, $G_\sigma(x) =
\exp(-x^2/2\sigma^2)$, and look at the combined brightness profile along the
line joining them:

$$
f(x) = G_\sigma(x-a) + G_\sigma(x+a), \qquad a = d/2.
$$

By symmetry, $f'(0)=0$ always — there is always a critical point exactly
halfway between the two sources. What that critical point *is* — a single
merged peak, or a dip between two separate humps — is decided by the sign of
$f''(0)$, precisely the one-dimensional case of Ch. 5's test. Differentiating
twice and evaluating at $x=0$:

$$
f''(0) = \frac{2}{\sigma^2}\left(\frac{a^2}{\sigma^2}-1\right)G_\sigma(a).
$$

Since $G_\sigma(a) > 0$ always, the sign of $f''(0)$ is the sign of
$a^2/\sigma^2 - 1$. For $a<\sigma$, $f''(0)<0$: a single local maximum sits
at the midpoint, and the two sources have merged into one blob — the
convolution has done to two nearby point sources exactly what a low-pass
filter does to two nearby high-frequency features. For $a>\sigma$,
$f''(0)>0$: the midpoint is a local *minimum*, a real dip separates two
distinguishable humps, and the pair is resolved. The boundary is

$$
d_{\min} = 2\sigma.
$$

This is the Sparrow resolution limit, derived rather than looked up: two
point sources under a Gaussian PSF stop being resolvable once their
separation drops below twice the PSF's own standard deviation.

Converting the survey's seeing from a FWHM to a $\sigma$ needs one more line —
$\mathrm{FWHM} = 2\sqrt{2\ln 2}\,\sigma \approx 2.3548\,\sigma$, derived from
scratch in Exercise 27.1 — which gives, at DESI's own measured seeing,

$$
\sigma = \frac{1.35''}{2.3548} \approx 0.573''
$$

<!-- check: ch27.sigma_arcsec = 0.5733 ± 0.001 -->
and therefore $d_{\min} \approx 1.147''$
<!-- check: ch27.d_min_arcsec = 1.1466 ± 0.001 -->. Two opposite images of a
ring of Einstein radius $\theta_{\mathrm{E}}$ sit a full diameter apart,
$d = 2\theta_{\mathrm{E}}$, so the wall translates directly into a floor on
$\theta_{\mathrm{E}}$ itself:

$$
\theta_{\mathrm{E},\min} = \frac{d_{\min}}{2} = \sigma \approx 0.573''.
$$

<!-- check: ch27.theta_e_wall_arcsec = 0.5733 ± 0.001 -->
The factor of two in "diameter" and the factor of two in "$d_{\min}=2\sigma$"
cancel exactly, so the resolution floor on an Einstein radius is just the
PSF's own $\sigma$ — a clean result, not a coincidence of this particular
algebra.

Now check it against numbers this book has already computed. A fiducial
massive elliptical — $\sigma_v = 250$ km/s at $z_l=0.5$, $z_s=2.0$, the same
system [Ch. 19](19-einstein-radius.md#theta-e-from-sigma-v) used — has
$\theta_{\mathrm{E}} = 1.145''$
<!-- check: ch19.theta_e_typical = 1.145 ± 0.001 -->, only

$$
\frac{1.145''}{0.573''} \approx 2.0
$$

<!-- check: ch27.ring_diameter_over_wall_typical = 1.9977 ± 0.001 -->
times the floor. A *typical* strong lens clears this wall by barely a factor
of two — not a comfortable margin for a survey whose every pixel already
carries correlated noise ([Ch. 11](11-observation.md#why-drizzle-correlates-noise))
and a bright foreground galaxy's own light.

The repo's own real system pressure-tests the idealization. `cikota-2023`
fits DESI-253.2534+26.8843 — the Einstein cross from
[Ch. 19](19-einstein-radius.md#theta-e-from-sigma-v) — on this exact
survey's blended $1.35''$-seeing pixels and recovers
$\theta_{\mathrm{E}} = 2.103''$
<!-- check: ch27.theta_e_cikota_imaging_fit = 2.103 ± 0.005 -->,
comfortably above the wall — a ratio of

$$
\frac{2 \times 2.103''}{1.147''} \approx 3.67
$$

<!-- check: ch27.ring_diameter_over_wall_cikota_imaging = 3.6683 ± 0.001 -->
— against the paper's own sharper, $0.6''$-seeing MUSE imaging, which gives
$\theta_{\mathrm{E}} = 2.520''$
<!-- check: ch27.theta_e_cikota_published = 2.520 ± 0.005 -->
(`reproductions/cikota-2023/papers/main.tex:41-42`). By the idealized
two-point criterion, this system should be easily resolved. And yet: "All
four images of the cross are visible, though partially blended at $1.35''$
seeing" (`reproductions/cikota-2023/papers/main.tex:126-127`). The
idealization is optimistic in two ways real data adds back: four images of a
quad are not evenly spaced around one circle of diameter $2\theta_{\mathrm{E}}$
— some pairs sit much closer than that — and they are not equal-brightness
points on a dark background but faint arcs sitting next to a much brighter
foreground galaxy's own Sérsic wings ([Ch. 10](10-galaxies.md#the-sersic-profile)).
The derived wall is a *lower bound* on what real detection needs, not the
whole story. Seeing does not only decide whether a ring is visible, either:
re-fitting the identical Legacy pixels with the paper's sharper $0.6''$ PSF
instead of the true $1.35''$ one moves the recovered radius from $2.103''$ to
$2.276''$
<!-- check: ch27.theta_e_cikota_ablation_06psf = 2.276 ± 0.005 -->,
closing only about half the gap to $2.520''$
(`reproductions/cikota-2023/README.md:39-41`) — the same seeing that hides a
ring also biases how big you measure it once you find it.

The wall's location is confirmed, empirically and independently, from
elsewhere in this repo. Of 17 DESI grade-C candidates the Euclid Q1 Discovery
Engine happened to re-observe at $0.1''$ resolution, 6
<!-- check: ch27.lensjudge_gradeC_to_gradeA_num = 6 ± 0 -->
jump all the way to grade A
(`reproductions/lensjudge/papers/main.tex:571-575`) — not because the object
changed, but because the wall moved: rerunning this exact derivation at
Euclid's sharper $\mathrm{FWHM}=0.1''$ drops the floor to
$\theta_{\mathrm{E},\min}\approx0.042''$
<!-- check: ch27.theta_e_wall_euclid_arcsec = 0.0425 ± 0.001 -->
(worked in Exercise 27.2) — below essentially every strong lens this book
quotes a number for. "C" was never a verdict about the source. It was a
statement about the pixels, and [Ch. 28](28-the-label.md#same-wall) shows it
is the *same* statement that makes a human grader's "C" unreliable too.

## The finders { #the-finders }

!!! tip "You already know this"
    A network that cannot improve past a resolution-set information ceiling,
    no matter how many parameters you give it, is the textbook signature of a
    **data-limited** regime rather than a compute-limited one — the same
    diagnosis that tells you when a bigger model, not more or better data,
    is the wrong lever to pull.

This repo's own finder lineage ran the parameter-count experiment for real,
across three papers. Huang 2020 re-implemented Lanusse 2018's ResNet-46
(`L18`) at $3{,}508{,}833$ parameters
<!-- check: ch27.params_l18 = 3508833 ± 0 -->. Huang 2021's controlled
comparison — same cutouts, positives, negatives, seed and split, architecture
the *only* variable — trained a "shielded" ResNet at $59{,}905$ parameters
<!-- check: ch27.params_shielded_60k = 59905 ± 0 -->, a

$$
\frac{3{,}508{,}833}{59{,}905} \approx 58.6\times
$$

<!-- check: ch27.param_ratio_l18_over_60k = 58.5733 ± 0.001 -->
reduction, and it did not lose accuracy: validation AUC actually rose slightly,
from $0.9983$ to $0.9989$
<!-- check: ch27.auc_l18_dr9_val = 0.9983 ± 0.0001 -->
<!-- check: ch27.auc_shielded60k_dr9_val = 0.9989 ± 0.0001 -->
(`reproductions/huang-2021/README.md:28-37`), a gap of
<!-- check: ch27.auc_gap_l18_shielded60k = 0.0006 ± 0.0001 -->
$0.0006$ in the *smaller* net's favor. Inchausti 2025 pushed the same
controlled comparison an order of magnitude further: a $194{,}501$-parameter
re-tuning of that shielded ResNet
<!-- check: ch27.params_shielded_194k = 194501 ± 0 -->
against an EfficientNetV2-S backbone at $20{,}543{,}145$ parameters
<!-- check: ch27.params_effnet = 20543145 ± 0 -->,

$$
\frac{20{,}543{,}145}{194{,}501} \approx 105.6\times
$$

<!-- check: ch27.param_ratio_effnet_over_shielded194k = 105.6197 ± 0.001 -->
more parameters, for the paper's own reported validation-AUC gain of
<!-- check: ch27.auc_gap_resnet_effnet_paper = 0.0003 ± 0.0001 -->
$0.9987-0.9984=0.0003$
(`reproductions/inchausti-2025/papers/main.tex:170-179`); stacking both
models through a 300-node meta-learner edges the shielded ResNet alone by
<!-- check: ch27.auc_gap_resnet_meta_paper = 0.0005 ± 0.0001 -->
$0.0005$.

| Model | Params | Val AUC | vs. the row above |
|---|---:|---:|---:|
| Lanusse-2018 ResNet-46 (Huang 2020) | 3,508,833 | 0.9983 | — |
| Shielded ResNet (Huang 2021) | 59,905 | 0.9989 | 58.6$\times$ fewer, $+0.0006$ AUC |
| Shielded ResNet, re-tuned (Inchausti 2025) | 194,501 | 0.9984 | baseline for the next row |
| EfficientNetV2-S (Inchausti 2025) | 20,543,145 | 0.9987 | 105.6$\times$ more, $+0.0003$ AUC |
| Stacking meta-learner (Inchausti 2025) | 1,201 | 0.9989 | $+0.0005$ AUC over 194K ResNet |

Read across the whole lineage, and ClaudeNet's own premise document states
the finding flatly: "at the deployment operating point, architecture is not
the bottleneck. A 194 K-param shielded ResNet ≈ a 20.5 M-param
EfficientNetV2-S within ±0.003 AUC" (`reproductions/claudenet/README.md:11-13`).
Every gap traced above — $0.0003$, $0.0005$, $0.0006$ — sits comfortably
inside that bound, by a factor of five to ten. Every one of these nets trains
on the same leakage-inflated positives, evaluated on the same easy validation
split, drawn from a survey whose seeing sets a hard information ceiling on
what a $101\times101$ cutout can say about a ring that thin (the previous
section, not a metaphor for it). Once the *pixels* cannot distinguish "faint
blue arc" from "faint blue smudge," no amount of extra network capacity
recovers information the input never carried — the data-limited regime this
chapter's tip box named.

## The operating point { #the-operating-point }

At $53{,}809{,}040$ galaxies, even an outstanding AUC produces an unusable
candidate list unless you also fix *where* on the ROC curve you deploy. A
finder run at a $1\%$ false-positive rate flags

$$
53{,}809{,}040 \times 0.01 \approx 538{,}090
$$

<!-- check: ch27.candidates_at_1pct_fpr = 538090.4 ± 1 -->
galaxies; even at $0.1\%$ FPR, still $53{,}809$
<!-- check: ch27.candidates_at_0_1pct_fpr = 53809.04 ± 1 -->. No
visual-inspection pipeline, human or agentic, clears either list in a
research career, so a real deployment always operates far out on the
low-FPR tail — exactly where AUC, an integral over the *entire* curve, is
least informative about the one point that matters.

The repo's own DR11-south sweep makes this concrete without touching
architecture at all. Five finder members score the same $53.8$M-galaxy
parent sample with the same trained weights; ranking their calibrated mean
against 2M random DR11 galaxies gives a threshold-free AUC of
$0.9955$
<!-- check: ch27.auc_mean_combiner = 0.9955 ± 0.0001 -->
(`reproductions/dr11-campaign/papers/main.tex:119-121`) — the members
separate real lenses from random galaxies almost perfectly. Yet at a fixed
candidate budget of $95{,}104$
<!-- check: ch27.n_survivor_budget_union = 95104 ± 0 -->
survivors — about
$95{,}104/53{,}809{,}040 \approx 0.18\%$
<!-- check: ch27.fpr_at_95k_budget = 0.001767 ± 0.00001 -->
of the parent sample — the *original* per-member union selector recovers
only $54\%$
<!-- check: ch27.recall_union_95k = 0.54 ± 0.0 -->
of held-out grade-A lenses, while swapping to a calibrated-*mean* combiner,
at the *identical* budget, recovers $75\%$
<!-- check: ch27.recall_mean_95k = 0.75 ± 0.0 -->; widening that same
mean-ranked budget to $150{,}000$
<!-- check: ch27.n_survivor_budget_mean = 150000 ± 0 -->
($\approx0.28\%$ FPR
<!-- check: ch27.fpr_at_150k_budget = 0.002788 ± 0.00001 -->)
reaches $80\%$
<!-- check: ch27.recall_mean_150k = 0.80 ± 0.0 -->, and $88\%$
<!-- check: ch27.recall_mean_heldout = 0.88 ± 0.0 -->
on the held-out subset
(`reproductions/dr11-campaign/papers/main.tex:124-138`). Same trained
weights. Same scores. Same AUC. A 34-percentage-point recall swing from
nothing but how five numbers get combined into one rank.

A second, independent confirmation comes from Inchausti 2025's own
negative-sampling ablation. The meta-learner's AUC barely moves across two
retraining stages — $0.9876$
<!-- check: ch27.auc_meta_stageC = 0.9876 ± 0.0001 -->
then $0.9919$
<!-- check: ch27.auc_meta_stageD = 0.9919 ± 0.0001 -->
— while recovery of the two published catalogs at a matched $1\%$ FPR moves
from
<!-- check: ch27.recovery_storfer_stageB_1pct = 0.118 ± 0.0 -->
$11.8\%$/
<!-- check: ch27.recovery_inchausti_stageB_1pct = 0.191 ± 0.0 -->
$19.1\%$ to
<!-- check: ch27.recovery_storfer_stageC_1pct = 0.836 ± 0.0 -->
$83.6\%$/
<!-- check: ch27.recovery_inchausti_stageC_1pct = 0.885 ± 0.0 -->
$88.5\%$ to
<!-- check: ch27.recovery_storfer_stageD_1pct = 0.908 ± 0.0 -->
$90.8\%$/
<!-- check: ch27.recovery_inchausti_stageD_1pct = 0.968 ± 0.0 -->
$96.8\%$ (Storfer/Inchausti catalogs respectively;
`reproductions/inchausti-2025/README.md:150-163`). What moved was not the
network; it was how many, and how realistic, the training negatives were.

Both stories teach the same lesson from opposite ends. Architecture buys
thousandths of an AUC point (the previous section). The **operating
point** — the combiner, the negative-sample composition, the threshold you
actually deploy at — buys tens of percentage points of recall, at *fixed*
AUC. At 53.8 million galaxies, the operating point is not one metric among
several worth reporting. It is the only one a real deployment decision can
be made from.

## Connect to the repo { #connect }

- `tools/desi-dr11-cookbook/dr11_collect.py` and `tools/desi-dr11-cookbook/README.md:20-64`
  — the DR11 → HDF5 cutout pipeline: the $(101,101,3)$ grz array, the
  five-filter object selection, and the transpose storage convention.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/tools/desi-dr11-cookbook/dr11_collect.py)
- `reproductions/huang-2020/03_download_decals_cutouts.py` and
  `reproductions/huang-2020/papers/main.tex:138` — the $101\times101$ grz
  cutout / $0.262''$ pixel-scale fact this whole lineage inherits.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/huang-2020/03_download_decals_cutouts.py)
- `reproductions/huang-2021/01b_shielded_resnet.py` and
  `reproductions/huang-2021/README.md:9-45` — the shielding architecture and
  its controlled, same-data AUC comparison against L18.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/huang-2021/01b_shielded_resnet.py)
- `reproductions/inchausti-2025/08_compare_models.py` and
  `reproductions/inchausti-2025/papers/main.tex:153-186` — the three-model
  params/AUC table (`tab:auc`) this chapter's finder lineage is built from.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/inchausti-2025/08_compare_models.py)
- `reproductions/inchausti-2025/22_fpr_operating_point.py` and
  `reproductions/inchausti-2025/README.md:126-163` — the Stage-C/D
  matched-false-positive-rate recovery arithmetic.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/inchausti-2025/22_fpr_operating_point.py)
- `reproductions/claudenet/README.md:9-21` — "architecture is not the
  bottleneck," the premise this chapter's finder table confirms in detail.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claudenet/README.md)
- `reproductions/dr11-campaign/papers/main.tex:90-141` — the DR11-south
  sweep and the union-vs-mean-combiner recall-recoverability story.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/dr11-campaign/papers/main.tex)
- `reproductions/cikota-2023/README.md:36-43` and
  `reproductions/cikota-2023/papers/main.tex:41-51,264-284` — the PSF/seeing
  ablation on a real, four-image system.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/cikota-2023/papers/main.tex#L264)
- `reproductions/lensjudge/papers/main.tex:562-596` — the DESI-vs-Euclid
  grade-C flip, the empirical confirmation of this chapter's derived wall.
  [(github)](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/papers/main.tex#L571)

## Exercises { #exercises }

??? question "Exercise 27.1 — the FWHM-to-$\sigma$ conversion, from scratch"
    A Gaussian $G_\sigma(x) = \exp(-x^2/2\sigma^2)$ has full width at half
    maximum $\mathrm{FWHM}$ defined by $G_\sigma(\mathrm{FWHM}/2) = 1/2$.
    Solve for $\mathrm{FWHM}$ in terms of $\sigma$, and evaluate the
    constant $\mathrm{FWHM}/\sigma$ to four decimal places.

    ??? success "Solution"

        $$
        \exp\!\left(-\frac{(\mathrm{FWHM}/2)^2}{2\sigma^2}\right) = \frac12
        \;\;\Longrightarrow\;\;
        \frac{(\mathrm{FWHM}/2)^2}{2\sigma^2} = \ln 2
        \;\;\Longrightarrow\;\;
        \mathrm{FWHM} = 2\sigma\sqrt{2\ln 2}.
        $$

        Numerically, $2\sqrt{2\ln 2} \approx 2.3548$
        <!-- check: ch27.fwhm_to_sigma_factor = 2.3548 ± 0.0001 -->,
        the constant used throughout [Deriving the wall](#deriving-the-wall)
        to convert the survey's quoted seeing (a FWHM, the astronomer's
        usual unit) into the $\sigma$ the second-derivative test needs.

??? question "Exercise 27.2 — rerun the wall at Euclid resolution"
    [Ch. 28](28-the-label.md#same-wall) reports that Euclid Q1 imaging
    resolves lenses DESI could not. Repeat this chapter's derivation with
    $\mathrm{FWHM} = 0.1''$ (Euclid VIS) instead of DESI's $1.35''$, and find
    the resolution floor $\theta_{\mathrm{E},\min}$. How does it compare to
    the $\theta_{\mathrm{E}} \gtrsim 1''$ scale of every lens this book has
    quoted a number for?

    ??? success "Solution"

        $$
        \sigma_{\mathrm{Euclid}} = \frac{0.1''}{2.3548} \approx 0.0425'',
        \qquad
        \theta_{\mathrm{E},\min} = \sigma_{\mathrm{Euclid}} \approx 0.0425''
        $$

        <!-- check: ch27.theta_e_wall_euclid_arcsec = 0.0425 ± 0.001 -->
        — $13.5\times$
        <!-- check: ch27.wall_ratio_desi_over_euclid = 13.5 ± 0.01 -->
        sharper than DESI's own $0.573''$ floor (the ratio of the two seeing
        FWHMs, since the wall is linear in $\sigma$), and about $27\times$
        <!-- check: ch27.typical_over_euclid_wall = 26.97 ± 0.01 -->
        *smaller* than the $1.145''$ fiducial Einstein radius this chapter
        derived the DESI wall against. At Euclid resolution the idealized wall essentially
        vanishes: every real lens sits far above it, which is exactly why 6
        of 17 DESI grade-C candidates
        <!-- check: ch27.lensjudge_gradeC_to_gradeA_num = 6 ± 0 -->
        jumped to grade A once regraded on Euclid imaging — the object never
        changed, only which side of the wall its pixels sat on.

??? question "Exercise 27.3 — candidate counts at three operating points"
    Using the DR11-south parent sample of $53{,}809{,}040$ galaxies, compute
    the number of flagged candidates at $1\%$, $0.1\%$, and $0.01\%$ false
    positive rate. At roughly how many candidates per day would a team need
    to grade to clear the $0.01\%$-FPR list within a single calendar year?

    ??? success "Solution"

        $$
        53{,}809{,}040 \times \{0.01,\ 0.001,\ 0.0001\}
        \approx \{538{,}090,\ 53{,}809,\ 5{,}381\}
        $$

        <!-- check: ch27.candidates_at_1pct_fpr = 538090.4 ± 1 -->
        <!-- check: ch27.candidates_at_0_1pct_fpr = 53809.04 ± 1 -->
        <!-- check: ch27.candidates_at_0_01pct_fpr = 5380.904 ± 1 -->
        Even the smallest of the three, $5{,}381$ candidates, needs about
        $5{,}381/365 \approx 15$ gradings a day, every day, for a year, to
        clear at the lowest operating point in this chapter — and that
        operating point is already $100\times$ more conservative than the
        $1\%$ FPR a naive threshold choice might reach for. This is the
        arithmetic reason [The operating point](#the-operating-point)
        insists that the threshold, not the AUC, is the number a deployment
        decision is made from.

??? question "Exercise 27.4 — verify the parameter ratios, and explain the flat AUC"
    Using $\text{params}_{\mathrm{L18}} = 3{,}508{,}833$,
    $\text{params}_{\mathrm{shielded,60K}} = 59{,}905$,
    $\text{params}_{\mathrm{shielded,194K}} = 194{,}501$, and
    $\text{params}_{\mathrm{EffNetV2\text{-}S}} = 20{,}543{,}145$, verify the
    $58.6\times$ and $105.6\times$ reduction/increase ratios quoted in
    [The finders](#the-finders). Then argue, from the resolution wall
    derived earlier in this chapter, why increasing parameter count by two
    orders of magnitude moved validation AUC by less than one part in a
    thousand in both directions.

    ??? success "Solution"

        $$
        \frac{3{,}508{,}833}{59{,}905} \approx 58.573,
        \qquad
        \frac{20{,}543{,}145}{194{,}501} \approx 105.620
        $$

        <!-- check: ch27.param_ratio_l18_over_60k = 58.5733 ± 0.001 -->
        <!-- check: ch27.param_ratio_effnet_over_shielded194k = 105.6197 ± 0.001 -->
        matching the table in [The finders](#the-finders) to four
        significant figures. The flat AUC is the data-limited regime made
        concrete: [Deriving the wall](#deriving-the-wall) showed that
        DESI's own seeing sets a hard floor on how much a $101\times101$
        cutout can say about a ring near that floor — a typical lens clears
        it by only about $2\times$. Once the *input* cannot distinguish a
        genuine tangential arc from a blended smudge of comparable size, no
        amount of additional network capacity can recover a distinction the
        pixels never encoded. Extra parameters can only fit patterns present
        in the training data; they cannot manufacture angular resolution the
        camera and the atmosphere never delivered.
