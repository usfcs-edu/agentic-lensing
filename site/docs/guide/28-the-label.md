# The label is the problem

You are reading this chapter second because it needs none of the calculus
Part I is about to build, and it carries the finding this guide treats as
the most consequential one for anyone who ships classifiers for a living.
[Chapter 1](01-orientation.md#the-two-halves) split this repository's work
into two disciplines — discovery, which decides whether a candidate is a
lens at all, and modeling, which measures one once confirmed — and
previewed discovery's central number: the human grade attached to a
candidate predicts later truth at AUC $0.577$, statistically
indistinguishable from a coin flip. This chapter earns that number rather
than quoting it, using only per-class precision and an ordinal agreement
statistic you can build from tools you already own, applied to every
follow-up outcome this repository's discovery lineage has published and to
three different machine systems tried against the same label.

!!! abstract "What you can skip"
    You already own precision, recall, AUC, and the confusion matrix — none
    of that gets rebuilt here. "Purity," as this program's own reports use
    the word, turns out to be exactly per-class precision, and the
    truth-referenced AUC comparisons are exactly the paired,
    non-inferiority-style comparison you would run if two model checkpoints
    differed only in a training-data source. What is genuinely new is
    quadratic-weighted kappa (QWK) — the standard way to score agreement on
    an *ordinal* label like a 1–4 lens grade — and this chapter builds it
    from the two-line "agreement beyond chance" idea behind Cohen's kappa, up
    to a number that matches this repository's own published statistic,
    entirely from data already sitting in its tables.

## The flat line { #the-flat-line }

Every discovery pipeline in this repository ends the same way: a candidate
gets a letter grade, usually A through D, meant to encode how confident the
grader — human or machine — is that the object is a genuine lens. A grade is
only useful if it is *informative*: among candidates that later got
independent follow-up (deeper imaging, a second redshift, spectroscopy), a
higher grade should correspond to a higher confirmed fraction.

!!! tip "You already know this"
    That confirmed fraction has a name you already use. Restrict a confusion
    matrix to the candidates predicted grade $g$, and ask what fraction of
    them are true positives (confirmed) rather than false positives
    (refuted) — that is precision, computed separately per predicted class.
    This program's reports call it "purity," but

    $$
    \text{purity}(g) \;=\; \frac{\#\{\text{confirmed}\mid\text{grade}=g\}}
    {\#\{\text{confirmed}\mid\text{grade}=g\} + \#\{\text{refuted}\mid\text{grade}=g\}}
    $$

    is precision restricted to grade $g$, nothing more.

If the grade carried real confidence information, purity should fall as the
grade drops — the way a classifier's precision falls as you relax its
decision threshold. Every published per-target follow-up outcome this
program could find — 551 targets across seven campaigns, cross-matched
against 4,354 uniquely graded candidates — gives 162 candidates with
follow-up, 130 with a decided (confirmed-or-refuted) outcome
(`reproductions/lensjudge/papers/human_baseline.tex:184`). Split by grade:

$$
\text{purity}(A) = \frac{79}{83} = 0.952,
\qquad
\text{purity}(B) = \frac{23}{25} = 0.920,
\qquad
\text{purity}(C) = \frac{20}{22} = 0.909.
$$

<!-- check: ch28.purity_A = 0.952 ± 0.001 -->
<!-- check: ch28.purity_B = 0.920 ± 0.001 -->
<!-- check: ch28.purity_C = 0.909 ± 0.001 -->

The spread from the best grade to the worst is $0.952 - 0.909 = 0.043$
<!-- check: ch28.purity_spread = 0.043 ± 0.001 -->
— four percentage points, well inside each grade's own sampling
uncertainty on a few dozen candidates. Grade A is not even flawless: four of
the 83 decided grade-A candidates were spectroscopically refuted by VLT/MUSE
follow-up (`reproductions/lensjudge/papers/human_baseline.tex:222`), every
one of them the same failure mode — an arc-shaped feature at the *wrong*
redshift: a spiral arm at the lens galaxy's own redshift, a tidal tail
between two merging galaxies, a blue arc that turns out to sit at the
deflector's redshift rather than behind it, and one "arc" that is a
foreground object entirely. A grade assigned from a $1.3''$-seeing cutout can
judge shape. It cannot see a redshift, and redshift is what actually decides
truth.

<figure markdown="span">
  ![Confirmation purity by human grade, with Jeffreys 95% intervals; the three grades are statistically indistinguishable](figures/ch28-flat-purity-light.svg#only-light){ width="90%" }
  ![Confirmation purity by human grade, with Jeffreys 95% intervals; the three grades are statistically indistinguishable](figures/ch28-flat-purity-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 28.1.** Confirmation purity among decided
  follow-ups, by consensus grade, with 95% Jeffreys intervals. The dotted
  line marks perfect purity. Grade A sits closest to it but is not on it — 4
  of 83 decided grade-A candidates were spectroscopically refuted — and all
  three grades' intervals overlap heavily. If the letter grade encoded
  confidence the way a classifier's score does, this line would slope down
  sharply from A to C. It is flat.</figcaption>
</figure>

Read the other direction and the same fact says something sharper: grade C
is not mostly junk. Twenty of its 22 decided candidates are real lenses —
purity indistinguishable from grade A's. Whatever "C" is measuring, it is not
"probably not a lens."

## The reliability ceiling { #the-reliability-ceiling }

Purity asks how well the grade predicts truth, conditional on a candidate
having been followed up at all. A prior question is sharper: how
reproducible is the grade *itself*? Show the same cutout to two trained
graders — would they even agree with each other?

Cohen's kappa answers this for a categorical label: let $p_o$ be the
observed fraction of exact agreement and $p_e$ the fraction two independent
graders would agree on by chance alone, given each grader's own marginal
distribution of scores. Agreement beyond chance is $(p_o-p_e)/(1-p_e)$. A
lens grade is not just categorical, though — it is *ordinal* (1 through 4,
or A through D), and a grader who says "3" when the other says "4" is closer
to right than one who says "1". Quadratic-weighted kappa (QWK) generalizes
Cohen's statistic to charge for *how far off* a disagreement is, not merely
whether it happened. Weight each possible pair of scores $(i,j)$ by squared
ordinal distance, $w_{ij} = (i-j)^2/(K-1)^2$ for $K$ score levels, and

$$
\mathrm{QWK} \;=\; 1 - \frac{\sum_{i,j} w_{ij}\,O_{ij}}{\sum_{i,j} w_{ij}\,E_{ij}},
$$

where $O$ is the observed joint distribution of the two graders' scores and
$E$ is the joint distribution you would see if the two graders' scores were
independent (the outer product of the marginals).

!!! tip "You already know this"
    That formula has a shape you have seen before. $R^2 = 1 -
    \mathrm{SS}_{\text{res}}/\mathrm{SS}_{\text{tot}}$ is a squared-error
    ratio against a null model's error; $\mathrm{QWK}$ is the identical
    ratio, built from a joint score distribution instead of a residual sum
    of squares, with "the null model" being *statistical independence
    between the two graders* rather than "predict the mean." It is exactly
    why leaderboards for ordinal-label problems (Kaggle's essay- and
    diagnosis-scoring competitions among them) use QWK as their metric: it
    behaves like $R^2$ for a discrete, ordered target.

**Worked example, built from the repository's own published numbers.**
`huang2021desi` (Paper II) had two graders independently score every
candidate 1–4; the public catalog keeps only the pair's *average* and
*absolute difference*, but that is enough to recover the unordered pair of
integer scores for 1,310 candidates. The full table of pair counts is
published as a flat list (`reproductions/lensjudge/parity/FINDINGS.md:30`):

| pair | count | pair | count | pair | count | pair | count |
|---|---|---|---|---|---|---|---|
| $\{2,2\}$ | 461 | $\{3,3\}$ | 165 | $\{4,4\}$ | 100 | $\{1,3\}$ | 48 |
| $\{2,3\}$ | 387 | $\{3,4\}$ | 115 | $\{2,4\}$ | 34 |  |  |

These sum to $1{,}310$ candidates
<!-- check: ch28.pair_n_candidates = 1310 ± 1 -->.
Grader identity was never published, so an unordered pair $\{a,b\}$ carries
no information about which grader gave which score; the standard fix is to
*symmetrize* — let each pair count once as $(a,b)$ and once as $(b,a)$ —
which is also why this statistic equals Krippendorff's interval $\alpha$,
the standard reliability measure when raters are interchangeable. Build the
resulting $4\times4$ joint table, apply the quadratic weights above, and
take the ratio:

$$
\mathrm{QWK} = 0.420
$$

<!-- check: ch28.qwk_derived_from_pairs = 0.420 ± 0.001 -->

built from nothing but the seven counts in the table above — and it matches
this program's own reported statistic
<!-- check: ch28.intergrader_qwk = 0.420 ± 0.001 --> to three decimal
places. The same seven counts, grouped by how far apart the two scores are,
also reproduce the raw agreement breakdown directly: exact agreement on
$726/1310 = 0.554$
<!-- check: ch28.pair_exact_frac = 0.554 ± 0.001 -->
of candidates, off by one on $0.383$
<!-- check: ch28.pair_one_step_frac = 0.383 ± 0.001 -->,
off by two — one grader ready to reject, the other confident enough to
publish — on $0.063$
<!-- check: ch28.pair_two_step_frac = 0.063 ± 0.001 -->.

Two trained graders, same rubric, same screen, same cutout, agree with each
other on the *exact* score barely more often than a coin flip, and QWK $=
0.42$ is "moderate" on the standard Landis–Koch scale — worse than the
$73\%$ intra-rater repeat consistency an independent study measured for a
single expert shown the same image twice
(`reproductions/lensjudge/papers/human_baseline.tex:125`): these two experts
disagree with each other more than one of them disagrees with their own past
self. There is a second, mechanically inflated anchor worth naming so you do
not mistake it for the real ceiling: an individual grader's score matches
the *published consensus* (the two-grader average) at QWK $=0.776$
<!-- check: ch28.grader_vs_consensus_qwk = 0.776 ± 0.001 -->
— but a grader who is literally half the number being compared against is
close to guaranteed a high score. The honest bar for "does a system
reproduce this team's grade" is the mutual $0.42$, not the self-referential
$0.776$.

This is a reliability ceiling in the same sense a noisy training label
imposes one on supervised learning: no model, however well built, can be
scored as "agreeing with the label" above what the label agrees with itself.
The ceiling is a property of the data-generating process, not of any
predictor's capacity.

## Nothing gets past the label { #nothing-gets-past-the-label }

This repository tried three structurally different machine systems against
exactly this task, and the pattern that emerges only sharpens the ceiling
argument. Distinguish two separate questions the way this program's own
methodology does: **E1**, does a machine's grade *agree with the human
label* (measured by QWK against the published consensus)? And **E2**, does a
machine's score *agree with truth* (measured by AUC against
confirmed-vs-refuted follow-up, paired against the human grade)?

On E1, nothing gets close. A 27-billion-parameter vision–language model,
fine-tuned directly on the human grade catalog — not distilled from another
model, trained on the actual letter grades a human team assigned — emits
grades that score QWK $=0.044$
<!-- check: ch28.student_qwk_vs_consensus = 0.044 ± 0.001 -->
against the published consensus, $n=162$. Prompt Claude Sonnet 5 with the
graders' own rubric and the same multi-band imaging context instead of
fine-tuning anything, and its frozen-gate grades score QWK $=0.025$
<!-- check: ch28.claude_matched_qwk_gate = 0.025 ± 0.001 -->.
Neither a purpose-trained model nor a frontier general model reproduces the
label a fraction as well as two trained humans reproduce each other.

A related, coarser contrast tells the same story from the vetting side: can
a machine at least separate human-graded A/B lenses from candidates humans
explicitly *rejected* (grade D)? Three unrelated architectures were run
against the identical frozen 259-row gate. A trained probe over classical,
engineered image features scores AUC $=0.425$
<!-- check: ch28.rep_probe_wall_auc = 0.425 ± 0.001 -->
— below chance. This repository's own published CNN ensemble scores
$0.646$
<!-- check: ch28.cnn_wall_auc = 0.646 ± 0.001 -->.
The fine-tuned 27B student, despite direct supervision on the human catalog,
scores $0.644$
<!-- check: ch28.student_wall_auc = 0.644 ± 0.001 -->
— statistically the same number as the published CNN, reached by a
completely different architecture and training recipe. Three tools built
three different ways converging on the same ceiling is itself evidence the
ceiling belongs to the pixels, not to any one model's limitations.

The twist that makes this sharper rather than softer: the same fine-tuned
student that cannot reproduce the letter grade discriminates *truth* on the
followed-up pool at AUC $=0.685$
<!-- check: ch28.student_truth_auc = 0.685 ± 0.001 -->
$[0.538, 0.819]$
<!-- check: ch28.student_truth_auc_lo = 0.538 ± 0.001 -->
<!-- check: ch28.student_truth_auc_hi = 0.819 ± 0.001 -->
— at least as well as the human grade's own $0.577$. Paired against the
human grade directly, the difference is $+0.108$
<!-- check: ch28.delta_auc_student_vs_human = 0.108 ± 0.001 -->
$[-0.028, +0.241]$
<!-- check: ch28.delta_auc_student_vs_human_lo = -0.028 ± 0.001 -->
<!-- check: ch28.delta_auc_student_vs_human_hi = 0.241 ± 0.001 -->
— formally *non-inferior* to the human grade at this program's pre-registered
margin, though the interval still touches zero, so *superiority* is not
established either. Read that pair of results together: a model can be bad
at imitating the label and simultaneously a legitimate — on this point
estimate, slightly ahead — detector of the thing the label was only ever
supposed to be a proxy for. The grade and the truth are not the same target,
and E1 tells you nothing at all about E2.

## The same wall { #same-wall }

[Chapter 27](27-discovery.md#deriving-the-wall) derives, from the survey's
own pixel scale, why a fixed resolution makes an Einstein ring's shape
disappear into the point-spread function past a certain radius — a limit no
architecture trains past, because the discriminating signal has been
convolved away before any classifier, however large, ever sees a pixel. This
chapter's ceiling is the identical physical fact, measured from the label
side instead of the classifier side.

The cleanest direct evidence: 17 candidates that DESI's own pipeline graded
C were independently rediscovered inside the Euclid Q1 survey's footprint
and re-scored there — not by DESI's $1.3''$-seeing grz imaging, but by a
$\sim 10$-expert panel working from Euclid's $0.1''$ VIS imaging
(`reproductions/lensjudge/papers/main.tex:571`). Of those 17 DESI grade-C
candidates, $53\%$
<!-- check: ch28.euclid_c_regrade_frac = 0.53 ± 0.001 -->
were re-graded A or B once the pixels sharpened, with 6 of the 17
<!-- check: ch28.euclid_c_to_a_count = 6 ± 0 -->
jumping all the way to grade A
<!-- check: ch28.euclid_c_regrade_n = 17 ± 0 -->
— about $6/17 \approx 0.353$
<!-- check: ch28.euclid_c_to_a_frac = 0.353 ± 0.001 -->
of the pool. Same object, same rubric, same sky. The grade moved because the
pixels changed, not because the lens changed.

That is the same wall Chapter 27 derives from first principles, seen from a
different instrument: a human grader staring at an under-resolved cutout
faces the identical information deficit a CNN's receptive field does. It is
why grade C's purity ($0.909$) sits statistically on top of grade A's
($0.952$) — "C" was never a verdict about the object, it was a statement
about the pixel budget available to render one. And it is why an engineered
probe, a published CNN, and a fine-tuned 27B model — three genuinely
different tools — all plateau in the same narrow band regardless of how they
are built: a classifier's ceiling is set by what information survives in its
inputs, not by its parameter count.

!!! tip "You already know this"
    This is the standard Bayes-error argument from supervised learning,
    applied to two different measuring instruments at once. If a feature map
    destroys the information a decision needs, no downstream model recovers
    it — that floor is a property of the representation, not of capacity.
    Chapter 27 computes that floor from the optics (a PSF convolution
    erasing a ring's shape); this chapter measures the *same* floor from the
    label side, by showing that two independent panels of experts, looking
    at the same lossy representation, cannot agree with each other about
    what they see there either.

## What to do instead { #what-to-do-instead }

Two different evaluation targets have been in play throughout this chapter,
and they behave nothing alike. Agreement-with-label (E1, QWK) has a hard
ceiling — $0.42$ here — set entirely by how reproducible the label is
between the humans who produced it; no predictor, however capable, can be
fairly scored above that ceiling, and training toward it optimizes a system
to imitate human noise. Agreement-with-truth (E2, AUC) has no such ceiling:
a model's score can legitimately match or exceed the human grade's own
truth-discrimination, exactly as the fine-tuned student's $0.685$ did
against the human grade's $0.577$.

The practical rule this sets, usable on day one: never evaluate — or train —
a vetting system by how well it reproduces a letter grade whose own
originators only agree with each other at $\mathrm{QWK}=0.42$. That target
caps your reported quality below what the underlying data can support and
rewards fitting human disagreement rather than physical truth. Build a
continuous score instead, calibrate it against held-out truth (confirmed
versus refuted), choose an operating point the same way any deployed
classifier's threshold gets chosen — against a validation ROC curve, not
against a categorical label someone else assigned — and validate with a
pre-registered, truth-referenced comparison against the human baseline
rather than an agreement statistic. This is precisely this program's own
stated conclusion: score-based vetting through a calibrated operating point
is deployable at DESI resolution; letter grades stay a human product.

Generalize it past this repository. Handed any "gold label" that is itself a
human rating, ask its own inter-rater reliability *before* touching a model.
A QWK of $0.42$ on a four-point scale is not a data-cleaning problem to fix
with more annotators — the two experts on this particular team are not
individually noisy, they simply do not agree with each other about the
underlying concept at ground-based resolution. The honest response is not a
better classifier for the label. It is changing what you evaluate against.

## Connect to the repo { #connect }

- [`reproductions/lensjudge/papers/human_baseline.tex:116`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/papers/human_baseline.tex#L116)
  (§ Inter-grader reliability) and
  [`:184`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/papers/human_baseline.tex#L184)
  (§ Grade-stratified confirmation rates) — the two tables this chapter's
  first two sections are built from, with every caveat about truncation and
  selection this chapter's numbers inherit.
- [`reproductions/lensjudge/parity/FINDINGS.md:30`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/parity/FINDINGS.md#L30)
  — the seven published score-pair counts the QWK worked example derives
  from scratch, and
  [`:107`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/parity/FINDINGS.md#L107)
  — the AUC(grade vs. truth) $=0.577$ power check.
- [`reproductions/lensjudge/parity/FINDINGS.md:190`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/parity/FINDINGS.md#L190)–`262`
  — the Phase C gate scoreboard (rep probe, CNN, 27B student) and Phase D
  parity comparison behind "nothing gets past the label."
- [`reproductions/lensjudge/papers/main.tex:571`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/lensjudge/papers/main.tex#L571)
  — the Euclid re-grade evidence behind "the same wall."
- `site/guide_src/worked_examples.py`, `ch28_labels()` — every number in
  this chapter, including the from-scratch QWK derivation, runnable with
  `worked_examples.py --show ch28`.

## Exercises { #exercises }

??? question "Exercise 28.1 — finish the agreement breakdown by hand"
    Using only the seven pair counts in the worked example's table (they sum
    to $1{,}310$), group them by $|a-b|$ and compute the fraction that are
    exact matches, off by one, and off by two. Then check your three
    fractions sum to $1$.

    ??? success "Solution"
        Exact ($a=b$): $461+165+100 = 726$, giving $726/1310 = 0.554$
        <!-- check: ch28.pair_exact_frac = 0.554 ± 0.001 -->.
        Off by one: $387 + 115 = 502$, giving $502/1310 = 0.383$
        <!-- check: ch28.pair_one_step_frac = 0.383 ± 0.001 -->.
        Off by two: $34 + 48 = 82$, giving $82/1310 = 0.063$
        <!-- check: ch28.pair_two_step_frac = 0.063 ± 0.001 -->.
        Sum: $0.554+0.383+0.063 = 1.000$. Two graders looking at the same
        cutout land on the exact same integer barely more than half the
        time, and disagree by a full two grade-levels — one ready to reject,
        one ready to publish — about one time in sixteen.

??? question "Exercise 28.2 — purity is precision; does it fall the way a precision curve should?"
    A well-calibrated classifier's precision falls as you relax its
    threshold and admit lower-confidence predictions. Using
    $\text{purity}(A)=0.952$, $\text{purity}(B)=0.920$, $\text{purity}(C)=0.909$,
    compute the drop from A to B and from B to C separately, then compare
    each to the $0.043$ total spread from A to C. Does the shape look like a
    classifier whose grade tracks confidence, or something else?

    ??? success "Solution"
        A$\to$B drops $0.952-0.920 = 0.032$; B$\to$C drops
        $0.920-0.909=0.011$. Both drops are small fractions of a purity that
        stays above $0.9$ throughout, and together they only account for the
        full $0.043$ spread — there is no point where purity falls off a
        cliff the way it would if "C" meant "probably a false positive."
        Combined with grade A's own $4/83$ refutation rate, the shape says
        the letter grade is not tracking confidence in the object; it is
        tracking something closer to constant, with grade-A's own
        imperfection as the tell.

??? question "Exercise 28.3 — E1 failing does not mean E2 fails"
    The fine-tuned 27B student scores $\mathrm{QWK}=0.044$ against the
    published human grade — essentially uncorrelated with the letter humans
    assigned. Its truth-referenced AUC is $0.685$, against the human grade's
    own $0.577$. Explain, in one or two sentences, why these two facts are
    not in tension, and why a deployment decision should be made on the
    second number rather than the first.

    ??? success "Solution"
        QWK measures agreement with a specific human artifact (the letter
        grade); AUC measures agreement with an independent, physically
        determined outcome (confirmed vs. refuted). A model can rank
        candidates by their true probability of being a lens in an order
        that has almost nothing to do with which of four discrete buckets a
        particular human team happened to sort them into — ordinal binning
        throws away exactly the fine-grained ranking information AUC can
        still see. Since the deployment question is "does this system find
        real lenses," not "does this system sound like our graders," E2 is
        the number to act on; a low E1 here is not evidence the system is
        bad, only evidence it is not imitating the humans.

??? question "Exercise 28.4 — reconciling 53% with 6 of 17"
    The text states that $53\%$ of 17 DESI grade-C candidates re-graded to A
    or B at Euclid resolution, and separately that 6 of the 17 jumped all
    the way to grade A. Use these two facts to work out how many of the 17
    moved specifically to grade B (not all the way to A).

    ??? success "Solution"
        $53\%$ of $17$ is $0.53\times17 \approx 9.0$
        <!-- check: ch28.euclid_c_regrade_frac = 0.53 ± 0.001 -->,
        so about 9 of the 17 candidates re-graded to A or B combined. Six of
        those nine
        <!-- check: ch28.euclid_c_to_a_count = 6 ± 0 --> went all the way to
        grade A, which leaves $9-6=3$ that moved only as far as grade B. The
        arithmetic matters here: a resolution fix does not uniformly promote
        everything to the top grade, it redistributes candidates along the
        same confidence scale the DESI grader was already trying, and
        failing, to encode from worse pixels.
