# Voice contract — the Beginner's Guide

Read this **in addition to** `../../contract/style.md`, which is shared and
still binding: the reader model, the hard syntax rules, the figure idiom, the
exercise idiom and the cross-link rules all come from there. This file only
records what is DIFFERENT about this book.

## What is different

**The reader is the same person** — Greg Benson, CS professor, directing this
lensing program. Keep the `!!! tip "You already know this"` bridges. What has
changed is only what he is assumed to know about *astronomy*: nothing.

**No calculus.** The main guide opens with derivatives; this one must not need
them. Arithmetic and light algebra only:

- $z = \Delta\lambda/\lambda$, $1+z = \lambda_{\rm obs}/\lambda_{\rm emit}$
- $v = H_0 d$
- inverse square, $F = GMm/r^2$
- powers of ten, ratios, log axes

If a point needs a derivative or an integral, **state the result and hand it to
the main guide by chapter link**. That handoff is a feature, not a failure — it
is the entire structure of this book.

**Scale is the argument.** The main guide has no ruler: it quotes a cluster mass
of $4.6\times10^{13}\,M_\odot$ and cannot say whether that is a lot, because it
never mentions the Milky Way. Ch. 1 makes the Milky Way this book's ruler, and
every later mass/size/distance is compared to something the reader already holds.
A number without a comparison is not doing its job here.

## Every chapter ends with "Unlocks"

Not "Connect to the repo" — this book connects to the *guide*:

```markdown
## Unlocks { #unlocks }

!!! success "What you can now read"
    - **[Ch. 12 — Redshifts and what a spectrum tells you](../guide/12-spectroscopy.md)**
      opens by treating $z$ as "a wavelength ratio you read off a spectrum, full
      stop", and defers the physical meaning. You now have the meaning, so that
      deferral costs you nothing.
```

Link with `../guide/NN-slug.md#anchor` — the mkdocs `validation.links.anchors`
setting **fails the build** on a wrong anchor, so these are checked. Use the
anchors listed in `../../contract/outline.yml`, never a guessed one.

## Tone

The main guide's register is "a colleague at a whiteboard". Keep it. Do not
switch to a popular-science register: no "imagine you are a photon", no
breathless awe, no exclamation marks. The reader is an expert in his own field
being briefed on someone else's, and he should be addressed that way.

**Do not oversell.** Where astronomy genuinely does not know something — what
dark matter *is*, why $H_0$ disagrees, why the isothermal conspiracy holds —
say so plainly. The main guide's culture is to retract its own numbers; match it.

## Honesty rules specific to this book

Simplification is the job, but a simplification that will have to be *unlearned*
is a bug. Two examples of the line:

- **Fine:** "the universe is expanding like raisin bread rising". Then say what
  the analogy breaks: the dough has an edge and a centre; the universe has
  neither.
- **Not fine:** "redshift is a Doppler shift". It is not, and Ch. 13 will
  contradict you. Introduce the Doppler picture as the *intuition*, then say
  explicitly that the real cause is the stretching of space itself, and that the
  naive $v = cz$ reading gives faster than light past $z = 1$ (Ch. 13 works that
  exact trap at $z = 1.432$).

Where a number is a crude estimate, **say so in the prose**. `pch03`'s stellar
density is a uniform-disk average and is ~4× the measured local value; the
emptiness argument survives that, but the prose must not claim precision the
model does not have.

## Length

1,500–3,000 words per chapter. Shorter than the main guide's chapters, because
concepts without derivations are shorter. Ch. 8 (redshift) and Ch. 16 (the
handoff) may run to 3,500.
