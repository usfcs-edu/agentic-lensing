# Style contract for the guide

Injected verbatim into every chapter-authoring prompt. Do not summarise it.

## The reader

One person: **Greg Benson, Professor of Computer Science at USF.** He guides
this research program; he is not an astrophysicist. Assume:

- **Owns**: algorithms, systems, PL, agentic AI, multi-agent systems, Python,
  ML as practised (backprop, autodiff, ResNets, transformers, ROC/AUC,
  regularisation, ensembles), probability as used in ML, linear algebra as used
  in ML, information theory as used in ML (cross-entropy, nats, KL).
- **Rusty on**: continuous calculus, vector calculus, PDEs, complex analysis.
- **Has none of**: astrophysics, cosmology, observational astronomy, GR.

So: never explain a `for` loop, a gradient, or what AUC is. Do explain what a
parsec is, why light bends twice as far as Newton predicted, and what the
Laplacian is doing in a lens equation.

He asked to "start from the basics of calculus" and to "reason about the math".
Take both literally. He does not want to be told the answer; he wants to be able
to derive it and to catch it when it is wrong.

## Voice

- Direct, technical, unhurried. A colleague explaining at a whiteboard.
- **Never condescend, never hype.** No "simply", "just", "obviously", "as we all
  know", "it turns out that". If it were obvious he would not be reading.
- Prefer the concrete number to the adjective. "191 nats" not "a huge swing".
- Second person is fine ("you can check this in three lines").
- Contractions are fine. Exclamation marks are not.
- When the repo got something wrong, say so plainly and say what it teaches.
  This repo retracts its own numbers in its own ledger; match that register.

## Structure of a chapter

```
# <Title>                                    <- H1, no anchor needed

<One-paragraph statement of what this chapter buys you. Name the destination.>

!!! abstract "What you can skip"
    <Explicit: what a CS PhD already owns here and need not read. Be specific
    and generous — his time is the scarce resource. Omit this box only in
    Ch 01, 25, 26, 29.>

## <Section> { #anchor }                     <- every H2/H3 needs { #anchor }

<Concept -> derivation -> worked example -> figure -> connect to the repo.>

## Connect to the repo { #connect }

<Which file(s) this chapter makes readable, with real paths and line numbers.>

## Exercises { #exercises }

??? question "Exercise <n>.<m> — <short phrase>"
    ...
    ??? success "Solution"
        ...
```

End every chapter that constrains gamma with a **gamma Ledger** row:

```
!!! note "γ Ledger"
    **What this chapter rules in or out about $\gamma = 1.103$:** ...
```

## Hard rules (the linter fails the build on these)

Read `contract/notation.yml`. The rules that catch people:

1. **Math is `$...$` and `$$...$$`.** Never `\(...\)`. It is the only syntax that
   works for the site (arithmatex), the PDF (pandoc→LaTeX) and the offline HTML
   (pandoc→MathML) simultaneously.
2. **No boundary whitespace**: `$ x $` is silently *not rendered* by arithmatex.
3. **`$\eqref{eq:foo}$`**, never bare `\eqref` — a bare one lands as literal text.
4. **`\eqref` only targets a `\label` in the same chapter.** Equation numbering is
   per-page. To cite another chapter, link in prose:
   `the lens equation ([Ch. 17](17-lens-equation.md#the-lens-equation))`.
5. **Every `##`/`###` carries `{ #kebab-anchor }`** — use the anchors listed for
   your chapter in `outline.yml`, and only those, plus `#connect`/`#exercises`.
6. **Every number is tagged** with `<!-- check: chNN.key = value ± tol -->` and
   must come from `worked_examples.py`. Untagged numbers get deleted.

## Figures

Use only the figure slugs assigned to your chapter in `outline.yml`. Both
variants, adjacent, identical alt text, `markdown="span"` on **both** the
`<figure>` and the `<figcaption>`:

```markdown
<figure markdown="span">
  ![Critical curves and caustics for an SIE lens](figures/ch18-sie-caustics-light.svg#only-light){ width="90%" }
  ![Critical curves and caustics for an SIE lens](figures/ch18-sie-caustics-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 18.1.** Critical curves ($\det A = 0$, left)
  and the caustics they map to (right), for an SIE with $\theta_{\mathrm{E}} = 1''$,
  $q = 0.7$. Nothing here is drawn: the curves come from numerically
  differentiating the deflection field.</figcaption>
</figure>
```

Number figures `**Figure <chapter>.<n>.**` by hand — there is no autonumbering.

## The "You already know this" box

```markdown
!!! tip "You already know this"
    Magnification is the change-of-variables factor. When a normalizing flow
    reports $\log q(x) = \log p(u) - \log|\det J|$, it is doing the identical
    arithmetic this repo does to decide how bright a lensed arc is.
```

Use where the bridge is real and load-bearing. Do not manufacture them.

## Cross-links

- To another chapter: `[Ch. 17](17-lens-equation.md#the-lens-equation)` (flat
  files, same directory).
- To a generated report page: **use its `#sec:*` anchor**, which comes from a
  `\label{sec:...}` in `main.tex` and is stable across rewording:
  `[the evidence flip](../current/claude-giga-lens/index.md#sec:realdata)`.
  Never link to a slugified heading on those pages.
- To source: cite the path in prose as `` `cgl/e2.py:110` `` and link to
  `https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L110`.
  Pin to `main`, never a SHA.

## Length

Aim 1,800–3,500 words per chapter. Ch 25 and 26 may reach 4,500. A chapter that
runs long is usually carrying a derivation that belongs in an exercise solution.

## The three things that matter most

1. **Every number reproduces.** This repo's own report carries a "~17σ" claim
   that reconciles with none of its quoted uncertainties. Do not add to that.
2. **The reader derives, then checks.** Structure worked examples so he computes
   the repo's number himself and *then* sees it matched.
3. **Teach through the real failures.** The drizzle chain and the saddle chain
   are better pedagogy than any clean result, and they are true.
