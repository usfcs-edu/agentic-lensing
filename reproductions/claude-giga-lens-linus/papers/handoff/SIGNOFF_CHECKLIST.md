# Sign-off checklist — borderline passages (adversarial audit, 2026-07-21; E1 addendum, 2026-07-22)

**Instruction: each characterizes unpublished code state — requires team sign-off before
ANY external release; internal use fine.**

The six passages below (passages 1–5 flagged by the adversarial audit of the campaign
report; passage 6 added with the E1 post-campaign epilogue) describe
the state of the GIGALens team's unpublished upstream development branches (the vendored
scene-API commit `80916d2` and the experimental sampler/SBC assets around it). Under the
campaign's pre-declared bright line (nothing derived from the upstream branches'
unpublished code or results appears externally without the team's sign-off), every one of
them must be explicitly signed off by the team before the report — or any excerpt
containing them — is released outside the collaboration. Line numbers refer to the tex
sources as of this revision.

---

## Passage 1 — abstract code-state sentence

`papers/sec_a_front_methods.tex:100-104`

> Meanwhile the \gigalens\ team's
> next-generation rewrite (``the upstream development branches'') rebuilt the
> forward model around a unified multi-plane, multi-band \emph{\sceneapi} whose
> validation suite was, at the commit we vendored, not yet run, and which
> contains neither a correlated-noise likelihood nor an evidence layer.

## Passage 2 — intro "three facts about the commit we vendored"

`papers/sec_a_front_methods.tex:190-197`

> Three facts about the commit
> we vendored (\S\ref{sec:substrate-a}) motivate the bridge. First, its
> validation suite was present but \emph{unrun} --- the forward model had no
> external certification. Second, it contains no correlated-noise
> likelihood, no analytic source marginalization with an Occam term, and no
> tempering/SMC evidence layer --- precisely the assets the companion
> campaign validated on the old stack. Third, its likelihood seam is
> documented and stable enough to carry an upstream-shaped port.

## Passage 3 — certification claim "unrun at the pin" (and the sec_b certification framing that rests on it)

`papers/sec_a_front_methods.tex:390-394`

> F1--F4 constitute, to our knowledge, the first
> external certification evidence for the \sceneapi\ forward model in the
> EPL+shear + 4$\times$S\'ersic / S\'ersic+shapelets ($n_{\max}=6$)
> configuration class --- the vendored validation suite was unrun at the
> pin.

Companion location: `papers/sec_b_results_mech.tex:24-33` (the Results I certification
framing — "the team's upstream development branch, vendored unpatched at commit
\dfile{80916d2}" — presents the same first-external-certification claim and rests on the
same unrun-suite characterization).

## Passage 4 — blackjax-internals note on the upstream experimental kernels

`papers/sec_a_front_methods.tex:523-525`

> Kernel logic is copied with attribution from the upstream experimental
> implementations, never imported (those modules reach into private
> \blackjax\ internals and the pre-rewrite simulator).

## Passage 5 — X2 generation-prior fact about the frozen SBC set

`papers/sec_c_bench_disc.tex:465-470`

> the pre-existing frozen SBC set carries, per its own provenance
> in the upstream development branches, a generation prior narrower than the
> current fitting prior --- under which SBC fails by construction (truth
> draws must follow the fitting prior). Such a mismatch is a legitimate
> robustness-testing design if intentional; the question for the team is
> whether a matched-prior set should exist.

## Passage 6 — E1 epilogue description of the vendored experimental MCLMC driver (added 2026-07-22)

`papers/sec_summary_modelers.tex` (item 3, the E1 sampler sentence)

> The E1 epilogue adds \textbf{MCLMC proper} ---
> the PI-requested vehicle; the team's own vendored implementation
> (\dfile{full\_mclmc\_with\_adapt\_sharded}: windowed mass-matrix
> adaptation, run with their Stan-style mass-matrix regularization option
> enabled)

Companion locations carrying the same characterization of the unpublished experimental
sampler's features and state: `papers/sec_appendix_convergence.tex` (group M header + rows —
"vendored windowed-adaptation MCLMC", "regularization ON", the two BLOCKED launch rows and
the VMA-check note quoted from the run jsons) and `papers/sec_c_bench_disc.tex` (the E1
epilogue paragraph in "What it closes": "the team's own vendored MCLMC ... windowed
mass-matrix adaptation and regularization option"). These describe features and defaults of
the unpublished `gigalens/jax/experimental/mclmc.py` driver (windowed mass-matrix adaptation;
opt-in Stan-style covariance regularization; the shard_map/VMA interaction worked around at
runtime with the vendored file byte-identical on disk) and the E1 result obtained with it.

---

Referenced from `papers/handoff/CLAIMS.md` and from the report's Reproducibility section
(external release is gated on this checklist).
