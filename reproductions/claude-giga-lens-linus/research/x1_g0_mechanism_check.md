# X1-G0 mechanism entry gate — VERDICT: FAIL-no-ordering (hypothesis structurally dead)

**Campaign**: claude-giga-lens-linus, PLAN §6 P4 pre-registered entry gate. **Cost**: 0 A100-h
(pure-numpy posterior/render analysis on on-disk artifacts). **Date**: 2026-07-15.

## Pre-registered hypothesis and gate

Hypothesis (P4/X1): the EPL single-power-law assumption drives the cross-product slope bracket
(fine-steep γ=1.816 / native-diag anchor γ=1.433 / binned-corr-low γ=1.103) because
whitening/binning reweight spatial frequencies so each product constrains the slope at a
different effective radius; a curved κ(r) then yields different local slopes per product.

Gate (pre-registered): the per-product effective radii r_eff(v3 fine), r_eff(v2d native),
r_eff(v3b binned) must be **strictly monotone in the same sense as the γ ordering** (either
direction) for ANY monotone κ-curvature to produce the observed sign pattern (fine ABOVE
anchor, binned BELOW anchor). No ordering ⇒ hypothesis structurally dead at zero cost.

## Verdict

**FAIL-no-ordering.** All 3 r_eff statistics (weighted mean / log-mean / median) are
non-monotone under the primary weighting, and **all 7 robustness variants × 3 statistics are
non-monotone as well (24/24 FAIL)**. The middle-γ product (native anchor) has the *smallest*
r_eff; the two γ-extremes (fine 1.816, binned 1.103) sit within **0.008″ of each other**.

| product (likelihood that produced γ) | γ | r_eff mean | log-mean | median | frac weight > θ_E |
|---|---|---|---|---|---|
| v3 fine 260² @0.04″, correlated (whitener_v3, M=20) | 1.816 ± 0.117 | **2.596″** | 2.578″ | 2.613″ | 0.563 |
| v2d native 80² @0.13″, **diagonal** (anchor) | 1.433 ± 0.035 | **2.501″** | 2.466″ | 2.543″ | 0.356 |
| v3b binned 130² @0.08″, correlated (whitener_v3b, M=10) | 1.103 ± 0.008 | **2.588″** | 2.571″ | 2.612″ | 0.475 |

Required γ ordering: 1.816 > 1.433 > 1.103. Observed r_eff ordering: v3 > v3b > v2d
(mean/log-mean/median all agree). **v2d is not between v3 and v3b ⇒ no monotone κ(r) exists.**

### The magnitude kill (stronger than the ordering kill)

Even ignoring ordering, the r_eff separations cannot carry the γ gaps. The local-slope
gradient a curved κ(r) would need:

- v3 vs v3b: |Δγ / Δln r_eff| = 0.712 / ln(2.5964/2.5882) ≈ **226 per e-fold** — the fine and
  binned products are the *same sky* and their slope information peaks at the *same radius*
  (Δr_eff ≈ 0.008″ < ¼ fine pixel), yet their γ differ by 0.71.
- v3 vs v2d: ≈ 10.3 per e-fold; v2d vs v3b: ≈ 9.7 per e-fold.

Physical profile curvature (NFW at r_s, dPIE break) gives |dγ_loc/dln r| = O(1). Even the
most charitable pairing is ~10× too weak; the fine-vs-binned pairing is ~200× too weak.

**Conclusion: the "different effective radius + curved κ(r)" mechanism is STRUCTURALLY DEAD.**
This kills the P4 mechanism premise at zero cost (the gate working as designed). Note what it
does *not* kill: a BPL/dPIE fit could still win on evidence (X1-G1) for other reasons — but the
pre-registered *mechanism* by which profile rigidity was supposed to generate the γ bracket is
excluded. The bracket must come from something that differs between the likelihoods at *fixed*
radial information — e.g. the noise model's spatial-frequency reweighting interacting with
source/PSF structure (the arcs' sub-arcsec profile), not with the radial κ profile. Recommend
re-pointing P4's budget accordingly (source-model/PSF systematics track).

## Method (documented weighting choice)

**Weighting = arc-brightness × noise (S/N)² — the pre-registered fallback** (a full Fisher
weighting needs per-pixel ∂m/∂γ forward evals; not free). Per product:

1. **Arc map**: per-product cold-MAP model render minus **elliptical** azimuthal-median profile
   (ellipse measured from the model's own central second moments, r<2.2″; consistent across
   products: q≈0.831/0.839/0.833, φ≈23.5°/24.4°/23.5°), clipped ≥0. A circular median leaves a
   lens-light quadrupole residual that contaminates the arc map (checked; it moved v2d by
   +0.4″ before the fix). Companion galaxy at (−2.34,−2.86)″ (model LL2/LL3 = lens light, not
   lensed source) excised to 1.2″.
2. **Diagonal weights** (v2d anchor): w = (arc/err)² on keep_mask pixels.
3. **Whitened weights** (v3, v3b): u = keep_w · corr(h, arc/err) with the on-disk cgl whitener
   taps and exact `cgl.whiten` convolution semantics (correlate, zero-padded, masked err
   = 1e10); w = u². Whitening is applied to the **full-field** arc map; the r∈[1.2,4.2]″ band
   and a 0.25″-dilated companion exclusion are applied to the *weights* (band-cutting before
   the conv manufactures band-edge high frequencies that the whitener amplifies — checked,
   it produced spurious edge spikes).
4. r_eff = w-weighted mean/log-mean/median of circular radius about each product's cold-MAP
   mass center on the mean-centered gigalens grid; whitened pixel values attributed to the
   pixel's radius (kernel supports ±0.8″, similar across v3/v3b).

**Robustness variants (all FAIL)**: tight band [1.8,3.6]″; ln²(r/θ_E) Fisher-lever
reweighting; companion excision 0.9″; elliptical-radius attribution; all-diagonal weighting
on all 3 products; all-whitened weighting on all 3 products. Earlier (rejected) analysis
configurations — circular median, pre-conv band cut, no companion excision — *also* failed
the gate; there is no analysis choice we found under which an ordering exists.

## Inputs (all on-disk)

- Cutouts + masks + err maps: `foundry-i/data/cutout_{v3,v3b,v2d}.npz`
- MAP model renders: `foundry-i/data/model_map_v3cold.npy`, `foundry-i/data/model_map_v3b_cold.npy`,
  `claude-giga-lens/data/model_map_v2d_cold.npy`
- Whitener bundles: `claude-giga-lens/data/whitener_v3.npz` (M=20), `whitener_v3b.npz` (M=10)
- γ per product: `e2_v3.json` steep basin (correlated HMC, R̂=1.02); foundry-i
  `hmc_v13_v2d.npz` (diagonal anchor 1.433 [1.400,1.469], recomputed); P1c converged SMC
  `e2_v3b_low_smc_canary_fix.npz` (γ=1.1032±0.0080, retrieved from the Perlmutter staged copy
  of the previous campaign, `~/claude-giga-lens/repo/reproductions/claude-giga-lens/data/results/`)

## Outputs

- `data/x1_g0_effective_radii.json` — all numbers (per-product stats, 7 variants × 3 statistics,
  gate verdicts, required-curvature diagnostic)
- `figs/x1_g0_radial_profiles.png` — radial (S/N)² weight profiles + γ-vs-r_eff gate panel
- `figs/x1_g0_arc_maps.png` — extracted arc maps (band-restricted) for the three products
- Analysis script (scratchpad, reproducible from the listed inputs): weighting exactly as above
