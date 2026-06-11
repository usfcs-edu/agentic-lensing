# Agentic Lensing

An agentic-AI research program at USF Computer Science, with USF Physics &
Astronomy and the DESI Strong Lens Foundry group: a complete public-data
reproduction of the 16-paper Huang-group strong gravitational lensing corpus —
discovery, modeling, and follow-up spectroscopy, plus the AION-1 astronomy
foundation model — executed end-to-end with coding agents, one LaTeX tech
report per paper. Three pieces of **current work** (LensJudge, ClaudeNet,
Redshifty) build on the reproduced stack.

Every report page on this site is generated directly from its `main.tex`; each
offers a **PDF download** and a link to the code and artifacts on **GitHub**.

## Current work — headline results

**[LensJudge](current/lensjudge/index.md)** — an agentic visual-inspection (VI)
grader for strong-lens candidates on the Claude Agent SDK. On the spectroscopic
task it is strong: 20/20 Hsu Table-2 Grade-A pairs re-graded plausible and 4/4
Foundry-II non-lenses rejected. On imaging grading of the CNN's hard high-score
pool, every configuration (lean Sonnet, lean Opus, judge panel, multi-agent)
lands near chance (AUC 0.32–0.51) against single-grader consensus A/B/C labels —
and paying 2–4× more per candidate buys nothing. The central finding is a
measurement problem: the missing multi-grader human ceiling, not the model.

**[ClaudeNet](current/claudenet/index.md)** — replaces the lineage's collapsed
meta-learner with a deliberately decorrelated ensemble (EfficientNet-family
backbones, an AION-1 frozen-embedding probe, two ResNets; member Spearman ~0.45
vs the lineage's ~1.0). It beats the published meta-learner at **all four
matched-FPR operating points** (+0.030 and +0.090 recovery vs Storfer at 1% and
0.1% FPR; +0.012 and +0.090 vs Inchausti), and adds certified conformal FDR
selection, uncertainty triage (selective error 0.022 → 0.0002 at 50% coverage),
and test-time equivariance (+0.04).

**[Redshifty](current/redshifty/index.md)** — reproduces the Approach-A
"redshift ignition" NERSC result on a single commodity GPU: sustained
`val_redshift_acc` ≥ 10% by step 9000 (peak 14.86%), AR/val ratio 0.73 vs the
NERSC reference's 0.74. A four-hypothesis diagnostic ladder shows the
sv3+main × bright+dark **data mix** is the load-bearing requirement, and the
accompanying NERSC scaling spec defines the go/no-go ladder to 1B parameters.

## Reproductions

**16/16 in-scope papers reproduced.** Each row condenses the verification index
in [REPRODUCTIONS.md](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/REPRODUCTIONS.md);
the report link carries full methods, results tables, and an honest "not
reproduced" section.

| Report | Original paper | What was reproduced | Agreement |
|---|---|---|---|
| [AION-1](reproductions/aion-1/index.md) | Parker+ 2025 — AION-1 omnimodal foundation model | All 11 downstream experiments from frozen public checkpoints (0.3/0.9/3B) on rebuilt Multimodal Universe benchmarks | Redshift R² 0.985 (image+spectrum) vs ~1.00; Gaia XP stellar-parameter residuals match or beat the paper; morphology/retrieval/segmentation qualitative (corpus-scale limited) |
| [Cikota 2023](reproductions/cikota-2023/index.md) | Einstein cross DESI-253.2534+26.8843 (arXiv 2307.12470) | GIGA-Lens model of the four-image cross on DESI Legacy imaging (proprietary MUSE skipped) | θ_E 2.10″ vs 2.52″ (offset ablated to 1.35″ seeing); σ_SIE 347 vs 379 km/s; μ 7.0 vs 10.47; χ²/px 0.90 |
| [Dawes 2022](reproductions/dawes-2022/index.md) | Multiply-lensed quasar search (ApJS 269 61) | Autocorrelation friends-of-friends finder on DESI DR1 spectroscopic QSOs (~1.6M vs paper's ~5M photometric targets) | 58/58 = 100% conditional recovery (apples-to-apples with the paper's 94/94); raw 14% is proxy-limited |
| [Foundry I](reproductions/foundry-i/index.md) | Huang+ 2025a — HST + GIGA-Lens (arXiv 2502.03455) | Genuine HMC modeling on HST F140W; 3 real upstream gigalens bugs found and documented | θ_E to 3%, e1 to 2.5%, shear PA exact; long 8-chain HMC brackets the paper's γ |
| [Foundry II](reproductions/foundry-ii/index.md) | Huang+ 2025b — DESI spectroscopy (arXiv 2509.18089) | 73/73 systems matched to public DR1 fibers + FastSpecFit | z_lens 70/72 and z_source 16/22 to <0.001; σ_v 65/71 (r = 0.80) |
| [Foundry III](reproductions/foundry-iii/index.md) | Agarwal+ 2025 — Keck NIRES follow-up | Blind Eq.-1 line-fit + Monte-Carlo validation (consistency reproduction; KOA serves raw L0 only) | 6/6 source redshifts to \|dz\| < 0.001 |
| [Foundry IV](reproductions/foundry-iv/index.md) | Lin+ 2025 — VLT/MUSE follow-up (arXiv 2509.18087) | 3 public ESO MUSE cubes; automated line-ID engine built | Auto z_lens 3/3 within dz < 0.02 (2/3 < 0.003); guided source z exact for Lens22 (0.821) |
| [Gu 2022](reproductions/gu-2022/index.md) | GIGA-Lens method (ApJ 935 49) | Method on 12 mock systems incl. convergence study | MAP χ² ≈ 1.0 on all; R̂ ≤ 1.017 with ESS ≫ 10⁴ on well-conditioned systems; mean ESS ≈ 11k (depth-limited on degenerate ones; paper used 4× A100) |
| [Hsu 2025](reproductions/hsu-2025/index.md) | Pairwise spectroscopic lens search (arXiv 2509.16033) | Full 28M-fiber DESI DR1 friends-of-friends search | 13,530 groups / 27,334 spectra vs 13,218 / 26,621 (+2.4%); 20/20 Table-2 Grade-A recovered within 3″ |
| [Huang 2020](reproductions/huang-2020/index.md) | DECaLS ResNet finder (arXiv 1906.00970) | From-scratch Lanusse ResNet-46; full 6.24M-galaxy DR7 sweep; leakage ablation | 83% Grade-A recall @ p ≥ 0.9 (paper-exact DR7-trained) |
| [Huang 2021](reproductions/huang-2021/index.md) | Shielded ResNet on DR8 (arXiv 2005.04730) | 59,905-parameter shielded net (58.6× smaller than L18); 17.3M-galaxy two-model DR8 sweep | AUC within ±0.002 of L18; leak-free recovery 50.4% @ p ≥ 0.9; north-calibration finding |
| [Inchausti 2025](reproductions/inchausti-2025/index.md) | Storfer 2024 + Inchausti 2025 DR9/DR10 ensemble (arXiv 2308.04603 / 2508.20087) | EfficientNetV2-S + shielded ResNet + 300-node FWLS meta-learner (meta ≈ simple average) | Recovery @ 1% FPR 91% / 97%; neg:pos ratio, not architecture, sets usability |
| [Sheu 2023](reproductions/sheu-2023/index.md) | Lensed supernova search (arXiv 2301.03578) | From-scratch Bramich-2008 difference imaging + SEP + SALT3 | Re-detected the Grade-A lensed SN (11 sub-detections; counter-image 1.48″ from lens); SALT3 μ 8.6 vs 8.2 |
| [Sheu 2024a](reproductions/sheu-2024a/index.md) | Variable lensed quasars (arXiv 2408.02670) | Reused the Sheu-2023 diff-imaging core; σ-metric validated to a few % on synthetic injections | Variability σ 0.34 vs 0.25 mag at both lensed images |
| [Sheu 2024b](reproductions/sheu-2024b/index.md) | Carousel cluster lens (arXiv 2408.10320) | Multi-source-plane model on public HST F140W/F200LP (5 source planes; per-plane θ_E scaling) | θ_E 12.96″ vs 13.03″ (0.5%); γ 1.53–1.67 vs 1.67; M(<θ_E) 4.62 vs 4.78 × 10¹³ M⊙ |
| [Silver 2025](reproductions/silver-2025/index.md) | ML lens-finding forecasts for JWST (ResNet/U-Net) | Model-1 (HST) ResNet on lenstronomy Sérsic-source MVP | Validation AUC 0.994 vs 0.998 |

Scope is honest throughout: algorithmic finder steps reproduce intermediate
counts (human VI grading is out of scope), modeling papers reproduce on public
imaging with published redshifts where spectroscopy is proprietary, and a few
are MVP/consistency reproductions where archive or compute limits apply.

## Other work

The corpus has also been ported to commodity Apple Silicon — see the
[Apple Silicon (M4 Max / MPS) ports](other/apple-silicon/index.md): the
Huang 2020/2021 lens finders and the Redshifty transformer re-run end-to-end
on a Mac Studio with machine-checked parity against the CUDA reference.
