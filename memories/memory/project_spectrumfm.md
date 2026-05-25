---
name: project-spectrumfm
description: "SpectrumFM — DOE Genesis Mission proposal for an extensible transformer foundation model on DESI spectra; Huang PI, Benson leads agentic-AI tooling"
metadata: 
  node_type: memory
  type: project
  originSessionId: 95b34e74-4971-42df-affe-07a9467ad83a
---

**SpectrumFM** is a DOE Genesis Mission Phase-I proposal (RFA-0003612, Topic 14: "Unifying Physics from Quarks to the Cosmos", Focus Area A): a transformer-based foundation model pre-trained from scratch on ~60M DESI spectra (~50M extragalactic + ~10M stellar) — a corpus 60× larger than any existing spectroscopic foundation model.

**Why:** DESI's production redshift-fitter Redrock relies on hand-crafted templates and breaks when new target classes (LAEs, LBGs for DESI-II; Spec-S5 successors) come online. SpectrumFM claims "one model, six classes" — a single encoder serving LRG, ELG, QSO, MWS at production grade plus few-shot LBG/LAE extensibility — and brings human alignment (RLHF-style) to scientific foundation models for the first time.

**How to apply:** Key architectural facts: transformer encoder-decoder operating at ~10:1 compression with a path to 7,081-pixel full resolution (vs AION-1's 273 tokens); auxiliary redshift head trained on every step (not bolted on like AION-1's downstream head); three-stage pipeline (MSM pretraining → SFT + preference learning on VI data → few-shot fine-tuning); compute on NERSC Perlmutter (A100s) → Doudna (NERSC-10) in Phase II. Downstream validation tasks include strong-lens identification (via Hsu et al. pairwise method) and SN typing. Phase I is 9 months structured into 3×3-month phases.

**Team:** Huang (PI, USF), Shan Wang (USF, co-leads architecture), **Greg Benson (USF, leads agentic-AI tooling)**, Stephen Bailey (LBNL, DESI corpus + Redrock baseline), Stephanie Juneau (NSF NOIRLab), Nao Suzuki + Eric Hsiao (FSU). Year 1 total budget: $605,922.

**Connection to lensing:** SpectrumFM's Hsu_2025-pairwise-spectroscopic-search downstream task connects directly to Huang's lens program — the encoder representations are validated on lens identification.

Related: [[project-huang-lensing]], [[user-role-benson]].
