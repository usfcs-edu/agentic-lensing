export const meta = {
  name: 'vhs-archive-lens-deepdive',
  description: 'Pressure-test the VHS (VISTA Hemisphere Survey) NIR strong-lens opportunity: verify the genuinely-fresh far-south sky fraction, whether NIR depth is sufficient to find lenses at all, prior NIR lens-search coverage, a concrete cutout pipeline, and an adversarially-verified yield.',
  phases: [
    { title: 'Research', detail: 'footprint/depth/fresh-sky + prior-NIR-coverage + access pipeline, in parallel' },
    { title: 'Verify', detail: 'adversarial red-team on the ~60%-fresh / ~10-100-yield claim and the depth-detectability risk' },
  ],
}

const CONTEXT = `
Program context: a strong-lensing ML group (X. Huang) runs an optical CNN lens-finder trained on
DESI Legacy grz cutouts. We are evaluating VHS (the VISTA Hemisphere Survey, ESO VISTA 4.1m/VIRCAM,
near-IR) as a drop-in target for that CNN, retrained for NIR bands/PSF. Two LOAD-BEARING claims
from a prior scan need verification: (1) "~60% of VHS's ~17,000-20,000 deg2 is genuinely FRESH
far-south sky (dec < -30) that has never had any deep optical OR blind-NIR strong-lens search";
(2) the realistic net-new yield is ~10-100 NIR-favored galaxy-galaxy lenses. The HUGE risk we must
test honestly: VHS is SHALLOW NIR (mostly 2-band J,Ks). Faint blue lensed arcs may be BELOW the VHS
detection limit, so VHS-only lens-finding could be near-impossible regardless of fresh sky.
Already-searched baseline to exclude: DESI Legacy DR7-10 (DECaLS+BASS+MzLS), DES Y6, DELVE DR1
(Zaborowski 2023), DESI DR1 spectroscopy. Be quantitative, cite URLs.
`;

const INVENTORY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    footprint_deg2: { type: 'string' },
    bands_and_subregions: { type: 'string' },
    depth_5sigma_per_band: { type: 'string' },
    depth_in_AB_vs_optical: { type: 'string' },
    seeing_pixscale: { type: 'string' },
    latest_release: { type: 'string' },
    fresh_sky_accounting: { type: 'string' },
    fresh_sky_fraction_estimate: { type: 'string' },
    overlap_with_searched_surveys: { type: 'string' },
    deep_optical_in_far_south: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['footprint_deg2', 'depth_5sigma_per_band', 'fresh_sky_fraction_estimate', 'overlap_with_searched_surveys', 'sources'],
};

const PRIOR_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    blind_nir_lens_searches: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { citation: { type: 'string' }, survey: { type: 'string' }, method: { type: 'string' }, n_lenses: { type: 'string' }, blind_or_counterpart: { type: 'string' }, url: { type: 'string' } },
      required: ['citation', 'survey', 'method', 'blind_or_counterpart'] } },
    is_there_any_blind_nir_galaxy_galaxy_search: { type: 'string' },
    nir_counterpart_uses: { type: 'string' },
    depth_detectability_verdict: { type: 'string' },
    nir_arc_detectability_basis: { type: 'string' },
    lens_surface_density_at_vhs_depth: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['is_there_any_blind_nir_galaxy_galaxy_search', 'depth_detectability_verdict', 'lens_surface_density_at_vhs_depth', 'sources'],
};

const PIPELINE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    access_mechanisms: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { name: { type: 'string' }, endpoint: { type: 'string' }, what: { type: 'string' } },
      required: ['name', 'endpoint', 'what'] } },
    recommended_data_product: { type: 'string' },
    band_strategy: { type: 'string' },
    pixscale_psf: { type: 'string' },
    optical_companion_strategy: { type: 'string' },
    cutout_tiling_approach: { type: 'string' },
    cnn_adaptation: { type: 'string' },
    practical_gotchas: { type: 'string' },
    effort_estimate: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['access_mechanisms', 'recommended_data_product', 'band_strategy', 'cnn_adaptation', 'practical_gotchas', 'sources'],
};

phase('Research');
const [inventory, prior, pipeline_] = await parallel([
  () => agent(
    `${CONTEXT}\nTASK A - VHS FOOTPRINT, DEPTH & GENUINELY-FRESH-SKY ACCOUNTING.\n` +
    `Quantify: (1) VHS total footprint (deg2), its sub-regions (VHS-DES, VHS-ATLAS/GCS, VHS-GPS) and bands (Y/J/H/Ks coverage varies). (2) 5-sigma point-source DEPTH per band (give both Vega and AB), and compare directly to optical lens surveys (DES grizY ~24 AB; DESI Legacy grz). (3) seeing/pixel scale. (4) Latest public release (DR5/DR6). (5) The CRITICAL fresh-sky accounting: of the ~20,000 deg2, how much is genuinely OUTSIDE the already-searched optical footprints (DES ~5000, DESI Legacy DR10-south ~14000 which ingested DELVE/DeROSITAS to dec +32, DELVE DR1 searched ~4000)? Specifically: what far-south sky (dec < -30, below the DES/Legacy reach) has VHS NIR but NO deep optical lens search? Quantify the fresh fraction and whether deep optical exists there (SkyMapper is shallow; is there anything deep?). Do 6-10 web searches; cite URLs and exact depth numbers.`,
    { label: 'A:footprint-depth', phase: 'Research', agentType: 'general-purpose', schema: INVENTORY_SCHEMA }),
  () => agent(
    `${CONTEXT}\nTASK B - PRIOR NIR LENS-SEARCH COVERAGE + THE DEPTH-DETECTABILITY QUESTION.\n` +
    `Two jobs. JOB 1: Find EVERY strong-lens search done on wide-area NIR imaging (VHS, VIKING, UKIDSS LAS, 2MASS) - classify each as a BLIND galaxy-galaxy lens search vs merely a NIR COUNTERPART/color-cut for optically- or submm-selected candidates (e.g. VIKING used for Herschel-ATLAS lens counterparts, Edge et al. 2013; lensed-quasar NIR selection). Is there ANY published blind galaxy-galaxy strong-lens DISCOVERY catalog from wide NIR imaging? JOB 2 (the load-bearing risk): Is VHS DEEP ENOUGH to find galaxy-galaxy lenses at all? Lensed arcs are typically faint blue star-forming sources, brightest in optical/UV rest-frame; in the NIR at VHS depth (Ks ~ 18-20 AB) can the ARC be detected, or only the deflector? Assess: what lens populations ARE NIR-favored (high-z sources, dusty/red arcs, red massive deflectors, submm-lens counterparts)? Give a realistic lens surface density (lenses/deg2) findable in VHS-depth NIR imaging, with a literature basis. Do 6-10 web searches; cite URLs.`,
    { label: 'B:prior+depth', phase: 'Research', agentType: 'general-purpose', schema: PRIOR_SCHEMA }),
  () => agent(
    `${CONTEXT}\nTASK C - DATA ACCESS & CUTOUT-GENERATION PIPELINE FOR VHS NIR.\n` +
    `Lay out a concrete pipeline. Cover: (1) ACCESS mechanisms with real endpoints - VISTA Science Archive (VSA/WFAU, horus.roe.ac.uk/vsa) SQL Freeform + image cutout/MultiGetImage; NOIRLab Astro Data Lab (vhsdr5, TAP/SIA cutout); ESO Science Archive Phase 3 (tile images + band-merged catalogs). (2) RECOMMENDED product (tile images vs pawprints vs band-merged source catalog). (3) BAND strategy: VHS is mostly 2-band (J, Ks), some H/Y - how to make a CNN input from 2 NIR bands (replicate, synth channels, or J/Ks/(J-Ks)); honest note that you lose the grz blue-arc discriminant entirely (worse than optical for arcs). (4) pixel scale (0.339 native, 0.34 tile) and seeing (~0.9-1.1) vs DESI Legacy 0.262/seeing 1.1 - so VHS is COARSER and shallower, the opposite of HST. (5) OPTICAL COMPANION strategy: since NIR-only is arc-poor, should this be a NIR+optical fusion (VHS Ks for red deflector + DES/DECaLS/SkyMapper g for blue arc) rather than NIR-only? Where deep optical is absent (far south) what's left? (6) Deflector seeding via the band-merged catalog (red massive galaxies). (7) CNN adaptation (retrain on JKs or fuse). (8) Gotchas (NIR sky variability, persistence, shallow depth, tile edges, confusion). (9) Effort. Do 6-10 web searches for endpoints; cite URLs.`,
    { label: 'C:pipeline', phase: 'Research', agentType: 'general-purpose', schema: PIPELINE_SCHEMA }),
]);

phase('Verify');
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['CONFIRMED', 'DOWNGRADED', 'REFUTED'] },
    fresh_sky_verdict: { type: 'string' },
    depth_detectability_verdict: { type: 'string' },
    nir_only_viable: { type: 'string' },
    realistic_net_new_yield: { type: 'string' },
    best_framing: { type: 'string' },
    biggest_risk_to_the_thesis: { type: 'string' },
    corroborating_sources: { type: 'integer' },
    evidence: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { claim: { type: 'string' }, source_url: { type: 'string' } }, required: ['claim', 'source_url'] } },
    rationale: { type: 'string' },
  },
  required: ['verdict', 'fresh_sky_verdict', 'depth_detectability_verdict', 'realistic_net_new_yield', 'biggest_risk_to_the_thesis', 'rationale', 'evidence'],
};
const redteam = await agent(
  `You are an adversarial red-team verifier. Thesis under test: "VHS offers ~60% genuinely-fresh far-south NIR sky never lens-searched, and our grz-trained CNN (retrained for NIR) could harvest ~10-100 NEW galaxy-galaxy lenses from it."\n\n` +
  `Footprint/depth finding: ${JSON.stringify(inventory)}\n\nPrior-coverage/depth finding: ${JSON.stringify(prior)}\n\n` +
  `Try HARD to FALSIFY, doing your own web searches:\n` +
  `1. DEPTH KILLER (most important): is VHS actually deep enough to detect lensed ARCS (not just deflectors)? At Ks~18-20 AB and ~1\" seeing, faint blue arcs may be invisible. If NIR-only cannot see arcs, the opportunity collapses to a NIR+optical fusion - and then the optical (DES/DECaLS) is the part already searched. Verify quantitatively.\n` +
  `2. FRESH SKY: is ~60% really fresh? Check overlap with DES, DESI Legacy DR10-south, DELVE, and whether the far-south VHS-only sky has ANY deep optical at all (or only shallow SkyMapper/Gaia).\n` +
  `3. Will Euclid (which covers the southern sky to VIS~26.5 over ~14000 deg2 eventually) and Rubin/LSST (south, ugrizy deep) simply harvest this far-south sky far better within a few years, making a shallow-NIR search obsolete?\n` +
  `4. Net it out: is NIR-ONLY viable? realistic NET-NEW galaxy-galaxy lens yield? best framing? single biggest risk?\n` +
  `Return CONFIRMED (the ~10-100 fresh-sky thesis holds), DOWNGRADED (real but smaller / needs optical fusion / niche), or REFUTED (VHS too shallow / not actually fresh). Require >=2 corroborating sources. Cite URLs.`,
  { label: 'redteam:vhs-yield', phase: 'Verify', agentType: 'general-purpose', schema: VERIFY_SCHEMA });

log(`VHS deep-dive done. Verdict: ${redteam?.verdict ?? 'N/A (red-team agent returned null)'}; depth: ${redteam?.depth_detectability_verdict ?? 'n/a'}; yield ${redteam?.realistic_net_new_yield ?? 'n/a'}.`);
return { inventory, prior, pipeline: pipeline_, redteam };
