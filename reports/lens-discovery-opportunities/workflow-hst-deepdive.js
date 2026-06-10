export const meta = {
  name: 'hst-archive-lens-deepdive',
  description: 'Pressure-test the HST general-archive strong-lens opportunity: quantify lens-searchable area, the never-ML-searched residual after prior searches, a concrete cutout-generation pipeline, and an adversarially-verified yield estimate.',
  phases: [
    { title: 'Research', detail: 'inventory + prior-coverage + access pipeline, in parallel' },
    { title: 'Verify', detail: 'adversarial red-team on the area & yield headline' },
  ],
}

const CONTEXT = `
Program context: a strong-lensing ML group (X. Huang) runs an optical CNN lens-finder trained on
DESI Legacy grz cutouts. We are evaluating the PUBLIC HST imaging archive (Hubble Legacy Archive /
MAST / Hubble Source Catalog + treasury & cluster mosaics) as a drop-in target for that CNN,
retrained for HST PSF/bands. We need to know (a) how much HST imaging is genuinely lens-SEARCHABLE,
(b) how much has ALREADY been searched for strong lenses (to compute the never-searched residual),
and (c) a concrete cutout-generation pipeline. Be quantitative and cite URLs. Galaxy-scale
galaxy-galaxy lenses are the primary target; cluster-scale arcs secondary.
`;

const INVENTORY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    total_unique_extragalactic_area_deg2: { type: 'string' },
    area_basis: { type: 'string' },
    contiguous_fields: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { name: { type: 'string' }, instrument_bands: { type: 'string' }, area_deg2: { type: 'string' }, depth: { type: 'string' }, contiguous: { type: 'string' }, notes: { type: 'string' } },
      required: ['name', 'instrument_bands', 'area_deg2'] } },
    lens_searchable_area_deg2: { type: 'string' },
    searchability_criteria: { type: 'string' },
    lens_surface_density_per_deg2: { type: 'string' },
    surface_density_basis: { type: 'string' },
    implied_total_lenses_in_archive: { type: 'string' },
    caveats: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['total_unique_extragalactic_area_deg2', 'lens_searchable_area_deg2', 'lens_surface_density_per_deg2', 'implied_total_lenses_in_archive', 'sources'],
};

const PRIOR_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    searches: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { citation: { type: 'string' }, fields_or_area_covered: { type: 'string' }, method: { type: 'string' }, n_lenses: { type: 'string' }, archive_wide: { type: 'string' }, url: { type: 'string' } },
      required: ['citation', 'fields_or_area_covered', 'method'] } },
    estimated_area_already_searched_deg2: { type: 'string' },
    archive_wide_serendipitous_coverage: { type: 'string' },
    never_searched_residual_deg2: { type: 'string' },
    residual_rationale: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['searches', 'estimated_area_already_searched_deg2', 'never_searched_residual_deg2', 'residual_rationale', 'sources'],
};

const PIPELINE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    access_mechanisms: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { name: { type: 'string' }, endpoint: { type: 'string' }, what: { type: 'string' }, bulk_or_cutout: { type: 'string' } },
      required: ['name', 'endpoint', 'what'] } },
    recommended_data_product: { type: 'string' },
    band_strategy: { type: 'string' },
    pixel_scale_and_psf: { type: 'string' },
    cutout_tiling_approach: { type: 'string' },
    cnn_adaptation: { type: 'string' },
    practical_gotchas: { type: 'string' },
    rough_data_volume: { type: 'string' },
    effort_estimate: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['access_mechanisms', 'recommended_data_product', 'band_strategy', 'cutout_tiling_approach', 'cnn_adaptation', 'practical_gotchas', 'sources'],
};

phase('Research');
const [inventory, prior, pipeline_] = await parallel([
  () => agent(
    `${CONTEXT}\nTASK A - HST IMAGING ARCHIVE INVENTORY & LENS-SEARCHABLE AREA.\n` +
    `Quantify: (1) total UNIQUE extragalactic sky area imaged by HST ACS/WFC3 (deg2) - distinguish all-pointings vs the contiguous deep/treasury fields, and state your basis (e.g. Hubble Source Catalog footprint, MAST exposure tallies). (2) The major CONTIGUOUS imaging fields suitable for arc-finding (COSMOS, CANDELS GOODS-N/S/EGS/UDS, GEMS, AEGIS, CLASH, RELICS, Frontier Fields, BUFFALO, CEERS-HST, etc.) with area/bands/depth each. (3) The LENS-SEARCHABLE subset (depth + resolution + enough contiguous area for arcs; note single-band ACS F814W is the workhorse). (4) The strong-lens SURFACE DENSITY at HST depth (lenses per deg2) WITH a literature basis (e.g. COSMOS lens counts, Collett 2015 forecasts scaled to HST depth). (5) Implied total galaxy-scale lenses sitting in the archive. Do 6-10 web searches; cite URLs.`,
    { label: 'A:inventory', phase: 'Research', agentType: 'general-purpose', schema: INVENTORY_SCHEMA }),
  () => agent(
    `${CONTEXT}\nTASK B - PRIOR HST STRONG-LENS SEARCH COVERAGE (the de-dup baseline).\n` +
    `Find EVERY published strong-lens search/catalog performed on HST imaging and what area/fields each covered, so we can compute the NEVER-searched residual. Cover at least: Faure et al. 2008 (COSMOS), Jackson 2008, Pourrahmani/LensFlow 2018 (COSMOS), Garvin et al. 2022 Hubble Asteroid Hunter II (archive-wide serendipitous via citizen science - quantify how much of the archive it effectively scanned), More et al. 2016, Sonnenfeld SL2S/HST follow-ups, cluster-lens programs (CLASH/Frontier Fields/RELICS lens models - these are targeted, not blind galaxy-galaxy searches), and any CNN/ML lens search on HST (note: COWLS is JWST, not HST - flag that). For each: method (visual/CNN/citizen/serendipitous/targeted-cluster), area or fields, number of lenses. Then estimate (a) total area already lens-searched, (b) how much of the contiguous archive remains NEVER blind-galaxy-galaxy-searched. Be skeptical: archive-wide serendipitous projects (Hubble Asteroid Hunter, SpaceWarps-HST) may already cover much of it. Do 6-10 web searches; cite URLs.`,
    { label: 'B:prior-coverage', phase: 'Research', agentType: 'general-purpose', schema: PRIOR_SCHEMA }),
  () => agent(
    `${CONTEXT}\nTASK C - DATA ACCESS & CUTOUT-GENERATION PIPELINE.\n` +
    `Lay out a concrete, practical pipeline to turn the public HST archive into CNN-ingestible cutouts. Cover: (1) ACCESS mechanisms with real endpoints - MAST (astroquery.mast Observations + cutouts), Hubble Legacy Archive (HLA) drizzled products + footprint service, Hubble Source Catalog (HSC) as a deflector/source catalog, MAST 'hapcut'/astrocut/cutout services, hips2fits for mosaic cutouts, the MAST AWS Open Data public bucket for bulk. (2) RECOMMENDED data product (e.g. HLA drizzled mosaics vs per-visit FLT/DRZ vs HAP single-visit/total-visit mosaics) and why. (3) BAND strategy: ACS/WFC3 are mostly single-band per field; F814W is the most common; where to get multi-band (CANDELS, COSMOS); how to make 3-channel inputs from few bands. (4) Pixel scale / PSF (ACS 0.05, WFC3-IR 0.13, drizzled 0.03-0.06) and how that differs from the grz-trained CNN. (5) TILING approach: how to enumerate footprints, pick deflector positions (HSC/photo-z galaxies vs blind grid), make ~64-100px cutouts at HST scale. (6) CNN ADAPTATION: what it takes to retrain the existing grz CNN for HST single-band/space-PSF (transfer learning, simulate HST lenses with lenstronomy/deeplenstronomy at HST PSF). (7) Practical gotchas (drizzle artifacts, cosmic rays, varying depth/PSF across the archive, chip gaps, single-band ambiguity). (8) Rough data volume and effort. Do 6-10 web searches for the access endpoints; cite URLs.`,
    { label: 'C:pipeline', phase: 'Research', agentType: 'general-purpose', schema: PIPELINE_SCHEMA }),
]);

phase('Verify');
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['CONFIRMED', 'DOWNGRADED', 'REFUTED'] },
    searchable_area_verdict: { type: 'string' },
    already_searched_verdict: { type: 'string' },
    net_residual_area_deg2: { type: 'string' },
    realistic_net_new_yield: { type: 'string' },
    biggest_risk_to_the_thesis: { type: 'string' },
    corroborating_sources: { type: 'integer' },
    evidence: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { claim: { type: 'string' }, source_url: { type: 'string' } }, required: ['claim', 'source_url'] } },
    rationale: { type: 'string' },
  },
  required: ['verdict', 'net_residual_area_deg2', 'realistic_net_new_yield', 'biggest_risk_to_the_thesis', 'rationale', 'evidence'],
};
const redteam = await agent(
  `You are an adversarial red-team verifier. The thesis under test: "the public HST imaging archive contains hundreds of galaxy-galaxy strong lenses that have never been systematically ML-searched, and our grz-trained CNN (retrained for HST) could harvest O(10^2) NEW lenses from it."\n\n` +
  `Inventory finding: ${JSON.stringify(inventory)}\n\nPrior-coverage finding: ${JSON.stringify(prior)}\n\n` +
  `Try HARD to FALSIFY or shrink the opportunity, doing your own web searches:\n` +
  `1. Is the lens-SEARCHABLE area really that large, or is HST so pencil-beam (a few deg2 of deep contiguous imaging) that the absolute lens count is small?\n` +
  `2. Has the archive ALREADY been effectively covered? Check archive-wide projects (Hubble Asteroid Hunter / Garvin 2022, any HST SpaceWarps, the Hubble Source Catalog-based searches) and the fact that the deepest fields (COSMOS, CANDELS, Frontier Fields) are the MOST-studied sky on the planet.\n` +
  `3. Will Euclid/JWST/ground (HSC) have already found the bright lenses in these exact fields?\n` +
  `4. Net it out: realistic NEVER-searched lens-searchable area (deg2) and realistic NET-NEW galaxy-scale lens yield. Give the single biggest risk to the thesis.\n` +
  `Return CONFIRMED (hundreds plausible), DOWNGRADED (real but tens, not hundreds), or REFUTED. Require >=2 corroborating sources. Cite URLs.`,
  { label: 'redteam:hst-yield', phase: 'Verify', agentType: 'general-purpose', schema: VERIFY_SCHEMA });

log(`HST deep-dive done. Red-team verdict: ${redteam.verdict}; net residual ${redteam.net_residual_area_deg2}; yield ${redteam.realistic_net_new_yield}.`);
return { inventory, prior, pipeline: pipeline_, redteam };
