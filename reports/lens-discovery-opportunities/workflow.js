export const meta = {
  name: 'lens-discovery-opportunity-scan',
  description: 'Scan the public sky-survey landscape for un-searched / unpublished strong-lens discovery opportunities across 8 modality families, adversarially verify each, and synthesize a priority-ranked opportunity set.',
  phases: [
    { title: 'Research', detail: 'one agent per survey family researches coverage, access, prior searches, gap' },
    { title: 'Verify', detail: 'adversarial red-team falsifies each flagged opportunity' },
    { title: 'Synthesize', detail: 'merge, dedup, priority-score, build tiers + watch-list' },
    { title: 'Critic', detail: 'completeness critic finds missed families/datasets' },
  ],
}

// ---------------------------------------------------------------------------
// Baseline: what the program has ALREADY searched. Agents must EXCLUDE these
// footprints from "opportunities" (they are the searched baseline, not a gap).
// ---------------------------------------------------------------------------
const BASELINE = `
ALREADY-SEARCHED BASELINE (exclude these as opportunities; they are the program's prior work):
- DESI Legacy Imaging Surveys DR7/DR8/DR9/DR10 (DECaLS + BASS + MzLS, ~14k-19k deg2, grz and DR10 +i):
  exhaustively searched with ResNet/EfficientNet CNN finders (Huang 2020, Huang 2021, Storfer 2024, Inchausti 2025) -> ~3,900 galaxy-scale candidates.
- DESI DR1 (Iron) spectroscopy (28M fiber redshifts): pairwise Friends-of-Friends spectroscopic search (Hsu 2025) + quasar autocorrelation (Dawes 2022) -> ~2,600 candidates.
- DECam per-exposure multi-epoch imaging (NOIRLab) over the DECaLS/DES strong-lens footprint: difference-imaging searches for lensed supernovae (Sheu 2023) and variable lensed quasars (Sheu 2024a).
- Follow-up/confirmation only (NOT discovery): HST/MAST (individual systems), Keck NIRES (KOA), VLT/MUSE (ESO), Euclid Q1 (used only as a benchmark - Q1 is fully picked over by the Euclid Strong Lensing Discovery Engine A-F).
THE PROGRAM'S THREE IN-HOUSE METHODS (use these for fit-to-pipeline):
  (1) optical-cnn: CNN/ViT finder on optical grz(i) imaging cutouts;
  (2) spectro-fof: Friends-of-Friends / redshift-pair search on a spectroscopic redshift catalog;
  (3) diff-imaging: difference-imaging time-domain search on multi-epoch exposures.
  Anything else = new-tooling.
ALREADY-SEARCHED-AND-PUBLISHED (rate RED, low opportunity): UNIONS (GLUE I 2025, Savary 2022), HSC-SSP PDR2/PDR3 (SuGOHI I-X, HOLISMOKES), KiDS DR4/DR5 (Li 2020/21, TEGLIE 2024), DES Y6 (Space Warps ViT / O'Donnell 2025), Euclid Q1 (Discovery Engine), bright Herschel/SPT/ACT submm lensing (Negrello, Vieira).
`;

const RESEARCH_TASK = `For EACH dataset, answer all of: (a) coverage = footprint deg2, depth/limiting-mag-or-flux, resolution/seeing, bands/modality; (b) access = mechanism (cutout API / TAP-ADQL / bulk / archive request), availability (public-now vs embargoed vs future) + the date if future, proprietary period, auth needed, approx data volume, and a CONCRETE endpoint URL; (c) prior_searches = EVERY published strong-lens search you can find (citation, method, area covered, completeness/yield) and CLASSIFY each as discovery (a real published lens catalog), forecast (predictions only), crossmatch (recovered known lenses only), method (technique paper, no blind catalog), or none; (d) gap = exactly what sky/depth/lens-class/modality is unsearched or whose results are unpublished; (e) fit_to_pipeline = which in-house method drops in (optical-cnn / spectro-fof / diff-imaging / new-tooling); (f) new_sky_fraction = estimated NON-overlap with the searched baseline (DECaLS/DES/DESI-DR1) AND with other already-searched surveys; (g) effort_to_access = 1 (public cutout API) to 5 (proprietary or needs new reduction/pipeline). Then assign rag: GREEN = largely un-searched/fresh & public-now; YELLOW = partially searched, or searched-but-results-not-public, or future-but-high-value, or re-searchable deeper; RED = exhaustively searched & published, low residual. Do 6-12 targeted web searches per family, concentrating on (c) prior searches (find the actual discovery paper, not just forecasts) and (b) public-availability status. Be skeptical and precise; cite URLs. Do NOT propose any baseline footprint above as a new opportunity.`;

const RESEARCH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    family: { type: 'string' },
    datasets: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        name: { type: 'string' },
        telescope_instrument: { type: 'string' },
        modality: { type: 'string' },
        footprint_deg2: { type: 'string' },
        depth: { type: 'string' },
        resolution: { type: 'string' },
        bands: { type: 'string' },
        latest_public_dr: { type: 'string' },
        access_mechanism: { type: 'string' },
        availability: { type: 'string', enum: ['public-now', 'embargoed', 'future'] },
        available_date: { type: 'string' },
        proprietary_period: { type: 'string' },
        auth: { type: 'string' },
        volume: { type: 'string' },
        endpoint_url: { type: 'string' },
        prior_searches: { type: 'array', items: {
          type: 'object', additionalProperties: false,
          properties: {
            citation: { type: 'string' }, method: { type: 'string' },
            area_covered: { type: 'string' }, completeness: { type: 'string' },
            type: { type: 'string', enum: ['discovery', 'forecast', 'crossmatch', 'method', 'none'] },
            url: { type: 'string' },
          }, required: ['citation', 'method', 'type'],
        } },
        gap: { type: 'string' },
        fit_to_pipeline: { type: 'string', enum: ['optical-cnn', 'spectro-fof', 'diff-imaging', 'new-tooling'] },
        new_sky_fraction: { type: 'string' },
        expected_yield_oom: { type: 'string' },
        effort_to_access: { type: 'integer' },
        rag: { type: 'string', enum: ['green', 'yellow', 'red'] },
        notes: { type: 'string' },
      },
      required: ['name', 'telescope_instrument', 'modality', 'footprint_deg2', 'latest_public_dr', 'availability', 'endpoint_url', 'prior_searches', 'gap', 'fit_to_pipeline', 'new_sky_fraction', 'effort_to_access', 'rag'],
    } },
  },
  required: ['family', 'datasets'],
};

const REDTEAM_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    dataset_name: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'DOWNGRADED', 'REFUTED'] },
    revised_rag: { type: 'string', enum: ['green', 'yellow', 'red'] },
    searched_verdict: { type: 'string' },
    public_now_verdict: { type: 'string' },
    new_sky_verdict: { type: 'string' },
    method_fit_verdict: { type: 'string' },
    corroborating_sources: { type: 'integer' },
    evidence: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { claim: { type: 'string' }, source_url: { type: 'string' } },
      required: ['claim', 'source_url'],
    } },
    rationale: { type: 'string' },
  },
  required: ['dataset_name', 'verdict', 'revised_rag', 'evidence', 'rationale'],
};

const FAMILIES = [
  { key: 'optical-south', title: 'Wide optical ground imaging - South',
    targets: 'DES Y6/DR2 (Blanco DECam), DELVE DR2 and DR3 (DECam, ~17000 deg2 south), the DECaLS-south DR10 i-band increment, VST-ATLAS DR4, SkyMapper DR4, and the southern portion of Pan-STARRS. Focus especially on DELVE (is there ANY published systematic CNN strong-lens search on DELVE? it overlaps DECaLS-south so quantify the NON-overlap new sky).' },
  { key: 'optical-north-hsc', title: 'Wide optical ground imaging - North & HSC',
    targets: 'HSC-SSP PDR3 AND the newer PDR4 (Subaru HSC, is PDR4 public yet? what new area/depth vs PDR3?), KiDS DR5 (VST), UNIONS (CFHT CFIS u/r + Pan-STARRS/HSC g/i/z), Pan-STARRS 3pi DR2. These are mostly already searched - find the discovery papers and identify only the residual (e.g. PDR4 increment, UNIONS z-band/footprint completion).' },
  { key: 'nir-ground', title: 'Wide-area NIR ground imaging',
    targets: 'VISTA VIKING (~1350 deg2, NIR ZYJHKs), VHS (~20000 deg2 south, JKs), VVV/VVVX (Galactic), UltraVISTA/VIDEO (deep small), UKIDSS LAS (UKIRT), and optionally J-PLUS/S-PLUS narrow-band optical. KEY QUESTION: has anyone done a BLIND systematic strong-lens search on wide NIR imaging, or is it only counterpart/proximity selection? NIR favors high-z and dusty lensing galaxies.' },
  { key: 'space-highres', title: 'Space high-resolution imaging',
    targets: 'Euclid Q1 (RED, benchmark) vs Euclid DR1 wide (future - when public?), HST archive mosaics not uniformly swept (COSMOS, CANDELS, CLASH, RELICS, Frontier Fields, SL2S fields, HLA), JWST PUBLIC fields (COSMOS-Web, CEERS, JADES, PRIMER, UNCOVER, NGDEEP) - any systematic vs serendipitous lens search?, Roman (future), Gaia GraL/GravLens lensed-quasar catalogs (how complete at small separation?).' },
  { key: 'spectroscopy', title: 'Optical spectroscopy',
    targets: 'DESI DR2 spectra (when public? only chains released 2025?), SDSS/BOSS/eBOSS legacy spectroscopy (SLACS/BELLS/BELLS-GALLERY/S4TM used emission-line-residual; has the FoF/redshift-PAIR method been exhaustively applied? -> methodological re-mine opportunity), LAMOST DR11/DR12, DESI-II/Spec-S5 (future), WEAVE, 4MOST, Subaru PFS (future), GAMA/2dF/6dF. Distinguish data-already-public (SDSS/LAMOST = re-minable now) from future.' },
  { key: 'radio', title: 'Radio continuum surveys',
    targets: 'VLASS (VLA 2-4 GHz, ~33885 deg2, 2.5"), LOFAR LoTSS DR2 (144 MHz, ~5700 deg2, sub-arcsec with ILT), ASKAP EMU (~20000 deg2 south) & RACS, MeerKAT MIGHTEE/MALS, legacy FIRST/NVSS. KEY: is there any BLIND real-data radio strong-lens DISCOVERY catalog, or only forecasts + known-lens crossmatches? Radio = modality mismatch -> new-tooling, but high novelty. Quantify.' },
  { key: 'submm-mm', title: 'Submillimeter / millimeter surveys',
    targets: 'Herschel H-ATLAS/HerMES/HeLMS (SPIRE 250/350/500um - bright S500>100mJy selection famously complete & published; is there a SUB-threshold or faint residual?), SPT-SZ/SPT-pol/SPT-3G (3G is deeper - residual?), ACT DR6, Planck PHz/PCCS, and the ALMA SCIENCE ARCHIVE (ALCS, ALMACAL, calibrator fields - serendipitous lenses, no exhaustive systematic search). Separate bright-source (RED) from faint/archive (YELLOW-GREEN).' },
  { key: 'time-domain', title: 'Time-domain / alert streams',
    targets: 'ZTF (P48, north, public DRs/alerts - lensed-SN & lensed-QSO searches), ATLAS (all-sky shallow, forced-phot public - under-mined for lensing?), Gaia Science Alerts, Rubin/LSST (future - the eventual prize, when DR?), and DECam multi-epoch OUTSIDE the Sheu 2023/2024a footprint (e.g. DES SN deep fields, DECaLS multi-epoch not yet diff-imaged). diff-imaging method fits here.' },
];

// ===========================================================================
// PHASE A+B: research -> red-team, pipelined per family (no barrier)
// ===========================================================================
const verifiedFamilies = await pipeline(
  FAMILIES,
  // STAGE 1: research the family
  (fam) => agent(
    `You are a strong-gravitational-lensing data-landscape researcher. Family: "${fam.title}".\n\n${BASELINE}\n\nResearch these datasets: ${fam.targets}\n\n${RESEARCH_TASK}\n\nReturn the structured object for family key "${fam.key}". Include every dataset you research, even RED ones (with the discovery-paper citation that makes them RED).`,
    { label: `research:${fam.key}`, phase: 'Research', agentType: 'general-purpose', schema: RESEARCH_SCHEMA }
  ),
  // STAGE 2: adversarially verify each GREEN/YELLOW flagged dataset
  async (research, fam) => {
    if (!research || !research.datasets) return { family: fam.key, title: fam.title, datasets: [], verdicts: [] };
    const flagged = research.datasets.filter(d => d.rag === 'green' || d.rag === 'yellow');
    const verdicts = await parallel(flagged.map(d => () =>
      agent(
        `You are an ADVERSARIAL red-team verifier. A researcher claims this strong-lens discovery OPPORTUNITY is "${d.rag}":\n\n` +
        `Dataset: ${d.name} (${d.telescope_instrument})\nModality: ${d.modality}\nClaimed footprint: ${d.footprint_deg2}; latest DR: ${d.latest_public_dr}; availability: ${d.availability} ${d.available_date || ''}\n` +
        `Claimed gap: ${d.gap}\nClaimed prior searches: ${JSON.stringify(d.prior_searches)}\nClaimed new-sky fraction: ${d.new_sky_fraction}\nClaimed fit: ${d.fit_to_pipeline}\n\n` +
        `Try HARD to FALSIFY the opportunity on four axes, doing your own web searches:\n` +
        `1. HAS IT ACTUALLY BEEN SEARCHED? Find any published strong-lens DISCOVERY catalog on this data (not a forecast, not a known-lens crossmatch). If one exists covering the claimed gap, the opportunity is weaker -> DOWNGRADE/REFUTE.\n` +
        `2. IS THE DATA PUBLIC NOW? Confirm the release/proprietary status and date. If it is not actually public yet, mark it future (DOWNGRADE to a watch-list YELLOW, not an actionable GREEN).\n` +
        `3. IS IT ACTUALLY NEW SKY? Check footprint overlap with the searched baseline (DECaLS/DES/DESI-DR1) and other searched surveys. If mostly overlapping already-searched sky, DOWNGRADE.\n` +
        `4. DOES A USABLE METHOD FIT? If it needs entirely new tooling (e.g. radio/submm/IFU), keep it but flag the build cost.\n\n` +
        `Require >=2 independent corroborating sources to let a GREEN stand. Return verdict CONFIRMED (claim holds), DOWNGRADED (real but lower rating - set revised_rag), or REFUTED (not a real opportunity - revised_rag=red). Cite source URLs for every claim.`,
        { label: `verify:${fam.key}:${d.name}`.slice(0, 60), phase: 'Verify', agentType: 'general-purpose', schema: REDTEAM_SCHEMA }
      ).then(v => v).catch(() => null)
    ));
    return { family: fam.key, title: fam.title, datasets: research.datasets, verdicts: verdicts.filter(Boolean) };
  }
);

const families = verifiedFamilies.filter(Boolean);
log(`Researched + verified ${families.length} families; ${families.reduce((n, f) => n + (f.datasets ? f.datasets.length : 0), 0)} datasets total.`);

// Merge red-team verdicts into each dataset (revised_rag wins).
for (const fam of families) {
  const vmap = {};
  for (const v of (fam.verdicts || [])) vmap[v.dataset_name] = v;
  for (const d of (fam.datasets || [])) {
    const v = vmap[d.name];
    if (v) { d.verified_rag = v.revised_rag; d.verdict = v.verdict; d.verify_rationale = v.rationale; d.verify_evidence = v.evidence; d.corroborating_sources = v.corroborating_sources; }
    else { d.verified_rag = d.rag; d.verdict = 'UNVERIFIED'; }
  }
}

// ===========================================================================
// PHASE C: synthesis (barrier) - priority scoring, tiers, cross-cutting
// ===========================================================================
phase('Synthesize');
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    master: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        name: { type: 'string' }, family: { type: 'string' }, modality: { type: 'string' },
        footprint_deg2: { type: 'string' }, availability: { type: 'string' },
        verified_rag: { type: 'string' }, prior_status: { type: 'string' },
        new_sky_fraction: { type: 'string' }, fit_to_pipeline: { type: 'string' },
        effort_to_access: { type: 'integer' }, expected_yield_oom: { type: 'string' },
        priority_score: { type: 'number' }, tier: { type: 'string', enum: ['dropin-now', 'newtooling-now', 'future-watch', 'low'] },
        one_line: { type: 'string' },
      },
      required: ['name', 'family', 'modality', 'verified_rag', 'fit_to_pipeline', 'priority_score', 'tier', 'one_line'],
    } },
    shortlist_dropin: { type: 'array', items: { type: 'string' } },
    shortlist_newtooling: { type: 'array', items: { type: 'string' } },
    shortlist_future: { type: 'array', items: { type: 'string' } },
    cross_cutting_overlap: { type: 'string' },
    cross_cutting_method_matrix: { type: 'string' },
    searched_but_unpublished: { type: 'array', items: { type: 'string' } },
    scoring_rubric: { type: 'string' },
  },
  required: ['master', 'shortlist_dropin', 'shortlist_newtooling', 'shortlist_future', 'cross_cutting_overlap', 'cross_cutting_method_matrix', 'searched_but_unpublished', 'scoring_rubric'],
};

const compact = families.map(f => ({ family: f.key, title: f.title, datasets: (f.datasets || []).map(d => ({
  name: d.name, modality: d.modality, footprint_deg2: d.footprint_deg2, availability: d.availability, available_date: d.available_date,
  fit_to_pipeline: d.fit_to_pipeline, new_sky_fraction: d.new_sky_fraction, effort_to_access: d.effort_to_access,
  expected_yield_oom: d.expected_yield_oom, original_rag: d.rag, verified_rag: d.verified_rag, verdict: d.verdict,
  gap: d.gap, prior_searches: d.prior_searches, verify_rationale: d.verify_rationale,
})) }));

const synthesis = await agent(
  `You are the synthesis lead for a strong-lens discovery opportunity report. Below is the verified research across ${families.length} survey families (each dataset has an original_rag and a red-team verified_rag/verdict).\n\n` +
  `DATA:\n${JSON.stringify(compact)}\n\n` +
  `Produce:\n` +
  `1) master: one row per dataset across ALL families. Compute priority_score 0-100 = f(verified_rag [green>yellow>red], fit_to_pipeline [optical-cnn & spectro-fof & diff-imaging are drop-in and score higher than new-tooling], new_sky area [more virgin sky = higher], expected_yield, and effort_to_access [lower effort = higher]). Assign tier: dropin-now (public-now AND fits an in-house method AND not exhausted), newtooling-now (public-now, real opportunity, but needs new tooling e.g. radio/submm/IFU), future-watch (not public yet but high ceiling), low (RED/exhausted). Give a crisp one_line per dataset. prior_status = short phrase like "no blind discovery search" or "DES Y6 ViT published".\n` +
  `2) shortlist_dropin / shortlist_newtooling / shortlist_future: ordered dataset names, best first.\n` +
  `3) cross_cutting_overlap: prose on which sky is multiply-searched vs virgin (footprint overlap), including the key dedup facts (DELVE vs DECaLS, VIKING vs KiDS, etc).\n` +
  `4) cross_cutting_method_matrix: prose mapping the three in-house methods to which opportunities they drop into vs which need a build.\n` +
  `5) searched_but_unpublished: datasets where a search likely happened but no public catalog exists (worth a direct inquiry).\n` +
  `6) scoring_rubric: one paragraph stating exactly how you computed priority_score.\n` +
  `Be decisive and rank clearly.`,
  { label: 'synthesize', phase: 'Synthesize', agentType: 'general-purpose', schema: SYNTH_SCHEMA }
);

// ===========================================================================
// PHASE D: completeness critic
// ===========================================================================
phase('Critic');
const CRITIC_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { missed: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: { name: { type: 'string' }, why_relevant: { type: 'string' }, likely_rag: { type: 'string' }, fit: { type: 'string' } },
    required: ['name', 'why_relevant'],
  } }, assessment: { type: 'string' } },
  required: ['missed', 'assessment'],
};
const covered = families.flatMap(f => (f.datasets || []).map(d => d.name));
const critic = await agent(
  `You are a completeness critic for a strong-lens discovery data-landscape report. We covered these datasets:\n${covered.join(', ')}\n\n` +
  `Across families: optical ground (N&S), wide-area NIR, space high-res, optical spectroscopy, radio, submm/mm, time-domain.\n\n` +
  `What PUBLIC (or imminent) telescope dataset / archive relevant to discovering strong gravitational lenses did we MISS? Think about: other ground imaging (e.g. DECaLS/DELVE-DEEP, NSC NOIRLab Source Catalog, J-PAS, Euclid deep fields, SDSS Stripe 82 coadds, DES deep fields, KiDZ), grism/slitless spectroscopy (Euclid NISP grism, JWST NIRISS, HST grism), other radio (GMRT TGSS, Apertif), microwave (CMB-S4 future), space NIR (WISE/unWISE, Spitzer SWIRE), and any all-sky catalog cross-match opportunities (Gaia x WISE quasar pairs). For each genuinely-missed item give name, why_relevant, likely_rag, and fit. Do a few web searches to check. Keep to items that are real gaps in our coverage; do not repeat what we already covered.`,
  { label: 'completeness-critic', phase: 'Critic', agentType: 'general-purpose', schema: CRITIC_SCHEMA }
);

log(`Synthesis complete: ${synthesis.master.length} ranked datasets. Critic flagged ${critic.missed.length} possibly-missed items.`);

return { families, synthesis, critic };
