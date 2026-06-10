export const meta = {
  name: 'gama-delve-lens-deepdive',
  description: 'Pressure-test the two surviving "existing-technique-now" drop-ins: GAMA DR4 (does multi-pass sampling really defeat the fiber-collision block for spectro-FoF lens pairs?) and DELVE DR2 (how much is genuinely fresh vs already in the group\'s own Legacy DR10 search?). 3 research agents + adversarial red-team per survey.',
  phases: [
    { title: 'Research', detail: 'per survey: method/coverage/access agents in parallel' },
    { title: 'Verify', detail: 'adversarial red-team on each survey\'s load-bearing claim' },
  ],
}

const BASELINE = `
Program context: X. Huang strong-lensing group. In-house methods: (1) optical-cnn on grz(i) cutouts;
(2) spectro-fof = Friends-of-Friends / redshift-PAIR search on a spectroscopic redshift catalog
(Hsu et al. 2025, arXiv:2509.16033, applied to DESI DR1) PLUS the single-fiber blended-spectrum
channel (two redshifts in one spectrum); (3) diff-imaging. Already-searched baseline (exclude):
DESI Legacy Imaging DR7-10 (DECaLS+BASS+MzLS) optical-CNN searched (Huang 2020/2021, Storfer 2024,
Inchausti 2025 / arXiv:2508.20087 - DR10 covered ~14,000 deg2 south of dec +32 with a z<20 deflector
cut, and INGESTED DELVE/DeROSITAS DECam exposures); DESI DR1 spectroscopy FoF-searched (Hsu 2025);
DECam multi-epoch diff-imaging (Sheu 2023/2024a). Be quantitative, cite URLs.
`;

const RESEARCH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    survey: { type: 'string' },
    role: { type: 'string' },
    summary: { type: 'string' },
    key_facts: { type: 'string' },
    load_bearing_verdict: { type: 'string' },
    risks: { type: 'string' },
    access_or_method: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['survey', 'role', 'summary', 'load_bearing_verdict', 'sources'],
};

const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    survey: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'DOWNGRADED', 'REFUTED'] },
    load_bearing_claim_holds: { type: 'string' },
    realistic_net_new_yield: { type: 'string' },
    best_framing: { type: 'string' },
    biggest_risk: { type: 'string' },
    corroborating_sources: { type: 'integer' },
    evidence: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { claim: { type: 'string' }, source_url: { type: 'string' } }, required: ['claim', 'source_url'] } },
    rationale: { type: 'string' },
  },
  required: ['survey', 'verdict', 'load_bearing_claim_holds', 'realistic_net_new_yield', 'biggest_risk', 'rationale', 'evidence'],
};

const SURVEYS = [
  {
    key: 'GAMA-DR4',
    thesis: 'GAMA DR4 is a clean drop-in for the spectro-FoF method because its multi-pass tiling defeats the SDSS/BOSS fiber-collision block; realistic net-new yield 10^1-10^2 galaxy-galaxy lenses.',
    roles: [
      { role: 'method-viability', prompt:
        `${BASELINE}\nSURVEY: GAMA DR4. ROLE: METHOD VIABILITY (the load-bearing question).\n` +
        `Does GAMA's multi-pass tiling ACTUALLY enable the spectro-FoF/redshift-PAIR lens search at the angular separations relevant to STRONG lensing? Determine: (1) GAMA's AAOmega/2dF fiber size and fiber-collision/minimum-separation limit per pass, and how many passes (multi-pass) GAMA used to reach close pairs - what is the resulting minimum resolvable two-fiber pair separation? (2) The SDSS/BOSS fiber-collision radius (55"/62") that BLOCKS two fibers on a galaxy-galaxy pair - does GAMA genuinely beat it, and DOWN TO WHAT SEPARATION? (3) CRUCIAL physics: galaxy-galaxy strong lenses have the source image at theta_E ~ 0.5-2" from the deflector, far BELOW any two-fiber separation - so the TWO-FIBER FoF channel (Hsu 2025) likely only catches WIDE pairs / group-galaxy-scale / "dimple" lenses, NOT classic galaxy-galaxy lenses. Is that right? (4) The OTHER channel - single-fiber BLENDED spectrum (deflector + lensed-source emission lines in one fiber) - is that what actually works on GAMA, and was it already done (Holwerda et al. 2015)? Conclude: which lens channel is genuinely OPEN and viable on GAMA, and at what separation regime. Do 6-10 web searches; cite URLs.` },
      { role: 'prior-coverage', prompt:
        `${BASELINE}\nSURVEY: GAMA DR4. ROLE: PRIOR LENS-SEARCH COVERAGE.\n` +
        `Find every strong-lens search done on GAMA spectroscopy: Holwerda et al. 2015 (blended-spectra galaxy-galaxy lens search in GAMA - what method, how complete, how many found), Knabel et al. 2020 (GAMA/KiDS lens comparison), any GAMA pair/FoF lens search, and GAMA's role in KiDS lens follow-up. Quantify what's already published and what residual a fresh FoF or blended-spectrum search would add. Note GAMA's full redshift catalog is public and ~98% complete to r<19.8. Do 6-10 web searches; cite URLs.` },
      { role: 'footprint-yield-access', prompt:
        `${BASELINE}\nSURVEY: GAMA DR4. ROLE: FOOTPRINT, YIELD & ACCESS.\n` +
        `Quantify: GAMA DR4 footprint (~286 deg2 across 5 fields G02/G09/G12/G15/G23), N spectra/redshifts (~300k), redshift completeness/depth (r<19.8), and a realistic strong-lens yield from a spectroscopic pair/blended search over that small, shallow-ish, low-z sample (compare to Hsu 2025's DESI DR1 yield scaled by N and depth). Is the absolute yield capped by the tiny area + bright limit? Data access (TAP/ADQL + bulk at gama-survey.org/dr4). Do 6-10 web searches; cite URLs.` },
    ],
    redteamPrompt:
      `Thesis under test: "GAMA DR4 is a clean spectro-FoF drop-in whose multi-pass sampling defeats the fiber-collision block, yielding 10^1-10^2 net-new galaxy-galaxy lenses." Try HARD to falsify: (1) METHOD - does the two-fiber FoF actually reach galaxy-galaxy lens separations (theta_E~0.5-2") or only wide/group pairs? Is the only real channel the single-fiber blended spectrum, and was that already done (Holwerda 2015)? (2) YIELD - is the absolute yield crushed by the tiny 286 deg2 area + bright r<19.8 limit + low-z sample? (3) FRESHNESS - is the residual real after Holwerda 2015 / Knabel 2020? Net out realistic net-new yield and best framing. Return CONFIRMED / DOWNGRADED / REFUTED, >=2 corroborating sources, cite URLs.`,
  },
  {
    key: 'DELVE-DR2',
    thesis: 'Re-running the optical CNN on DELVE DR2 griz harvests net-new southern lenses because most of its ~17-21k deg2 is deep, arc-capable, and not yet in a published lens catalog.',
    roles: [
      { role: 'fresh-sky-dedup', prompt:
        `${BASELINE}\nSURVEY: DELVE DR2. ROLE: FRESH-SKY DE-DUP (the load-bearing question).\n` +
        `How much of DELVE DR2 (~17,000-21,000 deg2 griz) is GENUINELY not already covered by a published optical-CNN lens search - specifically the Huang group's OWN DESI Legacy DR10 search (Inchausti 2025, ~14,000 deg2 south of dec +32, which INGESTED DELVE/DeROSITAS DECam exposures), DES Y6 (~5,000 deg2), and DELVE DR1 (Zaborowski 2023, ~4,000 deg2)? Quantify the overlap precisely (declination ranges, |b| cuts, the dec +32 boundary) and the genuinely-unsearched-AND-deep residual area in deg2. Is DELVE DR2 mostly the SAME DECam sky already searched via Legacy DR10, or is there a large fresh component (e.g. the far south below DES, low-|b| fill-in, dec < -18 not in DR10)? This is THE question. Do 6-10 web searches; cite URLs.` },
      { role: 'prior-coverage-reduction', prompt:
        `${BASELINE}\nSURVEY: DELVE DR2. ROLE: PRIOR COVERAGE + REDUCTION DIFFERENCE.\n` +
        `(1) What lens searches exist on DELVE? Zaborowski 2023 (DR1, ~4,000 deg2, 5-layer CNN, 581 candidates) - confirm scope; any DR2 search. (2) KEY second-order question: even where DELVE DR2 sky overlaps already-searched Legacy DR10, does DELVE DR2's INDEPENDENT DESDM/DECADE coadd reduction (different from Legacy Survey Tractor models, different coadd/PSF handling) plausibly surface DIFFERENT/NEW candidates on the same sky - i.e. is a re-search of overlapping sky scientifically justified, or just duplication? Cite evidence on how much candidate catalogs differ between independent reductions of the same DECam data. Do 6-10 web searches; cite URLs.` },
      { role: 'depth-access-pipeline', prompt:
        `${BASELINE}\nSURVEY: DELVE DR2. ROLE: DEPTH, ARC-CAPABILITY & CUTOUT PIPELINE.\n` +
        `Quantify DELVE DR2 depth/bands (griz 5-sigma AB: ~g24.3/r23.9/i23.5/z22.8?), seeing, vs DESI Legacy (is it deeper, comparable, or shallower? does the added i-band help?), and confirm it is arc-capable (unlike shallow NIR/SkyMapper). Then give the concrete cutout pipeline to re-run the existing grz(i) CNN: NOIRLab Astro Data Lab SIA cutouts (delve_dr3 / delve_dr2), TAP catalog (delve_dr2.objects) for deflector seeding (z<20 or fainter), bulk on Data Lab/S3, and what changes vs the Legacy-Survey pipeline the group already uses. Effort. Do 6-10 web searches; cite URLs.` },
    ],
    redteamPrompt:
      `Thesis under test: "Re-running the optical CNN on DELVE DR2 griz harvests net-new southern lenses because most of its ~17-21k deg2 is deep, arc-capable, and not yet in a published lens catalog." Try HARD to falsify: (1) DE-DUP - is most of DELVE DR2 actually the SAME DECam sky the group already searched via Legacy DR10 (which ingested DELVE exposures), so genuinely-fresh-AND-deep residual is small? (2) Even on overlapping sky, does an independent DESDM reduction justify a re-search, or is it duplication? (3) Is DELVE DR2 actually arc-capable depth (vs being shallower than Legacy)? (4) Will Rubin/LSST harvest this exact southern sky far deeper within ~2-3 years? Net out: realistic net-new yield (deg2 of fresh-deep sky x lens density) and best framing. Return CONFIRMED / DOWNGRADED / REFUTED, >=2 corroborating sources, cite URLs.`,
  },
];

phase('Research');
const results = await pipeline(
  SURVEYS,
  // STAGE 1: 3 research agents per survey, in parallel
  (s) => parallel(s.roles.map(r => () =>
    agent(`${r.prompt}\n\nReturn survey="${s.key}", role="${r.role}".`,
      { label: `res:${s.key}:${r.role}`.slice(0, 58), phase: 'Research', agentType: 'general-purpose', schema: RESEARCH_SCHEMA })
      .catch(() => null)
  )).then(research => ({ survey: s, research: research.filter(Boolean) })),
  // STAGE 2: adversarial red-team using the research
  async (bundle) => {
    if (!bundle) return null;
    const { survey, research } = bundle;
    const verdict = await agent(
      `You are an adversarial red-team verifier for strong-lens discovery opportunities.\n\n${BASELINE}\n\n` +
      `SURVEY: ${survey.key}. THESIS: ${survey.thesis}\n\n` +
      `Research findings (3 agents):\n${JSON.stringify(research)}\n\n${survey.redteamPrompt}\n\nReturn survey="${survey.key}".`,
      { label: `verify:${survey.key}`, phase: 'Verify', agentType: 'general-purpose', schema: VERIFY_SCHEMA }
    ).catch(() => null);
    return { survey: survey.key, research, verdict };
  }
);

const clean = results.filter(Boolean);
for (const r of clean) {
  log(`${r.survey}: ${r.verdict?.verdict ?? 'N/A (red-team null)'} - yield ${r.verdict?.realistic_net_new_yield ?? 'n/a'}`);
}
return { results: clean };
