export const meta = {
  name: 'atlas-ztf-timedomain-deepdive',
  description: 'Pressure-test the ATLAS/ZTF time-domain diff-imaging opportunity (the last unverified Tier-A lead): is the lensed-transient content genuinely unsearched, or already mined by dedicated lensed-SN/lensed-QSO teams and about to be dominated by Rubin? Separate lensed-SN vs lensed-QSO channels; adversarial red-team.',
  phases: [
    { title: 'Research', detail: 'lensed-SN saturation, lensed-QSO saturation, method-fit+obsolescence (parallel)' },
    { title: 'Verify', detail: 'adversarial red-team on the combined thesis' },
  ],
}

const CTX = `
Program context: X. Huang strong-lensing group. Relevant in-house method: DIFFERENCE IMAGING for
lensed transients - Sheu et al. 2023 (retrospective lensed-supernova search via Bramich difference
imaging at the positions of the 5,807 known DESI strong lenses, on archived DECam multi-epoch) and
Sheu et al. 2024a (variable lensed quasars, same DECam diff-imaging core + variability layer). The
forward-looking THESIS under test: "extend this diff-imaging approach to ZTF and ATLAS time-domain
data NOW to harvest net-new lensed transients, because the time-domain CONTENT (variability) is a
different, largely-unsearched axis from static-imaging coverage." Claimed yields: ATLAS = a handful
of highly-magnified glSNe; ZTF = tens of glSNe + tens-to-hundreds of lensed-QSO candidates.
ATLAS (Tonry): all-sky, shallow (~m19-19.5), public forced-photometry server. ZTF: northern,
deeper (~m20.5), public DRs + world-public broker alert stream (ALeRCE/Fink/Lasair/ANTARES).
Be quantitative (rates, depths, counts), cite URLs, and SEPARATE the lensed-SN and lensed-QSO cases.
`;

const RESEARCH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    channel: { type: 'string' },
    summary: { type: 'string' },
    prior_search_saturation: { type: 'string' },
    teams_and_programs: { type: 'string' },
    realistic_yield: { type: 'string' },
    is_content_actually_unsearched: { type: 'string' },
    key_numbers: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } },
  },
  required: ['channel', 'summary', 'prior_search_saturation', 'realistic_yield', 'is_content_actually_unsearched', 'sources'],
};

phase('Research');
const [lsn, lqso, method] = await parallel([
  () => agent(
    `${CTX}\nROLE: LENSED-SUPERNOVA (glSN) channel in ZTF & ATLAS.\n` +
    `How saturated is the lensed-supernova search in ZTF and ATLAS? Catalogue the dedicated programs/teams and discoveries: iPTF16geu (Goobar et al. 2017), SN Zwicky / SN 2022qmx (Goobar et al. 2023), SN H0pe, the ZTF systematic glSN searches (Sagues Carracedo et al. 2024 arXiv:2406.00052; Townsend et al. 2025 arXiv:2405.18589; Magee; Bag/Wojtak forecasts), any ATLAS lensed-SN work. Then: (1) is the ZTF/ATLAS glSN time-domain content genuinely UNSEARCHED, or is there an active crowded field with dedicated pipelines + brokers already scanning every alert? (2) Realistic glSN DISCOVERY RATE at ZTF (~m20.5) and ATLAS (~m19-19.5) depth - how many glSNe per year are actually findable/found (it is a famously rare, magnification-biased population)? (3) Would the program's retrospective-diff-imaging-at-known-lens-positions method add net-new glSNe beyond the live alert+classifier searches, or duplicate them? Quantify. 6-10 web searches; cite URLs. channel="lensed-SN".` ,
    { label: 'res:lensed-SN', phase: 'Research', agentType: 'general-purpose', schema: RESEARCH_SCHEMA }).catch(() => null),
  () => agent(
    `${CTX}\nROLE: LENSED-QUASAR (variability/time-delay) channel in ZTF & ATLAS.\n` +
    `How saturated is lensed-quasar discovery/monitoring via ZTF & ATLAS variability? Catalogue programs: COSMOGRAIL time-delay monitoring, ZTF lensed-quasar light-curve/time-delay work (e.g. Dux et al. 2026; Springer & Ofek 2021 arXiv:2110.15315 variability lens search), Gaia/GraL+ZTF, Chao/Shajib time-delay cosmography, the program's own Sheu 2024a variable-lensed-quasar approach. Then: (1) is the ZTF/ATLAS lensed-QSO variability content genuinely unsearched, or already worked by time-delay/variability teams? (2) Realistic net-new lensed-QSO candidate yield from a ZTF/ATLAS variability search at known-lens positions vs blind - is the claimed "tens-to-hundreds" defensible, and how many would be NET-NEW vs already-known (Gaia GraL/Lemon catalogs)? (3) Does diff-imaging/variability at known-lens positions discover NEW lensed quasars, or mostly re-characterize known ones (time delays)? Quantify. 6-10 web searches; cite URLs. channel="lensed-QSO".`,
    { label: 'res:lensed-QSO', phase: 'Research', agentType: 'general-purpose', schema: RESEARCH_SCHEMA }).catch(() => null),
  () => agent(
    `${CTX}\nROLE: METHOD-FIT, ACCESS PIPELINE & RUBIN OBSOLESCENCE.\n` +
    `(1) METHOD FIT: For ZTF/ATLAS, is the program's Sheu DIFFERENCE-IMAGING method (retrospective, at known-lens positions) actually the right tool, or is lensed-transient discovery in these surveys dominated by ALERT-STREAM photometric classification (brokers ALeRCE/Fink/Lasair/ANTARES; magnification-anomaly, host-offset, color, light-curve-shape selection) and forced photometry, NOT image-level diff-imaging (ZTF/ATLAS already deliver difference-image alerts/forced-phot, so the user would not re-do diff imaging)? What is the genuinely additive contribution the program could make (e.g., a SN-type/host-spectrum gate via SpectrumFM; monitoring known-lens positions for new transients)? (2) ACCESS: concrete endpoints - ZTF (IRSA TAP + IPAC forced-photometry service + brokers), ATLAS (fallingstar forced-photometry server), and the known-lens position list (the program's ~5,807 + ~6,500 candidates). (3) RUBIN OBSOLESCENCE: when does Rubin/LSST take over time-domain lensing, and at what scale (glSN forecasts ~hundreds, lensed-QSO forecasts), vs the ZTF/ATLAS precursor window (~2-3 yr)? Does a dedicated ZTF/ATLAS effort make sense as a precursor or is it marginal? Quantify. 6-10 web searches; cite URLs. channel="method-obsolescence".`,
    { label: 'res:method-obsolescence', phase: 'Research', agentType: 'general-purpose', schema: RESEARCH_SCHEMA }).catch(() => null),
]);

phase('Verify');
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['CONFIRMED', 'DOWNGRADED', 'REFUTED'] },
    lensed_sn_verdict: { type: 'string' },
    lensed_qso_verdict: { type: 'string' },
    content_unsearched_holds: { type: 'string' },
    method_fit_verdict: { type: 'string' },
    realistic_net_new_yield: { type: 'string' },
    best_framing: { type: 'string' },
    biggest_risk: { type: 'string' },
    corroborating_sources: { type: 'integer' },
    evidence: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { claim: { type: 'string' }, source_url: { type: 'string' } }, required: ['claim', 'source_url'] } },
    rationale: { type: 'string' },
  },
  required: ['verdict', 'lensed_sn_verdict', 'lensed_qso_verdict', 'content_unsearched_holds', 'realistic_net_new_yield', 'biggest_risk', 'rationale', 'evidence'],
};

const redteam = await agent(
  `You are an adversarial red-team verifier for strong-lens discovery opportunities.\n\n${CTX}\n\n` +
  `Research findings (3 agents - lensed-SN, lensed-QSO, method/obsolescence):\n` +
  `${JSON.stringify([lsn, lqso, method].filter(Boolean))}\n\n` +
  `Try HARD to FALSIFY the thesis "ATLAS/ZTF time-domain diff-imaging is a robust net-new lens-discovery opportunity with existing techniques." On these axes:\n` +
  `1. IS THE CONTENT ACTUALLY UNSEARCHED? Lensed-SN and lensed-QSO discovery in ZTF/ATLAS is a crowded, active field (Goobar group, time-delay teams, brokers scanning every alert). Quantify how much is already being done - is "time-domain content is unsearched" true or false?\n` +
  `2. METHOD FIT: ZTF/ATLAS already produce difference-image alerts + forced photometry, so the program would NOT re-run diff imaging - the discovery mechanism is alert-stream classification. Is the program's actual additive edge small (e.g. just a SpectrumFM host-spectrum gate / known-lens monitoring)?\n` +
  `3. YIELD: glSNe are rare/magnification-biased (single digits/yr even in ZTF); ATLAS shallower. Lensed-QSO net-new (vs known Gaia/Lemon) is small. Is the claimed "tens of glSNe + tens-to-hundreds lensed-QSO" realistic NET-NEW, or an overcount?\n` +
  `4. OBSOLESCENCE: Rubin/LSST takes over time-domain lensing at ~100x scale within ~2-3 yr; is a dedicated ZTF/ATLAS effort a worthwhile precursor or marginal?\n` +
  `Separate the lensed-SN and lensed-QSO verdicts. Return CONFIRMED (robust net-new opportunity), DOWNGRADED (real but small/niche/precursor-only), or REFUTED. Require >=2 corroborating sources; cite URLs. Be fair: if it survives better than HST/VHS/GAMA/DELVE did, say so.`,
  { label: 'redteam:atlas-ztf', phase: 'Verify', agentType: 'general-purpose', schema: VERIFY_SCHEMA }).catch(() => null);

log(`ATLAS/ZTF dive done. Verdict: ${redteam?.verdict ?? 'N/A'} | glSN: ${redteam?.lensed_sn_verdict ?? 'n/a'} | lensed-QSO: ${redteam?.lensed_qso_verdict ?? 'n/a'} | yield: ${redteam?.realistic_net_new_yield ?? 'n/a'}`);
return { research: { lsn, lqso, method }, redteam };
