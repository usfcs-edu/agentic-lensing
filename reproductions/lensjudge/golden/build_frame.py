#!/usr/bin/env python3
"""golden/build_frame.py — the LITE sampling frame (250 items) for the golden single-grader campaign.

WHAT. One row per exposure item Xiaosheng will grade, drawn from the JWST discovery run
(`jwst-strong-lens-search`, read-only) in seven strata fixed by the approved plan (Phase 1):

    T_verified   top-100 with verifier grade A/B/C (22 → 21: rank 14 is a 1.17" twin of rank 7
                 and is collapsed onto it as an alias; same pixels, two catalogue rows)
    T_U          top-100 never verified (78; ranks 16 & 17 are 8.78" apart — kept as two
                 stimuli, same `system_id`)
    K_cowls      all 31 blind COWLS positive controls injected into the run
    L_known      30 literature lenses (SIMBAD crossmatch < 2", not top-100, not COWLS):
                 every pipeline-D one + any A/B/C + a seeded stratified fill from U/unflagged
    D_refuted    40 verified-and-refuted (pipeline D) by `center_galaxy_type`:
                 10 merger / 10 ring+spiral / 10 highest-confidence elliptical ("near-miss")
                 / 10 random from the rest — the false-positive families
    U_tail       30 flagged-but-unverified below the top-100: 20 from ranks 101-300, 10 from
                 301-2,024 (rank = the run's own (score, confidence) ordering)
    N_unflagged  20 inspected-and-not-flagged, proposal-stratified to the flagged mix, with
                 anything within 2" of the DESI parity pool/benches excluded

WHY. The top-100 alone has zero refuted and zero unflagged items, so it cannot yield purity,
specificity or a random-negative stratum; the frame adds those while keeping every item the
agent actually surfaced. Exposure order is a separate seeded shuffle (build_kit.py); here the
`unit_id` is assigned in a seeded-shuffled order so it encodes nothing about the stratum.

EVERY row also carries: the run's own verdict as `pipe_grade_passcount` (a persona PASS-COUNT,
never a Huang grade — hence the column name), `prior_exposure` (2 = ranks 1-15 seen in the
annotated docx, 1 = ranks 16-100 seen on the contact sheet), `lit_known` (COWLS, L_known,
discovery_status known/field_match, or ANY row that carries a literature `known_lens_name` —
a SIMBAD-matched U_tail item is literature-known too), `layout` (gray when SW or LW is
missing OR the run's own cutout of that channel had < MIN_FINITE finite pixels —
the renderer's gate, read from `J/data/manifest.csv` finite_sw/finite_lw; one D_refuted
target sits in the masked region of an F444W coronagraph product and is gray_sw_only
although an LW observation exists), a 10" union-find `system_id`, and `desi_pool_overlap` against
`outputs/parity_train_pool.csv` + the two parity benches (flag only — 365 of the 371 overlapping
targets are DESI `random_neg`, i.e. a DESI-trained student was taught they are non-lenses).

Seed 2026 (`np.random.default_rng`), consumed in a fixed order: L_known fill -> D_refuted ->
U_tail -> N_unflagged -> unit_id shuffle. Positions come from results.csv (float64), never
parsed from the candidate id (which mangles precision).

    python lensjudge/golden/build_frame.py [--out-dir lensjudge/golden] [--jwst-repo J]

Outputs: golden/frame.csv (+ .sha via _util.pin) and golden/frame_summary.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.coordinates import SkyCoord, search_around_sky  # noqa: E402
import astropy.units as u  # noqa: E402

from lensjudge.golden import _util  # noqa: E402

SEED = _util.SEED
OUT_DIR = _util.HERE
OUTPUTS = _util.LENSJUDGE / "outputs"

# --- the contract schema (GOLDEN_CONTRACT.md "frame.csv"), in this exact order ---
FRAME_COLUMNS = (
    "unit_id", "candidate_id", "alias_ids", "system_id", "ra_deg", "dec_deg",
    "stratum", "substratum", "pipe_grade_passcount", "pipe_inspector_conf", "pipe_score",
    "center_galaxy_type", "rank_top100", "prior_exposure", "lit_known", "layout",
    "cowls_ranking", "cowls_theta_E", "known_lens_name", "known_type", "known_sep_arcsec",
    "desi_pool_overlap", "desi_pool_split",
    "sw_obs", "lw_obs", "sw_filter", "lw_filter", "proposal", "field_id",
)
STRATA = ("T_verified", "T_U", "K_cowls", "L_known", "D_refuted", "U_tail", "N_unflagged")

MATCH_ARCSEC = 2.0     # known-lens crossmatch, DESI-pool overlap, top-100 alias collapse
SYSTEM_ARCSEC = 10.0   # union-find radius for system_id
TOP_N = 100            # the run's published top-N
MIN_FINITE = 0.55      # render_cutout's finite-pixel gate (common/jwst_fetch.MIN_FINITE): a channel
                       # below it is dropped from the composite, so the layout goes gray
DOCX_RANKS = 15        # ranks reviewed in the annotated docx -> prior_exposure 2
TOP100_VERIFIED = ("A", "B", "C")
LIT_STATUS = ("known", "field_match")   # master discovery_status values that mean "in the literature"

# Quotas (plan Phase 1 table). Parametrised so a synthetic universe can exercise the same code.
DEFAULT_QUOTAS = {
    "L_known": 30,
    "D_refuted": 10,                       # per substratum
    "U_tail": ((101, 300, 20), (301, None, 10)),   # (rank_lo, rank_hi|None, n)
    "N_unflagged": 20,
}
N_TOTAL = 250
# D_refuted substrata in fill order; None = "everything not in the named families"
REFUTED_SUBSTRATA = (
    ("merger", ("merger",)),
    ("ring_spiral", ("ring", "spiral")),
    ("elliptical_nearmiss", ("elliptical",)),
    ("other", None),
)


# ============================================================================ helpers (pure)
def sky_match(ra, dec, cat_ra, cat_dec, radius_arcsec: float):
    """Nearest-neighbour match of (ra, dec) into a catalogue. Returns (idx, sep_arcsec, hit)
    with `hit = sep < radius`; empty catalogue -> all misses."""
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    cat_ra = np.asarray(cat_ra, float); cat_dec = np.asarray(cat_dec, float)
    if len(ra) == 0 or len(cat_ra) == 0:
        n = len(ra)
        return np.full(n, -1), np.full(n, np.inf), np.zeros(n, bool)
    c = SkyCoord(ra * u.deg, dec * u.deg)
    cat = SkyCoord(cat_ra * u.deg, cat_dec * u.deg)
    idx, sep, _ = c.match_to_catalog_sky(cat)
    sep = np.asarray(sep.arcsec, float)
    return np.asarray(idx), sep, sep < radius_arcsec


def union_find_systems(ra, dec, radius_arcsec: float = SYSTEM_ARCSEC) -> np.ndarray:
    """Group positions into systems: any pair closer than `radius_arcsec` shares a system
    (transitively). Returns dense 1-based ids numbered by first appearance in row order."""
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    n = len(ra)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]      # path halving
            i = parent[i]
        return i

    if n > 1:
        c = SkyCoord(ra * u.deg, dec * u.deg)
        i1, i2, _, _ = search_around_sky(c, c, radius_arcsec * u.arcsec)
        for a, b in zip(np.asarray(i1), np.asarray(i2)):
            if a < b:
                ra_, rb_ = find(a), find(b)
                if ra_ != rb_:
                    parent[max(ra_, rb_)] = min(ra_, rb_)
    roots = [find(i) for i in range(n)]
    ids: dict = {}
    return np.array([ids.setdefault(r, len(ids) + 1) for r in roots], dtype=int)


def dedup_known_lenses(known: pd.DataFrame, radius_arcsec: float = MATCH_ARCSEC) -> pd.DataFrame:
    """Non-COWLS `known_lens_recovery` rows within `radius_arcsec`, one per cutout: when several
    literature names fall on the same cutout the CLOSEST wins (ties -> name order)."""
    k = known[(known["lens_src"].astype(str) != "COWLS") & (known["sep_arcsec"] < radius_arcsec)]
    k = k.sort_values(["sep_arcsec", "lens_name"], kind="stable")
    return k.drop_duplicates("cutout_id", keep="first").reset_index(drop=True)


def largest_remainder(weights: dict, n: int, capacity: dict | None = None):
    """Hamilton (largest-remainder) integer allocation of `n` draws over keys, proportional to
    `weights`, capped per key by `capacity` (default unbounded). Draws a capped key cannot take
    are passed down the remainder order. Deterministic. Returns (alloc, unallocated)."""
    # a capacity dict is authoritative: a key it does not list has NO eligible rows
    cap = (lambda k: np.inf) if capacity is None else (lambda k: capacity.get(k, 0))
    keys = sorted((k for k in weights if weights[k] > 0 and cap(k) > 0), key=str)
    if not keys or n <= 0:
        return {}, n
    tot = float(sum(weights[k] for k in keys))
    quota = {k: n * weights[k] / tot for k in keys}
    alloc = {k: int(min(np.floor(quota[k]), cap(k))) for k in keys}
    left = n - sum(alloc.values())
    order = sorted(keys, key=lambda k: (-(quota[k] - np.floor(quota[k])), str(k)))
    while left > 0:
        progressed = False
        for k in order:
            if left == 0:
                break
            if alloc[k] < cap(k):
                alloc[k] += 1; left -= 1; progressed = True
        if not progressed:
            break
    return {k: v for k, v in alloc.items() if v > 0}, left


def draw(pool: pd.DataFrame, k: int, rng: np.random.Generator, how: str = "random",
         id_col: str = "id") -> pd.DataFrame:
    """Take k rows from `pool`: `random` = seeded choice without replacement over the id-sorted
    pool (so source row order never matters); `head` = the first k rows as given (caller sorted)."""
    k = int(min(k, len(pool)))
    if k <= 0:
        return pool.iloc[0:0]
    if how == "head":
        return pool.iloc[:k]
    srt = pool.sort_values(id_col, kind="stable")
    pick = rng.choice(len(srt), size=k, replace=False)
    return srt.iloc[np.sort(pick)]


def select_refuted(D: pd.DataFrame, rng: np.random.Generator, n_per: int = 10,
                   type_col: str = "center_galaxy_type"):
    """D_refuted: n_per from each REFUTED_SUBSTRATA family. merger / ring_spiral / other are
    random draws; elliptical_nearmiss takes the HIGHEST inspector confidence (ties: score, id).
    A short family is filled from the next family in order (wrapping), the fill rows keep the
    label of the family they were drawn from; `notes` records every fill."""
    ctype = D[type_col].fillna("").astype(str).str.strip().str.lower()
    named = {t for _, fam in REFUTED_SUBSTRATA if fam for t in fam}
    pools, how = {}, {}
    for name, fam in REFUTED_SUBSTRATA:
        if fam is None:
            pools[name] = D[~ctype.isin(named)]
        else:
            pools[name] = D[ctype.isin(fam)]
        how[name] = "random"
    pools["elliptical_nearmiss"] = pools["elliptical_nearmiss"].sort_values(
        ["confidence", "score", "id"], ascending=[False, False, True], kind="stable")
    how["elliptical_nearmiss"] = "head"

    order = [name for name, _ in REFUTED_SUBSTRATA]
    picks, notes, shortfall = [], [], 0
    for name in order:
        got = draw(pools[name], n_per, rng, how[name])
        pools[name] = pools[name].drop(got.index)
        picks.append(got.assign(substratum=name))
        if len(got) < n_per:
            shortfall += n_per - len(got)
            notes.append(f"{name}: only {len(got)}/{n_per} available")
    # carry the shortfall forward through the families (wrapping) until filled or exhausted
    for name in order * 2:
        if shortfall <= 0:
            break
        got = draw(pools[name], shortfall, rng, how[name])
        if len(got):
            pools[name] = pools[name].drop(got.index)
            picks.append(got.assign(substratum=name))
            notes.append(f"filled {len(got)} from {name}")
            shortfall -= len(got)
    if shortfall > 0:
        notes.append(f"UNFILLED shortfall of {shortfall} (D pool exhausted)")
    out = pd.concat(picks, ignore_index=True) if picks else D.iloc[0:0].assign(substratum="")
    return out, notes


def assign_unit_ids(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Seeded permutation of the rows, then unit_id = u0001.. in that order (so the id carries
    no stratum information) and system_id renumbered by first appearance."""
    perm = rng.permutation(len(frame))
    f = frame.iloc[perm].reset_index(drop=True)
    f["unit_id"] = [f"u{i + 1:04d}" for i in range(len(f))]
    f["system_id"] = union_find_systems(f["ra_deg"], f["dec_deg"], SYSTEM_ARCSEC)
    return f


# ============================================================================ sources
def load_sources(jwst_repo: Path = _util.JWST_REPO, outputs: Path = OUTPUTS) -> dict:
    """Read every input (all read-only) and derive the run's rank. 09_rank_report.py writes
    results.csv already sorted by (score, confidence) descending and defines
    `rank_of = row position + 1` over that same frame, so rank == 1-based row index; both
    facts are asserted here rather than assumed."""
    jwst_repo = Path(jwst_repo); R = jwst_repo / "results"
    results = pd.read_csv(R / "results.csv", dtype={"id": str, "grade": str}, low_memory=False)
    results["rank"] = np.arange(1, len(results) + 1)
    key = results[["score", "confidence"]].fillna(-np.inf).to_numpy()
    assert all(tuple(key[i]) >= tuple(key[i + 1]) for i in range(len(key) - 1)), \
        "results.csv is not in (score, confidence) descending order — rank derivation invalid"
    inspections = pd.read_csv(R / "inspections.csv", dtype={"id": str})
    master = pd.read_csv(R / "JWST_top100_master.csv", dtype={"candidate_id": str})
    controls = pd.read_csv(R / "control_recovery.csv", dtype={"id": str, "grade": str, "ranking": str})
    known = pd.read_csv(R / "known_lens_recovery.csv", dtype={"cutout_id": str, "grade": str})
    targets = pd.read_parquet(jwst_repo / "data" / "targets.parquet")
    # the run's render manifest: finite_sw / finite_lw per cutout (what decided its layout)
    rm = jwst_repo / "data" / "manifest.csv"
    render = (pd.read_csv(rm, dtype={"id": str}, usecols=["id", "finite_sw", "finite_lw"])
              if rm.exists() else None)
    desi = load_desi_catalog(outputs)
    # rank cross-checks against the two files that carry their own rank column
    rk = results.set_index("id")["rank"]
    assert (master["candidate_id"].map(rk).to_numpy() == master["rank"].to_numpy()).all(), \
        "master rank != results.csv row position"
    assert (controls["id"].map(rk).to_numpy() == controls["rank"].to_numpy()).all(), \
        "control_recovery rank != results.csv row position"
    return dict(results=results, inspections=inspections, master=master, controls=controls,
                known=known, targets=targets, desi=desi, render=render)


def load_desi_catalog(outputs: Path = OUTPUTS) -> pd.DataFrame:
    """parity_train_pool ∪ parity_bench_arm1 ∪ parity_bench_arm2 as one (ra, dec, split,
    label_source) table; bench rows get split = bench_arm{1,2}."""
    outputs = Path(outputs)
    pool = pd.read_csv(outputs / "parity_train_pool.csv",
                       usecols=["ra", "dec", "split", "label_source"], dtype={"split": str})
    parts = [pool]
    for arm in (1, 2):
        b = pd.read_csv(outputs / f"parity_bench_arm{arm}.csv", usecols=["ra", "dec"])
        parts.append(b.assign(split=f"bench_arm{arm}", label_source="bench"))
    cat = pd.concat(parts, ignore_index=True)
    return cat.dropna(subset=["ra", "dec"]).reset_index(drop=True)


# ============================================================================ the frame
def build_frame(src: dict, rng: np.random.Generator, quotas: dict = DEFAULT_QUOTAS):
    """Draw the strata in order (T -> K -> L -> D -> U_tail -> N), attach the run's columns,
    flag DESI overlap, shuffle, number. Returns (frame, summary dict)."""
    res = src["results"].copy()
    ins = src["inspections"].set_index("id")
    master = src["master"].sort_values("rank").reset_index(drop=True)
    controls, known, targets, desi = src["controls"], src["known"], src["targets"], src["desi"]
    S: dict = {"fills": [], "pop": {}}

    res["flagged"] = res["id"].map(ins["flagged"]).fillna(False).astype(bool).to_numpy()
    res["grade"] = res["grade"].fillna("").astype(str)
    assert (res["flagged"] == (res["grade"] != "")).all(), "flagged != (grade != '')"
    S["pop"]["results_rows"] = len(res)
    S["pop"]["flagged"] = int(res["flagged"].sum())
    S["pop"]["inspected_ok"] = int((ins["status"] == "ok").sum())
    S["pop"]["pipe_grade_counts"] = res.loc[res["flagged"], "grade"].value_counts().to_dict()

    taken: set = set()          # candidate ids already placed (or collapsed) — no row twice
    rows: list = []

    # ---- T_verified / T_U: the top-100, aliases collapsed onto the better rank --------------
    top_ids = set(master["candidate_id"])
    alias_of: dict = {}         # dropped id -> kept id
    if len(master) > 1:
        mc = SkyCoord(master["ra_deg"].to_numpy(float) * u.deg, master["dec_deg"].to_numpy(float) * u.deg)
        i1, i2, _, _ = search_around_sky(mc, mc, MATCH_ARCSEC * u.arcsec)
        for a, b in zip(np.asarray(i1), np.asarray(i2)):
            if a < b:   # master is rank-sorted, so a is the better rank
                alias_of[master.at[b, "candidate_id"]] = master.at[a, "candidate_id"]
    alias_ids: dict = {}
    for dropped, kept in alias_of.items():
        alias_ids.setdefault(kept, []).append(dropped)
    S["aliases"] = {kept: v for kept, v in alias_ids.items()}
    S["pop"]["top100_rows"] = len(master)
    S["pop"]["top100_grade_counts"] = master["verifier_grade"].value_counts().to_dict()
    for _, m in master.iterrows():
        cid = m["candidate_id"]
        if cid in alias_of:
            taken.add(cid); continue
        g = str(m["verifier_grade"])
        stratum = "T_verified" if g in TOP100_VERIFIED else "T_U"
        rows.append(dict(candidate_id=cid, stratum=stratum, substratum=g if stratum == "T_verified" else "",
                         alias_ids="|".join(alias_ids.get(cid, [])), rank_top100=int(m["rank"]),
                         discovery_status=str(m.get("discovery_status", ""))))
        taken.add(cid)

    # ---- K_cowls: every blind control ---------------------------------------------------------
    S["pop"]["cowls_rows"] = len(controls)
    S["pop"]["cowls_flagged"] = int(controls["flagged"].fillna(False).astype(bool).sum())
    for _, c in controls.sort_values("id").iterrows():
        cid = c["id"]
        assert cid not in taken, f"COWLS control {cid} already placed"
        rank_str = "" if pd.isna(c.get("ranking")) else str(c["ranking"])
        rows.append(dict(candidate_id=cid, stratum="K_cowls", substratum=_pipe_bucket(c.get("grade"), c.get("flagged")),
                         cowls_ranking=rank_str, cowls_theta_E=float(c.get("einstein_radius", np.nan)),
                         known_lens_name=str(c.get("cowls_code", "")), known_type="COWLS",
                         known_sep_arcsec=float(c.get("sep_arcsec", np.nan))))
        taken.add(cid)
    cowls_ids = set(controls["id"])

    # ---- L_known: literature lenses, closest name per cutout, fixed D/ABC + stratified fill ---
    kd = dedup_known_lenses(known, MATCH_ARCSEC)
    S["pop"]["known_rows"] = len(known)
    S["pop"]["known_noncowls_lt2_unique"] = len(kd)
    S["pop"]["known_in_top100"] = int(kd["cutout_id"].isin(top_ids).sum())
    S["pop"]["known_on_cowls_cutout"] = int(kd["cutout_id"].isin(cowls_ids).sum())
    kd = kd[~kd["cutout_id"].isin(top_ids) & ~kd["cutout_id"].isin(cowls_ids)].reset_index(drop=True)
    kd["bucket"] = [_pipe_bucket(g, f) for g, f in zip(kd["grade"], kd["flagged"])]
    S["pop"]["known_pool_by_bucket"] = kd["bucket"].value_counts().to_dict()
    fixed = kd[kd["bucket"].isin(["pipe_D", "pipe_A", "pipe_B", "pipe_C"])]
    n_fill = quotas["L_known"] - len(fixed)
    assert n_fill >= 0, f"L_known fixed rows ({len(fixed)}) exceed quota {quotas['L_known']}"
    fill_pool = kd[kd["bucket"].isin(["pipe_U", "unflagged"])]
    weights = fill_pool["bucket"].value_counts().to_dict()
    alloc, left = largest_remainder(weights, n_fill, capacity=weights)
    if left:
        S["fills"].append(f"L_known: fill short by {left} (pool U/unflagged = {weights})")
    S["L_known_alloc"] = {"fixed": fixed["bucket"].value_counts().to_dict(), "fill": alloc}
    picks = [fixed]
    for b in sorted(alloc):   # fixed bucket order -> fixed rng consumption order
        picks.append(draw(fill_pool[fill_pool["bucket"] == b], alloc[b], rng, "random", id_col="cutout_id"))
    for _, k in pd.concat(picks).iterrows():
        cid = k["cutout_id"]
        rows.append(dict(candidate_id=cid, stratum="L_known", substratum=k["bucket"],
                         known_lens_name=str(k["lens_name"]), known_type=str(k["lens_src"]),
                         known_sep_arcsec=float(k["sep_arcsec"])))
        taken.add(cid)

    # ---- D_refuted: the false-positive families ----------------------------------------------
    Dpool = res[(res["grade"] == "D") & ~res["id"].isin(taken)]
    S["pop"]["D_total"] = int((res["grade"] == "D").sum())
    S["pop"]["D_pool_after_KL"] = len(Dpool)
    S["pop"]["D_pool_by_type"] = Dpool["center_galaxy_type"].fillna("").value_counts().to_dict()
    dsel, notes = select_refuted(Dpool, rng, quotas["D_refuted"])
    S["fills"] += [f"D_refuted: {n}" for n in notes]
    for _, d in dsel.iterrows():
        rows.append(dict(candidate_id=d["id"], stratum="D_refuted", substratum=d["substratum"]))
        taken.add(d["id"])

    # ---- U_tail: flagged, unverified, below the top-100 --------------------------------------
    Upool = res[(res["grade"] == "U") & (res["rank"] > TOP_N) & ~res["id"].isin(taken)]
    S["pop"]["U_total"] = int((res["grade"] == "U").sum())
    S["pop"]["U_tail_bands"] = {}
    for lo, hi, n in quotas["U_tail"]:
        hi_eff = int(res.loc[res["flagged"], "rank"].max()) if hi is None else hi
        band = Upool[Upool["rank"].between(lo, hi_eff)]
        label = f"rank_{lo}_{hi_eff}"
        S["pop"]["U_tail_bands"][label] = len(band)
        got = draw(band, n, rng, "random")
        if len(got) < n:
            S["fills"].append(f"U_tail {label}: only {len(got)}/{n} available")
        for _, r in got.iterrows():
            rows.append(dict(candidate_id=r["id"], stratum="U_tail", substratum=label))
            taken.add(r["id"])

    # ---- N_unflagged: inspected negatives, proposal-stratified to the flagged mix -------------
    tg = targets.set_index("id")
    cand = ins[(ins["status"] == "ok") & ~ins["flagged"].fillna(False).astype(bool)]
    cand = cand[~cand.index.isin(taken)]
    _, _, hit = sky_match(cand["ra"], cand["dec"], desi["ra"], desi["dec"], MATCH_ARCSEC)
    S["pop"]["unflagged_inspected"] = int(((ins["status"] == "ok") & ~ins["flagged"].fillna(False).astype(bool)).sum())
    S["pop"]["unflagged_desi_overlap_excluded"] = int(hit.sum())
    cand = cand[~hit].reset_index()          # `id` back as a column
    cand["proposal"] = cand["proposal"].astype(str)
    flag_mix = ins.loc[ins["flagged"].fillna(False).astype(bool), "proposal"].astype(str).value_counts().to_dict()
    cap = cand["proposal"].value_counts().to_dict()
    alloc, left = largest_remainder(flag_mix, quotas["N_unflagged"], capacity=cap)
    if left:
        S["fills"].append(f"N_unflagged: short by {left} (no eligible unflagged rows left in the flagged proposals)")
    S["N_unflagged_alloc"] = alloc
    S["pop"]["flagged_proposals"] = len(flag_mix)
    S["pop"]["unflagged_eligible"] = len(cand)
    for prop in sorted(alloc, key=str):
        got = draw(cand[cand["proposal"] == prop], alloc[prop], rng, "random")
        for _, r in got.iterrows():
            rows.append(dict(candidate_id=r["id"], stratum="N_unflagged", substratum=f"proposal_{prop}"))
            taken.add(r["id"])

    # ---- attach the run's columns to every row --------------------------------------------
    frame = pd.DataFrame(rows)
    assert frame["candidate_id"].is_unique, "duplicate candidate_id in frame"
    frame, overlap_src = _attach(frame, res, tg, known, desi, src.get("render"))
    # overlap bookkeeping (stratum x pool split x DESI label_source) for the summary
    S["desi_overlap"] = (frame.assign(_ls=overlap_src)[frame["desi_pool_overlap"]]
                         .groupby(["stratum", "desi_pool_split", "_ls"]).size().to_dict())
    frame = assign_unit_ids(frame, rng)
    frame = frame.reindex(columns=list(FRAME_COLUMNS))
    S["n_rows"] = len(frame)
    return frame, S


def _pipe_bucket(grade, flagged) -> str:
    """Substratum label from the run's verdict: pipe_{A,B,C,D,U} or unflagged."""
    g = "" if pd.isna(grade) else str(grade).strip()
    return f"pipe_{g}" if g in ("A", "B", "C", "D", "U") else "unflagged"


def derive_layout(sw_obs: pd.Series, lw_obs: pd.Series, render: pd.DataFrame | None,
                  ids: pd.Series) -> np.ndarray:
    """color / gray_sw_only / gray_lw_only exactly as render_cutout decides it: a channel is
    usable when an observation exists AND its cutout has >= MIN_FINITE finite pixels (the run
    records finite_sw/finite_lw in data/manifest.csv; a target can sit on a mosaic's masked
    region with finite 0.0). Without the render manifest, presence alone decides."""
    sw_ok = np.asarray(sw_obs.notna(), dtype=bool)
    lw_ok = np.asarray(lw_obs.notna(), dtype=bool)
    if render is not None:
        rf = render.drop_duplicates("id").set_index("id").reindex(ids.to_numpy())
        sw_ok = sw_ok & (rf["finite_sw"].fillna(1.0).to_numpy(float) >= MIN_FINITE)
        lw_ok = lw_ok & (rf["finite_lw"].fillna(1.0).to_numpy(float) >= MIN_FINITE)
    if not (sw_ok | lw_ok).all():
        bad = ids[~(sw_ok | lw_ok)].tolist()
        raise ValueError(f"no renderable channel (SW and LW both missing/empty): {bad}")
    return np.select([~sw_ok, ~lw_ok], ["gray_lw_only", "gray_sw_only"], "color")


def _attach(frame: pd.DataFrame, res: pd.DataFrame, tg: pd.DataFrame, known: pd.DataFrame,
            desi: pd.DataFrame, render: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill the shared columns from results.csv (positions, pipeline verdict), targets.parquet
    (observations), the run's render manifest (finite fractions -> layout; presence-only when
    absent), known_lens_recovery (closest literature name for any row that has not already
    got one) and the DESI pool (overlap flag)."""
    r = res.set_index("id").loc[frame["candidate_id"]]
    f = frame.copy()
    f["ra_deg"] = r["ra"].to_numpy(float)
    f["dec_deg"] = r["dec"].to_numpy(float)
    f["pipe_grade_passcount"] = r["grade"].to_numpy()
    flagged = r["flagged"].to_numpy(bool)
    f["pipe_inspector_conf"] = np.where(flagged, r["confidence"].to_numpy(float), np.nan)
    f["pipe_score"] = np.where(flagged, r["score"].to_numpy(float), np.nan)
    f["center_galaxy_type"] = r["center_galaxy_type"].fillna("").astype(str).to_numpy()
    # defaults for the stratum-specific columns
    for c in ("alias_ids", "substratum", "cowls_ranking", "known_lens_name", "known_type", "discovery_status"):
        f[c] = f[c].fillna("").astype(str) if c in f.columns else ""
    for c in ("cowls_theta_E", "known_sep_arcsec", "rank_top100"):
        f[c] = f[c].astype(float) if c in f.columns else np.nan
    f["rank_top100"] = f["rank_top100"].astype("Int64")
    rk = f["rank_top100"].fillna(10 ** 9).to_numpy(dtype=int)
    f["prior_exposure"] = np.select([rk <= DOCX_RANKS, rk <= TOP_N], [2, 1], 0).astype(int)
    f["lit_known"] = (f["stratum"].isin(["K_cowls", "L_known"])
                      | f["discovery_status"].astype(str).isin(LIT_STATUS)).to_numpy(bool)
    # observations / layout from targets.parquet
    t = tg.loc[f["candidate_id"]]
    for c in ("sw_obs", "lw_obs", "sw_filter", "lw_filter", "proposal"):
        f[c] = t[c].fillna("").astype(str).to_numpy()
    f["field_id"] = t["field_id"].to_numpy()
    f["layout"] = derive_layout(t["sw_obs"], t["lw_obs"], render, f["candidate_id"])
    # closest literature name for rows that do not already carry one (e.g. top-100 known items)
    kd = dedup_known_lenses(known, MATCH_ARCSEC).set_index("cutout_id")
    need = (f["known_lens_name"] == "") & f["candidate_id"].isin(kd.index)
    if need.any():
        k = kd.loc[f.loc[need, "candidate_id"]]
        f.loc[need, "known_lens_name"] = k["lens_name"].astype(str).to_numpy()
        f.loc[need, "known_type"] = k["lens_src"].astype(str).to_numpy()
        f.loc[need, "known_sep_arcsec"] = k["sep_arcsec"].to_numpy(float)
    # a row that carries a literature name IS literature-known, whatever stratum drew it (a
    # SIMBAD-matched U_tail/D_refuted item must not be eligible for a hidden repeat or be
    # counted as non-literature in the subgroup rows)
    f["lit_known"] = (f["lit_known"] | (f["known_lens_name"].astype(str) != "")).to_numpy(bool)
    # DESI pool / bench overlap — flag only, never an exclusion for the fixed strata
    idx, sep, hit = sky_match(f["ra_deg"], f["dec_deg"], desi["ra"], desi["dec"], MATCH_ARCSEC)
    f["desi_pool_overlap"] = hit
    f["desi_pool_split"] = np.where(hit, desi["split"].astype(str).to_numpy()[np.clip(idx, 0, None)], "")
    overlap_src = np.where(hit, desi["label_source"].astype(str).to_numpy()[np.clip(idx, 0, None)], "")
    return f.drop(columns=["discovery_status"]), overlap_src


# ============================================================================ summary
def write_summary(frame: pd.DataFrame, S: dict, path: Path, src: dict, sha: str) -> None:
    """Human-readable provenance: what was drawn, from what, and every fill/dedup decision."""
    L: list = []
    L += ["# Golden LITE frame — build summary", "",
          f"`frame.csv` sha {sha} · {len(frame)} rows · seed {SEED} · built by `golden/build_frame.py`", ""]
    L += ["## Per-stratum / substratum counts", "", "| stratum | substratum | n |", "|---|---|---|"]
    cnt = frame.groupby(["stratum", "substratum"]).size()
    for s in STRATA:
        if s in cnt.index.get_level_values(0):
            for ss, n in cnt.loc[s].items():
                L.append(f"| {s} | {ss or '—'} | {n} |")
    L += ["", "| stratum | n |", "|---|---|"]
    for s in STRATA:
        L.append(f"| {s} | {int((frame['stratum'] == s).sum())} |")
    L += [f"| **total** | **{len(frame)}** |", ""]

    def _vc(col, title):
        L.extend(["", f"## {title}", "", "| value | n |", "|---|---|"])
        for k, n in frame[col].value_counts(dropna=False).sort_index().items():
            L.append(f"| {k} | {n} |")
    _vc("layout", "Layout (sw_obs / lw_obs present AND run finite fraction >= 0.55)")
    _vc("prior_exposure", "Prior exposure (2 = ranks 1-15 annotated docx; 1 = ranks 16-100 contact sheet)")
    _vc("pipe_grade_passcount", "Pipeline pass-count grade (`pipe_grade_passcount`; '' = unflagged)")
    _vc("lit_known", "Literature-known (COWLS, L_known, discovery_status known/field_match, or any row "
                     "with a literature known_lens_name)")

    L += ["", "## DESI pool / bench overlap (2\", flag only)", ""]
    ov = frame[frame["desi_pool_overlap"]]
    L += [f"{len(ov)} of {len(frame)} frame rows lie within {MATCH_ARCSEC}\" of `outputs/parity_train_pool.csv` ∪ "
          f"`parity_bench_arm1/2.csv` ({len(src['desi'])} positions).", ""]
    if len(ov):
        L += ["| stratum | split | desi label_source | n |", "|---|---|---|---|"]
        for (s, sp, ls), n in sorted(S["desi_overlap"].items()):
            L.append(f"| {s} | {sp} | {ls} | {n} |")
    L += ["", f"N_unflagged additionally EXCLUDED {S['pop'].get('unflagged_desi_overlap_excluded', 0)} "
          f"overlapping unflagged targets before drawing (fixed strata are flagged, not excluded)."]

    L += ["", "## Duplicates handled", ""]
    if S.get("aliases"):
        for kept, dropped in S["aliases"].items():
            rk = frame.loc[frame["candidate_id"] == kept, "rank_top100"]
            L.append(f"- top-100 alias collapse (< {MATCH_ARCSEC}\"): {', '.join(dropped)} → kept `{kept}` "
                     f"(rank {int(rk.iloc[0]) if len(rk) else '?'}); recorded in `alias_ids`.")
    else:
        L.append("- no top-100 aliases within 2\".")
    multi = frame.groupby("system_id").filter(lambda g: len(g) > 1).sort_values(["system_id", "unit_id"])
    if len(multi):
        L.append(f"- {multi['system_id'].nunique()} multi-member systems at {SYSTEM_ARCSEC}\" (union-find), "
                 f"{len(multi)} rows — share a split, excluded from repeats:")
        for sid, g in multi.groupby("system_id"):
            L.append("  - system " + str(sid) + ": " + ", ".join(
                f"`{c}` ({s}{'' if pd.isna(r) else f', rank {int(r)}'})"
                for c, s, r in zip(g["candidate_id"], g["stratum"], g["rank_top100"])))
    else:
        L.append(f"- no multi-member systems at {SYSTEM_ARCSEC}\".")
    L += [f"- L_known: {S['pop']['known_on_cowls_cutout']} SIMBAD match(es) fell on a COWLS cutout and "
          f"{S['pop']['known_in_top100']} on top-100 cutouts — both excluded from L_known (the cutout "
          f"is already in K_cowls / T_*; its literature name is still recorded in `known_lens_name`)."]

    L += ["", "## Fills / shortfalls", ""]
    L += [f"- {n}" for n in S["fills"]] or ["- none: every substratum filled from its own pool."]
    L += [f"- L_known allocation: fixed {S['L_known_alloc']['fixed']}, stratified fill {S['L_known_alloc']['fill']} "
          "(largest-remainder, proportional to the U/unflagged pool sizes)."]
    L += [f"- N_unflagged allocation by proposal (largest-remainder on the flagged proposal mix, "
          f"capped by eligible unflagged rows): {S['N_unflagged_alloc']}"]

    P = S["pop"]
    L += ["", "## Source populations observed", "",
          f"- `results.csv`: {P['results_rows']} rows; flagged {P['flagged']}; inspected ok {P['inspected_ok']}; "
          f"pipeline grades over flagged {P['pipe_grade_counts']}",
          f"- `JWST_top100_master.csv`: {P['top100_rows']} rows; verifier grades {P['top100_grade_counts']}; "
          f"aliases collapsed {sum(len(v) for v in S.get('aliases', {}).values())} → "
          f"{int(frame['stratum'].isin(['T_verified', 'T_U']).sum())} top-100 items",
          f"- `control_recovery.csv`: {P['cowls_rows']} COWLS controls, {P['cowls_flagged']} flagged",
          f"- `known_lens_recovery.csv`: {P['known_rows']} rows; non-COWLS < {MATCH_ARCSEC}\" unique cutouts "
          f"{P['known_noncowls_lt2_unique']}; in top-100 {P['known_in_top100']}; on COWLS cutout "
          f"{P['known_on_cowls_cutout']}; remaining pool by pipeline bucket {P['known_pool_by_bucket']}",
          f"- pipeline D: {P['D_total']} total, {P['D_pool_after_KL']} after removing K/L ids; by "
          f"center_galaxy_type {P['D_pool_by_type']}",
          f"- pipeline U: {P['U_total']} total; U_tail bands (after K/L removal) {P['U_tail_bands']}",
          f"- unflagged inspected: {P['unflagged_inspected']}; eligible after K/L + DESI-overlap removal "
          f"{P['unflagged_eligible']}; flagged rows span {P['flagged_proposals']} proposals",
          f"- DESI catalogue: {len(src['desi'])} positions (pool {int((src['desi']['label_source'] != 'bench').sum())} "
          f"+ bench {int((src['desi']['label_source'] == 'bench').sum())})",
          ""]
    L += ["## Conventions", "",
          "- `pipe_grade_passcount` is the run's persona PASS-COUNT (A=3/3 … D=0/3, U=never verified); "
          "it is not a Huang visual-inspection grade.",
          "- `pipe_inspector_conf` / `pipe_score` are NaN for unflagged rows (the run stores 0 there).",
          "- `rank_top100` is the run's rank only for top-100 rows; U_tail rank bands live in `substratum`.",
          "- `unit_id` order is a seeded shuffle of the assembled frame; `system_id` is numbered by first "
          "appearance in that order.",
          f"- rng consumption order (seed {SEED}): L_known fill → D_refuted → U_tail → N_unflagged → unit_id shuffle.",
          ""]
    Path(path).write_text("\n".join(L))


# ============================================================================ CLI
def build(out_dir: Path = OUT_DIR, jwst_repo: Path = _util.JWST_REPO, outputs: Path = OUTPUTS,
          quotas: dict = DEFAULT_QUOTAS, expect_total: int | None = N_TOTAL):
    """Load, draw, pin. Returns (frame, summary). `expect_total` is asserted when given."""
    src = load_sources(jwst_repo, outputs)
    rng = np.random.default_rng(SEED)
    frame, S = build_frame(src, rng, quotas)
    if expect_total is not None:
        assert len(frame) == expect_total, f"frame has {len(frame)} rows, expected {expect_total}"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    public = frame[list(FRAME_COLUMNS)]
    sha = _util.pin(public, out_dir / "frame.csv")
    write_summary(frame, S, out_dir / "frame_summary.md", src, sha)
    return public, S


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where frame.csv/.sha/frame_summary.md go")
    ap.add_argument("--jwst-repo", type=Path, default=_util.JWST_REPO, help="jwst-strong-lens-search checkout (read-only)")
    ap.add_argument("--outputs", type=Path, default=OUTPUTS, help="lensjudge/outputs (parity pool + benches)")
    a = ap.parse_args(argv)
    frame, S = build(a.out_dir, a.jwst_repo, a.outputs)
    print(f"frame: {len(frame)} rows -> {a.out_dir / 'frame.csv'} (sha {_util.sha_file(a.out_dir / 'frame.csv')})")
    print(frame["stratum"].value_counts().reindex(STRATA).to_string())
    print("layout:", frame["layout"].value_counts().to_dict())
    print("prior_exposure:", frame["prior_exposure"].value_counts().sort_index().to_dict())
    print("desi_pool_overlap:", int(frame["desi_pool_overlap"].sum()))
    for n in S["fills"]:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
