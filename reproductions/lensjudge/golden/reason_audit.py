#!/usr/bin/env python3
"""golden/reason_audit.py — regex taxonomy of refutation grounds (incumbent and new scheme).

Why: the diagnosis (diag_forensics §C3/§C4) found that the incumbent personas' fails rest on
grounds that are wrong for JWST lenses — an Einstein-radius prior ("a 3.5\" Einstein radius
would demand a group/cluster halo"), a colour prior ("redder than the host, wrong sign for a
lensed source") and the broken circular radial-profile subtraction in panel (f) ("textbook
bipolar bowtie ... a Sersic/ellipticity mismatch"). The new scheme forbids exactly those three
grounds in every critic brief. This module is the measurement: the same ten regex categories
are applied to the incumbent's `alternative | notes` (outputs/incumbent_replay.csv, verbatim
text) and to the new critics' `alternative_desc | notes`, so the pre-registered monitor
"forbidden-ground rate < 2 % per category" and the design-vs-incumbent comparison use ONE
ruler. The regexes are the ruler — they are shipped here so every quoted percentage is
recomputable (critique R8).

Categories (C4 order):  theta_e_prior, colour_only, over_subtraction, spiral_ring_disk,
companion, no_counter_image, not_tangential, diffraction_detector, low_sn, shell_tidal.
A text can hit several. `locates_feature(text)` is the companion test the critic briefs
demand ("point at pixels"): a radius (1.3\", 1.3 arcsec, r=1.3), a position-angle / quadrant
token (N, NE, ..., NNW, north/south/..., PA) or a "deg" value.

`audit_table(df)` -> per-category share of REFUTATIONS (incumbent: verdict == fail; new
scheme: a named alternative with no_opinion False), per persona/role and pooled, plus:
  any_ground         share with at least one category hit
  locates_feature    share that point at a radius / PA / quadrant
  forbidden_only     share whose ONLY tagged grounds are FORBIDDEN ones — a refutation with
                     nothing but a forbidden ground
The forbidden-only share is the pre-registered "forbidden-ground rate"; the per-category
shares of the three forbidden categories are reported beside it (a mention is not a sole
ground, so the category share is the upper bound).

The new scheme SANCTIONS three channels that the prose regexes alone would flag, so for
critic records (a frame with an `alternative` column) "forbidden" is derived from the
STRUCTURED record plus the prose, never the prose alone (the monitor must not fail when a
critic obeys its brief, nor push the design iterations to suppress the sanctioned channels):
  theta_e_prior_forbidden    the θ_E regex hits AND alternative != "scale_tension" (the brief's
                             rule (i) routes a scale argument into scale_tension at r <= 0.4)
  over_subtraction_forbidden the residual regex hits AND (alternative != "subtraction_residual",
                             OR it is "subtraction_residual" but covers an item the advocate
                             marked visible_in_direct — the channel is admissible only for
                             features absent from every direct panel)
  colour_only_forbidden      the colour regex hits with NO structural category in the prose AND
                             no structural alternative named (a named spiral_arm whose notes
                             also mention colour is a structural refutation)
  uses_scale_tension / uses_subtraction_residual  the sanctioned channels' usage, reported as
                             separate monitors
For the incumbent (no `alternative` enum; `verdict` column) the forbidden flags are the raw
regex categories, as before.

CLI (no API):
  python lensjudge/golden/reason_audit.py --incumbent outputs/incumbent_replay.csv
  python lensjudge/golden/reason_audit.py --votes outputs/preds_truth_a1_sonnet_design_r1_votes.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge.common import parse  # noqa: E402
from lensjudge.golden import _util  # noqa: E402

CATEGORIES = ("theta_e_prior", "colour_only", "over_subtraction", "spiral_ring_disk", "companion",
              "no_counter_image", "not_tangential", "diffraction_detector", "low_sn", "shell_tidal")
FORBIDDEN = ("theta_e_prior", "colour_only", "over_subtraction")
FORBIDDEN_FLAGS = tuple(f"{c}_forbidden" for c in FORBIDDEN)      # the structured-record versions
STRUCTURAL = tuple(c for c in CATEGORIES if c not in FORBIDDEN)    # prose grounds that are not forbidden
# alternatives that are structural claims about the host / field (a colour mention beside one
# of these is not a colour-only refutation); scale_tension and subtraction_residual are the
# sanctioned channels; "other" and None are neither
STRUCTURAL_ALTS = ("spiral_arm", "ring_galaxy", "shell_tidal", "merger", "edge_on_disk",
                   "companion_projection", "star_forming_clump", "diffraction_spike",
                   "detector_artifact", "psf_wing")
USAGE_FLAGS = ("uses_scale_tension", "uses_subtraction_residual")
PERSONAS_INCUMBENT = ("artifact", "geometry", "morphology")
_F = re.IGNORECASE

# Each category: a list of compiled patterns; a hit on any one tags the text. Spans between
# two terms are bounded to one clause (`_S{0,N}`: any run of characters that contains no ";"
# and no sentence-ending ". " — a decimal point inside 2.5" must NOT end the span).
_S = r"(?:(?!\.\s)[^;])"


def _c(pattern: str) -> re.Pattern:
    return re.compile(pattern.replace("{S}", _S), _F)


_PATTERNS: dict[str, list[re.Pattern]] = {
    "theta_e_prior": [
        # "a 3.5" Einstein radius would demand a group/cluster halo", "...is implausible"
        _c(r"einstein (radius|ring){S}{0,80}(implausib|impossib|too (large|big)|far too|unphysical|demand|requir|"
                   r"needs?\b|cannot|would need|no (plausible|viable|credible)|group|cluster|lone|single|isolated)"),
        _c(r"theta[_ ]?e\b{S}{0,80}(implausib|too (large|big)|far too|unphysical|demand|requir|needs?\b|"
                   r"cannot|would need|no (plausible|viable|credible)|group/cluster|lone|single galaxy|isolated)"),
        # "too large for this lone elliptical", "too low-mass to produce a 2" Einstein radius"
        _c(r"(implausib|impossib|too (large|big)|far too|unphysical|would (need|demand|require)|cannot produce|"
                   r"capable of producing|(far )?(beyond|outside) any|poor candidate for){S}{0,80}"
                   r"(einstein (radius|ring)|theta[_ ]?e\b|group[- /]|cluster[- ]|\bhalo\b|single galaxy|lone|isolated)"),
        _c(r"(too low[- ]mass|not (nearly )?massive enough|insufficient(ly)? mass|too little mass|"
                   r"(luminous|massive) enough to produce)"),
        _c(r"(group|cluster)[- ](scale|halo){S}{0,60}\b(not|no|nor)\b{S}{0,40}"
                   r"(single|lone|isolated|this galaxy|this elliptical)"),
        _c(r"(single|lone|isolated){S}{0,30}(galaxy|elliptical|deflector|lens){S}{0,60}"
                   r"(too (large|big)|cannot|implausib|not massive|insufficient|would (need|demand))"),
        _c(r"(implies|implying|imply|demands?|needs?|requires?) (a |an |>?\d\S* ?(msun|m_sun|solar)\S* )?(group|cluster)"),
        _c(r"(group|cluster)[- ](scale|halo|mass){S}{0,50}(not present|absent|no such|nowhere|none)"),
    ],
    "colour_only": [
        # "identical tan colour as the nucleus", "same warm colour", "host-coloured"
        _c(r"(same|identical|similar|matching|indistinguishable)[- ]?((warm|tan|orange|red|white|blue|"
                   r"yellow|pale)[- ])?(colou?r|hue|tint){S}{0,40}\b(as|to|of)\b{S}{0,30}"
                   r"(deflector|host|nucleus|galaxy|core|bulge|envelope|neighbou?r|member|disk|disc|lens)"),
        _c(r"(host|deflector|galaxy|nucleus|bulge)[- ]colou?red"),
        _c(r"\b(not|no|isn'?t|aren'?t|nor)\s+(distinctly |clearly |noticeably )?bluer"),
        _c(r"no (blue|colou?r) (contrast|excess|offset|difference|gradient)"),
        _c(r"redder than the (host|deflector|galaxy|core|nucleus|whiter|bluer?|central|blue|white)"),
        _c(r"wrong (sign|colou?r|way){S}{0,30}(lensed|lens|arc|source)"),
        _c(r"opposite of (a|the) (lensed[- ]arc|lens|expected){S}{0,20}colou?r"),
        _c(r"no blue (arc|feature|source|knot|counter|image|light|component|emission)"),
        _c(r"colou?r{S}{0,25}(identical|the same|matches|match|indistinguishable){S}{0,40}"
                   r"(nucleus|host|deflector|galaxy|bulge|core|neighbou?r|member)"),
        _c(r"(lacks?|lacking|without|absent){S}{0,15}(the )?(blue|colou?r) (contrast|excess|offset)"),
    ],
    "over_subtraction": [
        _c(r"butterfl|bow[- ]?tie|quadrupol|four[- ]lobed|bipolar (residual|over|pattern|lobe|structure)"),
        _c(r"(over|under)[- ]?subtract"),
        _c(r"subtraction[- ](residual|artifact|artefact|ringing|dipole|ring|lobe|pattern|error|mismatch|"
                   r"hook|ghost|signature)"),
        _c(r"residual[- ](ringing|dipole|lobe|bowtie|butterfly|ring|fan|hook|artefact|artifact|pattern)"),
        _c(r"s[eé]rsic{S}{0,30}(mis-?fit|mismatch|subtraction|residual|model)"),
        _c(r"(model|fit|ellipticity|centring|centering)[- ](residual|mismatch|mis-?fit|error)"),
        _c(r"psf[- ](subtraction|ringing)|mis-?cent(re|er)ed{S}{0,20}(model|subtraction)|"
                   r"concentric ring|radial[- ]profile subtraction|subtraction (artifact|artefact)"),
        _c(r"(is|are|as|just|only|merely|simply){S}{0,15}(a |the )?(subtraction|model) (residual|artifact|artefact)"),
    ],
    "spiral_ring_disk": [
        _c(r"spiral|\bbarred\b|\bbar\b|edge[- ]?on|\bneedle\b|\bdis[ck]\b|pseudo-?ring|"
                   r"(resonance|collisional|inner|outer|nuclear|ring) (ring|galaxy)|ring galaxy|"
                   r"h\s?ii (knot|region|complex)|star[- ]?forming (knot|clump|ring|arm|region)"),
    ],
    "companion": [
        _c(r"chance[- ](alignment|projection|superposition|coincidence)|projected (companion|galaxy|"
                   r"neighbou?r|background|foreground|dwarf|object)|\bcompanion\b|group member|"
                   r"field (object|galaxy|galaxies)|unrelated (radii|galaxy|galaxies|object|source|position|"
                   r"neighbou?r|system)|superpos|line[- ]of[- ]sight|foreground|background (galaxy|source|object|"
                   r"dwarf|spiral|disk)|separate (galaxy|galaxies|object|source|system)|neighbou?r(ing)? galax"),
    ],
    "no_counter_image": [
        _c(r"\b(no|without|lacks?|lacking|missing|absent|nor)\b{S}{0,25}counter[- ]?(image|arc|part)"),
        _c(r"counter[- ]?(image|arc){S}{0,30}(absent|missing|lacking|is not|are not|none|nowhere|"
                   r"cannot be|does not)"),
        _c(r"one[- ]sided|single[- ]image"),
    ],
    "not_tangential": [
        _c(r"not tangential|non-?tangential|radially (elongated|oriented|aligned|extended)|"
                   r"radial (elongation|orientation|streak|feature)"),
        _c(r"(curvature|concavity){S}{0,40}\b(not|off|away|elsewhere|wrong)\b{S}{0,40}"
                   r"(cent(re|er)|deflector|galaxy|nucleus|target)"),
        _c(r"cent(re|er) of curvature{S}{0,40}(off|not|away|elsewhere|sits|lands|lies){S}{0,40}"
                   r"(off|away|beyond|outside|\d)"),
        _c(r"concave away|convex toward|straight (streak|line|needle|feature|filament|ray|segment)|"
                   r"no (tangential )?curvature|not curved|(not|nor|isn'?t) concave|rising monotonically|"
                   r"inclined{S}{0,25}from the tangential|wrong (curvature|orientation)"),
    ],
    "diffraction_detector": [
        _c(r"diffraction|\bspikes?\b|snowball|cosmic[- ]ray|\bCRs?\b|persistence|mosaic (seam|edge)|"
                   r"\bdither|1/f|amplifier|strip(e|ing)|saturat|bleed|detector (artifact|artefact|edge|seam|"
                   r"glitch)|hot pixel|bad pixel|psf wing|\bwisp"),
    ],
    "low_sn": [
        _c(r"low[- ]s/?n|low signal|signal[- ]to[- ]noise|noise[- ]level|at the noise|below the noise|"
                   r"marginal(ly)? (detect|signific)|indistinguishable from noise|noise (peak|fluctuation|spike)|"
                   r"too faint|very faint|extremely faint|faint(ness)? (to|and)|"
                   r"only (visible|present|appears?|seen|exists?|shows?) in the (residual|subtract)|"
                   r"not (visible|present|seen|traceable|detect\w*) in (the )?(normal|deep|colour|direct|un-?subtract)"),
    ],
    "shell_tidal": [
        _c(r"\bshells?\b|tidal|merger|merging|interact|debris|\bstream|\btails?\b|umbrella|ripple|"
                   r"stripped|plume"),
    ],
}

# ---- locates_feature: a radius, a PA/quadrant token or a "deg" value ------------------
_RADIUS = re.compile(r"(\br\s*[=~≈]\s*\d|\d+(\.\d+)?\s*(\"|″|''|arcsec\b|\bas\b)|radius of \d|"
                     r"\b(at|to|from)\s+~?\d+(\.\d+)?\s*(\"|″|arcsec))", _F)
# compass tokens must stand alone (not "S/N", not inside a word); "N" alone counts only as a
# bare token, which is how the personas write "the N 'arc'" / "1.0\" W"
_COMPASS = re.compile(r"(?<![/\w])(N|NE|E|SE|S|SW|W|NW|NNE|ENE|ESE|SSE|SSW|WSW|WNW|NNW)(?![/\w])")
_COMPASS_WORDS = re.compile(r"\b(north|south|east|west|north-?east|north-?west|south-?east|south-?west)(ern|wards?)?\b", _F)
_DEG = re.compile(r"(\d+(\.\d+)?\s*(deg(rees?)?\b|°)|\bPA\s*[=~≈]?\s*-?\d|position[- ]angle)", _F)


def locates_feature(text) -> bool:
    """True when the text points at pixels: a radius in arcsec, a compass/PA token or a deg value."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return False
    t = str(text)
    return bool(_RADIUS.search(t) or _COMPASS.search(t) or _COMPASS_WORDS.search(t) or _DEG.search(t))


def tag_text(text) -> dict[str, bool]:
    """{category: hit} for one text (all ten categories) + locates_feature."""
    t = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    out = {c: any(p.search(t) for p in _PATTERNS[c]) for c in CATEGORIES}
    out["locates_feature"] = locates_feature(t)
    return out


def grounds(text) -> list[str]:
    """The categories a text hits, in C4 order."""
    tg = tag_text(text)
    return [c for c in CATEGORIES if tg[c]]


def forbidden_only(text, alternative=None, covers_direct=None) -> bool:
    """True iff the text hits at least one category and every hit is a forbidden ground —
    with the structured exemptions when `alternative` is given (a critic record): the θ_E
    regex is exempt under alternative "scale_tension", the residual regex under
    "subtraction_residual" unless `covers_direct`, the colour regex beside a structural
    alternative."""
    tg = tag_text(text)
    structural_prose = any(tg[c] for c in STRUCTURAL)
    if alternative is None:
        g = [c for c in CATEGORIES if tg[c]]
        return bool(g) and all(c in FORBIDDEN for c in g)
    flags = _forbidden_flags(tg, str(alternative), bool(covers_direct), structural_prose)
    return any(flags.values()) and not structural_prose


def _forbidden_flags(tg: dict, alt: str, covers_direct: bool, structural_prose: bool) -> dict:
    """The three structured forbidden flags for one critic record (see the module doc)."""
    alt = (alt or "").strip()
    return {
        "theta_e_prior_forbidden": bool(tg["theta_e_prior"]) and alt != "scale_tension",
        "over_subtraction_forbidden": bool(tg["over_subtraction"]) and (alt != "subtraction_residual" or covers_direct),
        "colour_only_forbidden": bool(tg["colour_only"]) and not structural_prose and alt not in STRUCTURAL_ALTS,
    }


# ------------------------------------------------------------------ frames in, table out
def incumbent_long(replay: pd.DataFrame) -> pd.DataFrame:
    """outputs/incumbent_replay.csv (wide, one row per id) -> one row per (id, persona) with
    columns id, persona, verdict, alternative, notes, text (= alternative | notes)."""
    rows = []
    for p in PERSONAS_INCUMBENT:
        if f"{p}_verdict" not in replay.columns:
            continue
        sub = replay[["id", f"{p}_verdict", f"{p}_alternative", f"{p}_notes"]].copy()
        sub.columns = ["id", "verdict", "alternative", "notes"]
        sub.insert(1, "persona", p)
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["id", "persona", "verdict", "alternative", "notes"])
    out["alternative"] = out["alternative"].fillna("").astype(str)
    out["notes"] = out["notes"].fillna("").astype(str)
    return out


def _pick(d: dict, *keys, default=""):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


CRITIC_COLS = ["id", "persona", "no_opinion", "alternative", "alternative_desc", "notes",
               "accounts_for", "covers_direct"]


def _record_of(r, raw_col, flat_keys):
    rec = None
    if raw_col is not None and isinstance(r[raw_col], str):
        rec = parse.extract_json_block(r[raw_col])
        if not isinstance(rec, dict):
            rec = None
    elif raw_col is not None and isinstance(r[raw_col], dict):
        rec = r[raw_col]
    if rec is None:      # columns already flattened by the runner
        rec = {k: r[k] for k in flat_keys if k in r.index}
    return rec or None


def votes_to_critics(votes: pd.DataFrame, roles=("artifact", "geometry", "morphology")) -> pd.DataFrame:
    """New-scheme `_votes.parquet` (one row per persona call with the raw model text) -> one
    row per critic call: id, persona, no_opinion, alternative, alternative_desc, notes,
    accounts_for, covers_direct (True when any item in accounts_for was marked
    visible_in_direct by the same id's advocate record; None when no advocate record is in
    the votes). The raw column is found by name (raw | raw_json | response | json | record)
    and read with `parse.extract_json_block` — the model's raw text is usually prose + the
    JSON object, not bare JSON (the runner stores what the model wrote, exactly as
    grader_direct parsed it); unparsable rows are dropped (they are parse failures, counted
    elsewhere)."""
    role_col = "role" if "role" in votes.columns else "persona"
    raw_col = next((c for c in ("raw", "raw_json", "response", "json", "record") if c in votes.columns), None)
    # the advocate's items per id: k -> visible_in_direct
    direct: dict[str, dict] = {}
    for _, r in votes.iterrows():
        if str(r.get(role_col, "")) != "advocate":
            continue
        rec = _record_of(r, raw_col, ())
        if rec and isinstance(rec.get("items"), list):
            cid = str(r.get("id", r.get("name", "")))
            direct[cid] = {}
            for it in rec["items"]:
                try:
                    direct[cid][int(it.get("k"))] = bool(it.get("visible_in_direct", True))
                except (TypeError, ValueError, AttributeError):
                    continue
    rows = []
    for _, r in votes.iterrows():
        role = str(r.get(role_col, ""))
        if role not in roles:
            continue
        rec = _record_of(r, raw_col, ("no_opinion", "alternative", "alternative_desc", "notes"))
        if not rec:
            continue
        cid = str(r.get("id", r.get("name", "")))
        acc = []
        for k in (_pick(rec, "accounts_for", default=[]) or []):
            try:
                acc.append(int(k))
            except (TypeError, ValueError):
                continue
        covers_direct = None
        if cid in direct:
            covers_direct = any(direct[cid].get(k, False) for k in acc)
        rows.append({"id": cid, "persona": role,
                     "no_opinion": bool(_pick(rec, "no_opinion", default=False)),
                     "alternative": _pick(rec, "alternative", default=None),
                     "alternative_desc": str(_pick(rec, "alternative_desc", default="")),
                     "notes": str(_pick(rec, "notes", default="")),
                     "accounts_for": acc, "covers_direct": covers_direct})
    return pd.DataFrame(rows, columns=CRITIC_COLS)


def refutation_mask(df: pd.DataFrame) -> pd.Series:
    """Which rows are refutations: incumbent = verdict == fail; new scheme = a named
    alternative (not None/'' ) with no_opinion False."""
    if "verdict" in df.columns:
        return df["verdict"].astype(str).str.lower().eq("fail")
    named = df["alternative"].map(lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))
                                  and str(v).strip() not in ("", "None", "null"))
    if "no_opinion" in df.columns:
        named &= ~df["no_opinion"].astype(bool)
    return named


def refutation_text(df: pd.DataFrame) -> pd.Series:
    """The text the regexes see: incumbent `alternative | notes`; new scheme
    `alternative_desc | notes` (the enum `alternative` is structured, not prose)."""
    if "alternative_desc" in df.columns:
        a = df["alternative_desc"].fillna("").astype(str)
    else:
        a = df["alternative"].fillna("").astype(str)
    return a + " | " + df["notes"].fillna("").astype(str)


def _alt_str(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s in ("None", "null", "nan") else s


def tag_frame(df: pd.DataFrame) -> pd.DataFrame:
    """df + one bool column per category + locates_feature + the three `*_forbidden` flags +
    the two `uses_*` flags + forbidden_only + is_refutation. On a critic frame (has an
    `alternative` column and no `verdict`) the forbidden flags carry the structured
    exemptions; on the incumbent they are the raw categories."""
    out = df.copy()
    text = refutation_text(out)
    tag_cols = list(CATEGORIES) + ["locates_feature"]
    tags = pd.DataFrame([tag_text(t) for t in text], index=out.index, columns=tag_cols)   # columns even when empty
    for c in tag_cols:
        out[c] = tags[c].astype(bool)
    structural_prose = out[list(STRUCTURAL)].any(axis=1) if len(out) else pd.Series(False, index=out.index, dtype=bool)
    new_scheme = "alternative" in out.columns and "verdict" not in out.columns
    if new_scheme:
        alts = out["alternative"].map(_alt_str)
        cov = out["covers_direct"].map(lambda v: bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False) \
            if "covers_direct" in out.columns else pd.Series(False, index=out.index)
        flags = [_forbidden_flags({c: bool(out.at[i, c]) for c in CATEGORIES}, alts[i], bool(cov[i]), bool(structural_prose[i]))
                 for i in out.index]
        fl = pd.DataFrame(flags, index=out.index, columns=list(FORBIDDEN_FLAGS))
        for c in FORBIDDEN_FLAGS:
            out[c] = fl[c].astype(bool) if len(out) else pd.Series(dtype=bool)
        out["uses_scale_tension"] = alts.eq("scale_tension")
        out["uses_subtraction_residual"] = alts.eq("subtraction_residual")
    else:
        for c, f in zip(FORBIDDEN, FORBIDDEN_FLAGS):
            out[f] = out[c].astype(bool)
        out["uses_scale_tension"] = pd.Series(False, index=out.index, dtype=bool)
        out["uses_subtraction_residual"] = pd.Series(False, index=out.index, dtype=bool)
    out["any_ground"] = out[list(CATEGORIES)].any(axis=1)
    out["forbidden_only"] = out[list(FORBIDDEN_FLAGS)].any(axis=1) & ~structural_prose
    out["is_refutation"] = refutation_mask(out)
    return out


def audit_table(df: pd.DataFrame, by: str = "persona") -> pd.DataFrame:
    """Per-category share of refutations, one column per persona/role plus `all`; final rows
    any_ground / locates_feature / forbidden_only / n_refutations. Shares are over the
    refutations in that column (NaN when there are none)."""
    tagged = tag_frame(df)
    ref = tagged[tagged["is_refutation"]]
    groups = [(g, sub) for g, sub in ref.groupby(by)] if by in ref.columns else []
    groups.append(("all", ref))
    stats = list(CATEGORIES) + list(FORBIDDEN_FLAGS) + list(USAGE_FLAGS) + ["any_ground", "locates_feature", "forbidden_only"]
    table = {}
    for name, sub in groups:
        col = {s: (float(sub[s].mean()) if len(sub) else float("nan")) for s in stats}
        col["n_refutations"] = float(len(sub))
        table[name] = col
    return pd.DataFrame(table).reindex(stats + ["n_refutations"])


def forbidden_rate(df: pd.DataFrame) -> float:
    """The pre-registered monitor: share of refutations whose only grounds are forbidden."""
    t = audit_table(df)
    return float(t.loc["forbidden_only", "all"])


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incumbent", type=Path, help="outputs/incumbent_replay.csv (wide per-persona text)")
    ap.add_argument("--votes", type=Path, nargs="*", default=[], help="new-scheme *_votes.parquet files")
    ap.add_argument("--out", type=Path, help="write the pooled audit table(s) as CSV")
    a = ap.parse_args(argv)
    tables = []
    if a.incumbent:
        rep = _util.read_pinned(a.incumbent, dtype={"id": str}) if a.incumbent.with_suffix(".csv.sha").exists() \
            else pd.read_csv(a.incumbent, dtype={"id": str})
        t = audit_table(incumbent_long(rep))
        print("incumbent refutations (verdict == fail), share per category:")
        print(t.round(4).to_string())
        tables.append(t.assign(source="incumbent"))
    for v in a.votes:
        crit = votes_to_critics(pd.read_parquet(v))
        t = audit_table(crit)
        print(f"\n{v.name}: named refutations, share per category:")
        print(t.round(4).to_string())
        tables.append(t.assign(source=v.stem))
    if not tables:
        ap.error("give --incumbent and/or --votes")
    if a.out:
        out = pd.concat([t.rename_axis("statistic").reset_index() for t in tables], ignore_index=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
