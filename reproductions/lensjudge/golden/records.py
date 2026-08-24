#!/usr/bin/env python3
"""golden/records.py — the per-role pydantic records of a finished run, rebuilt from its votes.

The deployment rule (REGISTRY.md "Deployment rule v2-deploy") makes every letter a
deterministic function of the STORED records plus one thresholds file: nothing is re-scored,
the records are re-read. A run stores two parquets — `preds_*.parquet` (one row per item,
the `schemas_panel.to_row` shape + the run tuple) and `preds_*_votes.parquet` (one row per
persona CALL: name, role, k, parse_ok, raw, cost_usd, system_sha16, written by
`run_truth_eval.vote_rows`). `raw` is the model's raw text exactly as the runner saw it —
```json fences, a prose preamble, trailing remarks — so this module parses it through the
SAME path the runner used at run time:

    panel._call → grader_jwst.grade_candidate(schema=…) → grader_direct.grade_candidate
      → common.parse.parse_model(raw, schema)      with schema = schemas_panel.SCHEMA_FOR_ROLE[role]

`parse_role_raw` IS that last call (imported, not re-implemented), so a raw string yields the
record the run scored — or None, the run's parse failure. Nothing is coerced here either: a
record that fails its `extra="forbid"` schema is None, never a default.

Two consumers: `records_from_votes` (votes → {name: {role: record | None}}; a role the run
never called is an ABSENT key, a failed one is None) and `panel_result_from_records`, the
glue onto `schemas_panel.assemble(advocate, critics, arbitrator, thresholds, ...)` that
rebuilds S / S_arb / letters / alternative_final from records + thresholds. `rebuild_rows` +
`compare_rebuild` check that the rebuilt rows reproduce the stored parquet column by column —
the zero-API validation `regrade_scrambled.py --reletter` and `transfer_check.py` rest on.

Run-time parity, stated once:
  * the stored `parse_ok` is honoured by default (`respect_parse_ok=True`): a row the run
    could not parse stays None even if the raw parses today, because the registered
    parse-failure policy (S NaN, excluded from recall/FPR) was applied to THAT row and the
    top-up of NaN rows is a registered re-score, not a re-parse. `parse_parity` reports
    both directions of disagreement per role (counts only, no ids).
  * a critic raw is validated as `CriticRecord`, whose `persona` Literal spans all three
    critics — exactly as at run time, a geometry reply stored under the artifact role would
    parse. `persona_matches` is the separate check (counted in `parse_parity`, never
    applied to the record).
  * a votes row whose `raw` is missing (NaN — seen for 2 items x 5 roles of the sonnet
    scrambled-100 dev run, parse_ok True, S stored) can be recovered from the run's per-role
    traces (`trace_dir/<safe_name>_<role>.jsonl`, the `direct_response` / parsed
    `direct_repair` text — exactly what the runner handed to parse_model) by passing
    `trace_dir=`; without it the row is None and `parse_parity` counts it as raw_missing.
  * the incumbent arm (a0) stores its three verdicts under the plain role names; pass
    `arm="a0"` (load_run reads it from the preds parquet) to validate them as
    `IncumbentVerdict`. `panel_result_from_records` refuses incumbent records (their score
    is `aggregate_v2.passcount_incumbent`, not `assemble`).

CLI (zero API, read-only on the run):
    python lensjudge/golden/records.py --preds outputs/<run>.parquet [--votes …] [--k 1] [--out DIR]
prints the parse-parity and rebuild-check count tables (no candidate ids) and, with --out,
pins them as CSVs (`parse_parity.csv`, `rebuild_check.csv`) in a NEW directory.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge.common import parse  # noqa: E402
from lensjudge.golden import _util, aggregate_v2, schemas_panel  # noqa: E402

CRITIC_ROLES = aggregate_v2.CRITIC_ROLES            # ("artifact", "geometry", "morphology")
ROLES = aggregate_v2.ROLES                          # advocate, critics, arbitrator
INCUMBENT_ARM = "a0"
VOTES_SUFFIX = "_votes.parquet"                     # run_truth_eval: out.with_name(out.stem + "_votes.parquet")
VOTE_COLS = ("name", "unit_id", "role", "k", "parse_ok", "raw", "cost_usd", "system_sha16")
THRESHOLD_COLS = ("tau0", "t_A", "t_B", "letter_source")
# the stored columns a rebuild from records must reproduce (everything to_row derives from
# the records + thresholds; not the run tuple, not the manifest join columns)
REBUILD_COLS = (
    "parse_ok", "error", "grade_pred", "p_lens", "confidence", "contaminant", "escalate", "rationale",
    *(f"crit_{c}" for c in schemas_panel.CRITERIA_V2),
    "S", "S_arb", "p_evidence", "scale_class", "scale_class_final", "letter_llm", "letter_arb",
    "letter_source", "alternative_final", "n_items", "n_surviving", "needs_human",
    *(f"a_{r}" for r in CRITIC_ROLES), *(f"r_{r}" for r in CRITIC_ROLES),
    *(f"alt_{r}" for r in CRITIC_ROLES), *(f"no_opinion_{r}" for r in CRITIC_ROLES),
    *(f"ruling_{r}" for r in CRITIC_ROLES), "parse_fail_roles", "calls",
)
PARITY_COLS = ("role", "parse_ok", "raw_missing", "parsed_now", "persona_ok", "n")
TRACE_EVENTS = ("direct_response", "direct_repair")   # grader_direct trace events that carry the raw text
REBUILD_CHECK_COLS = ("col", "n_compared", "n_mismatch")

Record = Union[schemas_panel.AdvocateRecord, schemas_panel.CriticRecord,
               schemas_panel.ArbitratorRecord, schemas_panel.IncumbentVerdict]


# ------------------------------------------------------------------ one raw → one record
def schema_for(role: str, arm: Optional[str] = None):
    """The pydantic record a role's reply validates into — `schemas_panel.SCHEMA_FOR_ROLE`
    (advocate / artifact / geometry / morphology / arbitrator / incumbent); under the
    incumbent arm every role is an IncumbentVerdict. Unknown role ⇒ ValueError (a wrong
    role name is a caller bug, not a model parse failure)."""
    if arm == INCUMBENT_ARM:
        return schemas_panel.IncumbentVerdict
    schema = schemas_panel.SCHEMA_FOR_ROLE.get(str(role))
    if schema is None:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(schemas_panel.SCHEMA_FOR_ROLE)}")
    return schema


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    try:
        return bool(pd.isna(x)) if not isinstance(x, (str, bytes, list, dict, tuple)) else False
    except (TypeError, ValueError):
        return False


def is_missing(x: Any) -> bool:
    """Public name of `_is_missing`: None / NaN / pandas NA (strings, lists, dicts never are)."""
    return _is_missing(x)


def parse_role_raw(role: str, raw: Any, arm: Optional[str] = None) -> Optional[Record]:
    """The run-time parse of one role's raw model text: `common.parse.parse_model(raw,
    SCHEMA_FOR_ROLE[role])` — fence / prose / trailing-text stripping and the extra=forbid
    validation included. None on any failure, and for a missing raw (None / NaN / "")."""
    schema = schema_for(role, arm)
    if _is_missing(raw) or not isinstance(raw, str) or not raw.strip():
        return None
    return parse.parse_model(raw, schema)


def persona_matches(role: str, record: Any) -> bool:
    """Whether a parsed record's `persona` names the role it was stored under (advocate /
    critic name / arbitrator). Advisory only: run-time validation does not check it."""
    if record is None:
        return False
    return str(getattr(record, "persona", None)) == str(role)


# ------------------------------------------------------------------ trace fallback
def trace_path_for(trace_dir, name: str, role: str) -> Path:
    """`panel._trace` naming: <trace_dir>/<_util.safe_name(name)>_<role>.jsonl."""
    return Path(trace_dir) / f"{_util.safe_name(str(name))}_{role}.jsonl"


def raw_from_trace(path) -> Optional[str]:
    """The raw text the run parsed, recovered from one per-role trace: the last
    `direct_response` text, replaced by the text of a `direct_repair` that parsed (the
    runner swaps raw for the repaired text — a later response starts over). None when the
    trace is absent or holds no response."""
    p = Path(path)
    if not p.exists():
        return None
    raw = None
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        ev, text = e.get("event"), e.get("text")
        if ev == "direct_response" and isinstance(text, str):
            raw = text
        elif ev == "direct_repair" and e.get("parse_ok") and isinstance(text, str):
            raw = text
    return raw


def _raw_for(row, trace_dir) -> Any:
    """The row's raw, or the trace's when the stored raw is missing and a trace_dir is given."""
    raw = row.raw
    if trace_dir is not None and (_is_missing(raw) or not isinstance(raw, str) or not raw.strip()):
        raw = raw_from_trace(trace_path_for(trace_dir, row.name, row.role))
    return raw


def _false(x: Any) -> bool:
    return (not _is_missing(x)) and (not bool(x))


# ------------------------------------------------------------------ votes → records
def _select_k(votes: pd.DataFrame, k: Optional[int]) -> pd.DataFrame:
    if k is None:
        if "k" in votes.columns and votes["k"].nunique(dropna=True) > 1:
            ks = sorted(int(x) for x in votes["k"].dropna().unique())
            raise ValueError(f"votes hold several replicates k={ks}; pass k=")
        return votes
    if "k" not in votes.columns:
        raise ValueError("k= given but the votes frame has no 'k' column")
    return votes[votes["k"] == int(k)]


def _check_cols(df: pd.DataFrame, cols: tuple, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} lacks columns {missing}")


def check_cols(df: pd.DataFrame, cols: tuple, what: str) -> None:
    """Public name of `_check_cols`: ValueError naming the columns `df` lacks."""
    _check_cols(df, cols, what)


def records_from_votes(votes: pd.DataFrame, k: Optional[int] = None, *, arm: Optional[str] = None,
                       respect_parse_ok: bool = True, trace_dir=None) -> dict:
    """{name: {role: record | None}} from a votes frame (one row per persona call). Roles the
    run never called for an item are ABSENT keys; a row whose raw does not parse is None.
    `k` selects one replicate (required when the frame holds several). With
    `respect_parse_ok` (default) a row the run marked parse_ok=False is None whatever its raw
    parses to today (the registered NaN policy was applied to that row); set False to
    re-parse purely from raw. A missing raw is read from `trace_dir` when given
    (`raw_from_trace`). Duplicate (name, role) rows within one replicate refuse."""
    _check_cols(votes, ("name", "role", "raw"), "votes")
    sub = _select_k(votes, k)
    if len(sub) == 0:
        return {}
    dup = sub.duplicated(["name", "role"], keep=False)
    if dup.any():
        roles = sorted(sub.loc[dup, "role"].astype(str).unique())
        raise ValueError(f"votes hold duplicate (name, role) rows for roles {roles}; one call per role per replicate")
    have_ok = "parse_ok" in sub.columns
    out: dict = {}
    for row in sub.itertuples(index=False):
        name, role = str(row.name), str(row.role)
        rec = parse_role_raw(role, _raw_for(row, trace_dir), arm)
        if respect_parse_ok and have_ok and _false(row.parse_ok):
            rec = None
        out.setdefault(name, {})[role] = rec
    return out


def parse_parity(votes: pd.DataFrame, k: Optional[int] = None, *, arm: Optional[str] = None,
                 trace_dir=None) -> pd.DataFrame:
    """Count table (role, parse_ok, raw_missing, parsed_now, persona_ok, n): the run's stored
    parse_ok against a parse of the stored raw today, per role — no ids. Every row with
    parse_ok True should read parsed_now True (the raw the run scored still parses);
    parse_ok False with parsed_now True is a raw that parses today but did not at run
    time; raw_missing marks rows whose votes raw is absent (parsed from the trace when
    `trace_dir` is given)."""
    _check_cols(votes, ("name", "role", "raw"), "votes")
    sub = _select_k(votes, k)
    rows = []
    for row in sub.itertuples(index=False):
        missing = _is_missing(row.raw) or not isinstance(row.raw, str) or not row.raw.strip()
        rec = parse_role_raw(str(row.role), _raw_for(row, trace_dir), arm)
        ok = bool(row.parse_ok) if "parse_ok" in sub.columns else None
        rows.append({"role": str(row.role), "parse_ok": ok, "raw_missing": bool(missing), "parsed_now": rec is not None,
                     "persona_ok": persona_matches(str(row.role), rec) if arm != INCUMBENT_ARM else rec is not None})
    if not rows:
        return pd.DataFrame(columns=list(PARITY_COLS))
    df = pd.DataFrame(rows)
    keys = ["role", "parse_ok", "raw_missing", "parsed_now", "persona_ok"]
    tab = df.groupby(keys, dropna=False).size().reset_index(name="n")
    order = {r: i for i, r in enumerate(ROLES)}
    tab["_o"] = tab["role"].map(lambda r: order.get(r, len(order)))
    tab = tab.sort_values(["_o"] + keys).drop(columns="_o")
    return tab[list(PARITY_COLS)].reset_index(drop=True)


# ------------------------------------------------------------------ a run on disk
def votes_path_for(preds_path) -> Path:
    """The sibling votes parquet of a preds parquet (run_truth_eval's naming)."""
    p = Path(preds_path)
    return p.with_name(p.stem + VOTES_SUFFIX)


def _single(df: pd.DataFrame, col: str):
    """The one value a column holds (None when absent, empty, or several)."""
    if col not in df.columns:
        return None
    vals = df[col].dropna().unique()
    return vals[0] if len(vals) == 1 else None


def single(df: pd.DataFrame, col: str):
    """Public name of `_single`: the one value `col` holds (None when absent, empty, or several)."""
    return _single(df, col)


def load_run(preds_path, votes_path=None, *, trace_dir=None, respect_parse_ok: bool = True) -> tuple:
    """(preds_df, records) for a stored run: the preds parquet as written and the records
    rebuilt from its votes parquet (default: the `_votes.parquet` sibling). The replicate
    index and the arm come from the preds parquet when it holds exactly one of each
    (k= filters the votes; arm a0 selects the incumbent schema). `trace_dir` recovers a
    missing votes raw from the run's per-role traces. Votes for a name the preds parquet
    does not carry refuse (a votes file from another run)."""
    preds_path = Path(preds_path)
    votes_path = votes_path_for(preds_path) if votes_path is None else Path(votes_path)
    preds = pd.read_parquet(preds_path)
    votes = pd.read_parquet(votes_path)
    _check_cols(preds, ("name",), f"preds {preds_path.name}")
    k = _single(preds, "k")
    k = None if k is None else int(k)
    arm = _single(preds, "arm")
    arm = str(arm) if arm is not None else None
    stray = set(votes["name"].astype(str)) - set(preds["name"].astype(str))
    if stray:
        raise ValueError(f"{votes_path.name}: {len(stray)} vote name(s) absent from {preds_path.name}")
    records = records_from_votes(votes, k, arm=arm, respect_parse_ok=respect_parse_ok, trace_dir=trace_dir)
    return preds, records


# ------------------------------------------------------------------ records → S, letters, row
def thresholds_from_row(row) -> dict:
    """The `aggregate_v2.resolve_thresholds`-shaped dict a stored preds row was lettered
    with: {tau0, t_A, t_B, letter_source} from its run-tuple columns."""
    get = row.get if hasattr(row, "get") else (lambda c, d=None: getattr(row, c, d))
    out: dict = {}
    for c in THRESHOLD_COLS:
        v = get(c)
        if c == "letter_source":
            out[c] = str(v) if not _is_missing(v) else "provisional"
        elif _is_missing(v):
            if c in ("t_A", "t_B"):
                raise ValueError(f"row lacks {c}; cannot letter without thresholds")
            out[c] = aggregate_v2.PROVISIONAL["tau0"]
        else:
            out[c] = float(v)
    return out


def _roles_of(name: str, records: dict) -> dict:
    if name not in records:
        raise KeyError(f"no records for {name!r} (no votes row for that item)")
    roles = records[name]
    if any(isinstance(r, schemas_panel.IncumbentVerdict) for r in roles.values()):
        raise ValueError("incumbent (a0) records are pass-counted by aggregate_v2.passcount_incumbent, "
                         "not assembled")
    return roles


def panel_result_from_records(name: str, records: dict, thresholds: Optional[dict] = None,
                              preds_row=None) -> schemas_panel.PanelResult:
    """Rebuild one item's `schemas_panel.PanelResult` (S, S_arb, letter, letter_arb,
    letter_source, a, r, parse_failures, ...) from `records[name]` = {role: record | None} via
    `schemas_panel.assemble(advocate, critics, arbitrator, thresholds, parse_failures=None,
    cost_usd, calls, system_sha16s, raw=None, meta, arbitrator_called)`. The critics dict holds
    exactly the critic roles the run called (present keys); the arbitrator counts as called
    iff its role is present; None entries become parse_failures. `thresholds` defaults to
    the stored row's (tau0, t_A, t_B, letter_source); `preds_row` also supplies cost_usd
    and the per-role system shas — nothing that touches the score."""
    roles = _roles_of(name, records)
    if thresholds is None:
        if preds_row is None:
            raise ValueError("pass thresholds= or preds_row= (the stored tau0/t_A/t_B/letter_source)")
        thresholds = thresholds_from_row(preds_row)
    advocate = roles.get("advocate")
    critics = {r: roles[r] for r in CRITIC_ROLES if r in roles}
    arbitrator = roles.get("arbitrator")
    cost, shas = 0.0, {}
    if preds_row is not None:
        get = preds_row.get if hasattr(preds_row, "get") else (lambda c, d=None: getattr(preds_row, c, d))
        c = get("cost_usd")
        cost = 0.0 if _is_missing(c) else float(c)
        for r in ROLES:
            s = get(f"system_sha16_{r}")
            if not _is_missing(s):
                shas[r] = str(s)
    return schemas_panel.assemble(
        advocate, critics, arbitrator, dict(thresholds), parse_failures=None, cost_usd=cost,
        calls=len(roles), system_sha16s=shas, raw=None,
        meta={"rebuilt_from": "votes", "roles": list(roles)}, arbitrator_called="arbitrator" in roles)


def deploy_from_roles(roles: Optional[dict], thresholds: dict, rule: str = "R1") -> dict:
    """`aggregate_v2.deploy_letters` on one item's {role: record | None} (the shape
    `records_from_votes` / `load_run` return) under the run-time parse-failure policy of
    `schemas_panel.assemble`: a role whose key is PRESENT with a None record was called and
    failed to parse. deploy_letters cannot tell a failed arbitrator from one that was not
    needed (a None arbitrator lets the critics' reports stand and may demote), so a
    called-but-failed arbitrator voids letter_final (None), veto ("") and S_arb (NaN) while
    letter_rank (advocate-only) is kept — the row is a parse failure at run time (S NaN) and
    must not gain a certified letter here. A failed advocate / critic already yields None
    letters inside deploy_letters. `roles` None (no records) -> every letter None."""
    if roles is None:
        return aggregate_v2.deploy_letters(None, {}, None, thresholds, rule)
    advocate = roles.get("advocate")
    critics = {r: roles[r] for r in aggregate_v2.CRITIC_ROLES if r in roles}
    dep = aggregate_v2.deploy_letters(advocate, critics, roles.get("arbitrator"), thresholds, rule)
    if "arbitrator" in roles and roles["arbitrator"] is None:
        dep = {**dep, "letter_final": None, "veto": "", "S_arb": float("nan")}
    return dep


def row_from_records(name: str, records: dict, thresholds: Optional[dict] = None, preds_row=None) -> dict:
    """`schemas_panel.to_row` of the rebuilt result: the stored row shape (ROW_COLS), with the
    manifest columns (grade_truth, catalog, region, p_meta) copied from `preds_row`."""
    result = panel_result_from_records(name, records, thresholds, preds_row)
    cand = {"name": name}
    if preds_row is not None:
        get = preds_row.get if hasattr(preds_row, "get") else (lambda c, d=None: getattr(preds_row, c, d))
        for src, dst in (("grade_truth", "grade"), ("catalog", "catalog"), ("region", "region"), ("p_meta", "p_meta")):
            v = get(src)
            cand[dst] = None if _is_missing(v) else v
    return schemas_panel.to_row(result, cand)


def rebuild_rows(preds: pd.DataFrame, records: dict, thresholds: Optional[dict] = None) -> pd.DataFrame:
    """One rebuilt row (ROW_COLS) per preds row whose name has records, in preds order;
    thresholds per row from the stored run-tuple columns unless one dict is given for all."""
    _check_cols(preds, ("name",), "preds")
    rows = []
    for _, r in preds.iterrows():
        name = str(r["name"])
        if name not in records:
            continue
        rows.append(row_from_records(name, records, thresholds, r))
    return pd.DataFrame(rows, columns=list(schemas_panel.ROW_COLS))


def _same(a: Any, b: Any, atol: float) -> bool:
    ma, mb = _is_missing(a), _is_missing(b)
    if ma or mb:
        return ma and mb
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return math.isclose(fa, fb, rel_tol=0.0, abs_tol=atol)


def compare_rebuild(preds: pd.DataFrame, rebuilt: pd.DataFrame, cols: tuple = REBUILD_COLS,
                    atol: float = 1e-9) -> pd.DataFrame:
    """Count table (col, n_compared, n_mismatch): stored vs rebuilt values, aligned on name,
    for every column of `cols` both frames carry. NaN/None agree with NaN/None; floats
    within `atol`; everything else by string equality. No ids."""
    _check_cols(preds, ("name",), "preds")
    _check_cols(rebuilt, ("name",), "rebuilt")
    a = preds.drop_duplicates("name").set_index(preds.drop_duplicates("name")["name"].astype(str))
    b = rebuilt.drop_duplicates("name").set_index(rebuilt.drop_duplicates("name")["name"].astype(str))
    names = [n for n in a.index if n in b.index]
    out = []
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        n_bad = sum(1 for n in names if not _same(a.at[n, c], b.at[n, c], atol))
        out.append({"col": c, "n_compared": len(names), "n_mismatch": int(n_bad)})
    return pd.DataFrame(out, columns=list(REBUILD_CHECK_COLS))


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", required=True, help="preds_*.parquet of a finished run")
    ap.add_argument("--votes", default=None, help="its votes parquet (default: the _votes.parquet sibling)")
    ap.add_argument("--k", type=int, default=None, help="replicate index when the votes hold several")
    ap.add_argument("--trace-dir", default=None,
                    help="the run's per-role traces: recovers a votes row whose raw is missing")
    ap.add_argument("--reparse", action="store_true",
                    help="ignore the stored parse_ok and rebuild purely from raw (default honours it)")
    ap.add_argument("--out", default=None, help="NEW directory for the pinned parse_parity.csv / rebuild_check.csv")
    args = ap.parse_args(argv)

    preds_path = Path(args.preds)
    votes_path = votes_path_for(preds_path) if args.votes is None else Path(args.votes)
    preds, records = load_run(preds_path, votes_path, trace_dir=args.trace_dir, respect_parse_ok=not args.reparse)
    votes = pd.read_parquet(votes_path)
    k = args.k if args.k is not None else (int(_single(preds, "k")) if _single(preds, "k") is not None else None)
    arm = _single(preds, "arm")
    arm = str(arm) if arm is not None else None
    parity = parse_parity(votes, k, arm=arm, trace_dir=args.trace_dir)
    n_votes = int(parity["n"].sum()) if len(parity) else 0
    n_parsed = int(parity.loc[parity["parsed_now"], "n"].sum()) if len(parity) else 0
    print(f"[records] {preds_path.name}: {len(preds)} preds rows, {n_votes} vote rows (k={k}, arm={arm}); "
          f"{n_parsed}/{n_votes} raws parse; {len(records)} items with records")
    print(parity.to_string(index=False))
    if arm == INCUMBENT_ARM:
        check = pd.DataFrame(columns=list(REBUILD_CHECK_COLS))
        print("[records] incumbent arm: no assemble() rebuild")
    else:
        rebuilt = rebuild_rows(preds, records)
        check = compare_rebuild(preds, rebuilt)
        bad = check[check["n_mismatch"] > 0]
        print(f"[records] rebuild: {len(rebuilt)} rows; {len(bad)}/{len(check)} columns with mismatches")
        print((bad if len(bad) else check).to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        s1 = _util.pin(parity, out / "parse_parity.csv")
        s2 = _util.pin(check, out / "rebuild_check.csv")
        print(f"[records] wrote {out / 'parse_parity.csv'} ({s1}) and {out / 'rebuild_check.csv'} ({s2})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
