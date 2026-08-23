#!/usr/bin/env python3
"""golden/run_golden_eval.py — registry-gated model runs on the golden JWST set (E1/E2/E3).

One invocation = one pre-registered tuple (arm, model, rubric sha16, n_exemplars, splits
sha16, system-prompt sha16, thinking, effort) scored k times on one half of the golden split,
each replicate to its own parquet `preds_golden_{arm}_{model}_{split}_r{k}.parquet` (+
`.meta.json` with the full provenance, written only when the replicate is COMPLETE) and its
own trace dir `outputs/traces_golden_{arm}_{model}_{split}_r{k}/`.

The gate (golden/REGISTRY.md): `--split validate` refuses unless the tuple is a registered
row AND no completed parquet (one with a .meta.json) for that tuple exists anywhere under
outputs/ or beside --out; a parquet without its meta is an interrupted replicate and is
resumed, not refused. `--force-rescore --rescore-reason "..."` appends a dated rescore row
and stamps the parquet `rescored=True`. `--split align` is ungated (it is where tuning
happens). Every scored unit is recorded in the exposure registry (golden/registry.py, WP-D)
with kind "eval"; E2 exemplars with kind "fewshot"; and before EVERY validate run — any arm,
any model, so an SFT student scored through --arm e1 is covered too —
`registry.assert_unexposed(validate units, kinds=("fewshot", "sft"))` must pass. The system
prompt (rubric + note) is run through the banned lexicon (golden/banned_lexicon.txt, built
by audit_traces.py --build-lexicon) before any call; a validate run without a lexicon is
refused, because a rubric is the one model-facing text the trace audit cannot read back.

Arms: e1 zero-shot (rubric_imaging_v2 + jwst_note); e2 few-shot (n exemplars per letter from
the ALIGN half via golden/fewshot.py — on `--split align` the exemplar units are left out
of the scored set); e3 rubric arm (`--rubric PATH`, zero-shot). The note is always appended
to the rubric (grader_jwst.with_note) unless --no-jwst-note.

`--smoke N` grades N frame items zero-shot with no split or registry requirement (plumbing
check before any labels exist); they are marked exposed with kind "eval", run_tag "smoke".
Once golden/splits.csv exists the validate half is excluded from the pick (a smoke run is
not a registered tuple). `--smoke-stratum K_cowls` restricts the pick to literature controls
(nothing pipeline-only to leak, the cheapest rows to burn).

  python lensjudge/golden/run_golden_eval.py --arm e1 --model sonnet --split align --k 3
  python lensjudge/golden/run_golden_eval.py --arm e2 --model sonnet --split validate \\
      --n-exemplars 3 --k 3 --concurrency 4
  python lensjudge/golden/run_golden_eval.py --smoke 3 --smoke-stratum K_cowls --model sonnet

Backend: a Claude alias (--model sonnet|opus|haiku) selects LENSJUDGE_BACKEND=anthropic
unless the env already says otherwise; any other --model is a served open-model id.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.golden import _util, audit_traces, fewshot, grader_jwst  # noqa: E402
from lensjudge.imaging import run_batch  # noqa: E402

ARMS = ("e1", "e2", "e3")
SPLITS = ("align", "validate")
CLAUDE_ALIASES = ("sonnet", "opus", "haiku")
REGISTRY_MD = _util.HERE / "REGISTRY.md"
SPLITS_CSV = _util.HERE / "splits.csv"
LABELS_CSV = _util.HERE / "golden_labels.csv"
FRAME_CSV = _util.HERE / "frame.csv"
KEYS_DIR = _util.HERE / "keys"
KITS_DIR = _util.HERE / "kits"
MANIFEST = config.OUT / "golden_jwst_manifest.csv"
BANNED = _util.HERE / "banned_lexicon.txt"
# {model} is the alias made path-safe (_util.safe_name): two models in one arm (E1 sonnet
# primary / opus secondary) must not share a parquet path or the gate would call the second
# "already scored"
OUT_TEMPLATE = str(config.OUT / "preds_golden_{arm}_{model}_{split}_r{k}.parquet")
REGISTERED_COLS = ("arm", "model", "rubric_sha16", "n_exemplars", "splits_sha16",
                   "system_sha16", "thinking", "effort")
# the markdown separator of the Registered-arms table (tests and --print-tuple share it)
TABLE_SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|"


def thinking_setting() -> str:
    """What the call will actually use (config.thinking_options reads the same variable)."""
    return os.environ.get("LENSJUDGE_THINKING") or "off"


def effort_setting() -> str:
    return os.environ.get("LENSJUDGE_EFFORT") or "default"


# ------------------------------------------------------------------ the tuple
@dataclass(frozen=True)
class RunTuple:
    arm: str
    model: str
    rubric_sha16: str
    n_exemplars: int
    splits_sha16: str
    # everything else that changes the validate answer (review S8): the FULL system prompt
    # (rubric + note, or a --no-jwst-note rubric alone) and the reasoning settings
    system_sha16: str = ""
    thinking: str = "off"
    effort: str = "default"

    def run_tag(self, split: str, k: int) -> str:
        return f"golden_{self.arm}_{_util.safe_name(self.model)}_{split}_r{k}"

    def row(self, rubric_name: str = "", k: int = 3, note: str = "") -> str:
        today = _dt.date.today().isoformat()
        return (f"| {today} | {self.arm} | {self.model} | {rubric_name} | {self.rubric_sha16} | "
                f"{self.n_exemplars} | {self.splits_sha16} | {self.system_sha16} | {self.thinking} | "
                f"{self.effort} | {k} | {note} |")


def _read(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.with_suffix(path.suffix + ".sha").exists():
        return _util.read_pinned(path)
    return pd.read_csv(path)


def splits_sha(splits_csv: Path) -> str:
    """The pin of golden/splits.csv (its .sha sidecar; recomputed from text if absent)."""
    p = Path(splits_csv)
    if not p.exists():
        raise SystemExit(f"{p} does not exist yet (golden/split_halves.py writes it after pass 1)")
    sha_path = p.with_suffix(p.suffix + ".sha")
    if sha_path.exists():
        return sha_path.read_text().strip()
    return _util.sha_text(p.read_text())


# ------------------------------------------------------------------ REGISTRY.md parsing
def _tables(md: str) -> dict[str, list[dict]]:
    """{section title: rows-as-dicts} for every markdown pipe table under a '## ' heading."""
    out: dict[str, list[dict]] = {}
    section, header = None, None
    for line in md.splitlines():
        if line.startswith("## "):
            section, header = line[3:].strip(), None
            out.setdefault(section, [])
            continue
        if section is None or not line.strip().startswith("|"):
            header = header if line.strip() else None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            continue
        out[section].append(dict(zip(header, cells)))
    return out


# The gate helpers take `section=` / `cols=` (golden defaults) so golden/run_truth_eval.py can
# run the same score-once rule against its own REGISTRY.md table and its own tuple type
# (TruthTuple) without a second implementation of the ledger parsing.
def registered_rows(registry_md: Path, section: str = "Registered arms") -> list[dict]:
    return _tables(Path(registry_md).read_text()).get(section, [])


def _cell_same(cell, value) -> bool:
    """Markdown cell vs tuple field: integers compare as integers ('3' == 3), everything
    else as exact strings (shas, aliases, settings)."""
    a, b = str(cell), str(value)
    try:
        return int(a) == int(b)
    except ValueError:
        return a == b


def is_registered(registry_md: Path, t, section: str = "Registered arms",
                  cols: tuple = REGISTERED_COLS) -> bool:
    """Every `cols` value must match the row exactly (a row lacking one of the columns
    cannot match: the table header is part of the pre-registration)."""
    for r in registered_rows(registry_md, section):
        if all(c in r and _cell_same(r.get(c), getattr(t, c)) for c in cols):
            return True
    return False


def append_rescore(registry_md: Path, t, reason: str, cols: tuple = REGISTERED_COLS,
                   section: str = "Rescores") -> None:
    """Insert `| date | <cols...> | reason |` as the last row of the `section` table (the
    Rescores table used to be the last thing in the file, but REGISTRY.md now carries the
    truth-eval tables after it, so the row is placed by section, not appended at EOF)."""
    today = _dt.date.today().isoformat()
    line = ("| " + " | ".join([today] + [str(getattr(t, c)) for c in cols]
                              + [reason.replace('|', '/')]) + " |")
    p = Path(registry_md)
    lines = p.read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == f"## {section}"), None)
    if start is None:
        raise SystemExit(f"[gate] {p} has no '## {section}' section to log the rescore in")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    last_row = max((i for i in range(start + 1, end) if lines[i].strip().startswith("|")), default=None)
    if last_row is None:
        raise SystemExit(f"[gate] '## {section}' in {p} has no table")
    lines.insert(last_row + 1, line)
    p.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ outputs + the gate
def out_paths(template: str, arm: str, split: str, k: int, model: str = "") -> list[Path]:
    """`--out` as a template: {arm}/{model}/{split}/{k} placeholders ({k} = the replicate
    index 1..k) plus {K} = the replicate COUNT, so two tuples that differ only in k (an arm
    scored once and the same arm's k=3 replicate study) never share a file name; a plain
    path gets _r{k}."""
    paths = []
    for i in range(1, k + 1):
        if any(ph in template for ph in ("{k}", "{K}", "{arm}", "{split}", "{model}")):
            paths.append(Path(template.format(arm=arm, model=_util.safe_name(model), split=split, k=i, K=k)))
        else:
            p = Path(template)
            paths.append(p if k == 1 else p.with_name(f"{p.stem}_r{i}{p.suffix}"))
    return paths


def meta_path(out: Path) -> Path:
    return out.with_name(out.stem + ".meta.json")


def _meta_tuple_matches(mp: Path, t, cols: tuple = REGISTERED_COLS) -> Optional[bool]:
    """True/False whether the .meta.json at mp records tuple t; None if unreadable."""
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return None
    tup = m.get("tuple", {})
    return all(_cell_same(tup.get(c), getattr(t, c)) for c in cols)


def scored_outputs(paths: list[Path], t, split: str, cols: tuple = REGISTERED_COLS,
                   prefix: str = "preds_golden_") -> list[Path]:
    """COMPLETED parquets that already hold this tuple: a target path whose .meta.json
    records it, plus EVERY `*.meta.json` found recursively under the target dirs or the
    default outputs/ dir whose recorded tuple and split are this run's — the file name is
    not consulted, so a replicate written under another name (`--out outputs/peek.parquet`)
    or into a sub-directory (`outputs/smoke/`) is found too. `prefix` only labels the
    message. A target parquet without its meta is an interrupted replicate (meta is written
    on completion) and is resumable, not scored; a target whose meta records ANOTHER tuple
    is a path collision and raises."""
    hits: list[Path] = []
    for p in paths:
        mp = meta_path(p)
        if p.exists() and mp.exists():
            same = _meta_tuple_matches(mp, t, cols)
            if same is False:
                raise SystemExit(f"[gate] {p} already holds a DIFFERENT registered tuple "
                                 f"(see {mp}); choose another --out")
            hits.append(p)
        elif p.exists():
            print(f"[gate] {p} exists without {mp.name}: an interrupted replicate — resuming it")
    dirs = {p.parent for p in paths} | {config.OUT}
    for d in dirs:
        if not d.is_dir():
            continue
        for mp in d.rglob("*.meta.json"):
            if ".pre_rescore_" in mp.name:
                continue
            if _meta_tuple_matches(mp, t, cols) and _meta_split(mp) in (split, None):
                pq = mp.with_name(mp.name[:-len(".meta.json")] + ".parquet")
                if pq.exists() and pq not in hits:
                    hits.append(pq)
    return hits


def _meta_split(mp: Path) -> Optional[str]:
    """The split a .meta.json was written for (None when it does not say — older metas)."""
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return None
    s = m.get("split")
    return str(s) if s else None


def check_validate_gate(registry_md: Path, t, paths: list[Path],
                        force: bool = False, reason: Optional[str] = None, *,
                        section: str = "Registered arms", cols: tuple = REGISTERED_COLS,
                        split: str = "validate", prefix: str = "preds_golden_",
                        rescore_section: str = "Rescores") -> bool:
    """The score-once rule. Returns True when this run is a sanctioned rescore.
    Raises SystemExit on an unregistered tuple or an already-scored one without --force.
    `section`/`cols`/`split`/`prefix` name the ledger table, the tuple fields, the gated
    half and the parquet prefix (golden defaults; run_truth_eval passes its own)."""
    if not is_registered(registry_md, t, section, cols):
        raise SystemExit(
            "[gate] REFUSED: tuple not registered in {md}:\n  {row}\n"
            "  Add that row (with a date) to the '{section}' table BEFORE scoring {split}."
            .format(md=registry_md, row=t.row(), section=section, split=split))
    done = scored_outputs(paths, t, split, cols, prefix)
    if done and not force:
        raise SystemExit(f"[gate] REFUSED: this tuple was already scored on {split}: "
                         + ", ".join(str(p) for p in done)
                         + "\n  Re-scoring needs --force-rescore --rescore-reason '...' (logged in REGISTRY.md).")
    if done and force:
        if not (reason and reason.strip()):
            raise SystemExit("[gate] --force-rescore requires --rescore-reason")
        append_rescore(registry_md, t, reason.strip(), cols, rescore_section)
        print(f"[gate] RESCORE logged in {registry_md} › {rescore_section}: {reason.strip()!r}")
        return True
    return False


# ------------------------------------------------------------------ exposure registry adapter
# golden/registry.py is WP-D's; the contract API is mark_exposed(unit_ids, run_tag, kind,
# path=) and assert_unexposed(unit_ids, kinds=(...), path=). Everything here goes through
# these two thin wrappers so an API difference is a two-line fix. Tests swap `_REGISTRY` for
# a fake; `_REGISTRY_PATH` (from --registry-csv) is forwarded as path= when set.
_REGISTRY = None
_REGISTRY_PATH: Optional[Path] = None


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        try:
            from lensjudge.golden import registry as _r
            _REGISTRY = _r
        except ImportError:
            _REGISTRY = False
    return _REGISTRY or None


def _path_kw() -> dict:
    return {"path": _REGISTRY_PATH} if _REGISTRY_PATH is not None else {}


def mark_exposed(unit_ids, run_tag: str, kind: str, required: bool) -> None:
    r = _registry()
    units = [str(u) for u in unit_ids if isinstance(u, str) and u]
    if r is None:
        msg = f"[registry] golden/registry.py not importable — exposure of {len(units)} units ({kind}, {run_tag}) NOT recorded"
        if required:
            raise SystemExit(msg + " (required for validate runs)")
        print(msg)
        return
    try:
        r.mark_exposed(units, run_tag, kind, **_path_kw())
    except KeyError as e:   # units not yet synced into the ledger (e.g. a smoke run before labels)
        msg = f"[registry] {e} — exposure ({kind}, {run_tag}) NOT recorded; run `python -m lensjudge.golden.registry sync`"
        if required:
            raise SystemExit(msg + " (required for validate runs)") from e
        print(msg)
        return
    print(f"[registry] marked {len(units)} units exposed: kind={kind} run_tag={run_tag}")


def assert_unexposed(unit_ids, kinds=("fewshot", "sft")) -> None:
    r = _registry()
    units = [str(u) for u in unit_ids if isinstance(u, str) and u]
    if r is None:
        raise SystemExit("[registry] golden/registry.py not importable — cannot prove the validate "
                         "units were never exemplars/SFT rows; refusing")
    r.assert_unexposed(units, kinds=tuple(kinds), **_path_kw())
    print(f"[registry] {len(units)} validate units verified unexposed to {kinds}")


# ------------------------------------------------------------------ data
def load_manifest(path: Path, split: Optional[str]) -> pd.DataFrame:
    """run_batch.build_set's manifest convention (grade_truth -> grade, catalog=survey_key)."""
    df = _read(path).rename(columns={"grade_truth": "grade"})
    if "catalog" not in df.columns:
        df["catalog"] = df.get("survey_key", "jwst")
    if split is not None:
        if "split" not in df.columns:
            raise SystemExit(f"{path} has no 'split' column (build it with golden/build_eval_manifest.py)")
        df = df[df["split"] == split].copy()
    if df.empty:
        raise SystemExit(f"no rows for split={split!r} in {path}")
    return df.reset_index(drop=True)


def smoke_frame(frame_csv: Path, keys_dir: Path, kits_dir: Path, n: int,
                stratum: Optional[str] = None, splits_csv: Optional[Path] = None) -> pd.DataFrame:
    """N frame items that have a served kit JPEG — zero-shot plumbing rows, no labels.
    `stratum` restricts the pick (e.g. K_cowls: literature controls, the cheapest items to
    burn on a plumbing check because they carry no pipeline-only information). Once the
    splits exist, validate units are never smoke rows (they are scored only through the
    registry gate)."""
    from lensjudge.golden.build_eval_manifest import image_paths, load_keys
    fr = _read(frame_csv)
    fr["unit_id"] = fr["unit_id"].astype(str)
    if stratum:
        fr = fr[fr["stratum"].astype(str) == stratum]
    if splits_csv is not None and Path(splits_csv).exists():
        sp = _read(Path(splits_csv))
        val = set(sp.loc[sp["split"].astype(str) == "validate", "unit_id"].astype(str))
        before = len(fr)
        fr = fr[~fr["unit_id"].isin(val)]
        print(f"[smoke] {before - len(fr)} validate units excluded from the smoke pick")
    keys = load_keys(keys_dir)
    fr["image_path"] = image_paths(fr, keys, kits_dir).values
    fr = fr[fr["image_path"] != ""].head(n)
    if fr.empty:
        raise SystemExit("smoke: no frame items with a kit JPEG (build the kit first)")
    return pd.DataFrame({"name": fr["candidate_id"].astype(str), "unit_id": fr["unit_id"],
                         "image_path": fr["image_path"], "survey_key": "jwst", "catalog": "jwst",
                         "grade": None, "region": fr.get("proposal", ""), "split": "smoke"})


# ------------------------------------------------------------------ scoring loop
async def score(df: pd.DataFrame, out: Path, trace_dir: Path, *, model: Optional[str],
                system_prompt: str, fewshot_blocks: list, exemplar_unit_ids: list,
                concurrency: int, extra_cols: dict) -> Path:
    """run_batch.run's loop with the jwst grader called directly (so fewshot_blocks can be
    passed); rows through run_batch._row_dict, resumable through run_batch._flush."""
    done = set(pd.read_parquet(out)["name"].tolist()) if out.exists() else set()
    todo = [r.to_dict() for _, r in df.iterrows() if r["name"] not in done]
    print(f"[golden] {len(todo)} to grade ({len(done)} already done) -> {out}")
    trace_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    rows, lock = [], asyncio.Lock()

    async def one(cand):
        async with sem:
            res = await grader_jwst.grade_candidate(
                cand, model=model, system_prompt=system_prompt,
                trace_path=str(trace_dir / f"{_util.safe_name(cand['name'])}.jsonl"),
                fewshot_blocks=fewshot_blocks, exemplar_unit_ids=exemplar_unit_ids)
        async with lock:
            row = run_batch._row_dict(res, cand)
            row.update({"unit_id": cand.get("unit_id"), "split": cand.get("split"),
                        "render_sha": res.meta.get("render_sha"),
                        "n_exemplars_seen": res.meta.get("n_exemplars"),
                        "system_sha16": res.meta.get("system_sha16"), **extra_cols})
            rows.append(row)
            if len(rows) % 5 == 0 or len(rows) == len(todo):
                run_batch._flush(rows, out, done)

    await asyncio.gather(*(one(c) for c in todo))
    if rows:
        run_batch._flush(rows, out, done)
    return out


def write_meta(out: Path, meta: dict) -> Path:
    """Written only once a replicate is complete: its presence is what the gate reads."""
    mp = meta_path(out)
    mp.write_text(json.dumps(meta, indent=2, default=str))
    return mp


def check_system_prompt(sysp: str, banned: Path, split: Optional[str],
                        gated_splits: tuple = ("validate",)) -> None:
    """Run the full system prompt through the banned lexicon (validate ids, PI comments,
    validate grades). Refuse on a hit; refuse a run on a gated split (validate; holdout for
    run_truth_eval) with no lexicon at all."""
    banned = Path(banned)
    if not banned.exists():
        if split in gated_splits:
            raise SystemExit(f"[embargo] REFUSED: {banned} does not exist; build it first "
                             f"(golden/audit_traces.py --build-lexicon) — the system prompt of a "
                             f"{split} run must be lexicon-checked before any call")
        print(f"[embargo] note: {banned} absent; system prompt not lexicon-checked ({split or 'smoke'} only)")
        return
    lex = audit_traces.load_lexicon(banned)
    hit = audit_traces.banned_hit(sysp, lex)
    if hit:
        raise SystemExit(f"[embargo] REFUSED: the system prompt contains a banned string "
                         f"(entry {hit[0][:60]!r}, window {hit[1]!r}); fix the rubric/note first")
    print(f"[embargo] system prompt clean against {len(lex)} lexicon entries")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, default=None)
    ap.add_argument("--model", default="sonnet", help="sonnet | opus | a served open-model id")
    ap.add_argument("--rubric", default=str(grader_jwst.RUBRIC_V2_PATH))
    ap.add_argument("--no-jwst-note", action="store_true", help="do not append prompts/jwst_note.md")
    ap.add_argument("--split", choices=SPLITS, default=None)
    ap.add_argument("--n-exemplars", type=int, default=0, help="exemplars PER LETTER (e2 only)")
    ap.add_argument("--k", type=int, default=3, help="replicate samples -> r1..rk parquets")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--labels", default=str(LABELS_CSV))
    ap.add_argument("--splits", default=str(SPLITS_CSV))
    ap.add_argument("--frame", default=str(FRAME_CSV))
    ap.add_argument("--keys-dir", default=str(KEYS_DIR))
    ap.add_argument("--kits-dir", default=str(KITS_DIR))
    ap.add_argument("--registry-md", default=str(REGISTRY_MD))
    ap.add_argument("--registry-csv", default=None,
                    help="exposure ledger path (default: golden/golden_registry.csv via registry.py)")
    ap.add_argument("--banned", default=str(BANNED),
                    help="banned lexicon (audit_traces --build-lexicon); required for validate runs")
    ap.add_argument("--out", default=OUT_TEMPLATE, help="template with {arm} {model} {split} {k}")
    ap.add_argument("--force-rescore", action="store_true")
    ap.add_argument("--rescore-reason", default=None)
    ap.add_argument("--smoke", type=int, default=0, help="grade N frame items zero-shot (plumbing)")
    ap.add_argument("--smoke-stratum", default=None,
                    help="restrict --smoke rows to one frame stratum (e.g. K_cowls)")
    ap.add_argument("--print-tuple", action="store_true", help="print the REGISTRY.md row and exit")
    ap.add_argument("--thinking", choices=("off", "adaptive"), default=None)
    ap.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default=None)
    args = ap.parse_args(argv)

    global _REGISTRY_PATH
    _REGISTRY_PATH = Path(args.registry_csv) if args.registry_csv else None
    fewshot.assert_env_clean()
    if args.thinking:
        os.environ["LENSJUDGE_THINKING"] = args.thinking
    if args.effort:
        os.environ["LENSJUDGE_EFFORT"] = args.effort
    if args.model in CLAUDE_ALIASES:
        os.environ.setdefault("LENSJUDGE_BACKEND", "anthropic")
    from lensjudge.common import llm_client
    backend = llm_client.get_backend()
    if backend == "anthropic":
        print("[golden] Anthropic path: single call at the API DEFAULT temperature (none passed); "
              "k replicates are real samples")

    rubric_path = Path(args.rubric)
    rubric_text = rubric_path.read_text()
    sysp = rubric_text if args.no_jwst_note else grader_jwst.with_note(rubric_text)
    rubric_sha = _util.sha_text(rubric_text)
    system_sha = _util.sha_text(sysp)
    if not args.print_tuple:
        check_system_prompt(sysp, Path(args.banned), args.split)

    # ---- smoke: no split, no registry gate (validate units excluded once splits exist)
    if args.smoke:
        df = smoke_frame(Path(args.frame), Path(args.keys_dir), Path(args.kits_dir), args.smoke,
                         stratum=args.smoke_stratum, splits_csv=Path(args.splits))
        out = (Path(args.out.format(arm="smoke", model=_util.safe_name(args.model), split="frame", k=1))
               if "{" in args.out else Path(args.out))
        trace_dir = out.parent / "traces_golden_smoke"
        extra = {"arm": "smoke", "model": args.model, "rubric_sha16": rubric_sha, "n_exemplars": 0,
                 "k": 1, "run_tag": "smoke", "rescored": False, "rescore_reason": None}
        asyncio.run(score(df, out, trace_dir, model=args.model, system_prompt=sysp, fewshot_blocks=[],
                          exemplar_unit_ids=[], concurrency=args.concurrency, extra_cols=extra))
        write_meta(out, {"tuple": extra, "system_sha16": system_sha, "backend": backend,
                         "jwst_note": not args.no_jwst_note, "thinking": thinking_setting(),
                         "effort": effort_setting(), "n": len(df)})
        mark_exposed(df["unit_id"].tolist(), "smoke", "eval", required=False)
        run_batch.summarize(out)
        return

    if not (args.arm and args.split):
        ap.error("--arm and --split are required (or --smoke N)")
    if args.arm == "e2" and args.n_exemplars <= 0:
        ap.error("--arm e2 needs --n-exemplars > 0")
    if args.arm != "e2" and args.n_exemplars:
        ap.error(f"--arm {args.arm} is zero-shot; drop --n-exemplars")

    t = RunTuple(args.arm, args.model, rubric_sha, int(args.n_exemplars), splits_sha(Path(args.splits)),
                 system_sha, thinking_setting(), effort_setting())
    if args.print_tuple:
        print(t.row(rubric_name=rubric_path.name, k=args.k))
        return
    paths = out_paths(args.out, args.arm, args.split, args.k, args.model)
    rescored = False
    if args.split == "validate":
        rescored = check_validate_gate(Path(args.registry_md), t, paths, args.force_rescore, args.rescore_reason)

    df = load_manifest(Path(args.manifest), args.split)
    splits = _read(Path(args.splits))
    splits["unit_id"] = splits["unit_id"].astype(str)
    align_units = splits.loc[splits["split"] == "align", "unit_id"].tolist()
    validate_units = splits.loc[splits["split"] == "validate", "unit_id"].tolist()

    # ---- validate units must never have been exemplars or SFT rows — for EVERY arm and
    # model (an SFT student is scored through --arm e1 --model <served id>; a corpus built
    # from an older splits.csv could have trained on units that are validate now)
    if args.split == "validate":
        assert_unexposed(validate_units, kinds=("fewshot", "sft"))

    # ---- E2 exemplars: from the ALIGN half only
    blocks, ex_ids = [], []
    if args.arm == "e2":
        labels = _read(Path(args.labels))
        from lensjudge.golden.build_eval_manifest import load_keys
        keys = load_keys(Path(args.keys_dir))
        blocks, ex_ids = fewshot.build_exemplar_blocks(
            labels, keys, n_per_grade=args.n_exemplars, seed=_util.SEED, embargo=True,
            eligible_units=align_units, kits_dir=Path(args.kits_dir))
        if not ex_ids:
            raise SystemExit("e2: no eligible exemplars in the align half (need confidence H, stable, colour)")
        print(f"[golden] {len(ex_ids)} exemplars from align: {ex_ids}")
        if args.split == "align":   # leave-exemplars-out within the tuning half
            before = len(df)
            df = df[~df["unit_id"].astype(str).isin(set(ex_ids))].reset_index(drop=True)
            print(f"[golden] align scoring set {before} -> {len(df)} after dropping exemplar units")

    for k, out in enumerate(paths, start=1):
        tag = t.run_tag(args.split, k)
        trace_dir = out.parent / f"traces_{tag}"   # beside the parquet (outputs/ by default)
        extra = {**asdict(t), "k": k, "run_tag": tag, "rescored": rescored,
                 "rescore_reason": (args.rescore_reason if rescored else None)}
        if args.arm == "e2":
            mark_exposed(ex_ids, tag, "fewshot", required=(args.split == "validate"))
        asyncio.run(score(df, out, trace_dir, model=args.model, system_prompt=sysp, fewshot_blocks=blocks,
                          exemplar_unit_ids=ex_ids, concurrency=args.concurrency, extra_cols=extra))
        write_meta(out, {"tuple": asdict(t), "split": args.split, "k": k, "run_tag": tag,
                         "rescored": rescored, "rescore_reason": extra["rescore_reason"],
                         "manifest": str(args.manifest), "manifest_sha16": _util.sha_text(Path(args.manifest).read_text()),
                         "rubric": str(rubric_path), "jwst_note": not args.no_jwst_note,
                         "system_sha16": system_sha, "jwst_note_sha16": _util.sha_text(grader_jwst.JWST_NOTE),
                         "backend": backend, "model": args.model,
                         "thinking": t.thinking, "effort": t.effort, "banned_lexicon": str(args.banned),
                         "exemplar_unit_ids": ex_ids, "n_exemplars_total": len(ex_ids),
                         "n": len(df), "trace_dir": str(trace_dir),
                         "scored_at": _dt.datetime.now(_dt.timezone.utc).isoformat()})
        mark_exposed(df["unit_id"].tolist(), tag, "eval", required=(args.split == "validate"))
        run_batch.summarize(out)


if __name__ == "__main__":
    main()
