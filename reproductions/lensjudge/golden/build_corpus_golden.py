#!/usr/bin/env python3
"""golden/build_corpus_golden.py — SFT corpus from the golden (single-grader, JWST) labels.

Mirrors finetune/build_corpus_desi.py for the golden campaign: the ONLY label source is
golden/golden_labels.csv (Xiaosheng's pass-1 score 1-4 + L/M/H confidence, collected by
golden/collect.py), and the ONLY image is the served kit JPEG — the byte-identical 6-panel v1
composite he graded (golden/kits/<kit_id>/items/NNN.jpg, sha-checked against the key). ONLY
the `align` half of golden/splits.csv enters this directory, in three disjoint parts: a
20% letter-stratified carve `valsel` (the swift-infer evalset WITHOUT assistant turns, for the
pre-registered init-selection rule — "start from whichever base scores best on the align
half" — and for a collapse check; NEVER for checkpoint selection, which the plan forbids at
this n: fixed epochs), a 3% loss-monitor carve `val`, and `train`. The `validate` half is
written NOWHERE here: a student is scored on it exactly once, through the registry gate of
golden/run_golden_eval.py (--arm e1 --model <served id>), never through `gate`. A 2" firewall
between the halves is asserted before anything is written.

Targets (per-example UNIQUE, the v5 mode-collapse lesson — distill_hsc._v5_target): every
target passes through build_sft_data._target_json with
  p_lens     = 0.5 + (SOFT[g] - 0.5) * (0.6 + 0.4 c)  +- 0.02 hash jitter, clipped, 2 dp
               (v5 SOFT from parity/build_train_splits — NOT build_sft_data.P_BY_GRADE —
               shrunk toward 0.5 by his confidence c = CONF_TO_C[L/M/H]; a less-sure D
               moves toward 0.5, never toward 0.01)
  confidence = CONF_TO_01[L/M/H] +- 0.02 hash jitter
  criteria   = the _v5_target hash channels with soft = SOFT[g]   (criteria_source =
               "synthetic_hash" in corpus_manifest.csv — he scored no criteria)
  rationale  = one of 3 hash-selected TRUTHFUL fact stems: coords + JWST filters + his
               score and confidence. No "committee"/"consensus" (one grader), no theta_E
               (none recorded), contaminant=None. Pipeline columns (pipe_*) never enter.

NO dihedral augmentation here (unlike build_corpus_desi / augment_sft_v5): the v1 composite
carries a compass, corner quadrant labels, scale bars and a fixed panel order, so a flipped or
rotated JPEG is a construction signature (the only images with a mirrored compass would be the
augmented copies) and the student would learn to read the annotation, not the sky.
Augmentation waits for the per-panel v2 renders (Campaign 2), which can be transformed in
array space before annotation.

Outputs (finetune/corpus_golden/, gitignored):
  sft_golden_train.jsonl / sft_golden_val.jsonl   ms-swift rows (val = 3% loss-monitor carve)
  valsel_golden.jsonl + valsel_golden_labels.csv  align-carve evalset (no assistant) + labels
  corpus_manifest.csv (+ .sha)                    provenance: every ALIGN row, split, target, sha

`gate` scores a parquet against valsel_golden_labels.csv (AUC of p_lens / p_lens_logprob /
s_exp for score>=3 vs rest; s_exp uses llm_client.ORDINAL_W, the same weights run_batch
writes) and refuses a labels file that names a validate unit.

  python lensjudge/golden/build_corpus_golden.py build --labels golden/golden_labels.csv \\
      --splits golden/splits.csv --frame golden/frame.csv --key golden/keys/<kit>_key.csv
  python lensjudge/golden/build_corpus_golden.py gate --preds outputs/preds_golden_x.parquet \\
      --labels finetune/corpus_golden/valsel_golden_labels.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.golden import _util  # noqa: E402
from lensjudge.finetune.build_sft_data import _target_json  # noqa: E402

OUT = config.HERE / "finetune" / "corpus_golden"
KITS_DIR = _util.HERE / "kits"
VAL_FRAC = 0.03            # loss-monitor carve, matches build_corpus_desi / distill_hsc sft-v5
VALSEL_FRAC = 0.20         # evalset carve of the ALIGN half (per letter, hash-ordered); never validate
SURVEY_KEY = "jwst"
LABEL_SOURCE = "golden_huang"
CRITERIA_SOURCE = "synthetic_hash"
FIREWALL_ARCSEC = 2.0
# v5 soft targets (parity/build_train_splits.SOFT) — the program convention; P_BY_GRADE is NOT used
SOFT = {"A": 0.95, "B": 0.80, "C": 0.45, "D": 0.05}
_LABEL = {"A": "lens", "B": "lens", "C": "softmid", "D": "nonlens"}   # build_corpus_desi._LABEL
CRIT_KEYS = ("blue_source", "low_surface_brightness", "curvature", "counter_images",
             "arc_morphology")
# words a single-grader rationale must never contain (tests grep for these)
BANNED_WORDS = ("committee", "consensus", "thetaE", "theta_E", "θ")

RUBRIC_V2 = (config.HERE / "prompts" / "rubric_imaging_v2.md").read_text()
JWST_NOTE_PATH = config.HERE / "prompts" / "jwst_note.md"
_JWST_NOTE_PLACEHOLDER = """

# IMPORTANT — JWST NIRCam composite (placeholder: prompts/jwst_note.md was not found at build time)
- One 6-panel composite is attached (10" field normal/deep/colour; 3.5" zoom deep/colour/residual).
- Apply the SAME A/B/C/D scale + v2 rubric. There are NO CNN scores.
"""
# the golden analogue of build_corpus_desi.USER_MSG: one image tag + the item-agnostic panel
# gloss the Claude grader sees (golden/grader_jwst.PANEL_GLOSS, so student and teacher read the
# same user text); a local copy stands in if grader_jwst is not importable.
_PANEL_GLOSS_FALLBACK = (
    "Grade this JWST NIRCam strong-lens candidate from the single composite below. "
    "Six panels, two rows, north up and east left. Row 1 (10\" field): normal stretch, "
    "deep stretch, two-band colour (red = long-wavelength channel, blue = short-wavelength "
    "channel). Row 2 (3.5\" zoom on the catalogued galaxy): deep stretch, two-band colour, "
    "deflector-subtracted residual. A 1\" scale bar sits in each row; four yellow ticks "
    "point at the catalogued galaxy (the putative deflector). No tools, scores or "
    "photometry are available; the composite is the complete evidence set.\n\n"
    "Respond with ONLY the JSON object for the required schema.")
try:
    from lensjudge.golden.grader_jwst import PANEL_GLOSS as _PANEL_GLOSS  # noqa: E402
except Exception:   # noqa: BLE001  (grader_jwst needs prompts/jwst_note.md at import time)
    _PANEL_GLOSS = _PANEL_GLOSS_FALLBACK
USER_MSG_JWST = "<image>\n" + _PANEL_GLOSS


def system_prompt(note_path: Path = JWST_NOTE_PATH) -> str:
    """rubric_imaging_v2 + prompts/jwst_note.md (WP-E writes the note; a placeholder + warning
    stands in until it lands so the corpus can be smoke-built without it)."""
    note_path = Path(note_path)
    if note_path.exists():
        note = note_path.read_text()
        if not note.startswith("\n"):
            note = "\n\n" + note
    else:
        warnings.warn(f"{note_path} not found — using a placeholder JWST note in the system prompt")
        note = _JWST_NOTE_PLACEHOLDER
    return RUBRIC_V2 + note


# ------------------------------------------------------------------ targets
def golden_p(grade: str, lmh: str, name: str, ndigits: int | None = 2) -> float:
    """Soft p_lens: v5 SOFT shrunk toward 0.5 by the grader's confidence, de-constanted by a
    +-0.02 deterministic jitter (same hash channel as distill_hsc: _hash01(name, "p")).
    ndigits=None skips the 2-dp rounding (tests check the raw ordering; C's confidence spread
    is only 0.013 end to end, below 2-dp resolution)."""
    c = _util.CONF_TO_C[lmh]
    p = 0.5 + (SOFT[grade] - 0.5) * (0.6 + 0.4 * c)
    p += (_util.hash01(name, "p") - 0.5) * 0.04
    p = min(0.99, max(0.01, p))
    return p if ndigits is None else round(p, ndigits)


def golden_confidence(lmh: str, name: str) -> float:
    """ImageGrade.confidence: the L/M/H mapping +- 0.02 jitter (hash channel "cf")."""
    conf = _util.CONF_TO_01[lmh] + (_util.hash01(name, "cf") - 0.5) * 0.04
    return round(min(0.99, max(0.05, conf)), 2)


def golden_criteria(grade: str, name: str) -> dict:
    """The _v5_target criteria channels (c{i}, o{i}) with soft = SOFT[grade]. Synthetic — the
    grader scored no criteria — hence criteria_source="synthetic_hash" in the manifest."""
    soft = SOFT[grade]
    crit = {}
    for i, k in enumerate(CRIT_KEYS):
        base = 10.0 * soft * (0.75 + 0.5 * _util.hash01(name, f"c{i}"))
        crit[k] = int(min(10, max(0, round(base + (_util.hash01(name, f"o{i}") - 0.5) * 2))))
    return crit


def golden_rationale(name: str, ra: float, dec: float, sw_filter, lw_filter,
                     score: int, lmh: str) -> str:
    """Short, TRUTHFUL, per-example fact stem (3 hash-selected phrasings of the same facts)."""
    bands = "/".join(str(f) for f in (sw_filter, lw_filter) if isinstance(f, str) and f.strip())
    bands = bands or "NIRCam"
    coord = f"({float(ra):.4f},{float(dec):.4f})"
    letter = _util.score_to_letter(score)
    stems = [f"{coord} JWST {bands}: expert score {score}/4, confidence {lmh}.",
             f"JWST {bands} composite at {coord}; expert score {score}/4 ({letter}), confidence {lmh}.",
             f"Expert score {score}/4 ({letter}), confidence {lmh}, for the JWST {bands} field at {coord}."]
    out = stems[int(_util.hash01(name, "r") * len(stems))]
    assert not any(w.lower() in out.lower() for w in BANNED_WORDS), out
    return out


def golden_target(r) -> str:
    """Per-example UNIQUE ImageGrade JSON for one labelled unit (row of load_rows())."""
    name, grade, lmh = str(r.name), str(r.grade), str(r.confidence_lmh)
    return _target_json(grade, contaminant=None,
                        rationale=golden_rationale(name, r.ra, r.dec, r.sw_filter, r.lw_filter,
                                                   int(r.score_1_4), lmh),
                        crit=golden_criteria(grade, name),
                        p_lens=golden_p(grade, lmh, name),
                        confidence=golden_confidence(lmh, name))


# ------------------------------------------------------------------- inputs
def _read(path: Path, **kw) -> pd.DataFrame:
    """Read a golden CSV, verifying its .sha pin when one exists (tracked artifacts are pinned;
    a hand-made test fixture is not)."""
    path = Path(path)
    if path.with_suffix(path.suffix + ".sha").exists():
        return _util.read_pinned(path, **kw)
    return pd.read_csv(path, **kw)


def load_rows(labels_csv: Path, splits_csv: Path, frame_csv: Path, key_csvs: list[Path],
              kits_dir: Path = KITS_DIR) -> pd.DataFrame:
    """Join labels x splits x frame x key -> one row per labelled unit with the served JPEG.

    Columns: name (candidate_id), unit_id, ra, dec, stratum, score_1_4, grade, confidence_lmh,
    sw_filter, lw_filter, layout, split (align|validate), kit_id, item_id, image, render_sha.
    Hard checks: strict 1-4 scores, letters consistent with scores, one pass-1 key row per unit,
    JPEG present and byte-faithful (sha16 == key.render_sha == labels.render_sha)."""
    lab = _read(labels_csv, dtype={"confidence_lmh": str, "grade_letter": str})
    spl = _read(splits_csv)
    frm = _read(frame_csv, dtype={"sw_filter": str, "lw_filter": str})
    key = pd.concat([_read(k, dtype={"item_id": str, "kit_id": str}) for k in key_csvs],
                    ignore_index=True)

    lab = lab.dropna(subset=["score_1_4"]).copy()
    lab["score_1_4"] = lab.score_1_4.astype(int)
    lab["confidence_lmh"] = lab.confidence_lmh.astype(str).str.strip().str.upper()
    bad = ~lab.confidence_lmh.isin(_util.CONF_LEVELS)
    assert not bad.any(), f"confidence_lmh outside L/M/H: {lab.loc[bad, 'unit_id'].tolist()[:5]}"
    lab["grade"] = lab.score_1_4.map(_util.score_to_letter)        # strict, never ImageGrade
    if "grade_letter" in lab.columns:
        mism = lab.grade_letter.astype(str).str.upper() != lab.grade
        assert not mism.any(), f"grade_letter != score_to_letter: {lab.loc[mism, 'unit_id'].tolist()[:5]}"

    spl = spl[["unit_id", "split"]]
    assert spl.split.isin(["align", "validate"]).all(), "split must be align|validate"
    assert not spl.unit_id.duplicated().any(), "duplicate unit_id in splits"
    df = lab.merge(spl, on="unit_id", how="inner", validate="one_to_one")
    if len(df) < len(lab):
        print(f"[inputs] {len(lab) - len(df)} labelled units have no split row -> dropped")

    fcols = ["unit_id", "sw_filter", "lw_filter", "layout"]
    if "ra_deg" not in df.columns:
        fcols += ["ra_deg", "dec_deg"]
    df = df.merge(frm[[c for c in fcols if c in frm.columns]], on="unit_id", how="left",
                  validate="one_to_one")
    for c in ("sw_filter", "lw_filter", "layout"):
        if c not in df.columns:
            df[c] = ""
    df["sw_filter"] = df.sw_filter.fillna("")
    df["lw_filter"] = df.lw_filter.fillna("")

    k1 = (key[key["pass"].astype(int) == 1][["unit_id", "kit_id", "item_id", "render_sha"]]
          .rename(columns={"render_sha": "render_sha_key"}))
    assert not k1.unit_id.duplicated().any(), "more than one pass-1 key row for a unit"
    if "render_sha" not in df.columns:
        df["render_sha"] = ""
    df = df.merge(k1, on="unit_id", how="inner", validate="one_to_one")
    df["item_id"] = df.item_id.astype(str).str.zfill(3)
    df["image"] = [str((Path(kits_dir) / str(r.kit_id) / "items" / f"{r.item_id}.jpg").resolve())
                   for r in df.itertuples()]   # absolute: ms-swift resolves nothing for us
    for r in df.itertuples():
        p = Path(r.image)
        assert p.exists(), f"served kit JPEG missing: {p}"
        sha = _util.sha_file(p)
        assert sha == str(r.render_sha_key), f"{p}: sha {sha} != key render_sha {r.render_sha_key}"
        if isinstance(r.render_sha, str) and r.render_sha:
            assert sha == r.render_sha, f"{p}: sha {sha} != labels render_sha {r.render_sha}"
    df["render_sha"] = df.render_sha_key
    df = df.rename(columns={"candidate_id": "name", "ra_deg": "ra", "dec_deg": "dec"})
    keep = ["name", "unit_id", "ra", "dec", "stratum", "score_1_4", "grade", "confidence_lmh",
            "sw_filter", "lw_filter", "layout", "split", "kit_id", "item_id", "image",
            "render_sha"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    assert not df.name.duplicated().any(), "duplicate candidate_id across units"
    return df


def firewall(df: pd.DataFrame, radius_arcsec: float = FIREWALL_ARCSEC) -> None:
    """Assert no align (train) row lies within `radius_arcsec` of any validate row, and no
    name is in both halves — the split_halves guarantee, re-checked at the consumer."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    tr, va = df[df.split == "align"], df[df.split == "validate"]
    both = set(tr.name.astype(str)) & set(va.name.astype(str))
    assert not both, f"NAME leakage across halves: {sorted(both)[:5]}"
    if len(tr) and len(va):
        _, sep, _ = SkyCoord(tr.ra.values * u.deg, tr.dec.values * u.deg).match_to_catalog_sky(
            SkyCoord(va.ra.values * u.deg, va.dec.values * u.deg))
        n_bad = int((sep.arcsec < radius_arcsec).sum())
        assert n_bad == 0, (f"POSITION leakage: {n_bad} align rows within {radius_arcsec}\" of a "
                            f"validate row: {tr.name[sep.arcsec < radius_arcsec].tolist()[:5]}")
    print(f"[firewall] {len(tr)} align vs {len(va)} validate rows: 0 name hits, "
          f"0 within {radius_arcsec}\" -> PASS")


# -------------------------------------------------------------------- build
def valsel_names(df: pd.DataFrame, frac: float = VALSEL_FRAC) -> set[str]:
    """The align-half evalset carve: per letter, the ceil(frac * n) rows with the lowest
    hash01(name, "valsel") — deterministic, no RNG state, every letter represented."""
    out: set[str] = set()
    al = df[df.split == "align"]
    for letter, g in al.groupby("grade"):
        names = sorted(g.name.astype(str), key=lambda n: (_util.hash01(n, "valsel"), n))
        k = int(np.ceil(frac * len(names))) if frac > 0 else 0
        out |= set(names[:k])
    return out


def build_records(df: pd.DataFrame, sys_prompt: str, valsel_frac: float = VALSEL_FRAC):
    """-> (train_recs, valsel_recs, valsel_labels). Both come from the ALIGN half: train rows
    carry assistant targets; valsel rows (the per-letter carve) carry none, plus their
    idx-aligned labels rows. Validate rows are skipped entirely."""
    train, valsel, vs_labels = [], [], []
    vs_names = valsel_names(df, valsel_frac)
    for r in df.itertuples():
        if r.split != "align":
            continue
        label = _LABEL[str(r.grade)]
        if str(r.name) in vs_names:
            valsel.append({"messages": [{"role": "system", "content": sys_prompt},
                                        {"role": "user", "content": USER_MSG_JWST}],
                           "images": [str(r.image)]})
            vs_labels.append({"idx": len(vs_labels), "label": label, "name": str(r.name),
                              "grade": str(r.grade), "label_source": LABEL_SOURCE,
                              "split": "align_valsel"})
        else:
            train.append({"messages": [{"role": "system", "content": sys_prompt},
                                       {"role": "user", "content": USER_MSG_JWST},
                                       {"role": "assistant", "content": golden_target(r)}],
                          "images": [str(r.image)], "label": label, "name": str(r.name)})
    assert not ({x["name"] for x in train} & {x["name"] for x in vs_labels}), "train/valsel overlap"
    # per-example-unique-target self-check (v5 standing rule: verify before every training run)
    targets = [x["messages"][2]["content"] for x in train]
    assert len(set(targets)) == len(targets), "target collision — per-example-unique rule violated"
    print(f"[targets] {len(targets)} train targets, all unique -> PASS")
    return train, valsel, vs_labels


def write_corpus(df: pd.DataFrame, train: list, valsel: list, vs_labels: list, out: Path,
                 val_frac: float = VAL_FRAC) -> pd.DataFrame:
    """Shuffle + carve the loss-monitor val split, write the jsonls/labels, pin the manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_util.SEED)
    train = list(train)
    rng.shuffle(train)
    n_val = int(len(train) * val_frac)
    splits = {"sft_golden_val": train[:n_val], "sft_golden_train": train[n_val:]}
    for stem, data in splits.items():
        with open(out / f"{stem}.jsonl", "w") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        counts = pd.Series([x["label"] for x in data]).value_counts().to_dict() if data else {}
        print(f"[sft] {stem}: {len(data)} ({counts})")
    with open(out / "valsel_golden.jsonl", "w") as fh:
        for rec in valsel:
            fh.write(json.dumps(rec) + "\n")
    pd.DataFrame(vs_labels, columns=["idx", "label", "name", "grade", "label_source", "split"]).to_csv(
        out / "valsel_golden_labels.csv", index=False)
    counts = pd.Series([x["label"] for x in vs_labels]).value_counts().to_dict() if vs_labels else {}
    print(f"[valsel] {len(valsel)} evalset rows from the ALIGN half ({counts}) + labels")

    corpus_split = {rec["name"]: "val" for rec in splits["sft_golden_val"]}
    corpus_split.update({rec["name"]: "train" for rec in splits["sft_golden_train"]})
    corpus_split.update({rec["name"]: "valsel" for rec in vs_labels})
    al = df[df.split == "align"].reset_index(drop=True)        # validate rows are written nowhere
    assert set(al.name.astype(str)) == set(corpus_split), "every align row must have a corpus split"
    man = pd.DataFrame({
        "name": al.name.astype(str), "unit_id": al.unit_id, "ra": al.ra, "dec": al.dec,
        "survey_key": SURVEY_KEY, "grade": al.grade, "score_1_4": al.score_1_4.astype(int),
        "confidence_lmh": al.confidence_lmh, "label_source": LABEL_SOURCE,
        "soft_target": al.grade.map(SOFT),
        "split": [corpus_split[str(n)] for n in al.name],
        "label": al.grade.map(_LABEL), "render_sha": al.render_sha,
        "criteria_source": CRITERIA_SOURCE,
        # extras after the contract columns: the p_lens actually written, and the served JPEG
        "p_lens_target": [golden_p(str(r.grade), str(r.confidence_lmh), str(r.name))
                          for r in al.itertuples()],
        "image_path": al.image,
    })
    sha = _util.pin(man, out / "corpus_manifest.csv")
    print(f"[corpus] manifest {len(man)} rows (sha {sha}) + jsonls -> {out}")
    return man


def _mark_registry(unit_ids, run_tag: str = "sft_golden") -> None:
    """Record SFT exposure in golden_registry.csv (golden/registry.py). Runs BEFORE the corpus
    is written so a registry failure never leaves an unrecorded corpus behind; unknown units
    mean the registry was not synced from labels/splits (or this is a synthetic smoke run ->
    --no-registry)."""
    from lensjudge.golden import registry
    try:
        registry.mark_exposed(list(unit_ids), run_tag, "sft")
    except KeyError as e:
        raise SystemExit(f"[registry] {e}\n  -> run registry.sync_from(labels, splits) first, "
                         f"or pass --no-registry for a synthetic/smoke build") from e
    print(f"[registry] marked {len(unit_ids)} align units exposed as sft ({run_tag})")


def stage_build(args):
    df = load_rows(Path(args.labels), Path(args.splits), Path(args.frame),
                   [Path(k) for k in args.key], Path(args.kits_dir))
    print(f"[inputs] {len(df)} labelled units with served JPEGs: "
          f"{df.groupby(['split', 'grade']).size().to_dict()}")
    firewall(df)
    train, valsel, vs_labels = build_records(df, system_prompt(Path(args.jwst_note)), args.valsel_frac)
    if not args.no_registry:
        _mark_registry(df.loc[df.split == "align", "unit_id"].tolist(), args.run_tag)
    write_corpus(df, train, valsel, vs_labels, Path(args.out), args.val_frac)
    if train:
        s = json.dumps(train[0])
        print(f"[sample] {s[:160]} ... {s[-300:]}")


# --------------------------------------------------------------------- gate
def assert_not_validate(labels: pd.DataFrame, golden_labels: Path | None, splits: Path | None) -> None:
    """A gate labels file must never name a validate unit: refuse on its own `split` column
    and, when golden_labels.csv + splits.csv exist, on the name -> unit -> split join."""
    if "split" in labels.columns and labels["split"].astype(str).str.contains("validate").any():
        raise SystemExit("[gate] REFUSED: labels carry validate rows; the validate half is scored "
                         "only through run_golden_eval.py --split validate (registry-gated)")
    if golden_labels and splits and Path(golden_labels).exists() and Path(splits).exists():
        gl = _read(Path(golden_labels))[["unit_id", "candidate_id"]]
        sp = _read(Path(splits))[["unit_id", "split"]]
        j = gl.merge(sp, on="unit_id")
        m = dict(zip(j["candidate_id"].astype(str), j["split"].astype(str)))
        bad = [n for n in labels["name"].astype(str) if m.get(n) == "validate"]
        if bad:
            raise SystemExit(f"[gate] REFUSED: {len(bad)} label row(s) are validate units "
                             f"({bad[:3]}...); the validate half is scored only through "
                             f"run_golden_eval.py --split validate")


def gate_aucs(preds: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """AUC of each available scorer (p_lens, p_lens_logprob, s_exp) for lens (score>=3, i.e.
    A/B) vs the rest (C+D). Joins on name when the parquet has one, else by idx/row order.
    A missing s_exp is recomputed from gp_* with llm_client.logprob_ordinal — the ONE
    definition of the expected ordinal (ORDINAL_W: A=1, B=2/3, C=1/3, D=0)."""
    from sklearn.metrics import roc_auc_score
    from lensjudge.common.llm_client import logprob_ordinal
    if "name" in preds.columns:
        j = labels.merge(preds, on="name", how="inner")
    else:
        assert len(preds) == len(labels), f"preds rows {len(preds)} != labels {len(labels)}"
        j = pd.concat([labels.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    gp_cols = [f"gp_{L}" for L in "ABCD"]
    if "s_exp" not in j.columns and all(c in j.columns for c in gp_cols):
        j["s_exp"] = [logprob_ordinal({L: (None if pd.isna(v) else float(v))
                                       for L, v in zip("ABCD", row)})
                      for row in j[gp_cols].to_numpy()]
    y = (j.label == "lens").astype(int).values
    out = {"n": int(len(j)), "n_lens": int(y.sum())}
    for col in ("p_lens", "p_lens_logprob", "s_exp"):
        if col not in j.columns:
            continue
        s = pd.to_numeric(j[col], errors="coerce").values
        ok = ~np.isnan(s)
        if ok.sum() < 2 or len(set(y[ok])) < 2:
            out[col] = float("nan")
            continue
        out[col] = float(roc_auc_score(y[ok], s[ok]))
    return out


def stage_gate(args):
    labels = pd.read_csv(args.labels)
    assert_not_validate(labels, Path(args.golden_labels), Path(args.splits))
    preds = pd.read_parquet(args.preds)
    res = gate_aucs(preds, labels)
    print(f"=== golden gate: {args.preds} vs {args.labels} ===")
    print(f"  n={res['n']} ({res['n_lens']} lens = score>=3 / {res['n'] - res['n_lens']} rest)")
    for col in ("p_lens", "p_lens_logprob", "s_exp"):
        if col in res:
            print(f"  AUC[{col}] = {res[col]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    b.add_argument("--labels", default=str(_util.HERE / "golden_labels.csv"))
    b.add_argument("--splits", default=str(_util.HERE / "splits.csv"))
    b.add_argument("--frame", default=str(_util.HERE / "frame.csv"))
    b.add_argument("--key", nargs="+", required=True, help="golden/keys/<kit>_key.csv (1+)")
    b.add_argument("--kits-dir", default=str(KITS_DIR))
    b.add_argument("--jwst-note", default=str(JWST_NOTE_PATH))
    b.add_argument("--out", default=str(OUT))
    b.add_argument("--val-frac", type=float, default=VAL_FRAC)
    b.add_argument("--valsel-frac", type=float, default=VALSEL_FRAC,
                   help="per-letter evalset carve of the ALIGN half (0 = no valsel)")
    b.add_argument("--no-registry", action="store_true",
                   help="do not record SFT exposure in golden_registry.csv (smoke runs only)")
    b.add_argument("--run-tag", default="sft_golden", help="exposure tag written to the registry")
    g = sub.add_parser("gate")
    g.add_argument("--preds", required=True, help="parquet with name or idx-order rows")
    g.add_argument("--labels", default=str(OUT / "valsel_golden_labels.csv"))
    g.add_argument("--golden-labels", default=str(_util.HERE / "golden_labels.csv"))
    g.add_argument("--splits", default=str(_util.HERE / "splits.csv"))
    args = ap.parse_args()
    {"build": stage_build, "gate": stage_gate}[args.stage](args)


if __name__ == "__main__":
    main()
