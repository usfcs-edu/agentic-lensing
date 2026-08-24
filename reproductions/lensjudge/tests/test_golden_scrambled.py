#!/usr/bin/env python3
"""No-network tests for golden/regrade_scrambled.py (the blind regrade of the scrambled
top-100 and its zero-API `--reletter`, REGISTRY.md "Deployment rule v2-deploy" item 8).

Covers, on synthetic keys / images / records only (no API, no real key file, no real run):
  1. the blind-cand builder emits EXACTLY {name, image_path, layout} — no candidate id,
     rank, coordinate, filter or any other key column can reach a model-facing row;
  2. filename -> layout from blank sw_filter / lw_filter alone (color; lw blank ->
     gray_sw_only; sw blank -> gray_lw_only, the build_frame.derive_layout naming; both
     blank refuses), NaN treated as blank;
  3. the de-scramble join (filename -> key -> candidate_id), the comparison-CSV shape (the
     original 17 columns, then the deploy columns), the p_evidence-descending order (ties by
     S, NaN last) and the agree_letter / agree_final logic, on canned rows;
  4. the frozen tuple per model (sonnet: off/default, thresholds gated; opus5: the same
     prompt shas, adaptive/xhigh, thresholds recorded not gated, provisional letters
     allowed), `apply_model_env` and `check_frozen` against the on-disk stack + refusals;
  5. `--dry-run` for opus5 on a synthetic 3-image kit in tmp_path: the resolved tuple is
     printed, budget refusal, nothing written, no call;
  6. `--reletter` on a synthetic run directory built from synthetic votes raws (advocate +
     one upheld critic + arbitrator, an abstaining stack, an advocate-only item and a
     parse-failure item): letters change with the thresholds file, S / S_arb / p_evidence
     do not, NaN stays NaN, the pre_reletter copy exists, the deploy columns are present,
     the CSV is ordered by p_evidence, R2 on top of R1, and the refusals (a tampered S, a
     wrong --model, a wrong key).

Runs under pytest:
    cd reproductions/lensjudge && ~/.venvs/lensjudge/bin/python -m pytest tests/test_golden_scrambled.py -q
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from lensjudge.golden import _util, aggregate_v2  # noqa: E402
from lensjudge.golden import records as R  # noqa: E402
from lensjudge.golden import regrade_scrambled as rs  # noqa: E402
from lensjudge.golden import run_truth_eval as rte  # noqa: E402
from lensjudge.golden import schemas_panel as sp  # noqa: E402

LEGACY_COLS = (
    "scrambled_item", "rank", "candidate_id", "nate_grade", "nate_n_pass",
    "nate_inspector_conf", "blind_theta_E_arcsec", "discovery_status", "our_S", "our_S_arb",
    "our_letter", "our_letter_llm", "our_p_evidence", "our_scale_class",
    "our_alternative_final", "our_needs_human", "agree_letter")
DEPLOY_CSV_COLS = ("our_letter_rank", "our_letter_final", "our_veto", "our_rule", "agree_final", "our_rationale")
SONNET_THR = {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318, "letter_source": "sonnet_api_calibrated",
              "thresholds_key": "sonnet_api"}
PROV = {"tau0": 0.15, "t_A": 0.8, "t_B": 0.5}


def _synthetic_key(n: int = 3) -> pd.DataFrame:
    """Scrambled rows with deliberately leak-shaped extra columns (synthetic ids)."""
    df = pd.DataFrame({
        "index": ["1", "2", "3", "4"],
        "filename": ["001.jpg", "002.jpg", "003.jpg", "004.jpg"],
        "rank": ["42", "3", "77", "9"],
        "candidate_id": ["JTEST0001+0000001", "JTEST0002+0000002", "JTEST0003+0000003", "JTEST0004+0000004"],
        "ra_deg": ["150.1", "150.2", "150.3", "150.4"],
        "dec_deg": ["2.1", "2.2", "2.3", "2.4"],
        "verifier_grade": ["U", "A", "C", "B"],
        "verifiers_pass": ["0", "3", "1", "2"],
        "inspector_confidence": ["26.0", "78.0", "", "55.0"],
        "discovery_status": ["new", "new", "known", "new"],
        "blind_theta_E_arcsec": ["1.4", "1.1", "", "0.9"],
        "sw_filter": ["F150W", "F150W2", np.nan, "F150W"],
        "lw_filter": ["F277W", "", "F444W", "F277W"],
        "evidence": ["secret free text", "more secret text", "even more", "still more"],
    })
    return df.iloc[:n].reset_index(drop=True)


def _make_kit(root: Path, key: pd.DataFrame) -> Path:
    from PIL import Image
    d = root / "kit"
    d.mkdir(parents=True, exist_ok=True)
    for fn in key["filename"]:
        Image.new("RGB", rs.IMG_SIZE, (10, 10, 12)).save(d / fn, quality=90)
    key.to_csv(d / "key.csv", index=False)
    return d


# ------------------------------------------------------------------ 1-3: blind side + de-scramble
def test_blind_build_layout_and_descramble(tmp_path):
    # ---- 2: layout from the filter blanks alone (the build_frame.derive_layout naming)
    assert rs.derive_layout("F150W", "F277W") == "color"
    assert rs.derive_layout("F150W2", "") == "gray_sw_only"          # lw blank -> SW present
    assert rs.derive_layout("", "F444W") == "gray_lw_only"           # sw blank -> LW present
    assert rs.derive_layout(float("nan"), "F277W") == "gray_lw_only" # NaN == blank
    for sw, lw in (("", np.nan), (" ", None), (None, "")):
        with pytest.raises(ValueError):
            rs.derive_layout(sw, lw)
    assert rs.scr_name("007.jpg") == "scr_007"
    assert rs.scr_to_filename("scr_007") == "007.jpg"
    for bad in ("7.jpg", "0007.jpg", "001.png"):
        with pytest.raises(ValueError):
            rs.scr_name(bad)

    key = _synthetic_key()
    d = _make_kit(tmp_path, key)
    # ---- 1: builder output carries nothing but name / image_path / layout
    cands = rs.build_blind_cands(key, d)
    assert list(cands.columns) == list(rs.BLIND_COLS)
    assert cands["name"].tolist() == ["scr_001", "scr_002", "scr_003"]
    assert cands["layout"].tolist() == ["color", "gray_sw_only", "gray_lw_only"]
    leak_values = set(key["candidate_id"]) | set(key["rank"]) | set(key["ra_deg"]) | \
        set(key["evidence"]) | set(key["verifier_grade"]) | {"F150W", "F277W", "F444W", "F150W2"}
    for _, row in cands.iterrows():
        for v in row.tolist():
            assert str(v) not in leak_values, (v, "key column leaked into a blind cand")
        assert "JTEST" not in str(row.tolist())
    # a wrong-size image refuses (the scrambled set is footer-stripped 752x540)
    from PIL import Image
    Image.new("RGB", (752, 562), (0, 0, 0)).save(d / "001.jpg", quality=90)
    with pytest.raises(ValueError):
        rs.build_blind_cands(key, d)

    # ---- 3: de-scramble join + comparison shape + agree logic, canned prediction rows
    assert rs.COMPARISON_COLS == LEGACY_COLS + DEPLOY_CSV_COLS          # 17 pinned, then the deploy columns
    preds = pd.DataFrame({
        "name": ["scr_001", "scr_002", "scr_003"],
        "S": [0.30, 0.05, float("nan")],
        "S_arb": [0.30, 0.05, float("nan")],
        "grade_pred": ["A", "C", None],                # scr_003 = parse failure
        "letter_llm": ["A", "C", None],
        "p_evidence": [0.85, 0.20, 0.10],
        "scale_class_final": ["galaxy", "galaxy", "none"],
        "alternative_final": [None, "merger", None],
        "needs_human": [False, False, False],
        "parse_ok": [True, True, False],
        "letter_source": ["sonnet_api_calibrated"] * 3,
        "cost_usd": [0.08, 0.06, 0.02],
    })
    comp = rs.descramble(preds, key)                  # no deploy columns in the preds: None / ""
    assert list(comp.columns) == list(rs.COMPARISON_COLS)
    assert comp["scrambled_item"].tolist() == ["scr_001", "scr_002", "scr_003"]
    assert comp["candidate_id"].tolist() == ["JTEST0001+0000001", "JTEST0002+0000002", "JTEST0003+0000003"]
    assert comp["rank"].tolist() == [42, 3, 77] and comp["rank"].dtype.kind == "i"
    assert comp["nate_grade"].tolist() == ["U", "A", "C"]
    assert comp["nate_n_pass"].tolist() == [0, 3, 1]
    assert math.isnan(comp["nate_inspector_conf"].iloc[2])       # blank conf -> NaN
    assert math.isnan(comp["blind_theta_E_arcsec"].iloc[2])
    assert comp["agree_letter"].tolist() == [False, False, False]  # U != A; A != C
    assert comp["our_letter_rank"].isna().all() and comp["our_letter_final"].isna().all()
    assert comp["our_veto"].tolist() == ["", "", ""] and comp["our_rule"].isna().all()
    assert comp["agree_final"].tolist() == [False] * 3 and comp["our_rationale"].isna().all()
    preds2 = preds.assign(grade_pred=["A", "A", "C"])            # scr_002 nate=A ours=A
    comp2 = rs.descramble(preds2, key).set_index("scrambled_item")
    assert bool(comp2.loc["scr_002", "agree_letter"]) is True
    assert bool(comp2.loc["scr_001", "agree_letter"]) is False   # U never agrees
    assert bool(comp2.loc["scr_003", "agree_letter"]) is True    # C == C
    # ORDER: REGISTRY item 2 ranks by p_evidence (R), ties by S, NaN last — not by S
    preds3 = preds.assign(p_evidence=[0.50, 0.90, float("nan")])
    assert rs.descramble(preds3, key)["scrambled_item"].tolist() == ["scr_002", "scr_001", "scr_003"]
    preds4 = preds.assign(p_evidence=[0.50, 0.50, 0.60], S=[0.05, 0.30, float("nan")])
    assert rs.descramble(preds4, key)["scrambled_item"].tolist() == ["scr_003", "scr_002", "scr_001"]
    # the deploy columns pass through; agree_final is on letter_final; rationale is one line
    preds5 = preds.assign(letter_rank=["A", "B", "B"], letter_final=["B", "A", None],
                          veto=["geometry:merger", "", None], rule=["R1"] * 3,
                          rationale=["line one\nline two\n  three", "x", None])
    comp5 = rs.descramble(preds5, key).set_index("scrambled_item")
    assert comp5.loc["scr_001", "our_letter_rank"] == "A" and comp5.loc["scr_001", "our_letter_final"] == "B"
    assert comp5.loc["scr_001", "our_veto"] == "geometry:merger" and comp5.loc["scr_003", "our_veto"] == ""
    assert comp5["agree_final"].tolist() == [False, True, False]   # scr_002: nate A == final A
    assert comp5.loc["scr_001", "our_rationale"] == "line one line two three"
    assert comp5["our_rule"].tolist() == ["R1"] * 3
    # a preds row without a key row refuses
    with pytest.raises(ValueError):
        rs.descramble(preds.assign(name=["scr_001", "scr_002", "scr_099"]), key)


# ------------------------------------------------------------------ 4: the frozen tuple per model
def _disk_stack():
    note = rte.NOTE_V2.read_text()
    _, full, _ = rte.role_prompts(rte.PERSONA_SET_DEFAULT, rs.ARM, note)
    return note, rte.system_shas(full, rs.ARM)


def test_frozen_tuple_per_model(monkeypatch):
    assert rs.MODELS == ("sonnet", "opus5") and rs.ARM == "a1"
    sonnet, opus5 = rs.FROZEN["sonnet"], rs.FROZEN["opus5"]
    assert sonnet["persona_set_sha16"] == "a26d972ecc0b4ee7" and sonnet["note_sha16"] == "754655a400f360e6"
    assert sonnet["system_sha16s"] == {"advocate": "c41d7f5787bdb472", "artifact": "f5ed259652e65ee2",
                                       "geometry": "a293ddddce11ee4a", "morphology": "26bde57ad0478237",
                                       "arbitrator": "44542114399ab277"}
    assert sonnet["render_desc_sha16"] == "28737c6083dc1978"
    assert sonnet["thresholds_sha16"] == "94d31c7b6979e0ca" == rte.thresholds_sha(SONNET_THR)
    assert (sonnet["thinking"], sonnet["effort"]) == ("off", "default")
    assert sonnet["letter_sources"] == ("sonnet_api_calibrated",)
    for k in ("persona_set_sha16", "note_sha16", "system_sha16s", "render_desc_sha16"):
        assert opus5[k] == sonnet[k], k                               # the same prompts
    assert opus5["thresholds_sha16"] is None                          # recorded, not gated
    assert (opus5["thinking"], opus5["effort"]) == ("adaptive", "xhigh")
    assert opus5["letter_sources"] == ("provisional", "opus5_api_calibrated")
    assert rte.COST_PER_CALL["opus5"] == 0.10 and rs.COST_CAP["opus5"] == pytest.approx(0.5)
    assert rs.COST_CAP["sonnet"] == rte.COST_CAP_DEFAULT and rs.DEPLOY_RULES == ("R1", "R2")
    assert rs.DEPLOY_COLS == ("letter_rank", "letter_final", "veto", "rule")

    # apply_model_env drives the run_truth_eval variables (effort "default" = unset)
    monkeypatch.setenv("LENSJUDGE_THINKING", "adaptive")
    monkeypatch.setenv("LENSJUDGE_EFFORT", "low")
    assert rs.apply_model_env("sonnet") == ("off", "default") and "LENSJUDGE_EFFORT" not in os.environ
    assert rs.apply_model_env("opus5") == ("adaptive", "xhigh")
    assert os.environ["LENSJUDGE_THINKING"] == "adaptive" and os.environ["LENSJUDGE_EFFORT"] == "xhigh"

    # check_frozen against the on-disk stack (the gate the CLI replays)
    note, shas = _disk_stack()
    got = rs.check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, SONNET_THR, "off", "default", "sonnet")
    assert got["thresholds_sha16"] == "94d31c7b6979e0ca" and got["system_sha16s"] == sonnet["system_sha16s"]
    disk = rte.load_thresholds(rte.THRESHOLDS, rte.model_key("sonnet"))
    assert rte.thresholds_sha(disk) == "94d31c7b6979e0ca"            # thresholds_v2.json still holds the frozen sonnet key
    prov = {**PROV, "letter_source": "provisional", "thresholds_key": "provisional"}
    cal = {"tau0": 0.15, "t_A": 0.31, "t_B": 0.2, "letter_source": "opus5_api_calibrated", "thresholds_key": "opus5_api"}
    for thr in (prov, cal):                                            # any thresholds sha passes for opus5
        g = rs.check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, thr, "adaptive", "xhigh", "opus5")
        assert g["thresholds_sha16"] == rte.thresholds_sha(thr) and g["thinking"] == "adaptive"
    # refusals: thinking / effort, a foreign letter_source, a moved sonnet threshold, a prompt sha
    with pytest.raises(SystemExit):
        rs.check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, SONNET_THR, "adaptive", "xhigh", "sonnet")
    with pytest.raises(SystemExit):
        rs.check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, prov, "off", "default", "opus5")
    with pytest.raises(SystemExit):
        rs.check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, prov, "adaptive", "high", "opus5")
    with pytest.raises(SystemExit):
        rs.check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, {**cal, "letter_source": "sonnet_api_uncalibrated"},
                        "adaptive", "xhigh", "opus5")
    with pytest.raises(SystemExit):
        rs.check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, prov, "off", "default", "sonnet")
    with pytest.raises(SystemExit):
        rs.check_frozen(rte.PERSONA_SET_DEFAULT, note, shas, {**SONNET_THR, "t_A": 0.5}, "off", "default", "sonnet")
    with pytest.raises(SystemExit):
        rs.check_frozen(rte.PERSONA_SET_DEFAULT, note, {**shas, "advocate": "0" * 16}, SONNET_THR, "off", "default", "sonnet")
    with pytest.raises(SystemExit):
        rs.check_frozen(rte.PERSONA_SET_DEFAULT, note + "\nedit", shas, SONNET_THR, "off", "default", "sonnet")


# ------------------------------------------------------------------ 5: --dry-run on a synthetic kit
def test_dry_run_opus5_synthetic_kit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LENSJUDGE_THINKING", "off")      # restored at teardown whatever main sets
    monkeypatch.setenv("LENSJUDGE_EFFORT", "low")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    key = _synthetic_key()
    d = _make_kit(tmp_path, key)
    out = tmp_path / "out"
    base = ["--scrambled-dir", str(d), "--key", "key.csv", "--out", str(out), "--dry-run"]
    rs.main(base + ["--model", "opus5", "--max-budget", "60"])
    text = capsys.readouterr().out
    assert "| a1 | opus5 |" in text and "| adaptive | xhigh |" in text
    assert "recorded, not gated" in text and "no call made" in text
    assert "worst-case ≈ $1.50" in text                       # 3 items x 5 calls x $0.10
    assert "'model_key': 'opus5_api'" in text and "rule R1" in text
    assert "3 blind items (1 color / 2 gray)" in text
    assert os.environ["LENSJUDGE_THINKING"] == "adaptive" and os.environ["LENSJUDGE_EFFORT"] == "xhigh"
    assert out.is_dir() and not list(out.glob("*.parquet")) and not (out / "traces_scrambled100_blind").exists()
    assert not list(out.glob("*.csv")) and not list(out.glob("*.json"))
    # budget refusal happens before any call
    with pytest.raises(SystemExit, match="REFUSED"):
        rs.main(base + ["--model", "opus5", "--max-budget", "1"])
    # the default --max-budget stays 10.0 (3 items x $0.50 fit; the real 100 items = $50 need --max-budget 60)
    rs.main(base + ["--model", "opus5"])
    assert "--max-budget" not in capsys.readouterr().out
    # sonnet under a stray adaptive env is forced back to the frozen off / default
    rs.main(base + ["--model", "sonnet"])
    text = capsys.readouterr().out
    assert "| a1 | sonnet |" in text and "| off | default |" in text and "94d31c7b6979e0ca" in text
    assert "sonnet_api_calibrated; gated" in text
    # a missing image refuses before the budget line
    (d / "002.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        rs.main(base + ["--model", "opus5", "--max-budget", "60"])


# ------------------------------------------------------------------ 6: --reletter on a synthetic run
def _advocate(p_ev: float, notes: str = "tangential arc east of the core"):
    return sp.AdvocateRecord(
        id="item", persona="advocate",
        criteria=sp.CriteriaV2(source_contrast=7, low_surface_brightness=6, curvature=8, counter_image=6,
                               arc_morphology=7),
        items=[sp.EvidenceItem(k=1, what="arc", panel="d", r_arcsec=1.3, pa_deg_from=40, pa_deg_to=170,
                               visible_in_direct=True, criteria=[3, 5]),
               sp.EvidenceItem(k=2, what="counter", panel="e", r_arcsec=1.1, pa_deg_from=230, pa_deg_to=260,
                               visible_in_direct=True, criteria=[4])],
        scale_class="galaxy", n_red_neighbours_10as=0, bcg_like_halo=False, deflector_is_centre=True,
        p_evidence=p_ev, notes=notes)


def _critic(role, alternative=None, r=0.0, no_opinion=False, reason=None, accounts=()):
    loc = sp.LocationBox(r_arcsec_from=1.0, r_arcsec_to=1.6, pa_deg_from=30, pa_deg_to=180) if alternative else None
    return sp.CriticRecord(id="item", persona=role, no_opinion=no_opinion, no_opinion_reason=reason,
                           alternative=alternative, alternative_desc="" if alternative is None else "desc",
                           location=loc, accounts_for=list(accounts), leaves_standing=[],
                           refutation_strength=r if alternative else None, notes=f"{role} note")


def _arbitrator(rulings, rationale="the arc\nstays"):
    return sp.ArbitratorRecord(id="item", persona="arbitrator",
                               rulings=[sp.Ruling(persona=p, ruling=ru, covers=[1], why="w") for p, ru in rulings],
                               surviving_items=[1, 2], letter_llm="B", scale_class_final="galaxy",
                               needs_human=False, rationale=rationale)


def _raw(rec) -> str:
    return "Here is my reading.\n\n```json\n" + json.dumps(rec.model_dump(mode="json"), indent=2) + "\n```\n"


def _thr_file(path: Path, opus5_entry) -> Path:
    path.write_text(json.dumps({"sonnet_api": {"tau0": 0.15, "t_A": 0.192, "t_B": 0.1318},
                                "opus5_api": opus5_entry, "provisional": PROV}, indent=2))
    return path


def _synthetic_run(run_dir: Path, key_csv: Path, thr_path: Path, model: str = "opus5") -> dict:
    """Four blind items: scr_001 advocate 0.7 + geometry shell_tidal r=0.6 covering item 1
    (a=0.5), upheld; scr_002 advocate 0.9 + three non-refuting critics + an arbitrator with
    no rulings; scr_003 advocate 0.1 alone (below tau0); scr_004 advocate 0.5 with a geometry
    raw that does not parse (parse_ok False -> S NaN). Preds rows through the assemble path
    (records.row_from_records) + the run-tuple columns; votes from the fenced raws."""
    run_dir.mkdir(parents=True, exist_ok=True)
    recs = {
        "scr_001": {"advocate": _advocate(0.7), "artifact": _critic("artifact", no_opinion=True, reason="image_quality"),
                    "geometry": _critic("geometry", "shell_tidal", r=0.6, accounts=(1,)),
                    "morphology": _critic("morphology"), "arbitrator": _arbitrator([("geometry", "upheld")])},
        "scr_002": {"advocate": _advocate(0.9, notes="clean arc"), "artifact": _critic("artifact", no_opinion=True, reason="outside_competence"),
                    "geometry": _critic("geometry"), "morphology": _critic("morphology"),
                    "arbitrator": _arbitrator([], rationale="nothing refuted")},
        "scr_003": {"advocate": _advocate(0.1, notes="faint smudge\nonly")},
        "scr_004": {"advocate": _advocate(0.5), "artifact": _critic("artifact"), "geometry": None,
                    "morphology": _critic("morphology")},
    }
    thr = rte.load_thresholds(thr_path, rte.model_key(model))
    sha = rte.thresholds_sha(thr)
    rows, votes = [], []
    for name, roles in recs.items():
        row = R.row_from_records(name, recs, thr)
        row.update({"layout": "color", "split": None, "arm": rs.ARM, "model": model,
                    "persona_set_sha16": rs.FROZEN_SHAS["persona_set_sha16"], "note_sha16": rs.FROZEN_SHAS["note_sha16"],
                    "system_sha16s": rte.join_shas(rs.FROZEN_SHAS["system_sha16s"], rs.ARM), "render_version": "jwst_v1",
                    "render_desc_sha16": rs.FROZEN_SHAS["render_desc_sha16"],
                    "splits_sha16": _util.sha_file(key_csv), "claim_mode": "none",
                    "thinking": rs.FROZEN[model]["thinking"], "effort": rs.FROZEN[model]["effort"], "k": 1,
                    "thresholds_sha16": sha, "run_tag": rs.RUN_TAG, "rescored": False, "rescore_reason": None,
                    "tau0": thr["tau0"], "t_A": thr["t_A"], "t_B": thr["t_B"], "cost_usd": 0.05})
        rows.append(row)
        for role, rec in roles.items():
            votes.append({"name": name, "unit_id": "", "role": role, "k": 1, "parse_ok": rec is not None,
                          "raw": _raw(rec) if rec is not None else "not json {", "cost_usd": 0.01,
                          "system_sha16": f"sha_{role}"})
    preds = pd.DataFrame(rows)
    out = rs.preds_path_for(run_dir, model)
    preds.to_parquet(out, index=False)
    pd.DataFrame(votes, columns=list(R.VOTE_COLS)).to_parquet(R.votes_path_for(out), index=False)
    meta = {"tuple": {"arm": rs.ARM, "model": model, "thresholds_sha16": sha}, "key_sha16": _util.sha_file(key_csv),
            "letter_source": thr["letter_source"], "thresholds_resolved": thr, "model": model, "n": len(preds)}
    Path(run_dir / out.name.replace(".parquet", ".meta.json")).write_text(json.dumps(meta, indent=2))
    return {"out": out, "preds": preds, "records": recs, "thr": thr, "sha": sha, "meta": meta}


def _score_cols_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    for c in rs.SCORE_COLS:
        for x, y in zip(a[c].tolist(), b[c].tolist()):
            if (pd.isna(x) or pd.isna(y)) and not (pd.isna(x) and pd.isna(y)):
                return False
            if not pd.isna(x) and abs(float(x) - float(y)) > 1e-12:
                return False
    return True


def test_reletter_synthetic_run(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    key = _synthetic_key(4)
    kit = _make_kit(tmp_path, key)
    key_csv = kit / "key.csv"
    thr_a = _thr_file(tmp_path / "thr_a.json", None)          # opus5_api null -> provisional (run time)
    thr_b = _thr_file(tmp_path / "thr_b.json", {"tau0": 0.15, "t_A": 0.6, "t_B": 0.3})
    run = _synthetic_run(tmp_path / "run", key_csv, thr_a)
    out, orig = run["out"], run["preds"]
    assert run["thr"]["letter_source"] == "provisional"
    o = orig.set_index("name")
    assert o.loc["scr_001", "S"] == pytest.approx(0.49) and o.loc["scr_001", "S_arb"] == pytest.approx(0.49)
    assert o.loc["scr_001", "grade_pred"] == "C" and o.loc["scr_002", "grade_pred"] == "A"
    assert o.loc["scr_003", "grade_pred"] == "C" and pd.isna(o.loc["scr_004", "S"]) and pd.isna(o.loc["scr_004", "grade_pred"])
    assert "letter_rank" not in orig.columns

    # ---- the pure re-letter on the run's own records, under the calibrated file
    preds0, records = R.load_run(out)
    thr_b_res = rte.load_thresholds(thr_b, "opus5_api")
    assert thr_b_res["letter_source"] == "opus5_api_calibrated"
    new, mism = rs.deploy_rows(preds0, records, thr_b_res, "R1")
    assert mism == [] and list(new["name"]) == list(preds0["name"])
    n = new.set_index("name")
    assert n.loc["scr_001", "grade_pred"] == "B" and n.loc["scr_001", "letter_arb"] == "B"      # 0.49 in [0.3, 0.6)
    assert n.loc["scr_001", "letter_rank"] == "A"                                               # advocate 0.7 >= 0.6, 3 strong criteria
    assert n.loc["scr_001", "letter_final"] == "B" and n.loc["scr_001", "veto"] == "geometry:shell_tidal"
    assert n.loc["scr_002", "letter_rank"] == "A" and n.loc["scr_002", "letter_final"] == "A" and n.loc["scr_002", "veto"] == ""
    assert n.loc["scr_003", "letter_rank"] == "C" and n.loc["scr_003", "letter_final"] == "C"
    assert pd.isna(n.loc["scr_004", "S"]) and pd.isna(n.loc["scr_004", "grade_pred"])          # NaN stays NaN
    assert n.loc["scr_004", "letter_rank"] == "B" and pd.isna(n.loc["scr_004", "letter_final"])
    assert set(n["rule"]) == {"R1"} and set(n["letter_source"]) == {"opus5_api_calibrated"}
    assert set(n["thresholds_sha16"]) == {rte.thresholds_sha(thr_b_res)} and set(n["t_A"]) == {0.6}
    assert _score_cols_equal(new, preds0)
    # the deploy letters equal aggregate_v2.deploy_letters on the same records
    for name, roles in records.items():
        d = aggregate_v2.deploy_letters(roles.get("advocate"), {r: roles[r] for r in rs.CRITIC_ROLES if r in roles},
                                        roles.get("arbitrator"), thr_b_res, "R1")
        assert (n.loc[name, "letter_rank"], n.loc[name, "letter_final"], n.loc[name, "veto"]) == \
            (d["letter_rank"], d["letter_final"], d["veto"])
    with pytest.raises(ValueError):
        rs.deploy_rows(preds0, records, thr_b_res, "R3")

    # ---- the CLI: --reletter RUN_DIR (zero API), rule R1
    args = ["--reletter", str(tmp_path / "run"), "--scrambled-dir", str(kit), "--key", "key.csv"]
    orig_bytes = out.read_bytes()
    rs.main(args + ["--thresholds", str(thr_b), "--rule", "R1"])
    text = capsys.readouterr().out
    assert "opus5_api_calibrated" in text and "1 grade_pred change(s)" in text and "1 NaN row(s) left NaN" in text
    pre = sorted((tmp_path / "run").glob("preds_scrambled100_a1_opus5.parquet.pre_reletter_*"))
    assert len(pre) == 1 and pre[0].read_bytes() == orig_bytes                          # the copy IS the original
    new = pd.read_parquet(out)
    assert set(rs.DEPLOY_COLS) <= set(new.columns) and _score_cols_equal(new, orig)
    n = new.set_index("name")
    assert n.loc["scr_001", "grade_pred"] == "B" and n.loc["scr_001", "letter_final"] == "B"
    assert n.loc["scr_001", "letter_rank"] == "A" and n.loc["scr_001", "veto"] == "geometry:shell_tidal"
    assert pd.isna(n.loc["scr_004", "S"]) and pd.isna(n.loc["scr_004", "S_arb"])
    assert set(n["letter_source"]) == {"opus5_api_calibrated"} and set(n["t_B"]) == {0.3} and set(n["rule"]) == {"R1"}
    meta = json.loads((tmp_path / "run" / "preds_scrambled100_a1_opus5.meta.json").read_text())
    assert meta["letter_source"] == "opus5_api_calibrated" and meta["rule"] == "R1" and meta["relettered_at_utc"]
    assert meta["thresholds_sha16"] == rte.thresholds_sha(thr_b_res) == meta["tuple"]["thresholds_sha16"]
    assert meta["thresholds_resolved"]["t_A"] == 0.6 and meta["pre_reletter"] == str(pre[0])
    assert len(meta["reletter_history"]) == 1 and meta["key_sha16"] == _util.sha_file(key_csv)
    comp = pd.read_csv(tmp_path / "run" / "scrambled100_comparison.csv")
    assert list(comp.columns) == list(rs.COMPARISON_COLS)
    assert comp["scrambled_item"].tolist() == ["scr_002", "scr_001", "scr_004", "scr_003"]   # by p_evidence, not S
    c = comp.set_index("scrambled_item")
    assert c.loc["scr_001", "our_veto"] == "geometry:shell_tidal" and c.loc["scr_001", "our_rationale"] == "the arc stays"
    assert c.loc["scr_003", "our_rationale"] == "faint smudge only"                     # advocate notes, one line
    assert bool(c.loc["scr_002", "agree_final"]) is True                                 # incumbent A == final A
    assert bool(c.loc["scr_004", "agree_final"]) is False and pd.isna(c.loc["scr_004", "our_letter_final"])
    assert c.loc["scr_004", "our_letter_rank"] == "B" and (comp["our_rule"] == "R1").all()

    # ---- R2 on top: letter_rank demoted only by the D rule (a=0.5 here: no demotion)
    rs.main(args + ["--thresholds", str(thr_b), "--rule", "R2"])
    capsys.readouterr()
    n2 = pd.read_parquet(out).set_index("name")
    assert n2.loc["scr_001", "letter_final"] == "A" and n2.loc["scr_001", "veto"] == "" and set(n2["rule"]) == {"R2"}
    assert n2.loc["scr_001", "grade_pred"] == "B"                                        # assemble letters unchanged by the rule
    assert _score_cols_equal(n2.reset_index(), orig)
    assert len(sorted((tmp_path / "run").glob("*.pre_reletter_*"))) >= 3               # parquet x2 + csv x1..2
    meta2 = json.loads((tmp_path / "run" / "preds_scrambled100_a1_opus5.meta.json").read_text())
    assert len(meta2["reletter_history"]) == 2 and meta2["rule"] == "R2"
    # back to the provisional file: allowed for opus5, letters revert
    rs.main(args + ["--thresholds", str(thr_a), "--rule", "R1"])
    capsys.readouterr()
    n3 = pd.read_parquet(out).set_index("name")
    assert n3.loc["scr_001", "grade_pred"] == "C" and n3.loc["scr_001", "letter_rank"] == "B"
    assert set(n3["letter_source"]) == {"provisional"}

    # ---- refusals: a wrong --model, a foreign key, a tampered S (nothing written)
    with pytest.raises(SystemExit, match="sonnet"):                # narrowed lookup: no sonnet parquet here
        rs.main(args + ["--thresholds", str(thr_b), "--model", "sonnet"])
    other = _synthetic_key(4).assign(rank=["1", "2", "3", "4"])
    other.to_csv(tmp_path / "other_key.csv", index=False)
    with pytest.raises(SystemExit, match="key_sha16"):
        rs.main(["--reletter", str(tmp_path / "run"), "--key", str(tmp_path / "other_key.csv"), "--thresholds", str(thr_b)])
    capsys.readouterr()
    tam_dir = tmp_path / "run_tampered"
    shutil.copytree(tmp_path / "run", tam_dir, ignore=shutil.ignore_patterns("*.pre_reletter_*"))
    tam_out = rs.preds_path_for(tam_dir, "opus5")
    tam = pd.read_parquet(tam_out)
    tam.loc[tam["name"] == "scr_001", "S"] = float(tam.loc[tam["name"] == "scr_001", "S"].iloc[0]) + 0.01
    tam.to_parquet(tam_out, index=False)
    before = tam_out.read_bytes()
    preds_t, records_t = R.load_run(tam_out)
    _, mism = rs.deploy_rows(preds_t, records_t, thr_b_res, "R1")
    assert [(m["name"], m["col"]) for m in mism] == [("scr_001", "S")]
    with pytest.raises(SystemExit, match="nothing written"):
        rs.main(["--reletter", str(tam_dir), "--scrambled-dir", str(kit), "--key", "key.csv", "--thresholds", str(thr_b)])
    text = capsys.readouterr().out
    assert "scr_001: S stored" in text
    assert tam_out.read_bytes() == before and not list(tam_dir.glob("*.pre_reletter_*"))
    # a scored row with no votes at all is a mismatch too ("records")
    _, mism2 = rs.deploy_rows(preds_t, {k: v for k, v in records_t.items() if k != "scr_002"}, thr_b_res, "R1")
    assert ("scr_002", "records") in [(m["name"], m["col"]) for m in mism2]
    # a scored row whose records do not rebuild KEEPS its stored letters / threshold columns
    # (only its deploy letters are blank): the non-strict scoring path never erases the record
    new_t, mism_t = rs.deploy_rows(preds_t, records_t, thr_b_res, "R1")
    t0 = preds_t.set_index("name").loc["scr_001"]
    t1 = new_t.set_index("name").loc["scr_001"]
    assert [m["name"] for m in mism_t] == ["scr_001"]
    assert (t1["grade_pred"], t1["letter_arb"], t1["letter_source"], t1["t_A"], t1["thresholds_sha16"]) == \
        (t0["grade_pred"], t0["letter_arb"], t0["letter_source"], t0["t_A"], t0["thresholds_sha16"])
    assert t1["letter_rank"] is None and t1["letter_final"] is None and t1["veto"] == "" and t1["rule"] == "R1"
    ok_row = new_t.set_index("name").loc["scr_002"]
    assert ok_row["letter_source"] == "opus5_api_calibrated" and ok_row["letter_rank"] == "A"
    # the same for a scored row with no votes at all (the raw-less sonnet dev rows)
    new_r, _ = rs.deploy_rows(preds_t, {k: v for k, v in records_t.items() if k != "scr_002"}, thr_b_res, "R1")
    r2 = new_r.set_index("name").loc["scr_002"]
    assert r2["grade_pred"] == preds_t.set_index("name").loc["scr_002", "grade_pred"] and r2["letter_final"] is None
    # attach_deploy non-strict reports, keeps the letters, blanks the deploy columns
    new_a, _, mism_a = rs.attach_deploy(tam_out, R.votes_path_for(tam_out), thr_b_res, "R1", strict=False)
    assert len(mism_a) == 1 and new_a.set_index("name").loc["scr_001", "grade_pred"] == t0["grade_pred"]
    assert tam_out.read_bytes() == before                                    # nothing written by attach_deploy
    text = capsys.readouterr().out
    assert "keep their stored grade_pred" in text
    # write_with_backup never replaces a run parquet in place
    bk = rs.write_with_backup(new_a, tam_out)
    assert bk is not None and bk.name.startswith(tam_out.name + rs.PRE_DEPLOY) and bk.read_bytes() == before
    assert tam_out.read_bytes() != before and not list(tam_dir.glob("*.tmp*"))
    bk2 = rs.write_with_backup(new_a, tam_out, now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc))
    assert bk2.name == tam_out.name + rs.PRE_DEPLOY + "20260101T000000Z"
    assert rs.write_with_backup(new_a, tam_dir / "fresh.parquet") is None and (tam_dir / "fresh.parquet").exists()
    assert rs.find_run_parquet(tam_dir) == tam_out                         # backups are not run parquets
    # a sonnet run may not be re-lettered with provisional thresholds
    son_dir = tmp_path / "run_sonnet"
    _synthetic_run(son_dir, key_csv, _thr_file(tmp_path / "thr_s.json", None), model="sonnet")
    with pytest.raises(SystemExit, match="letter_source"):
        rs.main(["--reletter", str(son_dir), "--scrambled-dir", str(kit), "--key", "key.csv",
                 "--thresholds", str(tmp_path / "thr_none.json")])       # missing file -> provisional


def test_reletter_refuses_a_run_outside_the_frozen_tuple(tmp_path, capsys):
    """--reletter presumes the stored records are the registered instrument's: every frozen
    tuple column (persona set, note, system shas, render, thinking, effort) must equal
    FROZEN[model]; a parquet carrying another tuple is refused with nothing written."""
    key = _synthetic_key(4)
    kit = _make_kit(tmp_path, key)
    thr_b = _thr_file(tmp_path / "thr_b.json", {"tau0": 0.15, "t_A": 0.6, "t_B": 0.3})
    run = _synthetic_run(tmp_path / "run", kit / "key.csv", _thr_file(tmp_path / "thr_a.json", None))
    out = run["out"]
    good = pd.read_parquet(out)
    assert rs.check_frozen_columns(good, "opus5")["persona_set_sha16"] == rs.FROZEN_SHAS["persona_set_sha16"]
    assert rs.frozen_columns("opus5")["system_sha16s"].startswith("advocate:c41d7f5787bdb472+artifact:")
    assert rs.frozen_columns("sonnet")["effort"] == "default" and rs.frozen_columns("opus5")["effort"] == "xhigh"
    args = ["--reletter", str(tmp_path / "run"), "--scrambled-dir", str(kit), "--key", "key.csv",
            "--thresholds", str(thr_b)]
    for col, val in (("persona_set_sha16", "0" * 16), ("note_sha16", "f" * 16), ("thinking", "off"),
                     ("effort", "low"), ("render_desc_sha16", "1" * 16),
                     ("system_sha16s", "advocate:" + "0" * 16)):
        bad = good.copy()
        bad[col] = val
        bad.to_parquet(out, index=False)
        before = out.read_bytes()
        with pytest.raises(SystemExit, match=col):
            rs.check_frozen_columns(bad, "opus5")
        with pytest.raises(SystemExit, match="frozen a1/opus5 tuple"):
            rs.main(args)
        assert out.read_bytes() == before and not list((tmp_path / "run").glob("*.pre_reletter_*"))
        assert not (tmp_path / "run" / "scrambled100_comparison.csv").exists()
    # a mixed column (two tuples in one parquet) and an absent column refuse too
    bad = good.copy()
    bad.loc[bad.index[0], "thinking"] = "off"
    with pytest.raises(SystemExit, match="thinking"):
        rs.check_frozen_columns(bad, "opus5")
    with pytest.raises(SystemExit, match="column absent"):
        rs.check_frozen_columns(good.drop(columns=["note_sha16"]), "opus5")
    # sonnet's frozen values differ from opus5's on thinking / effort only
    with pytest.raises(SystemExit, match="thinking"):
        rs.check_frozen_columns(good, "sonnet")
    # restored, the re-letter runs
    good.to_parquet(out, index=False)
    rs.main(args)
    capsys.readouterr()
    assert len(list((tmp_path / "run").glob("*.pre_reletter_*"))) >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
