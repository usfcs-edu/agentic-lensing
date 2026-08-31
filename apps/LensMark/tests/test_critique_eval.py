import json
from pathlib import Path

import pandas as pd
import pytest

from lensmark.critique import critique_path, list_critiques, submit_critique
from lensmark.evaluate import cli_eval, eval_rows, item_rows
from lensmark.model import Critique, CritiqueItem, CritiquePanel
from lensmark.store import Campaign
from synth_campaign import arrow, mask, ring, seed_run, set_fields

def _jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))



def _critique(image_id, run_id, items, *, reviewer="xhuang", lead=60.0, theta_h=None, fewshot=None, at="2026-08-30T10:00:00Z"):
    return Critique(image_id=image_id, run_id=run_id, reviewer=reviewer, reviewed_at=at, lead_time_s=lead,
                    items=[CritiqueItem(item_id=i, verdict=v, comment=c) for i, v, c in items],
                    panel=CritiquePanel(theta_e_human_arcsec=theta_h, would_use_as_fewshot=fewshot, completeness=4))


def seed_opus_and_sonnet(nine):
    """Two runs x two models with hand-checkable counts (see the assertions in test_eval_rows)."""
    c = Campaign(nine)
    # ---- opus/xhigh: deck-01 (r1) via the UI path (statuses set before the critique), deck-02 (r2) resolved by verdict
    seed_run(c, "deck-01", "r1", [
        arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "deflector", "green"),
        arrow("ann-arrow-002", [0.40, 0.60], [0.30, 0.75], "arc", "cyan"),
        mask("ann-mask-001", [0.1, 0.1], 1.0, "galaxy"),
        mask("ann-mask-002", [0.8, 0.2], 0.4, "star"),
        ring("ann-ring-001", [0.5, 0.5], 1.7),
    ], cost_usd=0.10, n_invalid=1, parse_ok=True, duration_s=30.0, proposed_theta_e=1.7,
        human_items=[arrow("ann-arrow-003", [0.6, 0.3], [0.75, 0.15], "counter-image cand.", "magenta")])
    set_fields(c, "deck-01", {
        "ann-arrow-001": {"status": "accepted"},
        "ann-arrow-002": {"status": "edited", "head": [0.45, 0.60], "edit_of": {"head": [0.40, 0.60], "tail": [0.30, 0.75]}},
        "ann-mask-001": {"status": "accepted"},
        "ann-mask-002": {"status": "rejected"},
        "ann-ring-001": {"status": "accepted"},
    })
    submit_critique(c, _critique("deck-01", "r1", [
        ("ann-arrow-001", "correct", ""), ("ann-arrow-002", "wrong_position", "head 0.8 arcsec off"),
        ("ann-mask-001", "correct", ""), ("ann-mask-002", "spurious", "noise"), ("ann-ring-001", "correct", ""),
        ("ann-arrow-003", "missed_by_model", "counter image"),
    ], lead=60.0, theta_h=1.5, fewshot=True))
    seed_run(c, "deck-02", "r2", [
        arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "deflector", "green"),
        arrow("ann-arrow-002", [0.3, 0.3], [0.2, 0.2], "arc", "cyan"),
        mask("ann-mask-001", [0.1, 0.9], 2.0, "galaxy"),
    ], cost_usd=0.30, n_invalid=0, parse_ok=False, duration_s=50.0)
    submit_critique(c, _critique("deck-02", "r2", [
        ("ann-arrow-001", "correct", ""), ("ann-arrow-002", "redundant", "same arc"), ("ann-mask-001", "wrong_size", "too big"),
    ], lead=120.0))
    # ---- sonnet/low: deck-03 (r3), deck-04 (r4)
    seed_run(c, "deck-03", "r3", [
        arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "deflector", "green"),
        arrow("ann-arrow-002", [0.2, 0.8], [0.1, 0.9], "arc", "cyan"),
        mask("ann-mask-001", [0.1, 0.1], 0.5, "star"), mask("ann-mask-002", [0.9, 0.1], 0.5, "star"),
        mask("ann-mask-003", [0.9, 0.9], 1.0, "galaxy"),
    ], model="claude-sonnet-5", effort="low", cost_usd=0.02, duration_s=8.0,
        human_items=[arrow("ann-arrow-003", [0.4, 0.6], [0.3, 0.8], "arc", "yellow")])
    submit_critique(c, _critique("deck-03", "r3", [
        ("ann-arrow-001", "correct", ""), ("ann-arrow-002", "spurious", ""), ("ann-mask-001", "spurious", ""),
        ("ann-mask-002", "spurious", ""), ("ann-mask-003", "correct", ""), ("ann-arrow-003", "missed_by_model", ""),
    ], lead=10.0))
    seed_run(c, "deck-04", "r4", [
        arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "deflector", "green"),
        ring("ann-ring-001", [0.5, 0.5], 2.5),
    ], model="claude-sonnet-5", effort="low", cost_usd=0.04, duration_s=9.0, proposed_theta_e=2.5)
    submit_critique(c, _critique("deck-04", "r4", [
        ("ann-arrow-001", "wrong_type", "that is a mask, not an arrow"), ("ann-ring-001", "correct", ""),
    ], lead=20.0, theta_h=2.0))
    return c


def test_submit_fills_counts_merges_reviews_and_computes_delta(nine):
    c = seed_opus_and_sonnet(nine)
    cs = list_critiques(c)
    assert [(x.image_id, x.run_id) for x in cs] == [("deck-01", "r1"), ("deck-02", "r2"), ("deck-03", "r3"), ("deck-04", "r4")]
    c1 = cs[0]
    assert c1.counts == {"proposed": 5, "accepted": 3, "edited": 1, "rejected": 1, "invalid": 0, "unreviewed": 0, "added_by_human": 1}
    assert c1.model == "claude-opus-5" and c1.effort == "xhigh"        # filled from the ProposalRun
    p = critique_path(c, c1)
    assert p.name == "deck-01.xhuang.r1.json" and p.exists() and _jload(p)["counts"]["proposed"] == 5
    f = c.load("deck-01")
    assert f.provenance.critiques == ["critiques/deck-01.xhuang.r1.json"]
    a2 = f.item("ann-arrow-002")
    assert a2.review.verdict == "wrong_position" and a2.review.reviewer == "xhuang" and a2.review.reviewed_at == c1.reviewed_at
    assert a2.review.delta_arcsec == pytest.approx(0.05 * 16.0)          # edit_of head [0.40,0.60] -> [0.45,0.60] on a 16" cutout
    assert next(ci for ci in c1.items if ci.item_id == "ann-arrow-002").delta_arcsec == pytest.approx(0.8)
    assert f.item("ann-mask-002").review.verdict == "spurious" and f.item("ann-arrow-003").review.verdict == "missed_by_model"
    assert f.item("ann-arrow-001").review.delta_arcsec is None
    log = c.read_log("deck-01")
    assert any(e["op"] == "critique" and e["source"] == "critique" and e["actor"] == "xhuang" for e in log)
    # deck-02 statuses were left 'proposed' -> resolved from the verdicts
    f2 = c.load("deck-02")
    assert [it.status for it in f2.items] == ["accepted", "rejected", "rejected"]
    assert cs[1].counts["proposed"] == 3 and cs[1].counts["unreviewed"] == 0


def test_submit_requires_saved_file_and_tolerates_unknown_ids(nine):
    c = Campaign(nine)
    with pytest.raises(FileNotFoundError):
        submit_critique(c, _critique("deck-09", "rX", [("ann-arrow-001", "correct", "")]))
    seed_run(c, "deck-09", "rX", [arrow("ann-arrow-001", [0.5, 0.5], [0.7, 0.5], "deflector", "green")])
    submit_critique(c, _critique("deck-09", "rX", [("ann-arrow-001", "correct", ""), ("ann-arrow-042", "spurious", "gone")]))
    assert c.load("deck-09").item("ann-arrow-001").status == "accepted"
    assert list_critiques(c, "deck-09")[0].counts["proposed"] == 1


def test_eval_rows_hand_checked(nine):
    c = seed_opus_and_sonnet(nine)
    rows = eval_rows(c, "model,effort")
    assert [(r["model"], r["effort"]) for r in rows] == [("claude-opus-5", "xhigh"), ("claude-sonnet-5", "low")]
    opus, sonnet = rows
    assert (opus["n_runs"], opus["n_images"], opus["proposed"]) == (2, 2, 8)
    assert (opus["TP"], opus["PARTIAL"], opus["FP"], opus["FN"]) == (4, 2, 2, 1)
    assert opus["precision_strict"] == pytest.approx(4 / 8)
    assert opus["precision_lenient"] == pytest.approx(6 / 8)
    assert opus["recall"] == pytest.approx(6 / 7)
    assert opus["spurious_mask_rate"] == pytest.approx(1 / 3)           # 1 spurious of 3 proposed masks
    assert opus["median_delta_arcsec"] == pytest.approx(0.8)
    assert opus["theta_e_abs_err"] == pytest.approx(0.2)                # |1.5 human - 1.7 proposed|
    assert opus["mean_cost_usd"] == pytest.approx(0.2)
    assert opus["parse_ok_rate"] == pytest.approx(0.5)
    assert opus["mean_n_invalid"] == pytest.approx(0.5)
    assert opus["mean_lead_time_s"] == pytest.approx(90.0)
    assert (sonnet["n_runs"], sonnet["proposed"]) == (2, 7)
    assert (sonnet["TP"], sonnet["PARTIAL"], sonnet["FP"], sonnet["FN"]) == (3, 1, 3, 1)
    assert sonnet["precision_strict"] == pytest.approx(3 / 7)
    assert sonnet["precision_lenient"] == pytest.approx(4 / 7)
    assert sonnet["recall"] == pytest.approx(4 / 5)
    assert sonnet["spurious_mask_rate"] == pytest.approx(2 / 3)
    assert sonnet["median_delta_arcsec"] is None
    assert sonnet["theta_e_abs_err"] == pytest.approx(0.5)
    assert sonnet["mean_cost_usd"] == pytest.approx(0.03) and sonnet["parse_ok_rate"] == 1.0
    assert sonnet["mean_lead_time_s"] == pytest.approx(15.0)
    # files
    out = c.exports_dir / "eval"
    items = pd.read_parquet(out / "items.parquet")
    assert len(items) == 6 + 3 + 6 + 2
    assert set(["image_id", "run_id", "model", "effort", "cost_usd", "parse_ok", "item_id", "verdict", "category", "delta_arcsec"]) <= set(items.columns)
    assert items[items.item_id == "ann-arrow-002"].iloc[0]["category"] == "PARTIAL"
    assert sorted(items.category.value_counts().to_dict().items()) == [("FN", 2), ("FP", 5), ("PARTIAL", 3), ("TP", 7)]
    runs = pd.read_csv(out / "runs.csv")
    assert list(runs.model) == ["claude-opus-5", "claude-sonnet-5"] and runs.iloc[0]["TP"] == 4
    per_run = pd.read_csv(out / "per_run.csv")
    assert len(per_run) == 4 and set(per_run.run_id) == {"r1", "r2", "r3", "r4"}


def test_eval_other_groupings_and_cli(nine, capsys):
    c = seed_opus_and_sonnet(nine)
    by_model = eval_rows(c, "model", write=False)
    assert [r["model"] for r in by_model] == ["claude-opus-5", "claude-sonnet-5"] and by_model[0]["proposed"] == 8
    by_rev = eval_rows(c, "reviewer", write=False)
    assert len(by_rev) == 1 and by_rev[0]["n_runs"] == 4 and by_rev[0]["proposed"] == 15
    with pytest.raises(ValueError):
        eval_rows(c, "nonsense", write=False)
    assert cli_eval(str(nine), by="model,effort") == 0
    out = capsys.readouterr().out
    assert "claude-opus-5" in out and "precision_strict" in out
    items, runs = item_rows(c)
    assert len(runs) == 4 and all("deltas" in r for r in runs)


def test_eval_empty_campaign(nine):
    c = Campaign(nine)
    assert eval_rows(c) == []
    assert (c.exports_dir / "eval" / "runs.csv").exists()
    assert len(pd.read_parquet(c.exports_dir / "eval" / "items.parquet")) == 0
