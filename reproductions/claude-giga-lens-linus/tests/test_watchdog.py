"""CPU-only unit tests for tools/watchdog.py — fake pollers injected, NO ssh.

Covers the pre-registered deliverable checks (dead-man watchdog task):
stall detection (pending/running past budget), vanished-job detection (with and
without the expected artifact), stale-state alert, resubmit-command EMISSION
(dry-run default — proven non-executing via a subprocess bomb), plus UNREACHABLE
handling, log dedup idempotence, corrupt-state survival, and the slurm parsers.

No jax, no cgl2 import, no network: any python3 + pytest runs this.
"""
from __future__ import annotations

import importlib.util
import json
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"

_spec = importlib.util.spec_from_file_location("cgl2_watchdog", TOOLS / "watchdog.py")
wd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wd)

NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ tiny fixtures

def iso_ago(hours: float) -> str:
    return (NOW - timedelta(hours=hours)).isoformat()


def job(**kw) -> dict:
    base = {
        "jobid": "111",
        "host": "perlmutter",
        "submitted_at": iso_ago(1.0),
        "max_pending_h": 24,
        "max_run_h": 6,
        "on_stall": "alert",
    }
    base.update(kw)
    return base


def write_state(tmp_path: Path, jobs: list, updated_ago_h: float = 0.0, **extra) -> Path:
    state = {"updated_at": iso_ago(updated_ago_h), "jobs": jobs}
    state.update(extra)
    p = tmp_path / "watchdog_state.json"
    p.write_text(json.dumps(state))
    return p


def fake_pollers(mapping: dict):
    """mapping: jobid -> poll dict; unlisted jobids come back VANISHED."""
    def poll(jobids):
        return {
            str(j): dict(mapping.get(
                str(j), {"state": "VANISHED", "detail": "not found", "elapsed_h": None}))
            for j in jobids
        }
    return {"perlmutter": poll, "phoenix": poll}


def run(tmp_path, state_path, mapping, execute=False, now=NOW):
    return wd.run_pass(
        state_path=state_path,
        log_path=tmp_path / "alerts.log",
        flag_path=tmp_path / "WATCHDOG_ALERT",
        pollers=fake_pollers(mapping),
        now=now,
        execute=execute,
        root=tmp_path,
    )


def conditions(alerts):
    return [a["condition"] for a in alerts]


def saved_jobs(state_path):
    return json.loads(Path(state_path).read_text())["jobs"]


@pytest.fixture
def bomb(monkeypatch):
    """Any subprocess use inside the watchdog module explodes the test."""
    def _bomb(*a, **kw):
        raise AssertionError(f"subprocess must not be invoked (args={a!r})")
    monkeypatch.setattr(wd.subprocess, "run", _bomb)


# ------------------------------------------------------------------ stall detection

def test_pending_too_long_alerts(tmp_path, bomb):
    sp = write_state(tmp_path, [job(submitted_at=iso_ago(30.0), max_pending_h=24)])
    alerts, _ = run(tmp_path, sp, {"111": {"state": "PENDING", "detail": "squeue:PENDING"}})
    assert conditions(alerts) == ["PENDING_TOO_LONG"]
    assert "30.0h > max_pending_h=24" in alerts[0]["msg"]
    assert (tmp_path / "WATCHDOG_ALERT").exists()
    assert "PENDING_TOO_LONG" in (tmp_path / "alerts.log").read_text()


def test_pending_within_window_is_clean(tmp_path, bomb):
    sp = write_state(tmp_path, [job(submitted_at=iso_ago(2.0))])
    alerts, info = run(tmp_path, sp, {"111": {"state": "PENDING", "detail": "squeue:PENDING"}})
    assert alerts == []
    assert not (tmp_path / "WATCHDOG_ALERT").exists()
    assert not (tmp_path / "alerts.log").exists()
    assert any("PENDING" in ln and "ok" in ln for ln in info)


def test_running_past_max_run_h(tmp_path, bomb):
    sp = write_state(tmp_path, [job(max_run_h=6)])
    alerts, _ = run(
        tmp_path, sp,
        {"111": {"state": "RUNNING", "detail": "squeue:RUNNING", "elapsed_h": 10.0}})
    assert conditions(alerts) == ["RUNNING_TOO_LONG"]
    assert "10.0h > max_run_h=6" in alerts[0]["msg"]


def test_running_falls_back_to_wall_age_without_elapsed(tmp_path, bomb):
    # phoenix-style: poller has liveness but no elapsed -> wall age since submit
    sp = write_state(tmp_path, [job(host="phoenix", jobid="4242",
                                    submitted_at=iso_ago(13.0), max_run_h=12)])
    alerts, _ = run(tmp_path, sp, {"4242": {"state": "RUNNING", "detail": "ps:python"}})
    assert conditions(alerts) == ["RUNNING_TOO_LONG"]
    assert "wall-age proxy" in alerts[0]["msg"]


def test_running_within_budget_is_clean(tmp_path, bomb):
    sp = write_state(tmp_path, [job(max_run_h=6)])
    alerts, _ = run(
        tmp_path, sp,
        {"111": {"state": "RUNNING", "detail": "squeue:RUNNING", "elapsed_h": 1.0}})
    assert alerts == []


# ------------------------------------------------------------- vanished / completed

def test_vanished_without_artifact_alerts(tmp_path, bomb):
    sp = write_state(tmp_path, [job(expect_artifact="data/out.npz")])
    alerts, _ = run(tmp_path, sp, {})  # fake poller: unlisted -> VANISHED
    assert conditions(alerts) == ["VANISHED_NO_ARTIFACT"]
    assert "data/out.npz" in alerts[0]["msg"]
    assert saved_jobs(sp)[0]["jobid"] == "111"  # kept for the operator to see


def test_vanished_with_artifact_deregisters_silently(tmp_path, bomb):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "out.npz").write_bytes(b"x")
    sp = write_state(tmp_path, [job(expect_artifact="data/out.npz")])
    alerts, info = run(tmp_path, sp, {})
    assert alerts == []
    assert any("assuming success" in ln for ln in info)
    assert saved_jobs(sp) == []  # auto-deregistered
    assert not (tmp_path / "WATCHDOG_ALERT").exists()


def test_completed_without_artifact_alerts(tmp_path, bomb):
    sp = write_state(tmp_path, [job(expect_artifact="data/out.npz")])
    alerts, _ = run(
        tmp_path, sp, {"111": {"state": "COMPLETED", "detail": "sacct:COMPLETED"}})
    assert conditions(alerts) == ["COMPLETED_NO_ARTIFACT"]


def test_completed_with_artifact_deregisters(tmp_path, bomb):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "out.npz").write_bytes(b"x")
    sp = write_state(tmp_path, [job(expect_artifact="data/out.npz")])
    alerts, _ = run(
        tmp_path, sp, {"111": {"state": "COMPLETED", "detail": "sacct:COMPLETED"}})
    assert alerts == []
    assert saved_jobs(sp) == []


def test_oom_failure_alerts(tmp_path, bomb):
    # the exact P1c failure signature: sacct OUT_OF_MEMORY
    sp = write_state(tmp_path, [job()])
    alerts, _ = run(
        tmp_path, sp, {"111": {"state": "FAILED", "detail": "sacct:OUT_OF_MEMORY"}})
    assert conditions(alerts) == ["JOB_FAILED"]
    assert "OUT_OF_MEMORY" in alerts[0]["msg"]


# ------------------------------------------------------------------ stale state file

def test_stale_state_alert(tmp_path, bomb):
    sp = write_state(tmp_path, [job()], updated_ago_h=60.0)  # default stale_state_h=48
    alerts, _ = run(
        tmp_path, sp,
        {"111": {"state": "RUNNING", "detail": "squeue:RUNNING", "elapsed_h": 1.0}})
    assert conditions(alerts) == ["STATE_STALE"]
    assert "orchestrating agent" in alerts[0]["msg"]


def test_fresh_state_no_stale_alert(tmp_path, bomb):
    sp = write_state(tmp_path, [job()], updated_ago_h=1.0)
    alerts, _ = run(
        tmp_path, sp,
        {"111": {"state": "RUNNING", "detail": "squeue:RUNNING", "elapsed_h": 1.0}})
    assert alerts == []


def test_stale_threshold_configurable(tmp_path, bomb):
    sp = write_state(tmp_path, [job()], updated_ago_h=10.0, stale_state_h=8)
    alerts, _ = run(
        tmp_path, sp,
        {"111": {"state": "RUNNING", "detail": "squeue:RUNNING", "elapsed_h": 1.0}})
    assert conditions(alerts) == ["STATE_STALE"]


def test_empty_watchlist_never_stale(tmp_path, bomb):
    sp = write_state(tmp_path, [], updated_ago_h=500.0)
    alerts, _ = run(tmp_path, sp, {})
    assert alerts == []


# --------------------------------------------------------------- resubmit emission

def test_resubmit_emitted_not_executed_by_default(tmp_path, bomb):
    # dry-run default: the exact command is emitted on the alert; the subprocess
    # bomb proves nothing was executed.
    sp = write_state(
        tmp_path,
        [job(on_stall="resubmit:/remote/foundry/slurm/p1c_v3b.slurm")])
    alerts, _ = run(
        tmp_path, sp, {"111": {"state": "FAILED", "detail": "sacct:OUT_OF_MEMORY"}})
    assert conditions(alerts) == ["JOB_FAILED"]
    cmd = alerts[0]["resubmit_cmd"]
    assert cmd == ("ssh gdbenson@perlmutter.nersc.gov "
                   "'sbatch /remote/foundry/slurm/p1c_v3b.slurm'")
    assert "dry-run, NOT executed" in alerts[0]["msg"]
    assert cmd in (tmp_path / "alerts.log").read_text()


def test_resubmit_not_offered_while_job_alive(tmp_path, bomb):
    # *_TOO_LONG means the job still exists -> resubmitting would double-submit
    sp = write_state(
        tmp_path,
        [job(submitted_at=iso_ago(30.0), on_stall="resubmit:/r/x.slurm")])
    alerts, _ = run(tmp_path, sp, {"111": {"state": "PENDING", "detail": "squeue:PENDING"}})
    assert conditions(alerts) == ["PENDING_TOO_LONG"]
    assert "resubmit_cmd" not in alerts[0]
    assert "sbatch" not in alerts[0]["msg"]


def test_resubmit_execute_updates_state_and_respects_cap(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(
            returncode=0, stdout="Submitted batch job 999\n", stderr="")

    monkeypatch.setattr(wd.subprocess, "run", fake_run)
    sp = write_state(tmp_path, [job(on_stall="resubmit:/r/x.slurm")])
    mapping = {"111": {"state": "FAILED", "detail": "sacct:OUT_OF_MEMORY"}}

    alerts, _ = run(tmp_path, sp, mapping, execute=True)
    assert len(calls) == 1 and "sbatch /r/x.slurm" in calls[0]
    entry = saved_jobs(sp)[0]
    assert entry["jobid"] == "999"          # now watching the resubmitted job
    assert entry["resubmit_count"] == 1
    assert "RESUBMITTED" in alerts[0]["msg"]

    # second failure: cap (max_resubmits default 1) blocks another execution
    alerts2, _ = run(tmp_path, sp, {"999": {"state": "FAILED", "detail": "sacct:FAILED"}},
                     execute=True, now=NOW + timedelta(hours=7))
    assert len(calls) == 1                  # no second sbatch
    assert "resubmit exhausted" in alerts2[0]["msg"]


# ------------------------------------------------------- unreachable / robustness

def test_unreachable_alerts_and_keeps_job(tmp_path, bomb):
    sp = write_state(tmp_path, [job(expect_artifact="data/out.npz",
                                    on_stall="resubmit:/r/x.slurm")])
    alerts, _ = run(
        tmp_path, sp,
        {"111": {"state": "UNREACHABLE",
                 "detail": "ssh failed (cert expired? run sshproxy): Permission denied"}})
    assert conditions(alerts) == ["UNREACHABLE"]
    assert "not treated as vanished" in alerts[0]["msg"]
    assert "resubmit_cmd" not in alerts[0]      # unknown status -> never resubmit
    assert saved_jobs(sp)[0]["jobid"] == "111"  # job kept


def test_corrupt_state_alerts_instead_of_crashing(tmp_path, bomb):
    sp = tmp_path / "watchdog_state.json"
    sp.write_text("{not json")
    alerts, _ = run(tmp_path, sp, {})
    assert conditions(alerts) == ["STATE_CORRUPT"]
    assert (tmp_path / "WATCHDOG_ALERT").exists()


def test_missing_state_file_is_clean(tmp_path, bomb):
    alerts, info = run(tmp_path, tmp_path / "nope.json", {})
    assert alerts == []
    assert any("nothing watched" in ln for ln in info)


def test_poller_crash_becomes_unreachable(tmp_path, bomb):
    def broken(jobids):
        raise RuntimeError("boom")
    sp = write_state(tmp_path, [job()])
    alerts, _ = wd.run_pass(
        state_path=sp, log_path=tmp_path / "alerts.log",
        flag_path=tmp_path / "WATCHDOG_ALERT",
        pollers={"perlmutter": broken, "phoenix": broken},
        now=NOW, root=tmp_path)
    assert conditions(alerts) == ["UNREACHABLE"]
    assert "poller crashed: boom" in alerts[0]["msg"]


def test_unknown_host_is_config_error(tmp_path, bomb):
    sp = write_state(tmp_path, [job(host="frontier")])
    alerts, _ = run(tmp_path, sp, {})
    assert conditions(alerts) == ["CONFIG_ERROR"]


# --------------------------------------------------------- dedup / idempotence

def test_log_dedup_but_flag_retouched(tmp_path, bomb):
    sp = write_state(tmp_path, [job(submitted_at=iso_ago(30.0))])
    mapping = {"111": {"state": "PENDING", "detail": "squeue:PENDING"}}

    run(tmp_path, sp, mapping)                                   # pass 1: logs
    (tmp_path / "WATCHDOG_ALERT").unlink()                       # orchestrator "cleared" it
    run(tmp_path, sp, mapping, now=NOW + timedelta(hours=1))     # pass 2: within realert_h

    log = (tmp_path / "alerts.log").read_text()
    assert log.count("PENDING_TOO_LONG") == 1                    # deduped in the log
    assert (tmp_path / "WATCHDOG_ALERT").exists()                # but re-flagged

    run(tmp_path, sp, mapping, now=NOW + timedelta(hours=7))     # pass 3: past realert_h=6
    assert (tmp_path / "alerts.log").read_text().count("PENDING_TOO_LONG") == 2


def test_clean_pass_is_idempotent_on_state(tmp_path, bomb):
    sp = write_state(tmp_path, [job()])
    mapping = {"111": {"state": "RUNNING", "detail": "squeue:RUNNING", "elapsed_h": 1.0}}
    run(tmp_path, sp, mapping)
    first = json.loads(sp.read_text())
    run(tmp_path, sp, mapping)
    assert json.loads(sp.read_text()) == first
    assert first["updated_at"] == iso_ago(0.0)   # watchdog never touches the heartbeat


# ----------------------------------------------------------------- parse helpers

def test_parse_slurm_elapsed():
    assert wd.parse_slurm_elapsed("1-02:00:00") == pytest.approx(26.0)
    assert wd.parse_slurm_elapsed("12:30:00") == pytest.approx(12.5)
    assert wd.parse_slurm_elapsed("45:00") == pytest.approx(0.75)   # squeue MM:SS
    assert wd.parse_slurm_elapsed("0:07") == pytest.approx(7 / 3600)
    assert wd.parse_slurm_elapsed("") is None
    assert wd.parse_slurm_elapsed("N/A") is None


def test_normalize_slurm_state():
    assert wd.normalize_slurm_state("PENDING") == "PENDING"
    assert wd.normalize_slurm_state("RUNNING") == "RUNNING"
    assert wd.normalize_slurm_state("COMPLETING") == "RUNNING"
    assert wd.normalize_slurm_state("COMPLETED") == "COMPLETED"
    assert wd.normalize_slurm_state("OUT_OF_MEMORY") == "FAILED"
    assert wd.normalize_slurm_state("CANCELLED by 12345") == "FAILED"
    assert wd.normalize_slurm_state("NODE_FAIL+") == "FAILED"
    assert wd.normalize_slurm_state("") == "VANISHED"
    assert wd.normalize_slurm_state("SOME_FUTURE_STATE") == "FAILED"  # fail-loud default
