#!/usr/bin/env python3
"""Dead-man watchdog for long autonomous runs (campaign task, PLAN §2 "watchdog armed").

Why this exists — the P1c incident (../claude-giga-lens/CAMPAIGN.md, 2026-07-13/14):
a Perlmutter SMC canary (55885270) OOM'd (120.4 GB alloc with 300 particles) AND the
babysitting agent died silently during a multi-day gap, so its SMC_PARTICLES=200
fallback never fired -> P1c stalled 2026-07-10 -> 07-14. Nobody caught it for 4 days.
This watchdog guards against that failure class: a dumb, stdlib-only, single-pass
poller that turns "job quietly dead / agent quietly dead" into a loud flag file.

Design (deliberately boring): stdlib only, no daemons, no threads, no backoff logic.
ONE poll pass per invocation — run it in a loop from tmux/nohup (see tools/README.md).
Any python3 (>=3.9) works; it never imports jax/cgl2, so the system python is fine.
Idempotent: re-running changes nothing except refreshed polls; repeated alerts for the
same (job, condition) are deduplicated in the log for `realert_h` (default 6 h) but
re-touch the flag file every pass so a cleared-but-unfixed alert re-flags.

Files (all under the campaign data/ dir, gitignored):
  data/watchdog_state.json   watched jobs — register with tools/watchdog_add.py
  data/watchdog_alerts.log   append-only alert history
  data/WATCHDOG_ALERT        flag file, (re)written whenever ANY alert is active.
                             The orchestrating agent checks for it each session and
                             DELETES it after triage — the watchdog never deletes it.

State file schema (JSON):
  {
    "updated_at": "<ISO8601, refreshed ONLY by watchdog_add.py add/remove/touch>",
    "stale_state_h": 48,          # optional, default DEFAULT_STALE_STATE_H
    "realert_h": 6,               # optional, default DEFAULT_REALERT_H
    "jobs": [
      {"jobid": "55885270",       # slurm job id (perlmutter) or local PID (phoenix)
       "host": "perlmutter",      # "perlmutter" | "phoenix"
       "submitted_at": "<ISO8601, naive == UTC>",
       "max_pending_h": 24,       # alert if queued longer than this
       "max_run_h": 6,            # alert if running longer than this
       "expect_artifact": "data/e2_v3b_low.npz",   # optional; relative to repo root
       "on_stall": "alert",       # or "resubmit:<path to .slurm ON THE JOB'S HOST>"
       "max_resubmits": 1,        # optional cap, default DEFAULT_MAX_RESUBMITS
       "note": "P1c v3b money"},  # optional, free text
      ...
    ],
    "alert_history": { ... }      # maintained by the watchdog (log dedup)
  }

Detected conditions -> alerts:
  PENDING_TOO_LONG       queued longer than max_pending_h
  RUNNING_TOO_LONG       running longer than max_run_h
  JOB_FAILED             slurm terminal-bad state (OUT_OF_MEMORY/FAILED/TIMEOUT/
                         CANCELLED/NODE_FAIL/...) — the exact P1c OOM case
  COMPLETED_NO_ARTIFACT  COMPLETED but expect_artifact missing on disk
  VANISHED_NO_ARTIFACT   job in neither squeue nor sacct (or PID gone) and
                         expect_artifact missing/unspecified
  UNREACHABLE            poll failure (ssh cert expiry, timeout, ...) — job status
                         UNKNOWN; alerts, never crashes, never treated as vanished
  STATE_STALE            state file updated_at older than stale_state_h while jobs
                         are watched — the "orchestrating agent died" tripwire
                         (heartbeat each session: python tools/watchdog_add.py touch)
  STATE_CORRUPT          state file unparseable
  CONFIG_ERROR           bad job entry (unknown host, bad submitted_at, ...)

Resubmit semantics: on_stall="resubmit:<file>" attaches the exact resubmit command to
dead-job alerts (JOB_FAILED / VANISHED_NO_ARTIFACT / COMPLETED_NO_ARTIFACT — never to
*_TOO_LONG, where the job is still alive). DRY-RUN BY DEFAULT: the command is emitted
(stdout + log + alert dict), NOT executed. Pass --execute to actually run it, capped
at max_resubmits per entry (budget discipline: an auto-resubmit spends A100-h; the
ledger row must exist BEFORE results are read — see CAMPAIGN.md house rule).

Exit code: 0 clean, 2 if any alert is active (loop scripts may ignore it).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # reproductions/claude-giga-lens-linus
DATA_DIR = ROOT / "data"
STATE_FILE = DATA_DIR / "watchdog_state.json"
ALERT_LOG = DATA_DIR / "watchdog_alerts.log"
ALERT_FLAG = DATA_DIR / "WATCHDOG_ALERT"

PERLMUTTER_SSH_DEST = "gdbenson@perlmutter.nersc.gov"
SSH_CONNECT_TIMEOUT_S = 30
SSH_TOTAL_TIMEOUT_S = 90
_SSH_MARKER = "---CGL2-WATCHDOG---"

DEFAULT_MAX_PENDING_H = 24.0
DEFAULT_MAX_RUN_H = 24.0
DEFAULT_STALE_STATE_H = 48.0
DEFAULT_REALERT_H = 6.0
DEFAULT_MAX_RESUBMITS = 1
_HISTORY_KEEP_DAYS = 14

# slurm state normalization (sacct states may carry suffixes: "CANCELLED by 123", "NODE_FAIL+")
_PEND_STATES = {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD", "SUSPENDED"}
_RUN_STATES = {"RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING", "RESIZING"}
_BAD_STATES = {
    "FAILED", "OUT_OF_MEMORY", "TIMEOUT", "CANCELLED", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT",
}
# conditions where the job is definitively dead (resubmit-eligible)
RESUBMIT_CONDITIONS = {"JOB_FAILED", "VANISHED_NO_ARTIFACT", "COMPLETED_NO_ARTIFACT"}


# --------------------------------------------------------------------------- helpers

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(s: str) -> datetime:
    """ISO8601 -> aware datetime; naive timestamps are UTC by contract."""
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_between(earlier: datetime, later: datetime) -> float:
    return (later - earlier).total_seconds() / 3600.0


def parse_slurm_elapsed(s: str):
    """'D-HH:MM:SS' | 'HH:MM:SS' | 'MM:SS' | 'SS' -> hours, or None if unparseable.

    Covers both sacct Elapsed ([DD-]HH:MM:SS) and squeue %M (days-hours:min:sec,
    where a 2-field value means MM:SS).
    """
    s = (s or "").strip()
    if not s:
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
    try:
        parts = [int(p) for p in s.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, (m, sec) = 0, parts
    elif len(parts) == 1:
        h, m, sec = 0, 0, parts[0]
    else:
        return None
    return days * 24.0 + h + m / 60.0 + sec / 3600.0


def normalize_slurm_state(raw: str) -> str:
    """Map a raw squeue/sacct state to PENDING|RUNNING|COMPLETED|FAILED|VANISHED.

    Unknown non-empty states map to FAILED (a false alert is the safe direction
    for a dead-man watchdog; the raw state rides along in the alert detail).
    """
    toks = (raw or "").strip().split()
    tok = toks[0].rstrip("+").upper() if toks else ""
    if not tok:
        return "VANISHED"
    if tok in _PEND_STATES:
        return "PENDING"
    if tok in _RUN_STATES:
        return "RUNNING"
    if tok == "COMPLETED":
        return "COMPLETED"
    return "FAILED"


def _unreachable(detail: str) -> dict:
    return {"state": "UNREACHABLE", "detail": detail, "elapsed_h": None}


# --------------------------------------------------------------------------- pollers
# A poller takes a list of jobid strings and returns
#   {jobid: {"state": PENDING|RUNNING|COMPLETED|FAILED|VANISHED|UNREACHABLE,
#            "detail": str, "elapsed_h": float|None}}
# Tests inject fakes here; run_pass(pollers=...) never touches ssh/ps itself.

def poll_perlmutter(jobids):
    """One batched ssh round-trip: squeue (live view) + sacct (terminal history).

    ssh failure (cert expiry, timeout, network) -> every job UNREACHABLE. We prove
    the remote shell actually ran via an echoed marker; ssh's own rc is unreliable
    because the compound remote command's rc is sacct's.
    """
    ids = ",".join(str(j) for j in jobids)
    remote = (
        f"squeue -h -j {ids} -o '%i|%T|%M' 2>/dev/null; "
        f"echo {_SSH_MARKER}; "
        f"sacct -n -X -P -j {ids} -o JobID,State,Elapsed 2>/dev/null"
    )
    cmd = [
        "ssh", "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
        PERLMUTTER_SSH_DEST, remote,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SSH_TOTAL_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return {str(j): _unreachable(f"ssh timeout after {SSH_TOTAL_TIMEOUT_S}s") for j in jobids}
    except OSError as e:
        return {str(j): _unreachable(f"ssh exec failed: {e}") for j in jobids}

    if _SSH_MARKER not in proc.stdout:
        err = (proc.stderr or "").strip().splitlines()
        detail = err[-1][:200] if err else f"ssh rc={proc.returncode}, no output"
        return {str(j): _unreachable(f"ssh failed (cert expired? run sshproxy): {detail}") for j in jobids}

    sq_txt, sa_txt = proc.stdout.split(_SSH_MARKER, 1)
    found = {}
    # sacct first (terminal states), then squeue overrides (live view wins)
    for src, txt in (("sacct", sa_txt), ("squeue", sq_txt)):
        for line in txt.strip().splitlines():
            f = line.split("|")
            if len(f) >= 3 and f[0].strip():
                found[f[0].strip()] = {
                    "state": normalize_slurm_state(f[1]),
                    "detail": f"{src}:{f[1].strip()}",
                    "elapsed_h": parse_slurm_elapsed(f[2]),
                }
    return {
        str(j): found.get(
            str(j),
            {"state": "VANISHED", "detail": "in neither squeue nor sacct", "elapsed_h": None},
        )
        for j in jobids
    }


def poll_phoenix(pids):
    """Local liveness via ps; nvidia-smi is best-effort garnish (on-GPU tag)."""
    gpu_pids = set()
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if q.returncode == 0:
            gpu_pids = {ln.strip() for ln in q.stdout.splitlines() if ln.strip()}
    except Exception:
        pass  # informational only; never blocks liveness detection

    out = {}
    for pid in pids:
        pid = str(pid)
        try:
            p = subprocess.run(
                ["ps", "-p", pid, "-o", "etimes=,comm="],
                capture_output=True, text=True, timeout=20,
            )
        except Exception as e:
            out[pid] = _unreachable(f"ps failed: {e}")
            continue
        line = p.stdout.strip()
        if p.returncode != 0 or not line:
            out[pid] = {"state": "VANISHED", "detail": "pid not in ps", "elapsed_h": None}
            continue
        parts = line.split(None, 1)
        try:
            elapsed_h = int(parts[0]) / 3600.0
        except (ValueError, IndexError):
            elapsed_h = None
        comm = parts[1] if len(parts) > 1 else "?"
        gpu = " on-GPU" if pid in gpu_pids else ""
        out[pid] = {"state": "RUNNING", "detail": f"ps:{comm}{gpu}", "elapsed_h": elapsed_h}
    return out


def default_pollers():
    return {"perlmutter": poll_perlmutter, "phoenix": poll_phoenix}


# --------------------------------------------------------------------------- state IO

def load_state(path=STATE_FILE):
    """-> (state|None, error|None). Missing file is (None, None); corrupt is (None, msg)."""
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text()), None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        return None, str(e)


def save_state(state, path=STATE_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)  # atomic on POSIX


# ------------------------------------------------------------------------- evaluation

def _mk_alert(cond: str, job: dict, msg: str) -> dict:
    return {
        "condition": cond,
        "jobid": str(job.get("jobid", "?")),
        "host": str(job.get("host", "?")),
        "msg": msg,
    }


def evaluate_job(job: dict, poll: dict, now: datetime, root=ROOT):
    """One job vs one poll result -> (alerts, keep_in_state, info_lines).

    Side-effect free except for the expect_artifact existence check.
    """
    alerts, info = [], []
    keep = True
    jid = str(job.get("jobid", "?"))
    state = poll.get("state", "UNREACHABLE")
    detail = poll.get("detail", "")
    elapsed_h = poll.get("elapsed_h")

    try:
        sub = parse_ts(job["submitted_at"])
    except (KeyError, ValueError, TypeError) as e:
        return [_mk_alert("CONFIG_ERROR", job, f"bad/missing submitted_at: {e}")], True, info
    age_h = hours_between(sub, now)

    art = job.get("expect_artifact")
    art_path = None
    if art:
        art_path = Path(art) if Path(art).is_absolute() else Path(root) / art
    art_ok = art_path.exists() if art_path is not None else None

    if state == "UNREACHABLE":
        alerts.append(_mk_alert(
            "UNREACHABLE", job,
            f"poll failed: {detail} — job status UNKNOWN (not treated as vanished)"))
    elif state == "PENDING":
        mp = float(job.get("max_pending_h", DEFAULT_MAX_PENDING_H))
        if age_h > mp:
            alerts.append(_mk_alert(
                "PENDING_TOO_LONG", job,
                f"pending {age_h:.1f}h > max_pending_h={mp:g} ({detail})"))
        else:
            info.append(f"job={jid} PENDING {age_h:.1f}h/{mp:g}h ok ({detail})")
    elif state == "RUNNING":
        mr = float(job.get("max_run_h", DEFAULT_MAX_RUN_H))
        run_h = elapsed_h if elapsed_h is not None else age_h
        proxy = "" if elapsed_h is not None else " [wall-age proxy incl. queue time]"
        if run_h > mr:
            alerts.append(_mk_alert(
                "RUNNING_TOO_LONG", job,
                f"running {run_h:.1f}h > max_run_h={mr:g}{proxy} ({detail})"))
        else:
            info.append(f"job={jid} RUNNING {run_h:.1f}h/{mr:g}h ok{proxy} ({detail})")
    elif state == "COMPLETED":
        if art_path is None or art_ok:
            suffix = f", artifact {art} present" if art else " (no artifact expected)"
            info.append(f"job={jid} COMPLETED{suffix} -> deregistered")
            keep = False
        else:
            alerts.append(_mk_alert(
                "COMPLETED_NO_ARTIFACT", job,
                f"COMPLETED but expected artifact missing: {art} ({detail})"))
    elif state == "FAILED":
        alerts.append(_mk_alert("JOB_FAILED", job, f"terminal state: {detail}"))
    elif state == "VANISHED":
        if art_path is not None and art_ok:
            info.append(f"job={jid} gone but artifact {art} present -> assuming success, deregistered")
            keep = False
        else:
            missing = f"missing: {art}" if art else "not specified"
            alerts.append(_mk_alert(
                "VANISHED_NO_ARTIFACT", job,
                f"job not found by poller and expected artifact {missing} ({detail})"))
    else:
        alerts.append(_mk_alert("CONFIG_ERROR", job, f"poller returned unknown state {state!r}"))

    return alerts, keep, info


def resubmit_command(job: dict):
    """The exact command that would resubmit this job, or None if on_stall is alert-only."""
    spec = str(job.get("on_stall", "alert"))
    if not spec.startswith("resubmit:"):
        return None
    target = spec.split(":", 1)[1].strip()
    if not target:
        return None
    if job.get("host") == "perlmutter":
        # target is the REMOTE path of the .slurm file on perlmutter
        return f"ssh {PERLMUTTER_SSH_DEST} 'sbatch {target}'"
    return f"bash {target}"  # phoenix: a local (re)launch script


def execute_resubmit(cmd: str):
    """Actually run an emitted resubmit command (only under --execute).

    -> (ok, msg, new_jobid|None); new_jobid parsed from 'Submitted batch job NNN'.
    """
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
    except Exception as e:
        return False, f"exec failed: {e}", None
    out = ((proc.stdout or "") + " " + (proc.stderr or "")).strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if proc.returncode == 0:
        return True, out[:200], (m.group(1) if m else None)
    return False, f"rc={proc.returncode}: {out[:200]}", None


# --------------------------------------------------------------------------- the pass

def run_pass(state_path=STATE_FILE, log_path=ALERT_LOG, flag_path=ALERT_FLAG,
             pollers=None, now=None, execute=False, root=ROOT):
    """One idempotent poll pass. Returns (alerts, info_lines).

    Tests inject `pollers` (dict host->callable) and `now`; production uses defaults.
    """
    now = now or utcnow()
    pollers = pollers if pollers is not None else default_pollers()
    state_path, log_path, flag_path = Path(state_path), Path(log_path), Path(flag_path)

    state, err = load_state(state_path)
    alerts, info = [], []
    kept = None

    if err is not None:
        state = None
        alerts.append({
            "condition": "STATE_CORRUPT", "jobid": "-", "host": "-",
            "msg": f"cannot parse {state_path}: {err} — fix by hand or re-register via watchdog_add.py",
        })
    elif state is None:
        info.append(f"no state file at {state_path} — nothing watched")
    else:
        jobs = state.get("jobs", [])
        if not isinstance(jobs, list):
            jobs = []
            alerts.append({
                "condition": "STATE_CORRUPT", "jobid": "-", "host": "-",
                "msg": f"'jobs' is not a list in {state_path}",
            })

        # --- dead-agent tripwire: nobody has add/remove/touch'ed the file recently
        if jobs:
            stale_h = float(state.get("stale_state_h", DEFAULT_STALE_STATE_H))
            try:
                upd_age = hours_between(parse_ts(state["updated_at"]), now)
            except (KeyError, ValueError, TypeError):
                upd_age = None
            if upd_age is None:
                alerts.append({
                    "condition": "STATE_STALE", "jobid": "-", "host": "-",
                    "msg": "state file has no parseable updated_at — re-register or "
                           "heartbeat via watchdog_add.py touch",
                })
            elif upd_age > stale_h:
                alerts.append({
                    "condition": "STATE_STALE", "jobid": "-", "host": "-",
                    "msg": f"state file updated_at is {upd_age:.1f}h old (> {stale_h:g}h) with "
                           f"{len(jobs)} job(s) watched — is the orchestrating agent alive? "
                           "(heartbeat: python tools/watchdog_add.py touch)",
                })

        # --- poll, batched per host (one ssh round-trip for all perlmutter jobs)
        by_host = {}
        for job in jobs:
            by_host.setdefault(str(job.get("host", "?")), []).append(str(job.get("jobid", "?")))
        polled = {}
        for host, ids in by_host.items():
            fn = pollers.get(host)
            if fn is None:
                for j in ids:
                    polled[(host, j)] = None  # unknown host -> CONFIG_ERROR below
                continue
            try:
                res = fn(ids)
            except Exception as e:  # a poller bug must not kill the watchdog
                res = {j: _unreachable(f"poller crashed: {e}") for j in ids}
            for j in ids:
                polled[(host, j)] = res.get(str(j), _unreachable("poller returned no entry"))

        # --- evaluate
        kept = []
        for job in jobs:
            host, jid = str(job.get("host", "?")), str(job.get("jobid", "?"))
            poll = polled.get((host, jid))
            if poll is None:
                alerts.append(_mk_alert(
                    "CONFIG_ERROR", job, f"unknown host {host!r} (need perlmutter|phoenix)"))
                kept.append(job)
                continue
            job_alerts, keep, job_info = evaluate_job(job, poll, now, root=root)

            # resubmit handling: only for definitively-dead conditions
            for al in job_alerts:
                if al["condition"] not in RESUBMIT_CONDITIONS:
                    continue
                cmd = resubmit_command(job)
                if cmd is None:
                    continue
                n = int(job.get("resubmit_count", 0))
                cap = int(job.get("max_resubmits", DEFAULT_MAX_RESUBMITS))
                if n >= cap:
                    al["msg"] += f" | resubmit exhausted ({n}/{cap}); manual action required"
                elif execute:
                    ok, msg, new_id = execute_resubmit(cmd)
                    job["resubmit_count"] = n + 1
                    if ok:
                        al["msg"] += f" | RESUBMITTED ({n + 1}/{cap}): {cmd} -> {msg}"
                        if new_id:
                            al["msg"] += f" | now watching new jobid {new_id}"
                            job["jobid"] = new_id
                        job["submitted_at"] = now.isoformat()
                    else:
                        al["msg"] += f" | RESUBMIT FAILED: {msg}"
                else:
                    al["resubmit_cmd"] = cmd
                    al["msg"] += f" | RESUBMIT (dry-run, NOT executed; pass --execute to run): {cmd}"

            alerts.extend(job_alerts)
            info.extend(job_info)
            if keep:
                kept.append(job)

    # --- emit: stdout always; log deduped per (host,jobid,condition) for realert_h;
    #           flag file (re)written on ANY active alert; state persisted.
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    for line in info:
        print(f"[watchdog {ts}] {line}")

    hist = dict(state.get("alert_history", {})) if isinstance(state, dict) else {}
    realert_h = (
        float(state.get("realert_h", DEFAULT_REALERT_H))
        if isinstance(state, dict) else DEFAULT_REALERT_H
    )
    log_lines = []
    for al in alerts:
        line = f"{ts} ALERT {al['condition']} job={al['jobid']} host={al['host']} :: {al['msg']}"
        print(f"[watchdog {ts}] ALERT {al['condition']} job={al['jobid']} host={al['host']} :: {al['msg']}")
        key = f"{al['host']}:{al['jobid']}:{al['condition']}"
        suppress = False
        if key in hist:
            try:
                suppress = hours_between(parse_ts(hist[key]), now) < realert_h
            except ValueError:
                pass
        if not suppress:
            log_lines.append(line)
            hist[key] = now.isoformat()
    if log_lines:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write("\n".join(log_lines) + "\n")
    if alerts:
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.write_text(
            f"{ts} {len(alerts)} active alert(s) — triage {log_path}, then delete this file\n"
            + "".join(f"- {a['condition']} job={a['jobid']} ({a['host']})\n" for a in alerts)
        )

    n_watched = len(state.get("jobs", [])) if isinstance(state, dict) else 0
    print(f"[watchdog {ts}] pass done: {n_watched} watched, {len(alerts)} alert(s)"
          + ("" if alerts else " — clean"))

    if isinstance(state, dict):
        if kept is not None:
            state["jobs"] = kept
        cutoff = now - timedelta(days=_HISTORY_KEEP_DAYS)
        pruned = {}
        for k, v in hist.items():
            try:
                if parse_ts(v) >= cutoff:
                    pruned[k] = v
            except ValueError:
                pass
        state["alert_history"] = pruned
        save_state(state, state_path)  # preserves updated_at: only watchdog_add touches it

    return alerts, info


# -------------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Dead-man watchdog: one poll pass over data/watchdog_state.json "
                    "(run in a loop; see tools/README.md).")
    ap.add_argument("--execute", action="store_true",
                    help="actually run emitted resubmit commands (default: dry-run emission only)")
    ap.add_argument("--state", default=str(STATE_FILE), help="state file path override")
    ap.add_argument("--log", default=str(ALERT_LOG), help="alert log path override")
    ap.add_argument("--flag", default=str(ALERT_FLAG), help="alert flag path override")
    args = ap.parse_args(argv)
    alerts, _ = run_pass(
        state_path=args.state, log_path=args.log, flag_path=args.flag, execute=args.execute
    )
    return 2 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
