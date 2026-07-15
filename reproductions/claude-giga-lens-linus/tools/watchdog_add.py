#!/usr/bin/env python3
"""Register/deregister watched jobs in data/watchdog_state.json (see tools/watchdog.py).

Call `add` right after EVERY sbatch / long nohup launch (to be wired into the slurm
submission wrappers). Every add/remove/touch refreshes the state file's updated_at —
that timestamp is the watchdog's dead-agent tripwire (STATE_STALE), so heartbeat with
`touch` at least once per agent session even when not submitting.

Usage:
  python tools/watchdog_add.py add --jobid 55885270 --host perlmutter \\
      --max-pending-h 24 --max-run-h 6 \\
      --expect-artifact data/e2_v3b_low.npz \\
      --on-stall 'resubmit:/global/homes/g/gdbenson/foundry-i/slurm/p1c_v3b.slurm' \\
      --note 'P1c v3b money product'
  python tools/watchdog_add.py add --jobid 412233 --host phoenix --max-run-h 12 \\
      --expect-artifact data/smc_b0_report.json --note 'B0 gates on L4 (PID)'
  python tools/watchdog_add.py remove --jobid 55885270
  python tools/watchdog_add.py list
  python tools/watchdog_add.py touch      # heartbeat only (refresh updated_at)

Notes:
- perlmutter --jobid = slurm job id; phoenix --jobid = local PID of the process.
- --on-stall is 'alert' (default) or 'resubmit:<path to .slurm ON THE JOB'S HOST>'
  (remote perlmutter path for perlmutter jobs; local script path for phoenix).
  Resubmit commands are only EMITTED by the watchdog unless it runs with --execute.
- Re-adding an existing (host, jobid) replaces the entry (idempotent).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# tools/ is not a package; load the sibling watchdog.py by path for the shared
# state-file helpers (single source of truth for schema and IO).
_spec = importlib.util.spec_from_file_location(
    "cgl2_watchdog", Path(__file__).resolve().with_name("watchdog.py"))
wd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wd)


def _load_or_new(path):
    state, err = wd.load_state(path)
    if err is not None:
        raise SystemExit(
            f"ERROR: state file {path} is corrupt ({err}); fix or delete it first.")
    if state is None:
        state = {"updated_at": wd.utcnow().isoformat(), "jobs": []}
    state.setdefault("jobs", [])
    return state


def _persist(state, path):
    state["updated_at"] = wd.utcnow().isoformat()
    wd.save_state(state, path)


def cmd_add(args) -> int:
    if not (args.on_stall == "alert" or args.on_stall.startswith("resubmit:")):
        raise SystemExit("ERROR: --on-stall must be 'alert' or 'resubmit:<slurm file>'")
    if args.submitted_at:
        try:
            wd.parse_ts(args.submitted_at)
        except ValueError as e:
            raise SystemExit(f"ERROR: bad --submitted-at: {e}")
    entry = {
        "jobid": str(args.jobid),
        "host": args.host,
        "submitted_at": args.submitted_at or wd.utcnow().isoformat(),
        "max_pending_h": args.max_pending_h,
        "max_run_h": args.max_run_h,
        "on_stall": args.on_stall,
    }
    if args.expect_artifact:
        entry["expect_artifact"] = args.expect_artifact
    if args.note:
        entry["note"] = args.note
    if args.max_resubmits is not None:
        entry["max_resubmits"] = args.max_resubmits

    state = _load_or_new(args.state)
    before = len(state["jobs"])
    state["jobs"] = [
        j for j in state["jobs"]
        if not (str(j.get("jobid")) == entry["jobid"] and j.get("host") == entry["host"])
    ]
    replaced = before != len(state["jobs"])
    state["jobs"].append(entry)
    _persist(state, args.state)
    print(f"{'replaced' if replaced else 'registered'}: job={entry['jobid']} host={entry['host']} "
          f"max_pending_h={entry['max_pending_h']:g} max_run_h={entry['max_run_h']:g} "
          f"on_stall={entry['on_stall']}"
          + (f" expect_artifact={entry.get('expect_artifact')}" if args.expect_artifact else ""))
    print(f"state: {args.state} ({len(state['jobs'])} job(s) watched)")
    return 0


def cmd_remove(args) -> int:
    state = _load_or_new(args.state)
    before = len(state["jobs"])
    state["jobs"] = [
        j for j in state["jobs"]
        if not (str(j.get("jobid")) == str(args.jobid)
                and (args.host is None or j.get("host") == args.host))
    ]
    removed = before - len(state["jobs"])
    _persist(state, args.state)
    if removed:
        print(f"deregistered {removed} entr{'y' if removed == 1 else 'ies'} for job={args.jobid}")
    else:
        print(f"job={args.jobid} was not watched (no-op)")  # idempotent
    print(f"state: {args.state} ({len(state['jobs'])} job(s) watched)")
    return 0


def cmd_list(args) -> int:
    state, err = wd.load_state(args.state)
    if err is not None:
        raise SystemExit(f"ERROR: state file {args.state} is corrupt ({err}).")
    if state is None or not state.get("jobs"):
        print(f"no jobs watched ({args.state})")
        return 0
    now = wd.utcnow()
    try:
        upd_age = wd.hours_between(wd.parse_ts(state["updated_at"]), now)
        print(f"updated_at: {state['updated_at']} ({upd_age:.1f}h ago)")
    except (KeyError, ValueError, TypeError):
        print("updated_at: MISSING/UNPARSEABLE (STATE_STALE will fire)")
    for j in state["jobs"]:
        try:
            age = f"{wd.hours_between(wd.parse_ts(j['submitted_at']), now):.1f}h ago"
        except (KeyError, ValueError, TypeError):
            age = "submitted_at BAD"
        print(f"  job={j.get('jobid')} host={j.get('host')} submitted={age} "
              f"max_pending_h={j.get('max_pending_h')} max_run_h={j.get('max_run_h')} "
              f"on_stall={j.get('on_stall', 'alert')} "
              f"artifact={j.get('expect_artifact', '-')} note={j.get('note', '-')}")
    return 0


def cmd_touch(args) -> int:
    state = _load_or_new(args.state)
    _persist(state, args.state)
    print(f"heartbeat: updated_at refreshed ({len(state['jobs'])} job(s) watched)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=str(wd.STATE_FILE), help="state file path override")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="register (or replace) a watched job")
    a.add_argument("--jobid", required=True, help="slurm job id (perlmutter) or PID (phoenix)")
    a.add_argument("--host", required=True, choices=["perlmutter", "phoenix"])
    a.add_argument("--submitted-at", default=None,
                   help="ISO8601 (naive == UTC); default: now")
    a.add_argument("--max-pending-h", type=float, default=wd.DEFAULT_MAX_PENDING_H)
    a.add_argument("--max-run-h", type=float, default=wd.DEFAULT_MAX_RUN_H)
    a.add_argument("--expect-artifact", default=None,
                   help="path that must exist once the job is done (relative to repo root)")
    a.add_argument("--on-stall", default="alert",
                   help="'alert' (default) or 'resubmit:<slurm file on the job's host>'")
    a.add_argument("--max-resubmits", type=int, default=None,
                   help=f"cap on --execute resubmits (default {wd.DEFAULT_MAX_RESUBMITS})")
    a.add_argument("--note", default=None)
    a.set_defaults(fn=cmd_add)

    r = sub.add_parser("remove", help="deregister a watched job (idempotent)")
    r.add_argument("--jobid", required=True)
    r.add_argument("--host", default=None, choices=["perlmutter", "phoenix"],
                   help="disambiguate if the same id exists on both hosts")
    r.set_defaults(fn=cmd_remove)

    l = sub.add_parser("list", help="show watched jobs")
    l.set_defaults(fn=cmd_list)

    t = sub.add_parser("touch", help="heartbeat: refresh updated_at (run each agent session)")
    t.set_defaults(fn=cmd_touch)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
