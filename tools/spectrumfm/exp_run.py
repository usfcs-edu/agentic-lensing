#!/usr/bin/env python3
"""
exp_run.py — launch a SpectrumFM training experiment from a yaml spec.

Reads experiments/specs/<name>.yaml, resolves the repo's venv, launches the
training script as a subprocess, tees stdout to a per-run log file while
parsing metric lines into a structured metrics.jsonl, and writes a final
summary.md in the style of redshifty/RESEARCH_LOG.md.

Usage:
    exp_run.py experiments/specs/redshifty_smoke.yaml
    exp_run.py experiments/specs/codecs_smoke.yaml --run-id my_label
    exp_run.py spec.yaml --dry-run                # print resolved command, don't launch

Spec yaml schema (see experiments/specs/ for examples):

    name: redshifty_smoke
    repo: redshifty | codecs                    # selects venv + cwd
    launch:
      command: [python, scripts/smoke_test.py]  # command[0] resolved against <venv>/bin/
    env: {WANDB_MODE: offline}                  # merged into subprocess env
    description: "free-text"
    tags: [smoke, track2-mvp]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "experiments" / "runs"

REPOS = {
    "redshifty": {
        "venv": Path.home() / ".venvs" / "redshifty",
        "cwd":  REPO_ROOT / "lensing-repos" / "redshifty",
    },
    "codecs": {
        "venv": Path.home() / ".venvs" / "codecs",
        "cwd":  REPO_ROOT / "lensing-repos" / "codecs",
    },
}


# ---------------------------------------------------------------------------
# Metric stream parsers — one per repo
# ---------------------------------------------------------------------------

# codecs lines look like:
#   "  step      2  train_loss=941.00  lr=1.00e-04  val_nll=161.8750  val_r2=-7.7109 ..."
CODECS_STEP_RE = re.compile(
    r"^\s*step\s+(?P<step>\d+)\s+(?P<rest>train_loss=.+)$"
)
CODECS_KV_RE = re.compile(r"(\w+)=([-\d.eE+nan]+)")

# redshifty/scripts/train.py epoch summary spans 3 lines:
#   "Epoch 2/3 (1.0s)"
#   "  Train: loss=17.8549, acc=0.297, redshift_acc=0.000"
#   "  Val:   loss=19.2841, acc=0.189, redshift_acc=0.000"
RED_EPOCH_RE = re.compile(r"^Epoch (?P<epoch>\d+)/(?P<total>\d+)\s+\((?P<sec>[\d.]+)s\)\s*$")
RED_KV_RE = re.compile(r"(\w+)=([-\d.eE+]+)")
RED_APPROACH_RE = re.compile(r"^Training Approach\s+(?P<a>[AB])\s*$")

# pretrain_tokenizer.py and train_transformer.py both print:
#   "[step   1000] key=val key=val ..."     (optional " [AR]" tag after the step)
#   "[val     500] key=val key=val ..."
# Permissive: capture step + remainder, harvest all `key=val` floats from rest.
RED_PRETRAIN_STEP_RE = re.compile(r"^\[step\s+(?P<step>\d+)(?:\s+\[AR\])?\]\s+(?P<rest>.+)$")
RED_PRETRAIN_VAL_RE = re.compile(r"^\[val\s+(?P<step>\d+)\]\s+(?P<rest>.+)$")


class MetricParser:
    """Stateful per-line parser. emit(line) returns a list of metric dicts."""

    def __init__(self, repo: str):
        self.repo = repo
        # redshifty epoch-summary state
        self._epoch_header: dict | None = None
        self._epoch_train: dict | None = None
        self._approach: str | None = None  # 'a' / 'b' / None

    def emit(self, line: str) -> list[dict[str, Any]]:
        line = line.rstrip("\n")
        if self.repo == "codecs":
            return self._emit_codecs(line)
        if self.repo == "redshifty":
            return self._emit_redshifty(line)
        return []

    def _emit_codecs(self, line: str) -> list[dict]:
        m = CODECS_STEP_RE.match(line)
        if not m:
            return []
        rec = {
            "kind": "step",
            "step": int(m.group("step")),
        }
        for k, v in CODECS_KV_RE.findall(m.group("rest")):
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v
        return [rec]

    def _emit_redshifty(self, line: str) -> list[dict]:
        # pretrain_tokenizer.py / train_transformer.py step line
        m = RED_PRETRAIN_STEP_RE.match(line)
        if m:
            rec: dict = {"kind": "step", "step": int(m.group("step"))}
            for k, v in RED_KV_RE.findall(m.group("rest")):
                try:
                    rec[f"train_{k}"] = float(v)
                except ValueError:
                    pass
            return [rec]

        # pretrain_tokenizer.py / train_transformer.py val line
        m = RED_PRETRAIN_VAL_RE.match(line)
        if m:
            rec = {"kind": "val", "step": int(m.group("step"))}
            for k, v in RED_KV_RE.findall(m.group("rest")):
                try:
                    rec[f"val_{k}"] = float(v)
                except ValueError:
                    pass
            return [rec]

        ap = RED_APPROACH_RE.match(line)
        if ap:
            self._approach = ap.group("a").lower()
            self._epoch_header = None
            self._epoch_train = None
            return []

        m = RED_EPOCH_RE.match(line)
        if m:
            self._epoch_header = {
                "epoch": int(m.group("epoch")),
                "total_epochs": int(m.group("total")),
                "epoch_sec": float(m.group("sec")),
            }
            self._epoch_train = None
            return []

        stripped = line.strip()
        if self._epoch_header is not None and stripped.startswith("Train:"):
            kvs = dict(RED_KV_RE.findall(stripped[6:]))
            self._epoch_train = {f"train_{k}": float(v) for k, v in kvs.items()}
            return []

        if self._epoch_header is not None and self._epoch_train is not None and stripped.startswith("Val:"):
            kvs = dict(RED_KV_RE.findall(stripped[4:]))
            val = {f"val_{k}": float(v) for k, v in kvs.items()}
            rec = {"kind": "epoch", **self._epoch_header, **self._epoch_train, **val}
            if self._approach is not None:
                rec["approach"] = self._approach
            # reset
            self._epoch_header = None
            self._epoch_train = None
            return [rec]

        return []


# ---------------------------------------------------------------------------
# Run-id + dirs
# ---------------------------------------------------------------------------

def short_hash(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_run_id(spec_name: str, override: str | None) -> str:
    if override:
        return override
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{spec_name}_{short_hash()}"


# ---------------------------------------------------------------------------
# Spec resolution + command build
# ---------------------------------------------------------------------------

def resolve_command(spec: dict) -> tuple[list[str], Path, dict[str, str]]:
    """Returns (argv, cwd, env_overrides). Resolves command[0] against the venv."""
    repo = spec.get("repo")
    if repo not in REPOS:
        raise SystemExit(f"spec.repo must be one of {list(REPOS)}, got {repo!r}")
    venv = REPOS[repo]["venv"]
    cwd = REPOS[repo]["cwd"]

    launch = spec.get("launch") or {}
    cmd = list(launch.get("command") or [])
    if not cmd:
        raise SystemExit("spec.launch.command is required and must be a non-empty list")

    head, *tail = cmd
    head_path = venv / "bin" / head
    if not head_path.exists():
        raise SystemExit(f"{head_path} not found — venv missing or wrong tool name")
    argv = [str(head_path), *tail]

    env_overrides = dict(spec.get("env") or {})
    return argv, cwd, env_overrides


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def summarize_metrics(records: list[dict]) -> dict:
    """Build a coarse summary stat: peak/final per known metric."""
    if not records:
        return {}
    summary: dict[str, Any] = {"n_records": len(records)}

    by_approach: dict[str | None, list[dict]] = {}
    for r in records:
        by_approach.setdefault(r.get("approach"), []).append(r)

    for ap, recs in by_approach.items():
        prefix = f"approach_{ap}_" if ap else ""
        # determine x-axis (step or epoch)
        for key in ("step", "epoch"):
            xs = [r[key] for r in recs if key in r]
            if xs:
                summary[f"{prefix}{key}_max"] = max(xs)
                summary[f"{prefix}{key}_n"] = len(xs)
                break

        # for every known metric, record final + best
        metric_names = set()
        for r in recs:
            for k in r:
                if k in ("kind", "step", "epoch", "total_epochs", "epoch_sec", "approach"):
                    continue
                if isinstance(r[k], (int, float)):
                    metric_names.add(k)

        for m in sorted(metric_names):
            vals = [r[m] for r in recs if m in r]
            if not vals:
                continue
            summary[f"{prefix}{m}_first"] = vals[0]
            summary[f"{prefix}{m}_final"] = vals[-1]
            summary[f"{prefix}{m}_min"]   = min(vals)
            summary[f"{prefix}{m}_max"]   = max(vals)

    return summary


def write_summary_md(run_dir: Path, spec: dict, summary: dict, argv: list[str],
                     started: str, finished: str, wallclock_s: float,
                     exit_code: int) -> None:
    name = spec.get("name", run_dir.name)
    desc = spec.get("description", "")
    tags = spec.get("tags") or []
    repo = spec.get("repo", "?")

    lines: list[str] = []
    lines.append(f"# {name}")
    lines.append("")
    if desc:
        lines.append(f"_{desc}_")
        lines.append("")
    lines.append(f"- **run dir:** `{run_dir}`")
    lines.append(f"- **repo:** {repo}")
    lines.append(f"- **started:** {started}")
    lines.append(f"- **finished:** {finished}")
    lines.append(f"- **wallclock:** {wallclock_s:.1f}s")
    lines.append(f"- **exit code:** {exit_code}")
    if tags:
        lines.append(f"- **tags:** {', '.join(tags)}")
    lines.append("")
    lines.append("## Command")
    lines.append("```")
    lines.append(" ".join(argv))
    lines.append("```")
    lines.append("")

    # Trajectory table — pick interesting columns per repo
    table_rows = _trajectory_rows(run_dir, repo)
    if table_rows:
        lines.append("## Trajectory")
        lines.append("")
        header = table_rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for row in table_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if summary:
        lines.append("## Summary stats")
        lines.append("")
        lines.append("```yaml")
        # render summary keys deterministically
        for k in sorted(summary):
            v = summary[k]
            if isinstance(v, float):
                lines.append(f"{k}: {v:.6g}")
            else:
                lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _trajectory_rows(run_dir: Path, repo: str) -> list[list[str]]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not recs:
        return []

    if repo == "codecs":
        header = ["step", "train_loss", "val_nll", "val_r2", "val_mask_bce", "val_perplexity"]
    else:  # redshifty
        header = ["approach", "epoch", "train_loss", "train_acc", "train_redshift_acc",
                  "val_loss", "val_acc", "val_redshift_acc"]
    rows = [header]
    for r in recs:
        row = []
        for col in header:
            if col not in r:
                row.append("")
                continue
            v = r[col]
            if isinstance(v, float):
                row.append(f"{v:.4f}")
            else:
                row.append(str(v))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path, help="Path to experiment yaml spec")
    ap.add_argument("--run-id", default=None, help="Override run id (default: timestamped)")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Where to write the run directory")
    ap.add_argument("--dry-run", action="store_true", help="Resolve + print the command but do not launch")
    ap.add_argument("--no-stream", action="store_true", help="Do not tee child stdout to this process's stdout")
    args = ap.parse_args()

    spec_path = args.spec.resolve()
    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        return 2
    spec = yaml.safe_load(spec_path.read_text())
    if not isinstance(spec, dict):
        print(f"spec yaml must be a mapping at top level: {spec_path}", file=sys.stderr)
        return 2

    argv, cwd, env_overrides = resolve_command(spec)

    run_id = make_run_id(spec.get("name", "exp"), args.run_id)
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # freeze inputs
    shutil.copy2(spec_path, run_dir / "spec.yaml")
    (run_dir / "command.txt").write_text(" ".join(argv) + "\n")

    if args.dry_run:
        print(f"[dry-run] run_dir = {run_dir}")
        print(f"[dry-run] cwd     = {cwd}")
        print(f"[dry-run] env     = {env_overrides}")
        print(f"[dry-run] argv    = {argv}")
        return 0

    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})

    parser = MetricParser(spec["repo"])
    metrics_path = run_dir / "metrics.jsonl"
    stdout_log = run_dir / "stdout.log"

    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    t0 = time.monotonic()

    print(f"[exp_run] launching {spec.get('name', run_id)} → {run_dir}", flush=True)
    print(f"[exp_run] cwd  = {cwd}", flush=True)
    print(f"[exp_run] argv = {argv}", flush=True)

    with metrics_path.open("w") as mf, stdout_log.open("w") as lf:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                if not args.no_stream:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                for rec in parser.emit(line):
                    mf.write(json.dumps(rec) + "\n")
                    mf.flush()
            exit_code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait(timeout=5)
            exit_code = 130
            raise
    wallclock_s = time.monotonic() - t0
    finished = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # build summary
    records = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    summary = summarize_metrics(records)
    write_summary_md(run_dir, spec, summary, argv, started, finished, wallclock_s, exit_code)

    print(f"\n[exp_run] done. exit={exit_code}  wallclock={wallclock_s:.1f}s  metrics={len(records)} records")
    print(f"[exp_run] summary: {run_dir / 'summary.md'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
