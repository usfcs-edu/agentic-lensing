"""``lensmark`` command line.

    lensmark serve DIR [--port 8765] [--bind 127.0.0.1] [--engine sdk|fixture] [--no-open]
    lensmark index DIR                      rebuild lensmark.manifest.json and print one row per image
    lensmark render DIR [--id ID] [--check] [--scale N]
    lensmark propose DIR --id ID [--model M] [--effort E] [--budget USD] [--fewshot BUNDLE] [--engine ...]
    lensmark eval DIR [--by model,effort]
    lensmark export DIR {coco|ds9|masks|fewshot} [--out PATH] [--k K]
    lensmark patch DIR --id ID --transcript "..."
    lensmark doctor [DIR]                   which claude binary the SDK resolves + one cheap smoke turn
    lensmark examples build [--out examples/nine] [--force]

Each sub-command imports its module lazily so the CLI stays fast and a missing optional dependency
only breaks the command that needs it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, config


def _add_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument("dir", help="campaign directory (a directory of cutout images)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lensmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"lensmark {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the local web app")
    _add_dir(p)
    p.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    p.add_argument("--bind", default=config.DEFAULT_BIND)
    p.add_argument("--engine", choices=["sdk", "fixture"], default=None, help="default: $LENSMARK_ENGINE or sdk")
    p.add_argument("--no-open", action="store_true", help="do not open the browser")
    p.add_argument("--reload", action="store_true", help="uvicorn auto-reload (development)")

    p = sub.add_parser("index", help="rebuild the manifest and list images")
    _add_dir(p)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("render", help="regenerate <id>.annot.png from <id>.lensmark.json")
    _add_dir(p)
    p.add_argument("--id", default=None)
    p.add_argument("--check", action="store_true", help="exit 1 if any annotated PNG is stale; write nothing")
    p.add_argument("--scale", type=float, default=1.0, help="render at N x the original size (export only)")
    p.add_argument("--out", default=None, help="output directory (default: beside the original)")

    p = sub.add_parser("propose", help="ask Claude for an annotation proposal")
    _add_dir(p)
    p.add_argument("--id", default=None, help="image id (default: every image without a proposal)")
    p.add_argument("--model", default=None, help=f"alias {sorted(config.FULL_ID)} or full id (default: campaign config)")
    p.add_argument("--effort", choices=config.EFFORTS, default=None)
    p.add_argument("--budget", type=float, default=None, help="max USD per call")
    p.add_argument("--fewshot", default=None, help="few-shot bundle directory (exports/fewshot)")
    p.add_argument("--engine", choices=["sdk", "fixture"], default=None)
    p.add_argument("--concurrency", type=int, default=2)

    p = sub.add_parser("eval", help="summarise critiques by model/effort")
    _add_dir(p)
    p.add_argument("--by", default="model,effort")
    p.add_argument("--out", default=None)

    p = sub.add_parser("export", help="export accepted annotations")
    _add_dir(p)
    p.add_argument("format", choices=["coco", "ds9", "masks", "fewshot"])
    p.add_argument("--out", default=None)
    p.add_argument("--k", type=int, default=6, help="few-shot: number of examples")
    p.add_argument("--require-flag", action="store_true", help="few-shot: only panels flagged would_use_as_fewshot")
    p.add_argument("--ids", default=None, help="comma-separated subset")

    p = sub.add_parser("patch", help="apply a natural-language / voice instruction via Claude (dry run prints ops)")
    _add_dir(p)
    p.add_argument("--id", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--engine", choices=["sdk", "fixture"], default=None)

    p = sub.add_parser("doctor", help="check the Claude engine (resolved binary, one cheap smoke turn)")
    p.add_argument("dir", nargs="?", default=None)
    p.add_argument("--no-call", action="store_true", help="only print the resolved binary/versions")
    p.add_argument("--budget", type=float, default=0.10)

    p = sub.add_parser("examples", help="example data")
    ex = p.add_subparsers(dest="examples_cmd", required=True)
    pb = ex.add_parser("build", help="split the deck grid into examples/nine")
    pb.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "examples" / "nine"))
    pb.add_argument("--force", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "examples":
        from . import examples_build
        paths = examples_build.build(Path(args.out), force=args.force)
        print(f"wrote {len(paths)} tiles to {Path(args.out)}")
        return 0
    if args.cmd == "index":
        from .store import Campaign
        rows = Campaign(args.dir).write_manifest()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                flag = "stale" if r["annot_stale"] else "ok"
                print(f"{r['id']:<20} {r['width']}x{r['height']} {r['cutout_arcsec']:.2f}\" ({r['scale_source']})"
                      f" items={r['n_items']} json={'y' if r['has_json'] else '-'} annot={flag}")
        return 0
    if args.cmd == "serve":
        from .server.app import serve
        return serve(args.dir, port=args.port, bind=args.bind, engine=args.engine, open_browser=not args.no_open,
                     reload=args.reload)
    if args.cmd == "render":
        from .render.draw import cli_render
        return cli_render(args.dir, image_id=args.id, check=args.check, scale=args.scale, out=args.out)
    if args.cmd == "propose":
        from .claude.propose import cli_propose
        return cli_propose(args.dir, image_id=args.id, model=args.model, effort=args.effort, budget=args.budget,
                           fewshot=args.fewshot, engine=args.engine, concurrency=args.concurrency)
    if args.cmd == "eval":
        from .evaluate import cli_eval
        return cli_eval(args.dir, by=args.by, out=args.out)
    if args.cmd == "export":
        from .exports import cli_export
        ids = args.ids.split(",") if args.ids else None
        return cli_export(args.dir, args.format, out=args.out, k=args.k, require_flag=args.require_flag, ids=ids)
    if args.cmd == "patch":
        from .voice.patch import cli_patch
        return cli_patch(args.dir, image_id=args.id, transcript=args.transcript, apply=args.apply,
                         model=args.model, engine=args.engine)
    if args.cmd == "doctor":
        from .claude.engine import cli_doctor
        return cli_doctor(args.dir, call=not args.no_call, budget=args.budget)
    return 2


if __name__ == "__main__":
    sys.exit(main())
