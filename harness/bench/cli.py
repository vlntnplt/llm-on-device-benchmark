"""`bench` — the harness CLI: argument parsing and dispatch only.

Each subcommand's logic lives in its own module (`bench.commands.*`, plus
`bench.fetch`); this file wires up argparse and hands off.

  plan       enumerate the cells (model × variant × provider) without running.
  run        probe each provider's ceilings, gate each cell on the brain-check,
             sweep + job the healthy ones, persist raw traces, aggregate to results.
  aggregate  re-derive results from a persisted raw trace — no re-inference.
  check      conformance-check a built backend against the contract.
  fetch      download model artifacts from the Hub into models/, per models.yaml.
  publish    stage a local run as a shareable submission under results/published/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config, fetch
from .commands.aggregate import cmd_aggregate
from .commands.check import cmd_check
from .commands.merge import cmd_merge
from .commands.plan import cmd_plan
from .commands.publish import cmd_publish
from .commands.run import cmd_run


def main() -> None:
    ap = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", required=True, help="backend key: ggml")
        p.add_argument("--models", type=Path, default=config.MODELS_DIR, help="models dir")
        p.add_argument("--tasks", type=Path, default=config.TASKS_DIR, help="tasks dir")

    p_plan = sub.add_parser("plan", help="enumerate cells without running")
    common(p_plan)
    p_plan.add_argument("--providers", nargs="*",
                        help="restrict to these device lanes (vulkan:0) or families (vulkan)")
    p_plan.add_argument("--model", nargs="*", help="restrict to these models.yaml keys")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="benchmark and write results")
    common(p_run)
    p_run.add_argument("--out", type=Path, default=config.RESULTS_DIR, help="results dir")
    p_run.add_argument("--iters", type=int, default=2, help="K: in-process job iters")
    p_run.add_argument("--spawns", type=int, default=1, help="S: job process re-spawns")
    p_run.add_argument(
        "--max-ms",
        type=int,
        default=30000,
        help="soft per-job-spawn time-box; the exe stops below K once hit (0 = off)",
    )
    p_run.add_argument(
        "--sweep-ms",
        type=int,
        default=90000,
        help="soft sweep budget: the instrumented pass stops between chunks once elapsed — "
        "the measured envelope shrinks on slow silicon instead of the time growing "
        "(0 = uncapped)",
    )
    p_run.add_argument(
        "--backstop-ms",
        type=int,
        default=120000,
        help="hard kill if even one job iteration outlives this — job marked too-slow",
    )
    p_run.add_argument("--providers", nargs="*",
                       help="restrict to these device lanes (vulkan:0) or families (vulkan)")
    p_run.add_argument("--model", nargs="*", help="restrict to these models.yaml keys")
    p_run.add_argument(
        "--machine", help="machine name for results.machine.host (default: hostname)"
    )
    p_run.set_defaults(func=cmd_run)

    p_agg = sub.add_parser("aggregate", help="re-derive results from persisted raw traces")
    p_agg.add_argument(
        "raw", nargs="+", type=Path, help="one or more <backend>-raw.json[.gz] files"
    )
    p_agg.add_argument("--machine", help="set/override results.machine.host for these traces")
    p_agg.set_defaults(func=cmd_aggregate)

    p_merge = sub.add_parser("merge", help="extend an existing run's raw trace with a newer one")
    p_merge.add_argument(
        "base", type=Path, help="existing raw trace (or dir holding one), e.g. a submission folder"
    )
    p_merge.add_argument("new", type=Path, help="raw trace with the cells to fold in")
    p_merge.add_argument("--out", type=Path, required=True, help="dir for merged raw + results")
    p_merge.set_defaults(func=cmd_merge)

    p_check = sub.add_parser("check", help="conformance-check a built backend")
    common(p_check)
    p_check.set_defaults(func=cmd_check)

    p_fetch = sub.add_parser("fetch", help="download model artifacts from the Hub into models/")
    p_fetch.add_argument("models", nargs="*", help="model names from models.yaml (default: all)")
    p_fetch.add_argument(
        "--models-dir",
        type=Path,
        default=config.MODELS_DIR,
        help="download target (default: models/)",
    )
    p_fetch.add_argument("--only", help="comma-separated backends to fetch: ggml")
    p_fetch.add_argument("--revision", help="pin a specific revision (applies to all repos)")
    p_fetch.set_defaults(func=fetch.cmd_fetch)

    p_pub = sub.add_parser("publish", help="stage a local run as a shareable submission")
    p_pub.add_argument("src", type=Path, help="local results dir holding <backend>-results.json")
    p_pub.add_argument("--name", help="submission folder name (default: src dir name)")
    p_pub.add_argument(
        "--published-dir",
        type=Path,
        default=config.RESULTS_DIR / "published",
        help="where submissions live (default: results/published)",
    )
    p_pub.add_argument("--force", action="store_true", help="overwrite an existing submission")
    p_pub.set_defaults(func=cmd_publish)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
