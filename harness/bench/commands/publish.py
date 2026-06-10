"""`bench publish` — stage a local run as a shareable submission.

A *submission* lives in `results/published/<name>/`, is committed via PR, and is
what the analysis notebook reads by default. This command takes a local results
dir (what `bench run --out` produced), copies its `<backend>-results.json` and
the matching `<backend>-raw.json.gz` into the submission folder, validates every
results file against the contract first (a malformed run never lands), and writes
a `README.md` summarizing the run spec — machine, sampling, and a per-backend
coverage table of `model × quant × provider` against tasks — so a reviewer sees
what was measured without opening the JSON.

It only moves and describes bytes the harness already produced; it never touches
the contract or re-derives anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .. import schema
from .._log import log

# Context ladder small→medium→large reads better than alphabetical; any task not
# in this list is appended sorted. Mirrors the notebook's ordering.
_TASK_ORDER = ["summarize-small", "summarize-medium", "summarize-large"]


def _machine_line(m: dict) -> list[str]:
    """The machine spec as markdown table rows."""
    cores, threads = m.get("cpu_cores"), m.get("cpu_threads")
    cpu = m.get("cpu", "?")
    if cores and threads:
        cpu += f" ({cores}C/{threads}T)"
    gpus = ", ".join(m.get("gpus") or []) or "—"
    return [
        f"| Host | {m.get('host', '?')} |",
        f"| OS | {m.get('os', '?')} |",
        f"| CPU | {cpu} |",
        f"| GPU | {gpus} |",
    ]


def _coverage(doc: dict) -> tuple[list[str], list[list[str]]]:
    """A `model × quant × provider` vs task status table for one backend doc.

    Returns (task columns in ladder order, rows). Each row is
    [model, quant, provider, status-per-task…]; an unhealthy run fills every task
    cell with `unhealthy`.
    """
    tasks: set[str] = set()
    for run in doc["runs"]:
        tasks.update(t["task"] for t in run.get("tasks") or [])
        tasks.update(run.get("timed_out_tasks") or [])
        tasks.update(run.get("errored_tasks") or [])
    cols = [t for t in _TASK_ORDER if t in tasks] + sorted(tasks - set(_TASK_ORDER))

    rows = []
    for run in doc["runs"]:
        status = {}
        if not run["healthy"]:
            status = {t: "unhealthy" for t in cols}
        else:
            status.update({t["task"]: "ok" for t in run.get("tasks") or []})
            status.update({t: "too_slow" for t in run.get("timed_out_tasks") or []})
            status.update({t: "errored" for t in run.get("errored_tasks") or []})
        rows.append(
            [run["model"], run["quant"], run["provider"], *(status.get(t, "—") for t in cols)]
        )
    rows.sort()
    return cols, rows


def _render_readme(name: str, docs: list[tuple[Path, dict]]) -> str:
    """The submission README: spec table + per-backend coverage."""
    machine = docs[0][1]["machine"]
    samplings = {(d["iters"], d["spawns"]) for _, d in docs}

    out = [f"# {name} — benchmark submission", ""]
    out += ["| | |", "|---|---|", *_machine_line(machine)]
    if len(samplings) == 1:
        i, s = next(iter(samplings))
        out.append(f"| Sampling | {i} iters × {s} spawns (timing n = {i * s} per cell) |")
    out.append("")
    out.append(
        "Status legend: `ok` (timed) · `too_slow` (timed out / below the floor) · "
        "`errored` (crash/OOM, no sample) · `unhealthy` (brain-check failed)."
    )

    for _path, doc in sorted(docs, key=lambda d: d[1]["backend"]):
        cols, rows = _coverage(doc)
        out += ["", f"## {doc['backend']}  ({len(doc['runs'])} runs)", ""]
        if len(samplings) > 1:
            out.append(f"_{doc['iters']} iters × {doc['spawns']} spawns_\n")
        header = ["model", "quant", "provider", *cols]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            out.append("| " + " | ".join(r) + " |")

    return "\n".join(out) + "\n"


def cmd_publish(args: argparse.Namespace) -> None:
    src: Path = args.src
    results = sorted(src.glob("*-results.json"))
    if not results:
        raise SystemExit(f"{src}: no *-results.json to publish")

    name = args.name or src.resolve().name
    dest = args.published_dir / name
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest} already exists — pass --force to overwrite")

    # Validate everything before writing anything, so a bad file never half-lands.
    docs: list[tuple[Path, dict]] = []
    for rp in results:
        doc = json.loads(rp.read_text())
        schema.validate_results(doc, label=str(rp))
        docs.append((rp, doc))

    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for rp, doc in docs:
        shutil.copy2(rp, dest / rp.name)
        copied.append(rp.name)
        raw = src / f"{doc['backend']}-raw.json.gz"  # raw trace is optional but encouraged
        if raw.exists():
            shutil.copy2(raw, dest / raw.name)
            copied.append(raw.name)

    (dest / "README.md").write_text(_render_readme(name, docs))
    log(f"published {name} → {dest}")
    log(f"  {len(copied)} files + README.md: {', '.join(copied)}")
    log("  review, then: git add + commit + PR")
