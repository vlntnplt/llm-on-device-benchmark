"""`bench publish` — stage a local run as a shareable submission.

A *submission* lives in `results/published/<name>/`, is committed via PR, and is
what the analysis notebook reads by default. This command takes a local results
dir (what `bench run --out` produced), copies its `<backend>-results.json` and
the matching `<backend>-raw.json.gz` into the submission folder, validates every
results file against the contract first (a malformed run never lands), and writes
a `README.md` summarizing the run spec — machine (incl. installed memory config),
the per-provider ceiling probes, and a `model × quant × provider` coverage table
of sweep/job status — so a reviewer sees what was measured without opening the
JSON.

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


def _memory_line(mem: dict | None) -> str:
    if not mem:
        return "—"
    parts = [f"{mem['total_gb']:g} GB"]
    if mem.get("channels"):
        parts.append(f"{mem['channels']}-channel")
    if mem.get("configured_mts"):
        speed = f"@ {mem['configured_mts']} MT/s"
        if mem.get("rated_mts") and mem["rated_mts"] != mem["configured_mts"]:
            speed += f" (rated {mem['rated_mts']})"
        parts.append(speed)
    if mem.get("rank"):
        parts.append(f"rank {mem['rank']}")
    return " ".join(parts)


def _machine_lines(m: dict) -> list[str]:
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
        f"| Memory | {_memory_line(m.get('memory'))} |",
    ]


def _probe_rows(doc: dict) -> list[str]:
    """One row per probed provider: the ceiling numbers a reviewer sanity-checks
    against the spec sheet."""
    rows = []
    for p in doc.get("probes") or []:
        if p["status"] != "ok":
            rows.append(f"| {p['provider']} | {p['device']} | — | — | {p['status']} |")
            continue
        gemm = max((g["tflops_p50"] for g in p["gemm"]), default=None)
        d2d = next((c["gbs_p50"] for c in p["copy"] if c["kind"] == "d2d"), None)
        rows.append(
            f"| {p['provider']} | {p['device']} "
            f"| {gemm if gemm is not None else '—'} | {d2d if d2d is not None else '—'} | ok |"
        )
    return rows


def _coverage(doc: dict) -> list[list[str]]:
    """A `model × quant × provider` vs sweep/job status table for one backend doc."""
    rows = []
    for run in doc["runs"]:
        if not run["healthy"]:
            sweep = job = "unhealthy"
        else:
            sweep = run["sweep"]["status"]
            job = run["job"]["status"]
        pts = len(run["sweep"]["prefill"]) + len(run["sweep"]["decode"])
        sweep_cell = f"{sweep} ({pts} pts)" if pts else sweep
        rows.append([run["model"], run["quant"], run["provider"], sweep_cell, job])
    rows.sort()
    return rows


def _render_readme(name: str, docs: list[tuple[Path, dict]]) -> str:
    """The submission README: spec table + probes + per-backend coverage."""
    machine = docs[0][1]["machine"]

    out = [f"# {name} — benchmark submission", ""]
    out += ["| | |", "|---|---|", *_machine_lines(machine)]
    spawns = {d["job_spawns"] for _, d in docs}
    if len(spawns) == 1:
        out.append(
            f"| Sampling | job: {next(iter(spawns))} spawns; sweep/probe: adaptive per point |"
        )
    out.append("")
    out.append(
        "Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · "
        "`errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed)."
    )

    for _path, doc in sorted(docs, key=lambda d: d[1]["backend"]):
        out += ["", f"## {doc['backend']}  ({len(doc['runs'])} runs)", ""]
        if doc.get("probes"):
            out += ["| provider | device | gemm TFLOP/s | d2d GB/s | probe |",
                    "|---|---|---|---|---|"]
            out += _probe_rows(doc)
            out.append("")
        header = ["model", "quant", "provider", "sweep", "job"]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in _coverage(doc):
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
