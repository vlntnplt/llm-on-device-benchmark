"""Task loading.

A brain-check is not a special *type* of task — it's an ordinary task JSON the exe
runs like any other. The only difference is its *role in the plan*: gate tasks run
once and decide provider health; timed tasks run S×K and get aggregated
(commands/run.py owns that). So there's one Task here, tagged by the file it
came from.

The harness inlines `{document: corpora/x.txt}` into the prompt so every backend
gets identical bytes and tokenizes them with its own tokenizer. We inline
in *full* and never trim: the harness can't tokenize, so any char-based trim would
risk cutting a pre-sized corpus. Instead the run path checks each cell's actual
rendered length (from the exe's own token counts) against max_context_length
and warns loudly on overrun (commands/run.py) — the corpus or prompt then gets
adjusted by hand. The corpora in tasks/ are authored to fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

GATE_FILE = "brain_check.yaml"
TIMED_FILE = "tasks.yaml"


@dataclass(frozen=True)
class Task:
    name: str
    role: str  # "gate" | "timed"
    spec: dict  # the resolved task JSON handed to the exe (--task)


def _resolve(task: dict, tasks_dir: Path) -> dict:
    """Inline any {document: corpora/x.txt} message content."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in task.items()}
    out["messages"] = []
    for msg in task["messages"]:
        content = msg.get("content")
        if isinstance(content, dict) and "document" in content:
            msg = {**msg, "content": (tasks_dir / content["document"]).read_text()}
        out["messages"].append(msg)
    return out


def load(tasks_dir: Path) -> list[Task]:
    """All tasks, gate first, documents inlined."""
    tasks: list[Task] = []
    for filename, role in ((GATE_FILE, "gate"), (TIMED_FILE, "timed")):
        for spec in yaml.safe_load((tasks_dir / filename).read_text()):
            tasks.append(Task(name=spec["name"], role=role, spec=_resolve(spec, tasks_dir)))
    return tasks
