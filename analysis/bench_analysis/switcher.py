"""Static, kernel-free switchers for the exported report.

The HTML export has no kernel, so a marimo UI element cannot switch anything:
clicking one asks the (absent) runtime for a new value and raises "Static
notebook — this notebook is not connected to a kernel". These switchers are
plain radios and labels instead, with every panel pre-rendered and the browser
doing the switching. Nothing here imports marimo — callers pass already-rendered
HTML — so the package keeps its light dependency footprint.

The scheme is all-static CSS: a group's radios are emitted *before* its bar and
body, so a general-sibling selector reaches both the panel to reveal and the
label to highlight. Those rules name concrete ids, so each group carries its own
scoped `<style>`; the shared look lives in `report.css`.
"""

import re

__all__ = ["slug", "tabs"]


def slug(text: str) -> str:
    """An id-safe token: lowercase, non-alphanumerics collapsed to single dashes."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", str(text).lower())).strip("-")


def tabs(panels: dict[str, str], *, group: str, active: str | None = None) -> str:
    """A tab bar over pre-rendered `{label: html}` panels, switched client-side.

    `group` must be unique in the page — it namespaces the radio group and the
    generated ids. `active` picks the initially-shown label (default: the first).
    """
    if not panels:
        raise ValueError("panels must not be empty")
    labels = list(panels)
    if active is None:
        active = labels[0]
    if active not in panels:
        raise ValueError(f"active {active!r} is not one of {labels}")

    gid = slug(group)
    ids = {label: f"{gid}-{slug(label)}" for label in labels}
    if len(set(ids.values())) != len(labels):
        raise ValueError(f"panel labels collide once slugified: {labels}")

    radios = "".join(
        f'<input class="sw-radio" type="radio" name="{gid}" id="{ids[label]}"'
        f'{" checked" if label == active else ""}>'
        for label in labels
    )
    bar = "".join(f'<label class="sw-tab" for="{ids[label]}">{label}</label>' for label in labels)
    body = "".join(
        f'<div class="sw-panel" data-p="{ids[label]}">{html}</div>'
        for label, html in panels.items()
    )
    # One rule pair per tab: reveal its panel, highlight its label.
    rules = "".join(
        f"#{i}:checked~.sw-body [data-p={i}]{{display:block}}"
        f"#{i}:checked~.sw-bar [for={i}]{{background:var(--sw-active-bg);"
        f"border-color:var(--sw-accent);color:var(--sw-active-fg);font-weight:600}}"
        for i in ids.values()
    )
    return (
        f"<style>{rules}</style>"
        f'<div class="sw-group">{radios}'
        f'<div class="sw-bar">{bar}</div>'
        f'<div class="sw-body">{body}</div>'
        f"</div>"
    )
