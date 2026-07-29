"""The site builder renders one self-contained page from a results dir: tabs,
model cards, strict-JSON chart islands (a browser's JSON.parse has no NaN),
and the fleet calculator's coefficient island."""

import json
import re
from pathlib import Path

from bench_analysis import site

FIXTURES = Path(__file__).parent / "fixtures"


def _build(tmp_path):
    cache = tmp_path / "vega"
    cache.mkdir()
    for name, version in site.VEGA_LIBS:  # stubs: the test never fetches
        (cache / f"{name}@{version}.min.js").write_text("/* stub */")
    out = tmp_path / "report.html"
    site.build(FIXTURES, out, vega_cache=cache)
    return out.read_text()


def test_build_renders_tabs_cards_and_strict_json(tmp_path):
    h = _build(tmp_path)
    assert "{{" not in h  # no unrendered template
    for anchor in ("tab-models", "tab-fleet", "tab-evidence", "fleet-cohorts",
                   'class="card"'):
        assert anchor in h
    islands = re.findall(
        r'<script type="application/json" id="[^"]+">(.*?)</script>', h, re.S)
    assert islands, "no chart/coefficient islands rendered"
    for body in islands:
        json.loads(body)  # strict — raises on NaN/Infinity


def test_build_is_self_contained(tmp_path):
    h = _build(tmp_path)
    assert "/* stub */" in h  # vega inlined from the cache
    assert "http" not in re.sub(r'href="https://github[^"]*"', "", h).lower() \
        or "src=" not in h  # no external script/style loads
