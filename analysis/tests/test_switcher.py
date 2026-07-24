"""The switchers must stay kernel-free: plain radios, CSS, and pre-rendered
panels. A regression here shows up only in the exported HTML, where there is no
runtime to complain, so the contract is pinned in tests instead."""

import pytest

from bench_analysis import switcher


def test_slug_is_id_safe():
    assert switcher.slug("summarize-small") == "summarize-small"
    assert switcher.slug("Ryzen 9 9950X + RTX 5080") == "ryzen-9-9950x-rtx-5080"
    assert switcher.slug("  a//b  ") == "a-b"


def test_tabs_marks_the_active_panel_and_no_other():
    html = switcher.tabs({"a": "<p>A</p>", "b": "<p>B</p>"}, group="g", active="b")
    # Count the attribute, not the ":checked" that also appears in the CSS rules.
    assert html.count(" checked>") == 1
    assert 'id="g-b" checked' in html
    assert 'id="g-a">' in html


def test_tabs_defaults_to_the_first_panel():
    html = switcher.tabs({"a": "x", "b": "y"}, group="g")
    assert 'id="g-a" checked' in html


def test_tabs_embeds_every_panel_body():
    html = switcher.tabs({"a": "<p>AAA</p>", "b": "<p>BBB</p>"}, group="g")
    assert "<p>AAA</p>" in html and "<p>BBB</p>" in html
    assert html.count('class="sw-panel"') == 2


def test_tabs_emits_a_rule_per_panel():
    html = switcher.tabs({"a": "x", "b": "y", "c": "z"}, group="g")
    assert html.count(":checked~.sw-body") == 3


def test_radios_precede_the_bar_and_body_so_sibling_rules_reach_them():
    # The all-static CSS depends on this ordering; a reshuffle silently breaks
    # every switch in the export.
    html = switcher.tabs({"a": "x", "b": "y"}, group="g")
    assert html.index('id="g-b"') < html.index('class="sw-bar"') < html.index('class="sw-body"')


def test_group_namespaces_ids_so_two_switchers_can_coexist():
    a = switcher.tabs({"small": "x"}, group="time-all")
    b = switcher.tabs({"small": "x"}, group="memory-all")
    assert 'id="time-all-small"' in a
    assert 'id="memory-all-small"' in b


def test_tabs_rejects_empty_and_unknown_active():
    with pytest.raises(ValueError):
        switcher.tabs({}, group="g")
    with pytest.raises(ValueError):
        switcher.tabs({"a": "x"}, group="g", active="nope")


def test_tabs_rejects_labels_that_collide_once_slugified():
    with pytest.raises(ValueError):
        switcher.tabs({"a b": "x", "a/b": "y"}, group="g")


def test_variants_tags_both_renderings():
    html = switcher.variants("<p>BOTH</p>", "<p>GGML</p>")
    assert '<div data-backend="all"><p>BOTH</p></div>' in html
    assert '<div data-backend="ggml"><p>GGML</p></div>' in html


def test_backend_filter_ids_match_the_selectors_in_report_css():
    from pathlib import Path

    html = switcher.backend_filter()
    css = (Path(__file__).resolve().parents[1] / "report.css").read_text()
    for element_id in ("backend-all", "backend-ggml"):
        assert f'id="{element_id}"' in html
        assert f"#{element_id}:checked" in css


def test_no_marimo_ui_element_is_produced():
    html = switcher.tabs({"a": "x"}, group="g") + switcher.backend_filter()
    assert "marimo-ui-element" not in html
    assert "marimo-tabs" not in html


def test_only_with_tjs_tags_content_for_the_all_state():
    assert switcher.only_with_tjs("<h2>4</h2>") == '<div data-backend="all"><h2>4</h2></div>'


def test_only_with_tjs_is_hidden_by_the_same_rule_variants_use():
    from pathlib import Path

    css = (Path(__file__).resolve().parents[1] / "report.css").read_text()
    # One rule drives both: whatever hides a variants() "all" branch also hides
    # a section wrapped by only_with_tjs.
    assert '#backend-ggml:checked) [data-backend="all"]' in css
    assert 'data-backend="all"' in switcher.only_with_tjs("x")
