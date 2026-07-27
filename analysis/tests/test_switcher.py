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


def test_no_marimo_ui_element_is_produced():
    html = switcher.tabs({"a": "x"}, group="g")
    assert "marimo-ui-element" not in html
    assert "marimo-tabs" not in html
