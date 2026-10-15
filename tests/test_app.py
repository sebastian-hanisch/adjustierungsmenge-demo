"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt den Regler an die Grenzen, wechselt Welt, Behandlung, Wirkung, Menge
und unbeobachtete Knoten und prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import ajm_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_default_verdict():
    at = _fresh_app()
    assert at.selectbox(key="world_select").value == C.DEFAULT_WORLD
    m = _metrics(at)
    assert m["Kriterium der Menge"] == "beide Kriterien" and m["Standardfehler gegen die genaueste Menge"] == "×1,46" and m["Gültige Mengen"] == "8 von 16"
    assert any("das 1,46fache" in w.value for w in at.warning)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_slider_at_its_min_and_max_does_not_raise():
    for edge in ("min", "max"):
        at = _fresh_app()
        s = at.slider(key="n_slider")
        s.set_value(int(s.min if edge == "min" else s.max)).run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.selectbox(key="world_select").value == preset["world"]
        assert at.selectbox(key="x_select").value == preset["x"] and at.selectbox(key="y_select").value == preset["y"]
        assert at.multiselect(key="z_select").value == list(preset["z"])
        assert at.multiselect(key="hidden_select").value == list(preset["hidden"])


def test_changing_the_world_resets_the_question():
    at = _fresh_app()
    at.selectbox(key="world_select").select("Präzisions-Welt").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert (at.selectbox(key="x_select").value, at.selectbox(key="y_select").value) == (4, 5)
    assert at.multiselect(key="z_select").value == [1]
    assert _metrics(at)["Gültige Mengen"] == "16 von 32"


def test_the_set_changes_the_verdict():
    at = _fresh_app()
    at.multiselect(key="z_select").set_value([2, 4]).run(timeout=TIMEOUT)
    assert _metrics(at)["Standardfehler gegen die genaueste Menge"] == "×1,00"
    assert any("am genauesten" in s.value for s in at.success)
    at.multiselect(key="z_select").set_value([4]).run(timeout=TIMEOUT)
    assert _metrics(at)["Kriterium der Menge"] == "ungültig"
    assert any("ungültig" in e.value for e in at.error)
    assert not at.exception, [str(e) for e in at.exception]


def test_hiding_a_node_removes_it_from_the_set_and_shows_the_unavailable_note():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Präzision dazu")].click().run(timeout=TIMEOUT)
    at.multiselect(key="hidden_select").set_value([2]).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert 2 not in at.multiselect(key="z_select").value
    assert any("nicht erreichbar" in i.value for i in at.info)


def test_choosing_another_treatment_repairs_the_effect_node():
    at = _fresh_app()
    at.selectbox(key="x_select").set_value(0).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="y_select").value in (2, 3, 4, 5) and at.multiselect(key="z_select").value == []


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_invalid_ones_cleaned():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["w"] = "Präzisions-Welt"
    at.query_params["x"] = "5"          # der Servicegrad hat in dieser Welt keinen Nachfahren: ungültig
    at.query_params["y"] = "5"
    at.query_params["z"] = "1,9,3"
    at.query_params["h"] = "3,7"
    at.query_params["n"] = "999999"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="world_select").value == "Präzisions-Welt"
    assert at.selectbox(key="x_select").value == 4 and at.selectbox(key="y_select").value == 5
    assert at.multiselect(key="hidden_select").value == [3]
    assert at.multiselect(key="z_select").value == [1]
    assert at.slider(key="n_slider").value == C.N_MAX


def test_unknown_world_in_the_permalink_is_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["w"] = "Nirgendwo"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="world_select").value == C.DEFAULT_WORLD


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.multiselect(key="z_select").set_value([2, 4]).run(timeout=TIMEOUT)
    assert at.query_params["z"] in ("2,4", ["2,4"])
