"""Pfade der App, die der Rauchtest nicht erreicht: jede Form, die sechs Experimente im Seiteninhalt, PDF, Eckfälle aller Regler zugleich (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kba_constants as C
import kba_forms as F

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        (at.sidebar.selectbox if key == "form_select" else at.sidebar.number_input if key == "seed_input" else at.sidebar.slider)(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_all_six_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Barriere-Parameter ν"] == "40" and values["Äußere Runden"] == "13" and values["Wahrer Fehler · t = m"] == "2,0" and values["Bewiesene Lücke ν/t"].endswith("⁻¹⁰")
    text = _texts(at)
    for part in ("Für die Kugel (Lorentz-Kegel) (ν = **40**)", "`t · Fehler` konstant", "**m ist eine kleine ganze Zahl**", "**Kugel:** Von n = 4 bis n = 320", "Die Kurzschritt-Regel der Theorie", "dann konvergiert Newton quadratisch", "bräuchte 1/s ein t von etwa"):
        assert part in text, part


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("form", F.FORMS)
def test_every_form_renders_without_error(form):
    at = _app(form_select=form)
    assert not at.error
    assert any("Die Rückumformung besteht" in s.value for s in at.success)


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_sliders_at_their_minimum_or_maximum_together(edge):
    pick = (lambda lo, hi: lo) if edge == "min" else (lambda lo, hi: hi)
    state = {"n_slider": pick(C.N_MIN, C.N_MAX), "stretch_slider": pick(C.STRETCH_MIN, C.STRETCH_MAX), "angle_slider": pick(C.ANGLE_MIN, C.ANGLE_MAX)}
    at = _app(**state)
    assert not at.exception and not at.error


def test_the_slider_corners_work_in_every_form():
    """Regression aus Stück 4 (Eckfall-Test fand dort einen Absturz): Streckung 6, 45 Grad, 20 Punkte in jeder Form."""
    for form in F.FORMS:
        at = _app(form_select=form, stretch_slider=C.STRETCH_MAX)
        assert not at.error, form
