"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import pytest

import kba_evaluation as ev
import kba_path as P


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1" and ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(float("inf")) == "unendlich"
    assert ev.factor_text(8.0) == "8" and ev.factor_text(1000.0) == "1·10³" and ev.mu_text(None) == "Kurzschritt" and ev.mu_text(1.5) == "1,5" and ev.mu_text(1e5) == "1·10⁵"


class _Run:
    """Minimale Hülle für verdict_run: nur die Felder, die das Verdikt liest."""

    def __init__(self, nu=40, rounds=13, newton=34, gap=1.7e-10, problems=None):
        self.nu = nu
        self.problems = problems or []
        self.result = type("R", (), {"rounds": rounds, "newton_steps": newton, "gap_bound": gap})()


def test_run_verdict_names_nu_rounds_newton_gap_and_the_ratio():
    kind, text = ev.verdict_run(_Run(), "Kugel (Lorentz-Kegel)", 2.0)
    assert kind == "success"
    for part in ("Kugel (Lorentz-Kegel) (ν = **40**)", "**13 Runden mit zusammen 34 Newton-Schritten**", "Lücke 1,7·10⁻¹⁰", "etwa **2,0/t**", "also ν/m = 20-mal kleiner", "Die Rückumformung besteht."):
        assert part in text, part
    kind, text = ev.verdict_run(_Run(problems=["x"]), "Kugel", 2.0)
    assert kind == "error" and "besteht nicht" in text
    assert "unendlich-mal" in ev.verdict_run(_Run(), "Kugel", 0.0)[1]


def test_path_verdict_names_first_last_and_the_constant_product():
    rows = [{"t": 0.1, "value": 400.1, "error": 393.0, "bound": 400.0, "m": 39.3}, {"t": 1e3, "value": 6.78, "error": 2e-3, "bound": 4e-2, "m": 2.052}, {"t": 1e7, "value": 6.78, "error": 2e-7, "bound": 4e-6, "m": 2.0}]
    kind, text = ev.verdict_path(rows, 40)
    assert kind == "info"
    for part in ("Bei kleinem t (1·10⁻¹)", "Wert 400,1, Fehler 3,9·10² bei einer Schranke von 4,0·10²", "`t · Fehler` konstant (2,00 bis 2,05)", "ν/t = 40/t", "Bei t = 1·10⁷ ist der Fehler 2,0·10⁻⁷"):
        assert part in text, part


def test_bounds_verdict_names_each_form_the_ratio_range_and_the_duplicate_check():
    rows = [{"name": "Würfel", "nu": 80, "m": 2.0, "ratio": 40.0}, {"name": "Ellipse", "nu": 24, "m": 5.0, "ratio": 4.8}]
    dup = {"nu_dup": 80, "deviation": 1.8e-15, "rounds_dup": 13, "rounds": 13, "newton_dup": 34, "newton": 34}
    kind, text = ev.verdict_bounds(rows, dup)
    assert kind == "info"
    for part in ("Würfel: ν = 80, m = 2,0; Ellipse: ν = 24, m = 5,0", "**m ist eine kleine ganze Zahl**", "um den Faktor 5 bis 40 zu groß", "verdoppelt sich ν auf 80", "Abweichung 1,8·10⁻¹⁵", "13 statt 13 Runden und 34 statt 34 Newton-Schritte"):
        assert part in text, part


def test_nu_verdict_names_the_flat_range_and_the_dimension_growth():
    rows = [{"n": 4, "nu": 8, "rows": 12, "newton": 30.0, "rounds": 12.0}, {"n": 40, "nu": 80, "rows": 120, "newton": 34.0, "rounds": 13.0}, {"n": 320, "nu": 640, "rows": 960, "newton": 33.0, "rounds": 13.0}]
    dims = [{"d": 2, "variables": 6, "newton": 42.0}, {"d": 8, "variables": 72, "newton": 68.0}]
    kind, text = ev.verdict_nu(rows, dims)
    assert kind == "success"
    for part in ("Von n = 4 bis n = 320 Punkten (ν von 8 auf 640, Zeilen von 12 auf 960)", "zwischen 30 und 34", "Runden zwischen 12 und 13", "Von d = 2 bis d = 8 wachsen die Variablen von 6 auf 72 (Faktor 12)", "die Schritte von 42 auf 68 (Faktor 1,6)", "Messbefund"):
        assert part in text, part


def _mu_rows():
    return [{"mu": 1.158, "rounds": 160, "newton": 343, "per_round": 2.1, "short": True}, {"mu": 4.0, "rounds": 18, "newton": 43, "per_round": 2.4, "short": False},
            {"mu": 32.0, "rounds": 8, "newton": 28, "per_round": 3.5, "short": False}, {"mu": 1e5, "rounds": 4, "newton": 30, "per_round": 7.5, "short": False}]


def test_mu_verdict_leaves_out_runs_that_ended_far_from_the_path():
    rows = _mu_rows()
    rows[3] = {**rows[3], "far_from_path": True, "newton": 12}
    kind, text = ev.verdict_mu(rows)
    assert kind == "warning" and "**Nicht gezählt:** bei μ = 1·10⁵ endete eine Zentrierung weit vom Pfad" in text and "falschen Wert" in text
    assert "μ von 4 bis 3·10¹ sind es nur 28 bis 43 Schritte" in text                      # die markierte Zeile geht nicht in die Spanne ein
    assert ev.verdict_mu(_mu_rows())[0] == "info" and "Nicht gezählt" not in ev.verdict_mu(_mu_rows())[1]


def test_mu_verdict_compares_the_short_step_with_the_long_steps():
    kind, text = ev.verdict_mu(_mu_rows())
    assert kind == "info"
    for part in ("(μ = 1,16)", "**160 Runden und 343 Newton-Schritte**", "μ von 4 bis 1·10⁵ sind es nur 28 bis 43 Schritte", "am wenigsten bei μ = 32: 28", "je Runde zwischen 2,4 und 7,5", "12-mal weniger als der Kurzschritt"):
        assert part in text, part


def test_decrement_verdict_lists_the_first_sequence_and_the_damped_steps():
    S = P.NewtonStep
    res = {"first": [S(8e-3, 1.0, 1.0), S(5e-6, 1.0, 1.0), S(2e-12, 1.0, 1.0)],
           "jumps": [{"factor": 8.0, "damped": 1, "total": 5, "steps": []}, {"factor": 1000.0, "damped": 3, "total": 6, "steps": []}]}
    kind, text = ev.verdict_decrement(res)
    assert kind == "info"
    for part in ("8·10⁻³ → 5·10⁻⁶ → 2·10⁻¹²", "Sprung um 8: 1 gedämpfte von 5 Schritten", "Sprung um 1·10³: 3 gedämpfte von 6 Schritten", "dann konvergiert Newton quadratisch"):
        assert part in text, part


def test_wrong_barrier_verdict_names_both_gaps_and_the_needed_t():
    res = {"log": [{"t": 1.0, "gap": 74.0}, {"t": 1e10, "gap": 2.0e-10}], "inverse": {1.0: [{"t": 1.0, "gap": 3.8}, {"t": 1e10, "gap": 1.414e-5}]}}
    kind, text = ev.verdict_wrong(res)
    assert kind == "warning"
    for part in ("Bei t = 1·10¹⁰ liegt die Lücke mit der log-Barriere bei 2,0·10⁻¹⁰ (≈ 2,0/t)", "mit φ = Σ 1/s bei 1,4·10⁻⁵ (≈ 1,41/√t)", "bräuchte 1/s ein t von etwa 2·10¹⁸, die log-Barriere 2·10⁹", "nicht der Aufwand"):
        assert part in text, part
