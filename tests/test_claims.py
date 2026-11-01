"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Sweeps laufen über feste Wolken (SWEEP_SEEDS); die Standardwolke der Primäransicht (Seed 35) ist ausdrücklich eine einzelne.
Bänder statt Punktwerte, wo ein Bibliotheksversion oder die Rundung auf Linux die letzte Stelle bewegen kann (CI installiert ungepinnt)."""

import math
from statistics import median

import pytest

import kba_constants as C
import kba_evaluation as ev
import kba_forms as F
import kba_path as P
from kba_scenario import make_cloud


@pytest.fixture(scope="module")
def std():
    return ev.analyse(ev.Settings())


@pytest.fixture(scope="module")
def bounds(std):
    return ev.bound_rows(std)


def test_standard_cloud_run(std):
    run = std.selected
    assert run.nu == 40 and run.result.rounds == 13 and 30 <= run.result.newton_steps <= 38 and run.result.gap_bound < 1e-9 and not run.problems
    assert {f: std.runs[f].nu for f in F.FORMS} == {"cube": 80, "diamond": 100, "ball_soc": 40, "ball_psd": 60, "ellipse_axis": 24, "ellipse_psd": 24}


def test_path_error_times_t_goes_to_a_small_integer(std):
    """README: t * Fehler = 39,3 (t = 0,1), 33,3 (1), 8,2 (10), 2,43 (100), 2,05 (1e3), 2,005 (1e4), danach 2,000 (Kugel, Standardwolke): m = 2 Stützpunkte, nicht ν = 40."""
    rows = {r["t"]: r["m"] for r in ev.path_rows(std.selected)}
    assert rows[0.1] == pytest.approx(39.3, abs=0.3) and rows[1.0] == pytest.approx(33.3, abs=0.3) and rows[10.0] == pytest.approx(8.2, abs=0.2) and rows[100.0] == pytest.approx(2.43, abs=0.05)
    assert rows[1e3] == pytest.approx(2.05, abs=0.02) and rows[1e4] == pytest.approx(2.005, abs=0.003) and all(rows[t] == pytest.approx(2.0, abs=2e-3) for t in (1e5, 1e6, 1e7))


def test_the_path_starts_at_radius_nu_over_t(std):
    """README: R(t) = ν/t für kleines t (Kugel hat keine obere Schranke für R): bei t = 0,1 ist der Wert 400 = 40/0,1 bis auf 0,1 %."""
    first = ev.path_rows(std.selected)[0]
    assert first["t"] == 0.1 and first["value"] == pytest.approx(40 / 0.1, rel=1e-3)


def test_bounds_table(bounds):
    """README: ν = 80 / 100 / 40 / 60 / 24 / 24 und m = 2 / 6 / 2 / 2 / 3 / 5 (Würfel, Raute, Kugel, Kugel als SDP, Ellipse achsenparallel, Ellipse allgemein); ν/m von 4,8 bis 40."""
    by = {r["form"]: r for r in bounds}
    assert [by[f]["nu"] for f in F.FORMS] == [80, 100, 40, 60, 24, 24]
    assert [by[f]["m"] for f in ("cube", "diamond", "ball_soc", "ball_psd")] == pytest.approx([2.0, 6.0, 2.0, 2.0], abs=0.01)
    assert [by[f]["active"] for f in ("cube", "diamond", "ball_soc", "ball_psd")] == [2, 6, 2, 2]
    assert by["ellipse_axis"]["m"] == pytest.approx(3.0, abs=0.1) and by["ellipse_psd"]["m"] == pytest.approx(5.0, abs=0.1)
    assert 4 <= min(r["ratio"] for r in bounds) <= 6 and 35 <= max(r["ratio"] for r in bounds) <= 45
    assert by["ball_soc"]["rounds"] == 13 and 28 <= by["ball_soc"]["newton"] <= 40


def test_duplicating_every_constraint(std):
    """README: jede Zeile zweimal: ν 40 -> 80, Abweichung x_dup(t) gegen x(t/2) unter 1e-10, Runden und Newton-Schritte gleich (13 und 34, Band ±3)."""
    d = ev.duplicate_check(std.selected)
    assert d["nu"] == 40 and d["nu_dup"] == 80 and d["deviation"] < 1e-10 and d["rounds_dup"] == d["rounds"] and abs(d["newton_dup"] - d["newton"]) <= 3 and d["value_diff"] < 1e-8


def test_newton_steps_are_flat_in_nu_for_the_ball():
    """README: Kugel mit n = 4 bis 320 (ν = 8 bis 640): Median 30 bis 34 Newton-Schritte, 12 bis 13 Runden."""
    rows = ev.nu_sweep()
    assert [r["n"] for r in rows] == list(C.N_SWEEP) and [r["nu"] for r in rows] == [2 * n for n in C.N_SWEEP]
    assert all(27 <= r["newton"] <= 38 and 12 <= r["rounds"] <= 14 for r in rows) and max(r["newton"] for r in rows) / min(r["newton"] for r in rows) < 1.3
    assert rows[-1]["rows"] == 960 and rows[-1]["variables"] == 3


def test_newton_steps_grow_slowly_with_the_dimension_of_the_ellipse():
    """README: allgemeine Ellipse d = 2 bis 8: Variablen 6 bis 72 (Faktor 12), Median 42 bis 68 Newton-Schritte (Faktor 1,6, Band 1,2 bis 2,2)."""
    rows = ev.dimension_sweep()
    assert [(r["d"], r["variables"], r["psd_size"], r["nu"]) for r in rows] == [(2, 6, 4, 14), (3, 12, 6, 19), (4, 20, 8, 24), (5, 30, 10, 29), (6, 42, 12, 34), (8, 72, 16, 44)]
    assert 1.2 <= rows[-1]["newton"] / rows[0]["newton"] <= 2.2 and 35 <= rows[0]["newton"] <= 50 and 55 <= rows[-1]["newton"] <= 80


def test_the_short_step_costs_an_order_of_magnitude_more(std):
    """README: Kurzschritt 1 + 1/√ν = 1,16: 160 Runden, 343 Newton-Schritte; μ = 8: 13 Runden, 34; μ = 32: 8 Runden, 28; für μ von 4 bis 1e5 28 bis 43 Schritte; Faktor 8 bis 15."""
    rows = ev.mu_rows(std.selected)
    short = next(r for r in rows if r["short"])
    assert short["mu"] == pytest.approx(1 + 1 / math.sqrt(40)) and 140 <= short["rounds"] <= 175 and 300 <= short["newton"] <= 380
    by = {r["mu"]: r for r in rows if not r["short"]}
    assert by[8.0]["rounds"] == 13 and 28 <= by[8.0]["newton"] <= 40 and by[32.0]["rounds"] == 8 and by[1e5]["rounds"] == 4
    long_rows = [r for r in rows if r["mu"] >= 4.0]
    assert 25 <= min(r["newton"] for r in long_rows) and max(r["newton"] for r in long_rows) <= 50 and 8 <= short["newton"] / min(r["newton"] for r in rows) <= 16
    assert all(r["converged"] for r in rows)


def test_damped_then_quadratic(std):
    """README: erste Zentrierung 8e-3 -> 5e-6 -> 2e-12 (Quadrierung, volle Schritte); Sprung um 8: 1 gedämpfter von 5 Schritten; Sprung um 1000: 3 gedämpfte von 6."""
    res = ev.decrement_experiment(std.selected)
    first = [s.lam2_half for s in res["first"]]
    assert len(first) == 3 and first[0] == pytest.approx(8e-3, rel=0.2) and first[1] < first[0] ** 1.5 and first[2] < 1e-10 and all(s.length > 0.99 for s in res["first"])
    small, large = res["jumps"]
    assert (small["damped"], small["total"]) == (1, 5) and (large["damped"], large["total"]) == (3, 6)
    assert large["steps"][0].length < 0.01 and all(s.length > 0.99 for s in large["steps"][3:])


def test_the_inverse_barrier_loses_to_the_logarithm(std):
    """README: log-Barriere: Lücke ≈ 2/t; φ = Σ 1/s: ≈ 1,414/√t, φ = Σ 1/s²: ≈ 1,587/t^(1/3); bei t = 1e10: 2e-10 gegen 1,4e-5; für 1e-9 bräuchte 1/s etwa t = 2e18."""
    res = ev.wrong_barrier_experiment(std.cloud.points)
    log_last = res["log"][-1]
    assert log_last["t"] == 1e10 and log_last["gap"] * log_last["t"] == pytest.approx(2.0, abs=5e-3) and log_last["gap"] == pytest.approx(2e-10, rel=0.01)
    for r in res["inverse"][1.0][2:]:
        assert r["gap"] * math.sqrt(r["t"]) == pytest.approx(math.sqrt(2.0), abs=0.01)           # p = 1, zwei aktive Zeilen mit z = 1/2: Lücke = 2 sqrt(z/t) = sqrt(2) / sqrt(t)
    for r in res["inverse"][2.0][2:]:
        assert r["gap"] * r["t"] ** (1.0 / 3.0) == pytest.approx(4.0 ** (1.0 / 3.0), abs=0.01)   # p = 2, zwei aktive Zeilen mit z = 1/2: Lücke = 4^(1/3) / t^(1/3) (Herleitung im 📐-Expander)
    inv_last = res["inverse"][1.0][-1]
    assert inv_last["gap"] == pytest.approx(1.414e-5, rel=0.01) and 1e18 <= (inv_last["gap"] * math.sqrt(inv_last["t"]) / 1e-9) ** 2 <= 3e18


def test_the_kernel_reports_far_from_path_only_for_huge_steps_and_there_the_value_is_wrong(std):
    """README: auf der Standardwolke meldet der Kern „weit vom Pfad“ nur bei μ = 10⁵ für die allgemeine Ellipse; dort ist der Wert falsch (100,96 statt 78,52), überall sonst stimmt er.""" 
    import kegel_kern as K

    flagged = {}
    for form in F.FORMS:
        run = std.runs[form]
        short = P.short_step_factor(run.problem)
        for m in C.MUS:
            res = K.solve_barrier(run.problem, tol=1e-9, mu=float(short if m is None else m), max_newton=5000)
            if res.far_from_path:
                flagged[(form, m)] = res
            else:
                assert abs(res.value - run.p_star) <= 1e-6 * (1.0 + abs(run.p_star)), (form, m)      # alle anderen Läufe treffen den Referenzwert
    assert set(flagged) == {("ellipse_psd", 1e5)}
    res, p_star = flagged[("ellipse_psd", 1e5)], std.runs["ellipse_psd"].p_star
    assert res.value == pytest.approx(100.96, abs=0.01) and p_star == pytest.approx(78.52, abs=0.01) and res.gap_bound < 1e-11          # README: falscher Wert bei winziger gemeldeter Lücke
