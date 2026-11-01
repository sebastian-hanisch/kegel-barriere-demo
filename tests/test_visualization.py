"""Abbildungen: Achsen gesperrt, Umrisse stimmen mit den Formen überein, Diagramme bauen an den Rändern."""

import numpy as np
import pytest

import kba_evaluation as ev
import kba_forms as F
import kba_visualization as V


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


@pytest.fixture(scope="module")
def a():
    return ev.analyse(ev.Settings(n=12))


def test_outlines_by_hand():
    x, y = V.outline(F.Shape("box", np.array([1.0, 2.0]), 3.0))
    assert (x.min(), x.max(), y.min(), y.max()) == (-2.0, 4.0, -1.0, 5.0)
    x, y = V.outline(F.Shape("diamond", np.array([0.0, 0.0]), 2.0))
    assert sorted(set(zip(x.round(6), y.round(6)))) == [(-2.0, 0.0), (0.0, -2.0), (0.0, 2.0), (2.0, 0.0)]
    x, y = V.outline(F.Shape("ball", np.array([1.0, 1.0]), 2.0))
    assert np.hypot(x - 1.0, y - 1.0) == pytest.approx(2.0)
    x, y = V.outline(F.Shape("ellipse", np.zeros(2), matrix=np.diag([1.0, 4.0])))
    assert (abs(x).max(), abs(y).max()) == pytest.approx((1.0, 0.5), abs=1e-3) and np.allclose(x ** 2 + 4 * y ** 2, 1.0)


def test_path_shapes_chart_has_one_line_per_path_point_the_solution_and_the_points(a):
    for form in F.FORMS:
        run = a.runs[form]
        shapes = ev.shape_path(run, a.cloud.points)
        fig = V.build_path_shapes(a.cloud.points, form, shapes, F.decode(form, a.cloud.points, run.result.x))
        names = [t.name for t in fig.data]
        assert names[:4] == [f"t = {t:g}" for t in (2.0, 5.0, 20.0, 100.0)] and names[4:] == ["Lösung", "Punkte"], form
        assert _locked(fig) and fig.layout.yaxis.scaleanchor == "x" and len(fig.data[-1].x) == 12


def test_experiment_charts_build_and_are_locked(a):
    run = a.selected
    rows = ev.path_rows(run)
    err = V.build_error(rows, run.nu, 2.0)
    assert _locked(err) and err.layout.xaxis.type == "log" and err.layout.yaxis.type == "log" and [t.name for t in err.data][1] == "wahrer Fehler"
    bounds = V.build_bounds(ev.bound_rows(a))
    assert _locked(bounds) and bounds.layout.yaxis.type == "log" and len(bounds.data) == 2 and len(bounds.data[0].y) == 6
    nu = V.build_nu(ev.nu_sweep(ns=(4, 6), seeds=(100000,)), ev.dimension_sweep(ds=(2, 3), seeds=(100000,)))
    assert _locked(nu) and nu.layout.yaxis.rangemode == "tozero" and len(nu.data) == 3
    mu = V.build_mu(ev.mu_rows(run, mus=(None, 8.0, 32.0)))
    assert _locked(mu) and mu.layout.xaxis.type == "log" and mu.data[-1].name == "Kurzschritt 1 + 1/√ν"
    dec = V.build_decrement(ev.decrement_experiment(run))
    assert _locked(dec) and dec.layout.yaxis.type == "log" and len(dec.data) == 3
    wrong = V.build_wrong(ev.wrong_barrier_experiment(a.cloud.points, ts=(1e2, 1e4), ps=(1.0, 2.0)))
    assert _locked(wrong) and wrong.layout.xaxis.type == "log" and len(wrong.data) == 4 and wrong.data[-1].name == "Ziel 10⁻⁹"


def test_decrement_chart_marks_damped_steps_with_larger_markers(a):
    res = ev.decrement_experiment(a.selected)
    fig = V.build_decrement(res)
    jump = fig.data[2]                                              # Sprung um 1000: die ersten drei Schritte sind gedämpft
    sizes = list(jump.marker.size)
    assert sizes[:3] == [16, 16, 16] and set(sizes[3:]) == {8}
