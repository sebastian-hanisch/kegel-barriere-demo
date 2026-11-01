"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import kba_evaluation as ev
import kba_forms as F
import kba_path as P
import kegel_kern as K
from kba_scenario import make_cloud


def test_two_points_have_a_path_in_the_ball_forms_and_t_times_error_is_two_or_three():
    """Zwei Punkte (1,2) und (-1,-2): Kugel R = sqrt 5, beide Punkte tragen: m = 2."""
    pts = np.array([[1.0, 2.0], [-1.0, -2.0]])
    for form in ("ball_soc", "ball_psd"):
        prob = F.build_form(form, pts)
        pt = P.trace_path(prob, (1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7))[-1]
        assert pt.centered and P.count_active(prob, pt.x) == 2
        assert P.error_times_t(prob, pt, 5 ** 0.5) == pytest.approx(2.0, abs=1e-3), form


def test_single_point_has_the_optimum_zero_and_the_path_stays_inside():
    prob = F.build_form("ball_soc", np.array([[3.0, -2.0]]))
    p = P.trace_path(prob, (1.0, 1e3, 1e6))
    assert all(q.value > 0 for q in p) and p[-1].value < 1e-5 and p[-1].value < p[0].value


def test_one_and_three_dimensions_run_through_every_form():
    p1 = np.array([[-2.0], [1.0], [3.0]])
    p3 = np.random.default_rng(5).normal(0, 1, (9, 3))
    for pts in (p1 - p1.mean(), p3 - p3.mean(axis=0)):
        for form in F.FORMS:
            run = ev.run_form(form, pts)
            assert not run.problems and run.result.gap_bound <= 1e-6, (form, pts.shape)


def test_a_path_that_is_not_centered_is_flagged_not_hidden():
    """Bei zu großem t sinkt die Rechengenauigkeit (Hesse-Matrix schlecht konditioniert): der Punkt trägt dann centered = False."""
    prob = F.build_form("ellipse_axis", make_cloud(20, 4.0, 45, 35).points)
    pts = P.trace_path(prob, (0.1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e9, 1e11, 1e13))
    assert any(not p.centered for p in pts) and all(p.centered for p in pts[:5])


def test_wrong_barrier_experiment_needs_a_cube_and_reports_its_series():
    res = ev.wrong_barrier_experiment(make_cloud(8, 3.0, 20, 100000).points, ts=(1e2, 1e4), ps=(1.0,))
    assert [r["t"] for r in res["log"]] == [1e2, 1e4] and set(res["inverse"]) == {1.0} and res["nu"] == 32
    assert res["inverse"][1.0][1]["gap"] < res["inverse"][1.0][0]["gap"] and res["log"][1]["gap"] < res["log"][0]["gap"]


def test_decrement_experiment_counts_damped_steps():
    run = ev.run_form("ball_soc", make_cloud(10, 3.0, 20, 100000).points)
    res = ev.decrement_experiment(run, t0=1e2, jumps=(2.0, 1e4))
    small, large = res["jumps"]
    assert small["factor"] == 2.0 and large["factor"] == 1e4
    assert large["damped"] > small["damped"] and large["total"] >= small["total"] and res["first"][0].lam2_half > 0


def test_mu_rows_marks_the_short_step():
    run = ev.run_form("ball_soc", make_cloud(10, 3.0, 20, 100000).points)
    rows = ev.mu_rows(run, mus=(None, 8.0))
    assert rows[0]["short"] and not rows[1]["short"] and rows[0]["mu"] < 1.5 and rows[1]["mu"] == 8.0


def test_sweeps_cover_the_documented_sizes():
    rows = ev.nu_sweep(ns=(4, 6), seeds=(100000, 100001))
    assert [r["n"] for r in rows] == [4, 6] and [r["nu"] for r in rows] == [8, 12] and [r["rows"] for r in rows] == [12, 18]
    d = ev.dimension_sweep(ds=(2, 3), seeds=(100000,))
    assert [(r["d"], r["points"], r["variables"], r["psd_size"]) for r in d] == [(2, 10, 6, 4), (3, 13, 12, 6)]


def test_invalid_inputs_are_rejected_with_a_message():
    with pytest.raises(ValueError, match="unbekannte Form"):
        F.build_form("dreieck", np.zeros((3, 2)))
    with pytest.raises(ValueError, match="mindestens zwei Punkte"):
        make_cloud(1, 2.0, 0, 1)
    prob = F.build_form("cube", make_cloud(8, 3.0, 20, 1).points)
    bad = K.ConeProblem(prob.c, prob.A, prob.b, prob.cones, prob.x0 * 0.0)
    with pytest.raises(ValueError, match="Start nicht streng zulässig"):
        K.solve_barrier(bad)
