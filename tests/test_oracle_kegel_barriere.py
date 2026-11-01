"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): die Barrierenmethode und ihr Pfad gegen einen anderen Rechenweg.

- cvxpy mit Clarabel: der Optimalwert jeder Form (der Pfad selbst ist nicht Teil einer Bibliotheksausgabe)
- eine unabhängige Kugel in der Ebene (Kreise durch zwei oder drei Punkte) für die Zahl der Stützpunkte, die m bestimmen
- finite Differenzen für die Stationarität auf dem Pfad
- Eigenschaften, die auch ein Fehler in beiden Rechenwegen nicht überlebt (Schranke nu/t, monotoner Wert, Skalierung)

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import itertools
import math

import numpy as np
import pytest

import kba_evaluation as ev
import kba_forms as F
import kba_path as P
import kegel_kern as K
from kba_scenario import make_cloud


def _cloud(seed, n, d=2):
    pts = np.random.default_rng(seed).normal(0.0, 1.0, (n, d))
    return pts - pts.mean(axis=0)


def _circle(pts):
    best = (math.inf, None)
    for i, j in itertools.combinations(range(len(pts)), 2):
        c, r = (pts[i] + pts[j]) / 2, np.linalg.norm(pts[i] - pts[j]) / 2
        if r < best[0] and np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9):
            best = (r, c)
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        a, b, c3 = pts[i], pts[j], pts[k]
        d = 2 * (a[0] * (b[1] - c3[1]) + b[0] * (c3[1] - a[1]) + c3[0] * (a[1] - b[1]))
        if abs(d) < 1e-12:
            continue
        ux = ((a @ a) * (b[1] - c3[1]) + (b @ b) * (c3[1] - a[1]) + (c3 @ c3) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c3[0] - b[0]) + (b @ b) * (a[0] - c3[0]) + (c3 @ c3) * (b[0] - a[0])) / d
        c = np.array([ux, uy])
        r = np.linalg.norm(a - c)
        if r < best[0] and np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9):
            best = (r, c)
    return best[1], best[0]


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_barrier_optimum_equals_the_cvxpy_value_for_all_six_forms():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    kw = dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    for seed in range(8):
        d = 2 + seed % 2
        pts = _cloud(seed, 8 + seed % 4, d)
        c, R = cp.Variable(d), cp.Variable()
        refs = {}
        for form, norm in (("cube", "inf"), ("diamond", 1), ("ball_soc", 2), ("ball_psd", 2)):
            cp.Problem(cp.Minimize(R), [cp.norm(p - c, norm) <= R for p in pts]).solve(**kw)
            refs[form] = float(R.value)
        pv = cp.Variable(d, nonneg=True)
        cp.Problem(cp.Minimize(cp.sum(cp.inv_pos(pv))), [(pts ** 2) @ pv <= 1]).solve(**kw)
        refs["ellipse_axis"] = float(np.sum(1 / pv.value))
        M = cp.Variable((d, d), PSD=True)
        cp.Problem(cp.Minimize(cp.tr_inv(M)), [cp.sum(cp.multiply(np.outer(p, p), M)) <= 1 for p in pts]).solve(**kw)
        refs["ellipse_psd"] = float(np.trace(np.linalg.inv(M.value)))
        for form in F.FORMS:
            run = ev.run_form(form, pts)
            tol = 1e-5 if form.startswith("ellipse") else 1e-6
            assert run.result.value == pytest.approx(refs[form], rel=tol), (form, seed)
            assert run.result.value - refs[form] <= run.result.gap_bound + 1e-7, (form, seed)          # die bewiesene Lücke deckt den Abstand zu cvxpy (dessen eigene Genauigkeit liegt bei bis zu 3e-8: „inaccurate“, der Abstand zu unserem Referenzwert bei 1e-10)


def test_the_number_of_active_rows_of_the_ball_is_the_number_of_support_points():
    for seed in range(15):
        pts = _cloud(seed, 7 + seed % 9)
        c_ref, r_ref = _circle(pts)
        support = int((np.linalg.norm(pts - c_ref, axis=1) >= r_ref * (1 - 1e-7)).sum())
        prob = F.build_form("ball_soc", pts)
        p_star = K.solve_barrier(prob, tol=1e-12).value
        pt = P.trace_path(prob, [1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])[-1]
        assert pt.centered and P.count_active(prob, pt.x) == support, seed
        assert P.error_times_t(prob, pt, p_star) == pytest.approx(support, abs=1e-2), seed             # t * Fehler -> Zahl der Stützpunkte, nicht nu


def test_t_times_error_is_the_number_of_active_rows_in_the_lp_forms():
    for seed in range(8):
        pts = _cloud(seed, 9, 2 + seed % 2)
        for form, expected in (("cube", 2), ("diamond", None)):
            prob = F.build_form(form, pts)
            p_star = K.solve_barrier(prob, tol=1e-12).value
            pt = P.trace_path(prob, [1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])[-1]
            m, active = P.error_times_t(prob, pt, p_star), P.count_active(prob, pt.x)
            assert m == pytest.approx(active, abs=2e-2), (form, seed)
            if expected:
                assert active == expected


def test_the_bound_covers_the_error_at_every_path_point_of_every_form():
    pts = make_cloud(14, 3.0, 25, 100003).points
    for form in F.FORMS:
        run = ev.run_form(form, pts)
        for p in P.trace_path(run.problem, (1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)):
            assert p.value - run.p_star <= p.bound * (1 + 1e-6) + 1e-9, (form, p.t)
            assert p.value - run.p_star >= -1e-9


def test_value_along_the_path_decreases_and_the_error_times_t_is_bounded_by_nu():
    pts = make_cloud(12, 2.0, 15, 100001).points
    for form in F.FORMS:
        run = ev.run_form(form, pts)
        path = [p for p in P.trace_path(run.problem, (1e1, 1e2, 1e3, 1e4, 1e5)) if p.centered]
        assert all(b.value < a.value + 1e-12 for a, b in zip(path, path[1:])), form
        assert all(P.error_times_t(run.problem, p, run.p_star) <= run.nu * (1 + 1e-6) + 1e-9 for p in path), form


def test_dual_certificate_from_the_path_matches_the_bound():
    """Stück 4 am Pfad: z = -grad phi / t gibt z^T s = nu / t, und c^T x(t) - p* <= nu / t."""
    prob = F.build_form("ball_psd", _cloud(1, 9))
    for p in P.trace_path(prob, [1.0, 1e2, 1e4]):
        s = prob.b - prob.A @ p.x
        z = np.zeros_like(s)
        for kind, m, nb, off in K.groups(prob.cones):
            d = K.cone_dim(kind, m)
            g, _ = K.group_grad_hess(kind, s[off:off + nb * d].reshape(nb, d), m, nb)
            z[off:off + nb * d] = (-g / p.t).reshape(-1)
        assert z @ s == pytest.approx(K.barrier_parameter(prob.cones) / p.t, rel=1e-6)


def test_scaling_the_points_scales_the_path_value_and_keeps_the_newton_steps():
    pts = _cloud(2, 10)
    for form in ("cube", "ball_soc"):
        a = ev.run_form(form, pts)
        b = ev.run_form(form, 3.0 * pts)
        assert b.result.value == pytest.approx(3.0 * a.result.value, rel=1e-6) and abs(b.result.newton_steps - a.result.newton_steps) <= 12, form
