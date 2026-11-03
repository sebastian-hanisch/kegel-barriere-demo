"""Der Pfad (`kba_path.py`) je Einheit gegen Handrechnung und Mini-Programme (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import kba_forms as F
import kba_path as P
import kegel_kern as K
from kba_scenario import HAND_POINTS, make_cloud


def test_path_of_the_orthant_mini_program_in_closed_form(tiny_cones):
    """min x mit x >= 1 (s = x - 1): f_t = t x - log(x - 1), f_t' = t - 1/(x - 1) = 0 => x(t) = 1 + 1/t; Lücke c x - p* = 1/t = nu/t (nu = 1)."""
    prob, opt = tiny_cones["lp"]
    for t in (0.5, 4.0, 1e3, 1e6):
        x, _ = P.path_point(prob, t, prob.x0)
        assert x[0] == pytest.approx(1.0 + 1.0 / t, rel=1e-9) and x[0] - opt == pytest.approx(K.barrier_parameter(prob.cones) / t, rel=1e-6)


def test_path_of_the_second_order_mini_program_by_hand(tiny_cones):
    """min t0 mit (t0; 3, 4) im Kegel: s0 = t0, q = t0^2 - 25. f = t t0 - log(t0^2 - 25); f' = t - 2 t0/(t0^2 - 25) = 0 => t t0^2 - 2 t0 - 25 t = 0 => t0 = (1 + sqrt(1 + 25 t^2))/t."""
    prob, opt = tiny_cones["soc"]
    for t in (0.1, 1.0, 10.0, 1e4):
        x, _ = P.path_point(prob, t, prob.x0)
        assert x[0] == pytest.approx((1.0 + math.sqrt(1.0 + 25.0 * t * t)) / t, rel=1e-9)
        assert x[0] - opt <= K.barrier_parameter(prob.cones) / t + 1e-12                         # nu = 2: die Schranke gilt


def test_path_of_the_psd_mini_program_by_hand(tiny_cones):
    """min x mit [[x, 1],[1, x]] >= 0: f = t x - log(x^2 - 1); f' = t - 2x/(x^2 - 1) = 0 => t x^2 - 2 x - t = 0 => x = (1 + sqrt(1 + t^2))/t."""
    prob, opt = tiny_cones["sdp"]
    for t in (0.5, 2.0, 100.0, 1e5):
        x, _ = P.path_point(prob, t, prob.x0)
        assert x[0] == pytest.approx((1.0 + math.sqrt(1.0 + t * t)) / t, rel=1e-9)
        assert x[0] - opt <= 2.0 / t + 1e-12


def test_stationarity_holds_at_every_path_point_of_the_ball():
    """t c = A^T grad phi(s): der Gradient von f_t verschwindet (finite Differenzen von f_t unabhängig von newton_system)."""
    prob = F.build_form("ball_soc", HAND_POINTS)
    x = prob.x0.copy()
    for t in (0.5, 5.0, 50.0):
        x, _ = P.path_point(prob, t, x)
        f = lambda v: t * prob.c @ v + K.barrier_value(prob.cones, prob.b - prob.A @ v)
        g = np.array([(f(x + 1e-6 * e) - f(x - 1e-6 * e)) / 2e-6 for e in np.eye(len(x))])
        assert np.abs(g).max() < 1e-5 * (1.0 + t)


def test_trace_path_is_continuation_and_flags_centering(tiny_cones):
    prob, _ = tiny_cones["lp"]
    pts = P.trace_path(prob, [1.0, 10.0, 100.0])
    assert [p.t for p in pts] == [1.0, 10.0, 100.0] and all(p.centered for p in pts)
    assert [p.bound for p in pts] == pytest.approx([1.0, 0.1, 0.01]) and all(b.value < a.value for a, b in zip(pts, pts[1:]))      # Wert fällt in t
    with pytest.raises(ValueError, match="aufsteigen"):
        P.trace_path(prob, [10.0, 1.0])
    with pytest.raises(ValueError, match="t muss positiv"):
        P.path_point(prob, 0.0, prob.x0)


def test_half_decrement_is_zero_on_the_path_and_infinite_when_singular(tiny_cones):
    prob, _ = tiny_cones["sdp"]
    x, _ = P.path_point(prob, 3.0, prob.x0)
    assert P.half_decrement(prob, x, 3.0) < 1e-18
    assert P.half_decrement(prob, prob.x0, 3.0) > 1e-3
    assert P.half_decrement(prob, np.array([1.0]), 3.0) == math.inf                          # x = 1: die Matrix [[1,1],[1,1]] ist singulär, keine Hesse-Matrix


def test_centering_trace_shows_damped_then_quadratic_steps():
    prob = F.build_form("ball_soc", make_cloud(20, 3.0, 30, 100000).points)
    x, _ = P.path_point(prob, 1e3, prob.x0)
    steps, _ = P.centering_trace(prob, x, 1e6)                                  # Sprung um das Tausendfache
    damped = [s for s in steps if s.length < 0.99]
    full = [s for s in steps if s.length >= 0.99]
    assert len(damped) >= 2 and len(full) >= 2 and steps.index(damped[-1]) < steps.index(full[0])   # erst gedämpft, dann voll
    assert all(s.decrease > 0 for s in steps[:-1])
    quad = [s.lam2_half for s in full]
    assert all(b < a for a, b in zip(quad, quad[1:])) and quad[-1] < 1e-3 * quad[0]            # das Dekrement fällt in der vollen Phase rasch


def test_centering_trace_at_the_path_is_empty(tiny_cones):
    prob, _ = tiny_cones["lp"]
    x, _ = P.path_point(prob, 5.0, prob.x0)
    steps, x2 = P.centering_trace(prob, x, 5.0)
    assert steps == [] and x2 == pytest.approx(x)


def test_error_times_t_by_hand(tiny_cones):
    prob, opt = tiny_cones["lp"]
    p = P.trace_path(prob, [100.0])[0]
    assert P.error_times_t(prob, p, opt) == pytest.approx(1.0, rel=1e-9)                       # x - 1 = 1/t, also t * Fehler = 1 (eine aktive Zeile)


def test_count_active_by_hand():
    prob = K.ConeProblem(np.array([1.0, 1.0]), np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]]), np.array([-1.0, -1.0, 5.0]), (("l", 3),), np.array([2.0, 2.0]))     # x1 >= 1, x2 >= 1, x1 <= 5
    x, _ = P.path_point(prob, 1e6, prob.x0)
    assert P.count_active(prob, x) == 2 and P.error_times_t(prob, P.trace_path(prob, [1.0, 1e6])[-1], 2.0) == pytest.approx(2.0, abs=1e-4)
    soc = F.build_form("ball_soc", HAND_POINTS)
    xs, _ = P.path_point(soc, 1e7, soc.x0)
    assert P.count_active(soc, xs) == 2                                                          # (1,1) und (-1,-1) tragen
    assert P.count_active(soc, xs, margin_tol=1e-12) == 0                                        # eine strenge Schwelle zählt nichts


def test_duplicating_every_constraint_doubles_nu_and_shifts_the_path_by_a_factor_two():
    prob = F.build_form("ball_soc", HAND_POINTS)
    dup = P.duplicate_constraints(prob)
    assert K.barrier_parameter(dup.cones) == 2 * K.barrier_parameter(prob.cones) and dup.A.shape[0] == 2 * prob.A.shape[0]
    for t in (2.0, 20.0, 2e3):
        x_orig, _ = P.path_point(prob, t / 2.0, prob.x0)
        x_dup, _ = P.path_point(dup, t, prob.x0)
        assert x_dup == pytest.approx(x_orig, abs=1e-8)
    assert K.solve_barrier(dup, tol=1e-9).value == pytest.approx(K.solve_barrier(prob, tol=1e-9).value, abs=1e-8)


def test_short_step_factor_and_mu_sweep_by_hand():
    prob = F.build_form("ball_soc", make_cloud(20, 3.0, 30, 100000).points)
    assert P.short_step_factor(prob) == pytest.approx(1.0 + 1.0 / math.sqrt(40))
    rows = P.mu_sweep(prob, [P.short_step_factor(prob), 8.0, 32.0])
    assert [r["mu"] for r in rows] == pytest.approx([1.158, 8.0, 32.0], abs=1e-3)
    assert rows[0]["rounds"] > 100 and rows[0]["newton"] > 8 * rows[1]["newton"] and rows[2]["newton"] < rows[1]["newton"] + 15
    assert all(r["converged"] and abs(r["value"] - rows[0]["value"]) < 1e-7 for r in rows)
    assert all(r["per_round"] == pytest.approx(r["newton"] / r["rounds"]) for r in rows)


def test_rounds_follow_the_outer_iteration_formula():
    """Runden = ceil(log_mu(nu / (tol t0))) + 1: die erste Zentrierung bei t0, dann k Multiplikationen bis t0 mu^k >= nu / tol (Boyd/Vandenberghe: Zahl der äußeren Iterationen)."""
    prob = F.build_form("ball_soc", make_cloud(20, 3.0, 30, 100000).points)
    nu, t0 = K.barrier_parameter(prob.cones), K.initial_t(prob)
    for mu, tol in ((8.0, 1e-9), (4.0, 1e-6), (32.0, 1e-9), (2.0, 1e-3)):
        res = K.solve_barrier(prob, tol=tol, mu=mu)
        assert res.rounds == math.ceil(math.log(nu / (tol * t0), mu)) + 1, (mu, tol)


def test_inverse_barrier_by_hand_in_one_dimension():
    """min x mit x >= 1, phi = 1/(x - 1): f = t x + 1/(x - 1), f' = t - 1/(x - 1)^2 = 0 => x = 1 + 1/sqrt(t); Fehler 1/sqrt(t), nicht 1/t."""
    prob = K.ConeProblem(np.array([1.0]), np.array([[-1.0]]), np.array([-1.0]), (("l", 1),), np.array([2.0]))
    for t in (1.0, 100.0, 1e6):
        x, _ = P.inverse_barrier_point(prob, t, prob.x0, 1.0)
        assert x[0] == pytest.approx(1.0 + 1.0 / math.sqrt(t), rel=1e-7)
    # p = 2: f' = t - 2/(x-1)^3 => x = 1 + (2/t)^(1/3)
    x, _ = P.inverse_barrier_point(prob, 1e3, prob.x0, 2.0)
    assert x[0] == pytest.approx(1.0 + (2.0 / 1e3) ** (1.0 / 3.0), rel=1e-7)


def test_inverse_barrier_gradient_and_hessian_match_finite_differences():
    prob = F.build_form("cube", HAND_POINTS)
    x, t, p = prob.x0.copy() + 0.1, 3.0, 1.5
    f = lambda v: t * prob.c @ v + float(np.sum((prob.b - prob.A @ v) ** (-p)))
    s = prob.b - prob.A @ x
    grad = t * prob.c + prob.A.T @ (p * s ** (-p - 1.0))
    hess = prob.A.T @ (prob.A * (p * (p + 1.0) * s ** (-p - 2.0))[:, None])
    num_g = np.array([(f(x + 1e-6 * e) - f(x - 1e-6 * e)) / 2e-6 for e in np.eye(len(x))])
    assert grad == pytest.approx(num_g, rel=1e-5, abs=1e-6)
    num_h = np.stack([[(f(x + 1e-4 * (ei + ej)) - f(x + 1e-4 * (ei - ej)) - f(x - 1e-4 * (ei - ej)) + f(x - 1e-4 * (ei + ej))) / 4e-8 for ej in np.eye(len(x))] for ei in np.eye(len(x))])
    assert hess == pytest.approx(num_h, rel=1e-3, abs=1e-4)


def test_inverse_barrier_only_exists_for_the_orthant_and_positive_exponents():
    with pytest.raises(ValueError, match="nur für Orthant-Programme"):
        P.inverse_barrier_point(F.build_form("ball_soc", HAND_POINTS), 1.0, F.build_form("ball_soc", HAND_POINTS).x0)
    cube = F.build_form("cube", HAND_POINTS)
    with pytest.raises(ValueError, match="p muss positiv sein"):
        P.inverse_barrier_point(cube, 1.0, cube.x0, 0.0)


def test_inverse_barrier_gap_falls_like_one_over_square_root_of_t_in_the_lp():
    prob = F.build_form("cube", make_cloud(20, 3.0, 30, 100000).points)
    p_star = K.solve_barrier(prob, tol=1e-12).value
    rows = P.inverse_barrier_path(prob, [1e4, 1e6, 1e8])
    gaps = [r["value"] - p_star for r in rows]
    assert [g * math.sqrt(r["t"]) for g, r in zip(gaps, rows)] == pytest.approx([math.sqrt(2.0)] * 3, rel=2e-3)        # 1,414/sqrt(t): ein aktives Paar, p = 1
    log_gap = P.trace_path(prob, [1e4, 1e6, 1e8])
    assert all(p.value - p_star < g * 1e-2 for p, g in zip(log_gap[1:], gaps[1:]))                                      # die log-Barriere liegt weit darunter (bei t = 1e6 um das Tausendfache, bei 1e8 um das Zehntausendfache)


def test_inverse_barrier_stops_at_the_rounding_limit_instead_of_counting_to_the_step_limit():
    """Bei sehr großem t hält die Rundung des Gradienten das Dekrement-Quadrat über der Toleranz (Würfel, p = 1, t = 10¹⁰; p = 2, t = 10⁸): früher zählte der Lauf bis 200,
    jetzt hält er an, sobald der Schritt x nicht mehr bewegt, und meldet „stagniert“. Der Punkt stimmt: Lücke gegen die Handformel 2 sqrt(z/t) mit z = 1/2 (p = 1) bzw. 4^(1/3)/t^(1/3) (p = 2)."""
    prob = F.build_form("cube", make_cloud(20, 4.0, 45, 35).points)
    p_star = K.solve_barrier(prob, tol=1e-12).value
    for p, ts, formula in ((1.0, (1e2, 1e4, 1e6, 1e8, 1e10), lambda t: math.sqrt(2.0 / t)), (2.0, (1e2, 1e4, 1e6, 1e8, 1e10), lambda t: 4.0 ** (1.0 / 3.0) / t ** (1.0 / 3.0))):
        x = prob.x0.copy()
        for t in ts:
            x, steps, stop = P.inverse_barrier_solve(prob, t, x, p)
            assert stop in (P.CONVERGED, P.STAGNATED) and steps < 30, (p, t, steps, stop)
            if t >= 1e4:                                                                                       # die Handformel ist asymptotisch
                assert float(prob.c @ x) - p_star == pytest.approx(formula(t), rel=2e-3), (p, t)


def test_inverse_barrier_reports_why_it_stopped():
    one = K.ConeProblem(np.array([1.0]), np.array([[-1.0]]), np.array([-1.0]), (("l", 1),), np.array([2.0]))          # min x mit x ≥ 1
    x, steps, stop = P.inverse_barrier_solve(one, 1e4, one.x0, 1.0)
    assert stop == P.CONVERGED and x[0] == pytest.approx(1.0 + 1.0 / math.sqrt(1e4), rel=1e-9) and steps == P.inverse_barrier_point(one, 1e4, one.x0, 1.0)[1]
    x, steps, stop = P.inverse_barrier_solve(one, 1e4, one.x0, 1.0, max_iter=2)
    assert (steps, stop) == (2, P.STEP_LIMIT)
    x, steps, stop = P.inverse_barrier_solve(one, 1e12, one.x0, 1.0)                                                  # t so groß, dass 1 + 10⁻⁶ gerade noch auflösbar ist
    assert stop in (P.CONVERGED, P.STAGNATED) and steps < 40 and x[0] == pytest.approx(1.0 + 1e-6, rel=1e-6)
    rows = P.inverse_barrier_path(one, [1e2, 1e12], 1.0)
    assert [r["stop"] for r in rows][0] == P.CONVERGED and all(r["steps"] < 40 and "stop" in r for r in rows)
