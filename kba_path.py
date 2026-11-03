"""Der zentrale Pfad und was ihn kostet (nur numpy, benutzt den gemeinsamen Kern `kegel_kern.py` unverändert).

Barriere: phi(s) = Summe der Barrieren der Kegel (-log s, -log(s0^2 - |s_bar|^2), -log det S), nu = Summe der Barriere-Parameter.
Zentraler Pfad: x(t) minimiert f_t(x) = t c^T x + phi(b - A x). Auf ihm gilt z^T s = nu / t für das Dual z = -grad phi / t (Stück 4), also c^T x(t) - p* <= nu / t.
Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b)."""

import contextlib
import functools
import math
from dataclasses import dataclass

import numpy as np

import kegel_kern as K

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - threadpoolctl ist optional
    threadpool_limits = None


def single_thread():
    """Kontext, der die BLAS-Threads auf eins setzt: die Matrizen sind klein, und mehrere Threads kosten auf einem ausgelasteten Rechner ein Vielfaches (Stück 4: 2,8 s statt 0,12 s)."""
    return threadpool_limits(limits=1) if threadpool_limits is not None else contextlib.nullcontext()


def single_threaded(fn):
    """Dekorator: die Funktion läuft mit einem BLAS-Thread."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with single_thread():
            return fn(*args, **kwargs)
    return wrapper


@dataclass(frozen=True, eq=False)
class PathPoint:
    t: float
    x: np.ndarray
    value: float        # c^T x(t)
    bound: float        # nu / t: die bewiesene Obergrenze für c^T x(t) - p*
    steps: int          # Newton-Schritte, um diesen Punkt von seinem Vorgänger aus zu erreichen
    centered: bool      # halbes Dekrement-Quadrat am Punkt unter 1e-6 (sonst ist der Punkt nicht auf dem Pfad: Rundung bei sehr großem t)


@dataclass(frozen=True)
class NewtonStep:
    lam2_half: float    # halbes Quadrat des Newton-Dekrements vor dem Schritt
    length: float       # Schrittlänge der Rücksetzsuche (1 = voller Newton-Schritt)
    decrease: float     # Abnahme von f_t


# ----------------------------------------------------------------------------- Pfadpunkte

def path_point(problem: K.ConeProblem, t: float, x_start: np.ndarray, tol: float = 1e-20, max_iter: int = 300) -> tuple[np.ndarray, int]:
    """Der Punkt x(t) des zentralen Pfades: Newton auf f_t bis das halbe Dekrement-Quadrat unter `tol` liegt (oder kein Schritt mehr hilft). Gibt (x, Schritte) zurück."""
    if t <= 0:
        raise ValueError("t muss positiv sein")
    x, steps = x_start.copy(), 0
    for _ in range(max_iter):
        try:
            grad, hess = K.newton_system(problem, x, t)
            dx, dec2 = K.newton_step(grad, hess)
        except np.linalg.LinAlgError:
            break
        if dec2 / 2.0 <= tol:
            break
        x, gain = K.backtrack(problem, x, dx, t, dec2)
        if gain <= 0.0:
            break
        steps += 1
    return x, steps


def half_decrement(problem: K.ConeProblem, x: np.ndarray, t: float) -> float:
    """Halbes Quadrat des Newton-Dekrements am Punkt x für dieses t; unendlich, wenn die Hesse-Matrix numerisch singulär ist."""
    try:
        grad, hess = K.newton_system(problem, x, t)
        return K.newton_step(grad, hess)[1] / 2.0
    except np.linalg.LinAlgError:
        return math.inf


def trace_path(problem: K.ConeProblem, ts) -> list[PathPoint]:
    """Pfadpunkte für aufsteigende t, jeder vom vorigen aus gerechnet (Fortsetzung). Das erste t wird vom streng zulässigen Start aus erreicht."""
    ts = [float(t) for t in ts]
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("t muss aufsteigen")
    nu = K.barrier_parameter(problem.cones)
    x, out = problem.x0.copy(), []
    for t in ts:
        x, steps = path_point(problem, t, x)
        out.append(PathPoint(t, x.copy(), float(problem.c @ x), nu / t, steps, half_decrement(problem, x, t) <= 1e-6))
    return out


def centering_trace(problem: K.ConeProblem, x: np.ndarray, t: float, max_iter: int = 40, stop: float = 1e-14) -> tuple[list[NewtonStep], np.ndarray]:
    """Die Newton-Schritte einer Zentrierung bei festem t einzeln: Dekrement, Schrittlänge, Abnahme. Gedämpfte Phase (Länge unter 1), dann quadratische (Länge 1, Dekrement quadriert sich)."""
    steps = []
    for _ in range(max_iter):
        try:
            grad, hess = K.newton_system(problem, x, t)
            dx, dec2 = K.newton_step(grad, hess)
        except np.linalg.LinAlgError:
            break
        if dec2 / 2.0 <= stop:
            break
        x_new, gain = K.backtrack(problem, x, dx, t, dec2)
        norm = float(np.linalg.norm(dx))
        steps.append(NewtonStep(dec2 / 2.0, float(np.linalg.norm(x_new - x)) / norm if norm > 0 else 0.0, gain))
        if gain <= 0.0:
            break
        x = x_new
    return steps, x


def error_times_t(problem: K.ConeProblem, point: PathPoint, p_star: float) -> float:
    """t * (c^T x(t) - p*): geht gegen eine kleine ganze Zahl m (Zahl der aktiven Bedingungen), nicht gegen nu."""
    return point.t * (point.value - p_star)


# ----------------------------------------------------------------------------- aktive Bedingungen

def count_active(problem: K.ConeProblem, x: np.ndarray, margin_tol: float = 1e-3) -> int:
    """Zahl der aktiven Bedingungen am Punkt x (für großes t): Orthant-Zeilen mit Schlupf unter `margin_tol`, dazu jeder Lorentz- und PSD-Kegel, dessen Randabstand unter `margin_tol` liegt."""
    s = problem.b - problem.A @ x
    active, off = 0, 0
    for kind, m in problem.cones:
        d = K.cone_dim(kind, m)
        blk = s[off:off + d]
        if kind == "l":
            active += int(np.sum(blk < margin_tol))
        else:
            active += int(K.cone_margin(kind, blk, m) < margin_tol)
        off += d
    return active


# ----------------------------------------------------------------------------- doppeltes nu

def duplicate_constraints(problem: K.ConeProblem) -> K.ConeProblem:
    """Dasselbe Problem, jede Nebenbedingung zweimal hingeschrieben: gleiche zulässige Menge und gleiches Optimum, aber nu doppelt. Der Pfad verschiebt sich zu x_dup(t) = x(t/2)."""
    return K.ConeProblem(problem.c, np.vstack([problem.A, problem.A]), np.concatenate([problem.b, problem.b]), tuple(problem.cones) + tuple(problem.cones), problem.x0)


# ----------------------------------------------------------------------------- Schrittfaktor

def short_step_factor(problem: K.ConeProblem) -> float:
    """Die Schrittweite der Kurzschritt-Theorie: mu = 1 + 1/sqrt(nu)."""
    return 1.0 + 1.0 / math.sqrt(K.barrier_parameter(problem.cones))


def mu_sweep(problem: K.ConeProblem, mus, tol: float = 1e-9, max_newton: int = 5000) -> list[dict]:
    """Je Faktor mu: äußere Runden, Newton-Schritte gesamt und je Runde, erreichte Lücke."""
    rows = []
    for mu in mus:
        res = K.solve_barrier(problem, tol=tol, mu=float(mu), max_newton=max_newton)
        rows.append({"mu": float(mu), "rounds": res.rounds, "newton": res.newton_steps, "per_round": res.newton_steps / res.rounds, "converged": res.converged, "gap": res.gap_bound, "value": res.value, "far_from_path": res.far_from_path})
    return rows


# ----------------------------------------------------------------------------- falsche Barriere (nur Orthant)

CONVERGED, STAGNATED, STEP_LIMIT, NO_LINE = "konvergiert", "stagniert", "Schrittgrenze", "Schrittsuche erfolglos"
STAGNATION_FACTOR = 8.0        # ein Schritt, der x um höchstens 8 · eps · max(1, |x|) bewegt, liegt an der Rundungsgrenze: Newton kommt nicht weiter


def inverse_barrier_solve(problem: K.ConeProblem, t: float, x_start: np.ndarray, p: float = 1.0, tol: float = 1e-16, max_iter: int = 200) -> tuple[np.ndarray, int, str]:
    """Newton auf t c^T x + Summe_i s_i^(-p) (statt -Summe log s_i): nur für reine Orthant-Programme. Gibt (x, Schritte, Grund des Abbruchs) zurück.

    Gründe: „konvergiert“ (halbes Dekrement-Quadrat unter `tol`), „stagniert“ (der Schritt bewegt x um weniger als STAGNATION_FACTOR · eps · max(1, |x|): die Rundung des
    Gradienten bei sehr großem t hält das Dekrement-Quadrat über `tol`, ohne dass der Punkt noch wandert; der Pfadpunkt ist dann erreicht), „Schrittsuche erfolglos“ und „Schrittgrenze“."""
    if any(kind != "l" for kind, _ in problem.cones):
        raise ValueError("die Fremdbarriere gibt es nur für Orthant-Programme (LP)")
    if p <= 0:
        raise ValueError("p muss positiv sein")
    x, steps = x_start.copy(), 0
    eps = float(np.finfo(float).eps)
    for _ in range(max_iter):
        s = problem.b - problem.A @ x
        grad = t * problem.c + problem.A.T @ (p * s ** (-p - 1.0))
        hess = problem.A.T @ (problem.A * (p * (p + 1.0) * s ** (-p - 2.0))[:, None])
        dx = -np.linalg.solve(hess + 1e-14 * np.eye(len(x)), grad)
        dec2 = float(-grad @ dx)
        if dec2 / 2.0 <= tol:
            return x, steps, CONVERGED
        f0 = t * problem.c @ x + float(np.sum(s ** (-p)))
        step = 1.0
        while step > 1e-14:
            xn = x + step * dx
            sn = problem.b - problem.A @ xn
            if np.all(sn > 0) and t * problem.c @ xn + float(np.sum(sn ** (-p))) <= f0 - 0.25 * step * dec2:
                break
            step *= 0.5
        if step <= 1e-14:
            return x, steps, NO_LINE
        if float(np.max(np.abs(step * dx))) <= STAGNATION_FACTOR * eps * max(1.0, float(np.max(np.abs(x)))):
            return x, steps, STAGNATED
        x, steps = x + step * dx, steps + 1
    return x, steps, STEP_LIMIT


def inverse_barrier_point(problem: K.ConeProblem, t: float, x_start: np.ndarray, p: float = 1.0, tol: float = 1e-16, max_iter: int = 200) -> tuple[np.ndarray, int]:
    """Wie `inverse_barrier_solve`, ohne den Grund des Abbruchs: (x, Schritte)."""
    x, steps, _ = inverse_barrier_solve(problem, t, x_start, p, tol, max_iter)
    return x, steps


def inverse_barrier_path(problem: K.ConeProblem, ts, p: float = 1.0) -> list[dict]:
    """Pfadpunkte der Fremdbarriere für aufsteigende t: (t, Wert, Newton-Schritte, Grund des Abbruchs)."""
    x, out = problem.x0.copy(), []
    for t in ts:
        x, steps, stop = inverse_barrier_solve(problem, float(t), x, p)
        out.append({"t": float(t), "value": float(problem.c @ x), "steps": steps, "stop": stop})
    return out
