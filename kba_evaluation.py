"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

import kba_constants as C
import kba_forms as A
import kba_path as P
import kegel_kern as K
from kba_scenario import Cloud, make_cloud


@dataclass(frozen=True)
class Settings:
    form: str = C.DEFAULT_FORM
    n: int = C.DEFAULT_N
    stretch: float = C.DEFAULT_STRETCH
    angle: int = C.DEFAULT_ANGLE
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True, eq=False)
class FormRun:
    form: str
    problem: K.ConeProblem
    result: K.BarrierResult     # Barrierenmethode bis 1e-9
    p_star: float               # Referenzwert (Lücke 1e-12)
    problems: list              # Rückumformung (Kegel und Ausgangsaufgabe)
    nu: int


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    cloud: Cloud
    runs: dict                  # Form -> FormRun

    @property
    def selected(self) -> FormRun:
        return self.runs[self.settings.form]


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def factor_text(x: float) -> str:
    """Faktor für Texte: bis 999 als ganze Zahl, darüber in Zehnerpotenz."""
    return de(x, 0) if x < 1000 else sci(x, 0)


def mu_text(mu: float | None) -> str:
    """Schrittfaktor für Tabellen: None heißt Kurzschritt."""
    return "Kurzschritt" if mu is None else sci(mu, 0) if mu >= 1000 else de(mu, 1)


# ----------------------------------------------------------------------------- Analyse der eingestellten Wolke

def run_form(form: str, points: np.ndarray) -> FormRun:
    problem = A.build_form(form, points)
    result = K.solve_barrier(problem, tol=1e-9)
    ref = K.solve_barrier(problem, tol=1e-12)
    pts = np.asarray(points, dtype=float)
    problems = K.check_solution(problem, result.x) + A.original_violations(form, pts, result.x, result.value)
    return FormRun(form, problem, result, ref.value, problems, K.barrier_parameter(problem.cones))


@P.single_threaded
def analyse(settings: Settings) -> Analysis:
    cloud = make_cloud(settings.n, settings.stretch, settings.angle, settings.seed)
    return Analysis(settings, cloud, {f: run_form(f, cloud.points) for f in A.FORMS})


@P.single_threaded
def path_rows(run: FormRun, ts=C.PATH_TS) -> list[dict]:
    """Pfadpunkte der gewählten Form: t, Wert, wahrer Fehler, bewiesene Schranke nu/t, t mal Fehler, Newton-Schritte."""
    points = P.trace_path(run.problem, ts)
    return [{"t": p.t, "value": p.value, "error": p.value - run.p_star, "bound": p.bound, "m": P.error_times_t(run.problem, p, run.p_star), "steps": p.steps, "x": p.x} for p in points]


@P.single_threaded
def shape_path(run: FormRun, points: np.ndarray, ts=C.SHAPE_TS) -> list[tuple[float, A.Shape]]:
    """Die Form der gewählten Aufgabe an einigen Pfadpunkten (zum Zeichnen)."""
    return [(p.t, A.decode(run.form, points, p.x)) for p in P.trace_path(run.problem, ts)]


# ----------------------------------------------------------------------------- Experiment 2: Schranke gegen wahren Fehler

@P.single_threaded
def bound_rows(a: Analysis) -> list[dict]:
    """Je Form: nu, Runden und Newton-Schritte des Lösers, m = t * Fehler bei t = ERROR_T, Zahl der aktiven Bedingungen, Verhältnis nu / m."""
    rows = []
    for f in A.FORMS:
        run = a.runs[f]
        pt = next(p for p in P.trace_path(run.problem, C.PATH_TS) if p.t == C.ERROR_T)
        m = P.error_times_t(run.problem, pt, run.p_star)
        active = P.count_active(run.problem, pt.x, C.ACTIVE_MARGIN)
        rows.append({"form": f, "name": A.FORM_NAMES[f], "nu": run.nu, "rounds": run.result.rounds, "newton": run.result.newton_steps, "m": m, "active": active, "centered": pt.centered,
                     "ratio": run.nu / m if m > 0 else float("inf")})
    return rows


@P.single_threaded
def duplicate_check(run: FormRun, t: float = 10.0) -> dict:
    """Jede Nebenbedingung zweimal: x_dup(t) gegen x(t/2), nu und Lückenschranke, Runden und Newton-Schritte des Lösers."""
    dup = P.duplicate_constraints(run.problem)
    x_orig, _ = P.path_point(run.problem, t / 2.0, run.problem.x0)
    x_dup, _ = P.path_point(dup, t, run.problem.x0)
    res = K.solve_barrier(dup, tol=1e-9)
    return {"t": t, "deviation": float(np.abs(x_orig - x_dup).max()), "nu": run.nu, "nu_dup": K.barrier_parameter(dup.cones), "rounds": run.result.rounds,
            "rounds_dup": res.rounds, "newton": run.result.newton_steps, "newton_dup": res.newton_steps, "value_diff": abs(res.value - run.result.value)}


# ----------------------------------------------------------------------------- Experiment 3: Newton-Schritte gegen nu und Dimension

@P.single_threaded
def nu_sweep(ns=C.N_SWEEP, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Kugel (Lorentz-Form) auf wachsender Punktzahl: nu = 2 n, Zeilen 3 n, Variablen 3."""
    rows = []
    for n in ns:
        steps, rounds = [], []
        for seed in seeds:
            prob = A.build_form("ball_soc", make_cloud(n, 3.0, 30, seed).points)
            res = K.solve_barrier(prob, tol=1e-9)
            steps.append(res.newton_steps)
            rounds.append(res.rounds)
        rows.append({"n": n, "nu": K.barrier_parameter(prob.cones), "rows": prob.A.shape[0], "variables": prob.A.shape[1], "newton": float(median(steps)), "newton_min": min(steps),
                     "newton_max": max(steps), "rounds": float(median(rounds))})
    return rows


@P.single_threaded
def dimension_sweep(ds=C.D_SWEEP, seeds=C.D_SWEEP_SEEDS) -> list[dict]:
    """Allgemeine Ellipse in wachsender Dimension d (3 d + 4 Punkte): Variablen d (d + 1), PSD-Block 2 d."""
    rows = []
    for d in ds:
        steps = []
        for seed in seeds:
            pts = np.random.default_rng(seed).normal(0.0, 1.0, (3 * d + 4, d))
            pts -= pts.mean(axis=0)
            prob = A.build_form("ellipse_psd", pts)
            steps.append(K.solve_barrier(prob, tol=1e-9).newton_steps)
        rows.append({"d": d, "points": 3 * d + 4, "variables": prob.A.shape[1], "psd_size": 2 * d, "nu": K.barrier_parameter(prob.cones), "newton": float(median(steps)), "newton_min": min(steps), "newton_max": max(steps)})
    return rows


# ----------------------------------------------------------------------------- Experiment 4 und 5: Schrittfaktor, Dekrement-Folge

@P.single_threaded
def mu_rows(run: FormRun, mus=C.MUS) -> list[dict]:
    """Schrittfaktor gegen Aufwand für die gewählte Form; mu = None steht für den Kurzschritt 1 + 1/sqrt(nu)."""
    short = P.short_step_factor(run.problem)
    rows = P.mu_sweep(run.problem, [short if m is None else m for m in mus])
    for row, m in zip(rows, mus):
        row["short"] = m is None
    return rows


@P.single_threaded
def decrement_experiment(run: FormRun, t0: float = C.DECREMENT_T0, jumps=C.DECREMENT_JUMPS) -> dict:
    """Newton-Folge einer ersten Zentrierung vom Start aus und von Sprüngen t0 -> t0 mal Faktor: Dekrement und Schrittlänge je Schritt."""
    x_start, _ = P.path_point(run.problem, t0, run.problem.x0)
    first, _ = P.centering_trace(run.problem, run.problem.x0, K.initial_t(run.problem))
    out = {"first": first, "jumps": []}
    for factor in jumps:
        steps, _ = P.centering_trace(run.problem, x_start, t0 * factor)
        damped = sum(1 for s in steps if s.length < 0.99)
        out["jumps"].append({"factor": factor, "steps": steps, "damped": damped, "total": len(steps)})
    return out


# ----------------------------------------------------------------------------- Experiment 6: falsche Barriere

@P.single_threaded
def wrong_barrier_experiment(points: np.ndarray, ts=C.WRONG_TS, ps=C.WRONG_PS) -> dict:
    """Der Würfel (LP) mit der log-Barriere gegen φ = Σ s^(-p): Lücke c^T x(t) - p* über t."""
    prob = A.build_form("cube", points)
    p_star = K.solve_barrier(prob, tol=1e-12).value
    log_rows, x = [], prob.x0.copy()
    for t in ts:
        x, steps = P.path_point(prob, t, x)
        log_rows.append({"t": t, "gap": float(prob.c @ x) - p_star, "steps": steps})
    inv = {}
    for p in ps:
        inv[p] = [{"t": r["t"], "gap": r["value"] - p_star, "steps": r["steps"], "stop": r["stop"]} for r in P.inverse_barrier_path(prob, ts, p)]
    return {"nu": K.barrier_parameter(prob.cones), "p_star": p_star, "log": log_rows, "inverse": inv}


# ----------------------------------------------------------------------------- Verdikte (rein)

def verdict_run(run: FormRun, name: str, m: float) -> tuple[str, str]:
    """Kernfrage der Primäransicht: die Barrierenmethode, ihr Aufwand, die Schranke gegen den wahren Fehler."""
    r = run.result
    base = (f"Für die {name} (ν = **{run.nu}**) braucht die Barrierenmethode **{r.rounds} Runden mit zusammen {r.newton_steps} Newton-Schritten** bis zur bewiesenen Lücke {sci(r.gap_bound)}. "
            f"Die Schranke ist `ν/t`; der wahre Fehler auf dem Pfad ist bei großem t nur etwa **{de(m, 1)}/t**, also ν/m = {de(run.nu / m, 0) if m > 0 else 'unendlich'}-mal kleiner.")
    if run.problems:
        return "error", base + " Die Rückumformung besteht nicht: Das darf nicht passieren."
    return "success", base + " Die Rückumformung besteht."


def verdict_path(rows: list[dict], nu: int) -> tuple[str, str]:
    """Experiment 1: Pfad, Schranke und wahrer Fehler."""
    first, last = rows[0], rows[-1]
    ms = [r["m"] for r in rows if r["t"] >= 1e3]
    return "info", (f"Bei kleinem t ({sci(first['t'], 0)}) ist der Pfad weit weg von der Lösung (Wert {de(first['value'], 1)}, Fehler {sci(first['error'])} bei einer Schranke von {sci(first['bound'])}). "
                    f"Ab t = 10³ ist `t · Fehler` konstant ({de(min(ms), 2)} bis {de(max(ms), 2)}): der Fehler fällt wie **m/t**, die Schranke wie ν/t = {nu}/t. Bei t = {sci(last['t'], 0)} ist der Fehler {sci(last['error'])}.")


def verdict_bounds(rows: list[dict], dup: dict) -> tuple[str, str]:
    """Experiment 2: ν gegen m über alle Formen und das Verdoppeln."""
    parts = "; ".join(f"{r['name']}: ν = {r['nu']}, m = {de(r['m'], 1)}" for r in rows)
    return "info", (f"{parts}. **m ist eine kleine ganze Zahl** (bei Würfel, Raute und Kugeln gleich der Zahl der aktiven Bedingungen), ν dagegen wächst mit der Zahl der Zeilen: "
                    f"die Schranke ν/t ist um den Faktor {de(min(r['ratio'] for r in rows), 0)} bis {de(max(r['ratio'] for r in rows), 0)} zu groß. "
                    f"Schreibt man jede Nebenbedingung zweimal hin, verdoppelt sich ν auf {dup['nu_dup']}, der Pfad verschiebt sich aber nur zu x(t/2) (Abweichung {sci(dup['deviation'])}); "
                    f"der Löser braucht {dup['rounds_dup']} statt {dup['rounds']} Runden und {dup['newton_dup']} statt {dup['newton']} Newton-Schritte.")


def verdict_nu(rows: list[dict], dim_rows: list[dict]) -> tuple[str, str]:
    """Experiment 3: Newton-Schritte gegen ν und Dimension."""
    lo, hi = min(r["newton"] for r in rows), max(r["newton"] for r in rows)
    d0, d1 = dim_rows[0], dim_rows[-1]
    return "success", (f"**Kugel:** Von n = {rows[0]['n']} bis n = {rows[-1]['n']} Punkten (ν von {rows[0]['nu']} auf {rows[-1]['nu']}, Zeilen von {rows[0]['rows']} auf {rows[-1]['rows']}) liegt der Median der Newton-Schritte "
                       f"zwischen {de(lo, 0)} und {de(hi, 0)}, die Runden zwischen {de(rows[0]['rounds'], 0)} und {de(rows[-1]['rounds'], 0)}. "
                       f"**Allgemeine Ellipse:** Von d = {d0['d']} bis d = {d1['d']} wachsen die Variablen von {d0['variables']} auf {d1['variables']} (Faktor {de(d1['variables'] / d0['variables'], 0)}), "
                       f"die Schritte von {de(d0['newton'], 0)} auf {de(d1['newton'], 0)} (Faktor {de(d1['newton'] / d0['newton'], 1)}). Das ist ein Messbefund dieser Implementierung; die Theorie nennt nur eine Schranke mit √ν.")


def verdict_mu(rows: list[dict]) -> tuple[str, str]:
    """Experiment 4: Schrittfaktor gegen Gesamtaufwand."""
    flagged = [r for r in rows if r.get("far_from_path")]
    rows = [r for r in rows if not r.get("far_from_path")]
    short = next(r for r in rows if r["short"])
    long_rows = [r for r in rows if r["mu"] >= 4.0]
    best = min(rows, key=lambda r: r["newton"])
    text = (f"Die Kurzschritt-Regel der Theorie (μ = {de(short['mu'], 2)}) braucht **{short['rounds']} Runden und {short['newton']} Newton-Schritte**. "
            f"Für μ von {de(min(r['mu'] for r in long_rows), 0)} bis {sci(max(r['mu'] for r in long_rows), 0)} sind es nur {min(r['newton'] for r in long_rows)} bis {max(r['newton'] for r in long_rows)} Schritte "
            f"(am wenigsten bei μ = {de(best['mu'], 0)}: {best['newton']}), je Runde zwischen {de(min(r['per_round'] for r in long_rows), 1)} und {de(max(r['per_round'] for r in long_rows), 1)}: "
            f"der Kurzschritt braucht für diese Form und Wolke etwa {de(short['newton'] / best['newton'], 0)}-mal so viele Schritte (andere nicht ausgewertet).")
    if flagged:
        mus = ", ".join(sci(r["mu"], 0) if r["mu"] >= 1000 else de(r["mu"], 1) for r in flagged)
        return "warning", (text + f" **Nicht gezählt:** bei μ = {mus} endete eine Zentrierung weit vom Pfad (der Kern meldet `far_from_path`): der Lauf liefert dort einen falschen Wert, "
                                  "die gemeldete Lücke gilt nicht, und seine Schrittzahl sagt nichts über die Methode.")
    return "info", text


def verdict_decrement(res: dict) -> tuple[str, str]:
    """Experiment 5: gedämpfte Schritte, dann quadratische Konvergenz."""
    first = res["first"]
    parts = "; ".join(f"Sprung um {factor_text(j['factor'])}: {j['damped']} gedämpfte von {j['total']} Schritten" for j in res["jumps"])
    seq = " → ".join(sci(s.lam2_half, 0) for s in first)
    return "info", (f"Die erste Zentrierung vom Start aus: halbes Dekrement-Quadrat {seq} (bei vollem Schritt quadratische Konvergenz).{parts}. "
                    "Zuerst kostet ein großer Sprung gedämpfte Schritte (Länge unter 1), dann konvergiert Newton quadratisch.")


def verdict_wrong(res: dict) -> tuple[str, str]:
    """Experiment 6: log-Barriere gegen 1/s im LP."""
    last_log, last_inv = res["log"][-1], res["inverse"][1.0][-1]
    t_need = (last_inv["gap"] * math.sqrt(last_inv["t"]) / 1e-9) ** 2 if last_inv["gap"] > 0 else float("nan")
    t_log = last_log["gap"] * last_log["t"] / 1e-9
    steps_log = max((r.get("steps", 0) for r in res["log"]), default=0)
    steps_inv = max((r.get("steps", 0) for rows in res["inverse"].values() for r in rows), default=0)
    stagnated = {p: next(r["t"] for r in rows if r.get("stop") == P.STAGNATED) for p, rows in res["inverse"].items() if any(r.get("stop") == P.STAGNATED for r in rows)}
    limit = any(r.get("stop") == P.STEP_LIMIT for rows in res["inverse"].values() for r in rows) or (steps_inv > 20 and not stagnated)
    effort = f"Newton-Schritte je t: bis {steps_log} mit der log-Barriere, bis {steps_inv} mit der Fremdbarriere"
    if stagnated:
        where = ", ".join(f"{'Σ 1/s' if p == 1.0 else f'Σ s^(−{p:g})'} ab t = {sci(t, 0)}" for p, t in stagnated.items())
        effort += f"; bei sehr großem t stagniert die Fremdbarriere an der Rundungsgrenze ({where}): der Punkt bewegt sich nicht mehr, Newton hält dort an, die Lücke bleibt davon unberührt."
    elif limit:
        effort += "; bei sehr großem t erreicht die Fremdbarriere die Schrittgrenze, die Lücke bleibt davon unberührt."
    else:
        effort += ": der Aufwand ist hier klein, das Problem der falschen Barriere ist die Lücke."
    return "warning", (f"Bei t = {sci(last_log['t'], 0)} liegt die Lücke mit der log-Barriere bei {sci(last_log['gap'])} (≈ {de(last_log['gap'] * last_log['t'], 1)}/t), "
                       f"mit φ = Σ 1/s bei {sci(last_inv['gap'])} (≈ {de(last_inv['gap'] * math.sqrt(last_inv['t']), 2)}/√t). Um 10⁻⁹ zu erreichen, bräuchte 1/s ein t von etwa {sci(t_need, 0)}, die log-Barriere {sci(t_log, 0)}. "
                       + effort)
