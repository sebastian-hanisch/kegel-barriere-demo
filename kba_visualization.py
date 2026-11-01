"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import kba_constants as C
import kba_forms as A


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def outline(shape: A.Shape) -> tuple[np.ndarray, np.ndarray]:
    """Umriss der Form als Linienzug (x, y): Würfel = Quadrat, Raute = Quadrat auf der Spitze, Kugel = Kreis, Ellipse aus x^T P x = 1."""
    t = np.linspace(0.0, 2.0 * math.pi, 241)
    cx, cy = shape.center
    if shape.kind == "ball":
        return cx + shape.radius * np.cos(t), cy + shape.radius * np.sin(t)
    if shape.kind == "box":
        r = shape.radius
        return np.array([cx - r, cx + r, cx + r, cx - r, cx - r]), np.array([cy - r, cy - r, cy + r, cy + r, cy - r])
    if shape.kind == "diamond":
        r = shape.radius
        return np.array([cx + r, cx, cx - r, cx, cx + r]), np.array([cy, cy + r, cy, cy - r, cy])
    eig, vec = np.linalg.eigh(shape.matrix)
    pts = vec @ (np.stack([np.cos(t), np.sin(t)]) / np.sqrt(eig)[:, None])
    return cx + pts[0], cy + pts[1]


def build_path_shapes(points: np.ndarray, form: str, path: list[tuple[float, A.Shape]], final: A.Shape):
    """Die Form der gewählten Aufgabe an einigen Pfadpunkten (graue Linien, beschriftet mit t) und die Lösung (Farbe der Form)."""
    fig = go.Figure()
    for t, shape in path:
        xs, ys = outline(shape)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"t = {t:g}", line=dict(width=1.5, dash="dot"), hoverinfo="skip"))
    xs, ys = outline(final)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Lösung", line=dict(color=C.COLORS[form], width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", name="Punkte", marker=dict(size=8, color=C.COLOR_POINT, line=dict(width=1, color="#444")),
                             hovertemplate="(%{x:.2f}; %{y:.2f})<extra></extra>"))
    pts = np.vstack([points.min(axis=0), points.max(axis=0)])
    span = float(np.ptp(pts, axis=0).max())
    mid = pts.mean(axis=0)
    fig.update_xaxes(title_text="x", range=[mid[0] - 1.1 * span, mid[0] + 1.1 * span])
    fig.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1, range=[mid[1] - 1.1 * span, mid[1] + 1.1 * span])
    return _base(fig, 440)


def build_error(rows: list[dict], nu: int, m: float):
    """Wahrer Fehler und bewiesene Schranke nu/t über t (doppelt logarithmisch), dazu die Gerade m/t."""
    ts = [r["t"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=[max(r["bound"], 1e-300) for r in rows], mode="lines+markers", name=f"bewiesene Schranke ν/t (ν = {nu})", line=dict(color=C.COLOR_BOUND, width=2.5)))
    fig.add_trace(go.Scatter(x=ts, y=[max(r["error"], 1e-300) for r in rows], mode="lines+markers", name="wahrer Fehler", line=dict(color=C.COLOR_ERROR, width=2.5)))
    fig.add_trace(go.Scatter(x=ts, y=[m / t for t in ts], mode="lines", name=f"m/t mit m = {m:.1f}", line=dict(color=C.COLOR_REF, width=1.5, dash="dash")))
    fig.update_xaxes(title_text="t (logarithmisch)", type="log")
    fig.update_yaxes(title_text="Fehler und Schranke (logarithmisch)", type="log")
    return _base(fig)


def build_bounds(rows: list[dict]):
    """ν und m je Form (Balken, logarithmisch): ν wächst mit den Zeilen, m bleibt eine kleine ganze Zahl."""
    names = [r["name"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r["nu"] for r in rows], name="ν (Schranke ν/t)", marker_color=C.COLOR_BOUND))
    fig.add_trace(go.Bar(x=names, y=[r["m"] for r in rows], name="m (wahrer Fehler m/t)", marker_color=C.COLOR_ERROR))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Wert (logarithmisch)", type="log")
    return _base(fig)


def build_nu(rows: list[dict], dim_rows: list[dict]):
    """Links: Newton-Schritte der Kugel über ν (mit Spannweite); rechts: Newton-Schritte der Ellipse über die Zahl der Variablen. Beide Achsen für die Schritte beginnen bei null."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["nu"] for r in rows], y=[r["newton"] for r in rows], mode="lines+markers", name="Kugel (ν = 2n, 3 Variablen)",
                             error_y=dict(type="data", symmetric=False, array=[r["newton_max"] - r["newton"] for r in rows], arrayminus=[r["newton"] - r["newton_min"] for r in rows]),
                             line=dict(color=C.COLORS["ball_soc"], width=3)))
    fig.add_trace(go.Scatter(x=[r["nu"] for r in dim_rows], y=[r["newton"] for r in dim_rows], mode="lines+markers", name="Ellipse allgemein (d = 2 bis 8)",
                             error_y=dict(type="data", symmetric=False, array=[r["newton_max"] - r["newton"] for r in dim_rows], arrayminus=[r["newton"] - r["newton_min"] for r in dim_rows]),
                             line=dict(color=C.COLORS["ellipse_psd"], width=3)))
    fig.add_trace(go.Scatter(x=[r["nu"] for r in rows], y=[30.0 * math.sqrt(r["nu"] / rows[0]["nu"]) for r in rows], mode="lines", name="zum Vergleich: wachsend wie √ν", line=dict(color=C.COLOR_REF, dash="dash", width=1.5)))
    fig.update_xaxes(title_text="Barriere-Parameter ν (logarithmisch)", type="log")
    fig.update_yaxes(title_text="Newton-Schritte gesamt", rangemode="tozero")
    return _base(fig)


def build_mu(rows: list[dict]):
    """Gesamtzahl der Newton-Schritte und der äußeren Runden über den Schrittfaktor μ (logarithmisch); der Kurzschritt links."""
    mus = [r["mu"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mus, y=[r["newton"] for r in rows], mode="lines+markers", name="Newton-Schritte gesamt", line=dict(color=C.COLOR_ERROR, width=3)))
    fig.add_trace(go.Scatter(x=mus, y=[r["rounds"] for r in rows], mode="lines+markers", name="äußere Runden", line=dict(color=C.COLOR_BOUND, width=2, dash="dot")))
    short = next(r for r in rows if r["short"])
    fig.add_trace(go.Scatter(x=[short["mu"]], y=[short["newton"]], mode="markers", name="Kurzschritt 1 + 1/√ν", marker=dict(symbol="diamond", size=14, color=C.COLOR_REF)))
    flagged = [r for r in rows if r.get("far_from_path")]
    if flagged:
        fig.add_trace(go.Scatter(x=[r["mu"] for r in flagged], y=[r["newton"] for r in flagged], mode="markers", name="weit vom Pfad (Wert falsch)", marker=dict(symbol="x", size=13, color="#b34a4a", line=dict(width=2))))
    fig.update_xaxes(title_text="Schrittfaktor μ (t wird je Runde mit μ multipliziert, logarithmisch)", type="log")
    fig.update_yaxes(title_text="Anzahl (logarithmisch)", type="log")
    return _base(fig)


def build_decrement(res: dict):
    """Halbes Dekrement-Quadrat je Newton-Schritt (logarithmisch): erste Zentrierung und die beiden Sprünge; Marker größer, wo der Schritt gedämpft war."""
    fig = go.Figure()
    series = [("erste Zentrierung", res["first"])] + [(f"Sprung um {j['factor']:g}", j["steps"]) for j in res["jumps"]]
    for name, steps in series:
        fig.add_trace(go.Scatter(x=list(range(1, len(steps) + 1)), y=[max(s.lam2_half, 1e-300) for s in steps], mode="lines+markers", name=name,
                                 marker=dict(size=[16 if s.length < 0.99 else 8 for s in steps]),
                                 hovertemplate="Schritt %{x}: %{y:.1e}<extra></extra>"))
    fig.update_xaxes(title_text="Newton-Schritt (große Marker: gedämpft, Länge unter 1)", dtick=1)
    fig.update_yaxes(title_text="halbes Dekrement-Quadrat (logarithmisch)", type="log")
    return _base(fig)


def build_wrong(res: dict):
    """Lücke über t (doppelt logarithmisch): log-Barriere gegen Σ 1/s^p; dazu die Referenzgeraden 2/t."""
    fig = go.Figure()
    ts = [r["t"] for r in res["log"]]
    fig.add_trace(go.Scatter(x=ts, y=[max(r["gap"], 1e-300) for r in res["log"]], mode="lines+markers", name="log-Barriere", line=dict(color=C.COLOR_ERROR, width=3)))
    for p, rows in res["inverse"].items():
        fig.add_trace(go.Scatter(x=[r["t"] for r in rows], y=[max(r["gap"], 1e-300) for r in rows], mode="lines+markers", name=f"φ = Σ s^(−{p:g})", line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=[ts[0], ts[-1]], y=[1e-9, 1e-9], mode="lines", name="Ziel 10⁻⁹", line=dict(color=C.COLOR_REF, dash="dash", width=1.5)))
    fig.update_xaxes(title_text="t (logarithmisch)", type="log")
    fig.update_yaxes(title_text="Lücke c^T x(t) − p* (logarithmisch)", type="log")
    return _base(fig)
