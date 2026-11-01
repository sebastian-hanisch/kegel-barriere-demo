"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import kba_forms as A

DEFAULT_FORM = "ball_soc"
N_MIN, N_MAX, DEFAULT_N = 8, 30, 20
STRETCH_MIN, STRETCH_MAX, DEFAULT_STRETCH = 1.2, 6.0, 4.0
ANGLE_MIN, ANGLE_MAX, DEFAULT_ANGLE = 0, 90, 45
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(100000, 100005))   # feste Wolken für alle Sweeps: nie ein einzelner Seed als Befund

PATH_TS = (1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7)      # Pfadpunkte der Tabelle und des Fehlerdiagramms
SHAPE_TS = (2.0, 5.0, 20.0, 100.0)                             # Pfadpunkte, an denen die Form gezeichnet wird (dazu die Lösung)
ERROR_T = 1e7                                                  # t, bei dem t * Fehler gelesen wird (m)
ACTIVE_MARGIN = 1e-3                                           # Schlupf unter dem eine Bedingung als aktiv zählt (bei t = 1e7)

N_SWEEP = (4, 6, 10, 20, 40, 80, 160, 320)                     # Punktzahlen der Kugel (nu = 2 n)
D_SWEEP = (2, 3, 4, 5, 6, 8)                                   # Dimensionen der allgemeinen Ellipse (PSD-Block 2 d)
D_SWEEP_SEEDS = SWEEP_SEEDS[:3]
MUS = (None, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 100.0, 1e3, 1e5)  # None = Kurzschritt 1 + 1/sqrt(nu)
DECREMENT_T0 = 1e3                                             # Pfadpunkt, von dem aus die Sprünge starten
DECREMENT_JUMPS = (8.0, 1e3)                                   # Sprung in t um diesen Faktor
WRONG_TS = (1.0, 1e2, 1e4, 1e6, 1e8, 1e10)
WRONG_PS = (1.0, 2.0)                                          # Exponenten der Fremdbarriere sum s^(-p)

COLORS = {"cube": "#4c78a8", "diamond": "#f58518", "ball_soc": "#2F6B65", "ball_psd": "#7a9e9a", "ellipse_axis": "#c0504d", "ellipse_psd": "#8a2b27"}
COLOR_ERROR = "#2F6B65"
COLOR_BOUND = "#c0504d"
COLOR_REF = "#444444"
COLOR_POINT = "rgba(150,150,150,0.85)"

PRESETS = {
    "Standardfall: Kugel (Lorentz-Kegel)": {"form": DEFAULT_FORM, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Der Würfel: ein LP": {"form": "cube", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Dieselbe Kugel als SDP": {"form": "ball_psd", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Allgemeine Ellipse: der Pfad dreht sich": {"form": "ellipse_psd", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Viele Punkte, viel ν": {"form": DEFAULT_FORM, "n": 30, "stretch": 2.0, "angle": 10, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall: Kugel (Lorentz-Kegel)": "20 Punkte, in die Länge gezogen und um 45 Grad gedreht; der Pfad der kleinsten Kugel schrumpft von sehr groß auf die Lösung.",
    "Der Würfel: ein LP": "Die l-unendlich-Kugel ist ein reines LP mit 80 Zeilen: die Barriere ist die Summe von 80 Logarithmen, ν = 80.",
    "Dieselbe Kugel als SDP": "Dieselbe Kugel als 3 x 3-Matrizen im PSD-Kegel: ν = 60 statt 40, die Newton-Schritte bleiben ähnlich.",
    "Allgemeine Ellipse: der Pfad dreht sich": "Die Ellipse mit PSD-Kegel: ihr Pfad richtet sich beim Wachsen von t nach den Punkten aus.",
    "Viele Punkte, viel ν": "30 Punkte, rundere Wolke: ν = 60 für die Kugel, mehr aktive Stützpunkte möglich.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["form"] in A.FORMS for p in PRESETS.values())
