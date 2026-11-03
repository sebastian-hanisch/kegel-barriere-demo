"""Kegel-Barriere – der zentrale Pfad und was die Newton-Schritte kosten - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der Reihe „Konvexe und konische Optimierung“: Die Barrierenmethode aus Stück 2 wird geöffnet. Die Barriere macht aus den Kegelbedingungen einen glatten Strafterm, der zentrale Pfad ist die Spur seiner
Minimierer, und auf ihm kennt man die Lücke ν/t. Die Demo misst, wovon die Zahl der Newton-Schritte abhängt (kaum von ν), was ein großer Schritt in t kostet, wie weit der wahre Fehler unter der Schranke liegt
und warum die Barriere ein Logarithmus sein muss. Der gemeinsame Kern (kegel_kern.py) ist unverändert aus Stück 2.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kba_constants as C
import kba_forms as A
from kba_evaluation import (
    Settings,
    analyse,
    bound_rows,
    de,
    decrement_experiment,
    dimension_sweep,
    duplicate_check,
    factor_text,
    mu_rows,
    mu_text,
    nu_sweep,
    path_rows,
    shape_path,
    sci,
    verdict_bounds,
    verdict_decrement,
    verdict_mu,
    verdict_nu,
    verdict_path,
    verdict_run,
    verdict_wrong,
    wrong_barrier_experiment,
)
from kba_layout import dunkles_design, kopf_css
from kba_pdf_export import build_pdf
from kba_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kba_visualization import build_bounds, build_decrement, build_error, build_mu, build_nu, build_path_shapes, build_wrong

st.set_page_config(page_title="Kegel-Barriere – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _path(settings):
    a = _analysis(settings)
    return path_rows(a.selected), shape_path(a.selected, a.cloud.points)


@st.cache_data(show_spinner=False)
def _bounds(settings):
    a = _analysis(settings)
    return bound_rows(a), duplicate_check(a.selected)


@st.cache_data(show_spinner=False)
def _sweeps():
    return nu_sweep(), dimension_sweep()


@st.cache_data(show_spinner=False)
def _mu(settings):
    return mu_rows(_analysis(settings).selected)


@st.cache_data(show_spinner=False)
def _decrement(settings):
    return decrement_experiment(_analysis(settings).selected)


@st.cache_data(show_spinner=False)
def _wrong(settings):
    return wrong_barrier_experiment(_analysis(settings).cloud.points)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kba_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Kegel-Barriere – der zentrale Pfad und was die Newton-Schritte kosten")
st.markdown(
    """
**Die Barriere macht aus Nebenbedingungen einen glatten Strafterm.** Statt `min cᵀx` unter `s = b − A x ∈ K` minimiert man `t·cᵀx + φ(s)` für wachsendes `t`; `φ` ist die Summe der Barrieren der Kegel (−Σ log s, −log(s₀² − ‖s̄‖²), −log det S),
ihr Parameter **ν** zählt die Kegel (Orthant: Zeilen, Lorentz-Kegel: 2, PSD-Kegel: Matrixgröße). Die Minimierer bilden den **zentralen Pfad**, und auf ihm gilt: der Fehler ist höchstens `ν/t`.
Stück 2 hat diese Methode als Rechenwerk benutzt, Stück 4 ihr Dual gelesen; hier wird sie **geöffnet**: Wovon hängt die Zahl der Newton-Schritte ab, was kostet ein großer Schritt in `t`, wie weit liegt der wahre Fehler unter der Schranke, und warum muss die Barriere ein Logarithmus sein?
Stück 5 der Reihe „Konvexe und konische Optimierung“.

Weiter unten: **sechs Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Barrierenmethode", expanded=True):
    st.markdown(
        """
1. **Strafterm:** Am Rand eines Kegels wird `φ` unendlich. Deshalb bleibt jeder Newton-Schritt im Inneren, und `t` sagt, wie wichtig das Ziel gegen den Rand ist.
2. **Zentrierung:** Für festes `t` minimiert das Newton-Verfahren `f_t(x) = t·cᵀx + φ(b − A x)`. Das Ergebnis `x(t)` liegt auf dem zentralen Pfad. Mit dem Dual `z = −∇φ/t` (Stück 4) ist `zᵀs = ν/t`, also `cᵀx(t) − p* ≤ ν/t`.
3. **Pfad folgen:** `t` wird in jeder Runde mit dem Faktor `μ` multipliziert; der letzte Punkt dient als Start der nächsten Zentrierung. Gestoppt wird bei `ν/t ≤` Ziel.
4. **Zwei Phasen je Zentrierung:** Ist `x` weit vom Pfad weg, sind die Newton-Schritte gedämpft (Länge unter 1); in der Nähe konvergiert das Dekrement quadratisch.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Form**")
    form = st.selectbox("Gesuchte Form", A.FORMS, key="form_select", format_func=A.FORM_NAMES.get,
                        help="Für welche der sechs Formen das Hauptdiagramm und die Experimente 1, 4, 5 gelten; die Tabelle in Experiment 2 vergleicht immer alle sechs.")
    st.markdown("**Punktwolke**")
    n = st.slider("Punkte n", *bounds("n_slider"), key="n_slider", help="Wie viele Punkte die Wolke hat.")
    stretch = st.slider("Streckung", *bounds("stretch_slider"), step=0.1, key="stretch_slider", help="Wie viel breiter die Wolke entlang ihrer Längsachse ist als quer dazu.")
    angle = st.slider("Drehung (Grad)", *bounds("angle_slider"), key="angle_slider", help="Winkel der Längsachse gegen die x-Achse.")
    seed = st.number_input("Zufalls-Seed der Wolke", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Wolke generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"form_select": form, "n_slider": int(n), "stretch_slider": float(stretch), "angle_slider": int(angle), "seed_input": int(seed)})

settings = Settings(form, int(n), float(stretch), int(angle), int(seed))
a = _analysis(settings)
run = a.selected
path, shapes = _path(settings)
m_final = path[-1]["m"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if run.problems:
    st.error("Die Rückumformung besteht nicht: " + "; ".join(run.problems))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Barriere-Parameter ν", run.nu, help="Summe der Barriere-Parameter der Kegel: Orthant je Zeile 1, Lorentz-Kegel 2, PSD-Kegel die Matrixgröße.")
m2.metric("Äußere Runden", run.result.rounds, help="Wie viele Zentrierungen nötig waren (t wächst je Runde um den Faktor 8; Ziel: bewiesene Lücke 1e-9).")
m3.metric("Newton-Schritte gesamt", run.result.newton_steps, help="Summe der Newton-Schritte aller Zentrierungen in dieser Implementierung.")
m4.metric("Bewiesene Lücke ν/t", sci(run.result.gap_bound), help="Obergrenze für cᵀx − p* am Ende.")
m5.metric("Wahrer Fehler · t = m", de(m_final, 1), help="t mal der wahre Fehler bei t = 10⁷: geht gegen eine kleine ganze Zahl (Zahl der aktiven Bedingungen), nicht gegen ν.")
st.plotly_chart(build_path_shapes(a.cloud.points, form, shapes, A.decode(form, a.cloud.points, run.result.x)), width="stretch", key="main_chart")
st.caption("Gepunktet: die Form auf dem zentralen Pfad bei einigen t; sie zieht sich von weit außen auf die Lösung zusammen. Alle Formen sind um den Mittelwert der Wolke zentriert; die Ellipsen haben ihren Mittelpunkt im Ursprung.")

if st.button("📄 PDF der aktuellen Wolke erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a, path))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kegel-barriere.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was kostet die Barriere?")
verdict(verdict_run(run, A.FORM_NAMES[form], m_final))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Der Pfad und sein Fehler")
st.caption("Pfadpunkte der gewählten Form von t = 0,1 bis 10⁷, jeder vom vorigen aus gerechnet. Der Fehler wird gegen den Referenzwert (Lücke 10⁻¹²) gemessen.")
e1, e2 = st.columns([3, 2])
e1.plotly_chart(build_error(path, run.nu, m_final), width="stretch", key="error_chart")
e2.dataframe(pd.DataFrame([{"t": sci(r["t"], 0), "Wert": round(r["value"], 5), "Fehler": sci(r["error"]), "ν/t": sci(r["bound"]), "t · Fehler": round(r["m"], 3), "Newton": r["steps"]} for r in path]),
             hide_index=True, width="stretch")
verdict(verdict_path(path, run.nu))

st.markdown("---")

st.subheader("🔬 Experiment 2: Die Schranke ν/t und der wahre Fehler m/t")
st.caption("Alle sechs Formen derselben Wolke: ν wächst mit der Zahl der Zeilen, `m = t · Fehler` bei t = 10⁷ bleibt eine kleine ganze Zahl. Dazu: jede Nebenbedingung der gewählten Form zweimal hingeschrieben.")
brows, dup = _bounds(settings)
st.plotly_chart(build_bounds(brows), width="stretch", key="bounds_chart")
st.dataframe(pd.DataFrame([{"Form": r["name"], "ν": r["nu"], "m = t·Fehler": round(r["m"], 3), "aktive Bedingungen": r["active"], "ν/m": round(r["ratio"], 1), "Runden": r["rounds"], "Newton": r["newton"],
                            "zentriert": "ja" if r["centered"] else "nein"} for r in brows]), hide_index=True, width="stretch")
verdict(verdict_bounds(brows, dup))

st.markdown("---")

st.subheader("🔬 Experiment 3: Newton-Schritte gegen ν und Dimension")
st.caption(f"Kugel auf {C.N_SWEEP[0]} bis {C.N_SWEEP[-1]} Punkten (Median über {len(C.SWEEP_SEEDS)} feste Wolken) und allgemeine Ellipse in d = {C.D_SWEEP[0]} bis {C.D_SWEEP[-1]} Dimensionen (Median über {len(C.D_SWEEP_SEEDS)} Wolken).")
nrows, drows = _sweeps()
st.plotly_chart(build_nu(nrows, drows), width="stretch", key="nu_chart")
st.dataframe(pd.DataFrame([{"Kugel: Punkte": r["n"], "ν": r["nu"], "Zeilen": r["rows"], "Newton (Median)": r["newton"], "min–max": f"{r['newton_min']}–{r['newton_max']}", "Runden": r["rounds"]} for r in nrows]),
             hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"Ellipse: d": r["d"], "Punkte": r["points"], "Variablen": r["variables"], "PSD-Größe": r["psd_size"], "ν": r["nu"], "Newton (Median)": r["newton"], "min–max": f"{r['newton_min']}–{r['newton_max']}"}
                           for r in drows]), hide_index=True, width="stretch")
verdict(verdict_nu(nrows, drows))

st.markdown("---")

st.subheader("🔬 Experiment 4: Der Schrittfaktor μ")
st.caption("Die gewählte Form mit verschiedenen Faktoren μ, mit denen `t` je Runde multipliziert wird (Ziel 10⁻⁹). Der Kurzschritt `μ = 1 + 1/√ν` ist die Regel der Theorie.")
mrows = _mu(settings)
st.plotly_chart(build_mu(mrows), width="stretch", key="mu_chart")
st.dataframe(pd.DataFrame([{"μ": mu_text(None) if r["short"] else mu_text(r["mu"]), "Faktor": round(r["mu"], 3), "Runden": r["rounds"], "Newton gesamt": r["newton"], "Newton je Runde": round(r["per_round"], 2), "weit vom Pfad": "ja" if r.get("far_from_path") else "–"} for r in mrows]),
             hide_index=True, width="stretch")
verdict(verdict_mu(mrows))

st.markdown("---")

st.subheader("🔬 Experiment 5: Gedämpft, dann quadratisch")
st.caption(f"Die Newton-Schritte einzelner Zentrierungen der gewählten Form: die erste vom Start aus und Sprünge von t = {sci(C.DECREMENT_T0, 0)} auf das {factor_text(C.DECREMENT_JUMPS[0])}fache und das {factor_text(C.DECREMENT_JUMPS[1])}fache.")
dec = _decrement(settings)
st.plotly_chart(build_decrement(dec), width="stretch", key="dec_chart")
verdict(verdict_decrement(dec))

st.markdown("---")

st.subheader("🔬 Experiment 6: Warum ein Logarithmus?")
st.caption("Der Würfel (ein LP) der eingestellten Wolke mit der log-Barriere und mit `φ = Σ 1/s` und `Σ 1/s²`. Bis t = 10⁶ ist Newton unauffällig (4 bis 10 Schritte); die Lücke nicht. Bei sehr großem t (10⁸ bis 10¹⁰) stagniert die Fremdbarriere an der Rundungsgrenze: der Punkt bewegt sich nicht mehr, und Newton hält nach wenigen Schritten an (das Verdikt nennt, ab welchem t).")
wrong = _wrong(settings)
st.plotly_chart(build_wrong(wrong), width="stretch", key="wrong_chart")
verdict(verdict_wrong(wrong))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Newton-Schritte hängen kaum an ν** | Das ist ein Messbefund dieser Implementierung (Faktor 8, Start durch Konstruktion, Zentrieren auf 10⁻⁶), keine Eigenschaft der Barriere. Die Theorie nennt für den Kurzschritt eine Schranke mit √ν · log(1/ε) und zeigt nicht, dass sie scharf ist. | – |
| **Der Start ist streng zulässig** | Ohne einen solchen Punkt gibt es keine Barriere am Start; Phase I und die Einbettung lösen das. | Stück 6 |
| **Der Pfad existiert** | Ohne streng zulässiges Dual läuft er davon (Stück 4). | – |
| **t bleibt unter etwa 10¹⁰ bis 10¹¹** | Die Hesse-Matrix wird numerisch singulär; der Löser beendet den Lauf mit der zuletzt bewiesenen Lücke, und Experiment 2 meldet nicht zentrierte Punkte. | Stück 6 |
| **Die Barriere ist eine selbstkonkordante Logarithmus-Barriere** | Für andere Barrieren (Experiment 6) gilt die Schranke ν/t nicht; der Fehler fällt viel langsamer. | – |
| **Aktive Bedingungen zählen m** | Für Würfel, Raute und die Kugeln ist `m` die Zahl der aktiven Bedingungen (bzw. Stützpunkte); bei den Ellipsen zählt statt der Blöcke der Rang des Dualblocks (allgemeine Ellipse: 3 Punkte + Rang 2 = 5), die Zählung im Code folgt dieser Regel nicht, dort stehen beide Zahlen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Barrieren.** Orthant: $\phi(s) = -\sum_j \log s_j$, $\nu = m$. Lorentz-Kegel: $\phi(s) = -\log(s_0^2 - \lVert \bar s \rVert^2)$, $\nu = 2$. PSD-Kegel der Größe $k$: $\phi(S) = -\log\det S$, $\nu = k$. Für ein Produkt von Kegeln addieren sich $\phi$ und $\nu$.
Alle drei sind selbstkonkordant; $\nu$ ist der kleinste Parameter, für den $\nabla\phi(s)^\top (\nabla^2\phi(s))^{-1} \nabla\phi(s) \le \nu$ gilt.

**Zentraler Pfad.** $x(t) = \arg\min_x\; t\,c^\top x + \phi(b - Ax)$. Stationarität: $t\,c = A^\top \nabla\phi(s)$ für $s = b - Ax(t)$. Mit $z(t) = -\nabla\phi(s)/t \in K$ gilt $A^\top z(t) = -c$ (dual zulässig) und
$z(t)^\top s = -\nabla\phi(s)^\top s / t = \nu / t$, denn $\nabla\phi(s)^\top s = -\nu$ für alle drei Barrieren. Also $c^\top x(t) + b^\top z(t) = \nu/t$ und $c^\top x(t) - p^* \le \nu/t$.

**Wahrer Fehler.** Im LP ist $c^\top x(t) - p^* = \sum_{i \text{ aktiv}} 1/t \cdot (1 + o(1))$: jede aktive Zeile trägt $1/t$ bei, weil $s_i z_i = 1/t$ und $z_i \to z_i^*$ (nichtentartet). Deshalb ist $t\,(c^\top x(t) - p^*) \to m$, die Zahl der aktiven Zeilen, unabhängig von $\nu$ (alle Zeilen). Für Lorentz- und PSD-Kegel gilt der Gedanke blockweise mit dem Rang: ein aktiver Block trägt den Rang seines Dualblocks bei (Kugeln: 1; allgemeine Ellipse: PSD-Block mit Rang 2, also $m = 3 + 2 = 5$); die Ellipsen zeigen deshalb kleine ganze Zahlen.

**Verdoppeltes $\nu$.** Schreibt man jede Nebenbedingung zweimal hin, ist $\phi_{\text{dup}} = 2\phi$ und $f_t^{\text{dup}} = t\,c^\top x + 2\phi = 2\bigl((t/2)\,c^\top x + \phi\bigr)$: der Pfad ist derselbe, nur mit $t/2$ statt $t$; die Lückenschranke ist $2\nu/t = \nu/(t/2)$.

**Newton.** Das Dekrement $\lambda(x)^2 = \nabla f_t^\top (\nabla^2 f_t)^{-1}\nabla f_t$. Für selbstkonkordante $f_t$ gilt: ist $\lambda$ groß, senkt ein gedämpfter Schritt $f_t$ um mindestens eine Konstante; ist $\lambda$ klein, konvergiert das volle Newton-Verfahren quadratisch. Die Demo misst $\lambda^2/2$ und die Schrittlänge je Schritt.

**Äußere Runden.** Ab $t_0$ mit Faktor $\mu$: die letzte Runde hat $t_0\mu^k \ge \nu/\varepsilon$, also $k = \lceil \log_\mu(\nu/(\varepsilon t_0)) \rceil$ und $k + 1$ Runden, wenn die erste Zentrierung bei $t_0$ mitgezählt wird.

**Falsche Barriere im LP.** Für $\phi(s) = \sum_i s_i^{-p}$ ist Stationarität $t\,c = A^\top p\, s^{-p-1}$ (mit $A$ wie in $s = b - Ax$), also hat das Dual $z_i = p\, s_i^{-p-1}/t$ und für aktive Zeilen $s_i = (p/(t z_i))^{1/(p+1)}$. Die Lücke ist $\sum_i z_i s_i = \tfrac{p}{t}\sum_i s_i^{-p}$, also
$\sum_i \tfrac{p}{t}\,(t z_i / p)^{p/(p+1)} \propto t^{-1/(p+1)}$: für $p = 1$ wie $t^{-1/2}$ (im Würfel mit zwei aktiven Zeilen, $z_i = 1/2$: $2\sqrt{z_i/t} = \sqrt{2}/\sqrt{t}$, gemessen 1,414), für $p = 2$ wie $t^{-1/3}$. Mit dem Logarithmus ($p \to 0$) wie $t^{-1}$.

**Literatur.** Y. Nesterov und A. Nemirovskii: *Interior-Point Polynomial Algorithms in Convex Programming*, SIAM Studies in Applied Mathematics 13, SIAM 1994 (Selbstkonkordanz, Barriere-Parameter, Iterationsschranken mit $\sqrt{\nu}$). S. Boyd und L. Vandenberghe: *Convex Optimization*, Cambridge University Press 2004, Kapitel 11 (Barrierenmethode, zentraler Pfad, Komplexitätsanalyse über Selbstkonkordanz; dort auch die Beobachtung, dass der Gesamtaufwand für $\mu$ ab etwa 10 wenig davon abhängt).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html)."
)
