# Kegel-Barriere – der zentrale Pfad und was die Newton-Schritte kosten – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-barriere-demo.streamlit.app/)** -->

Stück 5 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Vorgänger sind [konvexitaet-zertifikat-demo](https://github.com/sebastian-hanisch/konvexitaet-zertifikat-demo) (Stück 1), [kegelform-demo](https://github.com/sebastian-hanisch/kegelform-demo) (Stück 2: der Löser als Rechenwerk) und [kegel-dualitaet-demo](https://github.com/sebastian-hanisch/kegel-dualitaet-demo) (Stück 4: sein Dual); Stück 3 (DCP, optional) ist übersprungen. Nachfolger ist der Primal-Dual-Schritt (Stück 6). Die Linien-Spezifikation und die Einbindung in die Website folgen, wenn die Reihe vollständig ist.

## Kernfrage

**Was kostet die Barriere?** Die Barrierenmethode minimiert `t·cᵀx + φ(b − A x)` für wachsendes `t`; `φ` ist die Summe der Barrieren der Kegel (−Σ log s, −log(s₀² − ‖s̄‖²), −log det S), ihr Parameter **ν** zählt die Kegel (Orthant: Zeilen, Lorentz-Kegel: 2, PSD-Kegel: Matrixgröße). Auf dem zentralen Pfad gilt `zᵀs = ν/t` und damit `cᵀx(t) − p* ≤ ν/t`. Die Demo öffnet die Methode, die Stück 2 als Rechenwerk benutzt und Stück 4 ausgelesen hat, und misst, **wovon die Zahl der Newton-Schritte abhängt, was ein großer Schritt in `t` kostet, wie weit der wahre Fehler unter der Schranke liegt und warum die Barriere ein Logarithmus sein muss.**

Das Beispiel bleibt die kleinste umschließende Form einer Punktwolke in den sechs Formen aus Stück 2 (Würfel, Raute, Kugel als Lorentz- und PSD-Kegel, Ellipse achsenparallel und allgemein; die Ellipsen um den Ursprung mit Ziel kleinste Summe der quadrierten Halbachsen).

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2 und 4, unverändert): Barrieren der drei Kegel, Newton-Verfahren mit Rücksetzen, Barrierenmethode mit festem Faktor `μ` für `t` (hier 8).
- **Pfad** (`kba_path.py`): Pfadpunkte `x(t)` durch Newton bis zum halben Dekrement-Quadrat 10⁻²⁰, Fortsetzung vom vorigen Punkt; Dekrement-Folge einer Zentrierung (Dekrement und Schrittlänge je Schritt); Zahl der aktiven Bedingungen; jede Nebenbedingung zweimal (ν doppelt); Schrittfaktor-Reihe; Fremdbarriere `Σ s^(−p)` nur für Orthant-Programme, mit eigenem Newton-Verfahren.
- **Rückumformung** wie in Stück 2: das Ergebnis wird in der Ausgangsaufgabe und im Kegelprogramm geprüft.
- **Kein Eingriff in den Kern:** alles Neue steht in `kba_path.py`; BLAS-Threads laufen mit einem Thread (`threadpoolctl`), weil die Matrizen klein sind.

## Methodik

Sechs Experimente mit live berechnetem Verdikt: (1) der Pfad und sein Fehler (dazu die Form an einigen Pfadpunkten); (2) die Schranke `ν/t` gegen den wahren Fehler `m/t` für alle sechs Formen, mit dem Verdoppeln von ν; (3) Newton-Schritte gegen ν und Dimension (Kugel bis 320 Punkte, allgemeine Ellipse bis d = 8); (4) der Schrittfaktor μ (Kurzschritt gegen große Schritte); (5) gedämpft, dann quadratisch (Dekrement-Folge); (6) die falsche Barriere `Σ 1/s` und `Σ 1/s²` gegen den Logarithmus im LP.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung auf Linux, steht die gemessene Zahl trotzdem hier); Sweeps laufen über feste Wolken `SWEEP_SEEDS`. Die Standardwolke (20 Punkte, Streckung 4, 45 Grad, Seed 35) ist ausdrücklich eine einzelne.

| Frage | Befund |
|---|---|
| Standardwolke, Kugel (Lorentz-Form) | ν = 40, **13 Runden, 34 Newton-Schritte** bis zur bewiesenen Lücke 1,7·10⁻¹⁰. ν der sechs Formen: Würfel 80, Raute 100, Kugel 40, Kugel als SDP 60, Ellipse achsenparallel 24, Ellipse allgemein 24 |
| Der Fehler auf dem Pfad (Kugel) | `t · Fehler` = 39,3 (t = 0,1), 33,3 (1), 8,2 (10), 2,43 (100), 2,05 (10³), 2,005 (10⁴), danach 2,000 bis t = 10⁷. Der Fehler fällt also wie **2/t**, die bewiesene Schranke wie 40/t: ν/m = 20. Bei kleinem t ist R(t) ≈ ν/t (400 bei t = 0,1): der Pfad kommt aus dem Unendlichen, weil R keine obere Schranke hat |
| `m = t · Fehler` bei t = 10⁷ über die sechs Formen (Standardwolke) | Würfel **2,0** (2 aktive Zeilen), Raute **6,0** (6), Kugel **2,0** (2 Stützpunkte), Kugel als SDP **2,0** (2), Ellipse achsenparallel **3,0** (3), Ellipse allgemein **5,0** (4 Blöcke gezählt; der aktive PSD-Block trägt den Rang seines Dualblocks bei, hier 2: 3 + 2 = 5). Die Schranke ν/t ist um den Faktor 4,8 bis 40 zu groß. Die Ellipse achsenparallel ist bei t = 10⁷ nicht mehr genau zentriert (m = 2,97, Tabelle meldet es) |
| Jede Nebenbedingung zweimal hinschreiben (Kugel) | ν 40 → 80; der Pfad verschiebt sich exakt: `x_dup(t) = x(t/2)` (Abweichung 1,8·10⁻¹⁵); Runden (13) und Newton-Schritte (34) bleiben gleich, der Wert auch |
| Newton-Schritte gegen ν (Kugel, n = 4 bis 320 Punkte, ν = 8 bis 640, Zeilen 12 bis 960) | **Median 30, 32, 33, 33, 32, 34, 34, 33** (je 5 feste Wolken; min 28, max 39), 12 bis 13 Runden: praktisch flach, die Rundenzahl wächst nur mit log ν |
| Newton-Schritte gegen Dimension (allgemeine Ellipse, d = 2 bis 8, Variablen 6 bis 72, ν = 14 bis 44) | Median 42, 46, 57, 61, 58, 68: Variablen **× 12**, Schritte **× 1,6** (Median aus 3 Wolken; mit den Daten ist ein Faktor zwischen etwa 1,2 und 1,9 vereinbar) |
| Schrittfaktor μ (Kugel, Standardwolke, Ziel 10⁻⁹) | Kurzschritt 1 + 1/√ν = 1,16: **160 Runden, 343 Newton-Schritte**; μ = 1,5: 174; 2: 62; 4: 43; **8: 34**; 16: 29; **32: 28**; 100: 42; 1000: 29; 10⁵: 30 (4 Runden). Für μ von 4 bis 10⁵ liegt der Gesamtaufwand bei 28 bis 43 Schritten, je Runde 2,4 bis 7,5: auf der Kugel der Standardwolke braucht der Kurzschritt etwa 12-mal so viele Schritte wie μ = 32 (andere Wolken und Formen nicht ausgewertet) |
| Gedämpft, dann quadratisch (Kugel, Standardwolke) | Erste Zentrierung vom Start: halbes Dekrement-Quadrat 8·10⁻³ → 5·10⁻⁶ → 2·10⁻¹² (volle Schritte, quadratische Konvergenz: λ₊ ≈ 0,2·λ² mit dem Dekrement λ). Sprung t × 8: **1 gedämpfter von 5 Schritten** (Länge 0,13). Sprung t × 1000: **3 gedämpfte von 6** (Länge 0,00, 0,25, 0,5) |
| Falsche Barriere im LP (Würfel, Standardwolke) | **log:** Lücke = 2/t (2,0·10⁻¹⁰ bei t = 10¹⁰). **φ = Σ 1/s:** 1,414/√t (1,4·10⁻⁵ bei t = 10¹⁰); **φ = Σ 1/s²:** 1,587/t^(1/3) = 4^(1/3)/t^(1/3). Für 10⁻⁹ bräuchte `Σ 1/s` ein t von etwa 2·10¹⁸. Newton-Schritte je t sind bis t = 10⁶ klein (4 bis 10); bei sehr großem t erreicht die Fremdbarriere die Rundungsgrenze und **stagniert**: die Rundung des Gradienten hält das Dekrement-Quadrat über der Toleranz 10⁻¹⁶, obwohl der Punkt sich nicht mehr bewegt. Früher lief der Zähler dann bis an die Schrittgrenze 200 (Σ 1/s² bei t = 10⁸, Σ 1/s bei t = 10¹⁰); jetzt hält Newton an, sobald ein Schritt x um weniger als 8 · eps · max(1, ‖x‖∞) bewegt, und meldet „stagniert“ (Σ 1/s bei t = 10¹⁰: 6 Schritte, Σ 1/s² bei t = 10⁸: 8 und bei t = 10¹⁰: 14; alle kleineren t konvergieren mit unveränderten Schrittzahlen); der Pfadpunkt und die Lücke sind gleich geblieben und passen weiter zur Formel |

## Befunde und Korrekturen gegenüber dem Plan

- **Der Exponent der Fremdbarriere.** Ich hatte für `Σ s^(−p)` einen Lückenabfall `t^(−p/(p+1))` hergeleitet; für p = 1 fällt das mit dem richtigen Wert `t^(−1/2)` zusammen. Die Messung für p = 2 zeigte `t^(−1/3)`: richtig ist `t^(−1/(p+1))` (aus `z_i = p s_i^(−p−1)/t` und Lücke `Σ z_i s_i`). Die Herleitung im 📐-Expander ist korrigiert, die Konstanten (1,414 und 1,587) folgen aus zwei aktiven Zeilen mit `z = ½`.
- **Die Ellipsen:** Der Plan nannte `m` = 3 bis 5; die Standardwolke gibt 3,0 und 5,0. Die Zahl der gezählten aktiven Bedingungen passt bei der Ellipse achsenparallel (3), bei der allgemeinen nicht (4 gezählt, m = 5). Die Abweichung ist erklärbar: ein aktiver PSD-Block trägt den Rang seines Dualblocks bei (hier 2: Eigenwerte von Z 45,6 und 5162, zwei Nullen), nicht 1; 3 aktive Punkte + Rang 2 = 5. Die Tabelle zeigt beide Zahlen.
- **Der Plan nannte Newton-Schritte „30 bis 34“;** der Test prüft das Band 27 bis 38 (Rundung auf Linux).
- **Zentrierung am Pfadende:** Die Ellipse achsenparallel ist bei t = 10⁷ in der Standardwolke nicht mehr auf dem Pfad (Rundung, Stück 4). Die Tabelle trägt dafür die Spalte „zentriert“.
- **Keine Quellen aus dem Gedächtnis:** Nesterov und Nemirovskii (SIAM Studies in Applied Mathematics 13, 1994) und Boyd und Vandenberghe (2004, Kapitel 11) sind geprüft (Buchangaben, Inhalt der Kapitel); die genaue Form der √ν-Schranke habe ich nicht an der Quelle nachgelesen und beweise sie hier nicht.
- **Port:** 9302 (Netstat und `launch.json` gelesen, frei).

## Ehrliche Grenzen

- **„Die Newton-Schritte hängen kaum an ν“ ist ein Messbefund dieser Implementierung** (Faktor 8 für `t`, Start durch Konstruktion, Zentrieren auf 10⁻⁶), keine Eigenschaft der Barriere. Die Theorie nennt für den Kurzschritt eine Schranke mit √ν · log(1/ε) und zeigt nicht, dass sie scharf ist; dass 160 Runden für den Kurzschritt viel zu viele sind, ist ebenfalls nur gemessen.
- **`m` als Zahl der aktiven Bedingungen** gilt für Würfel, Raute und Kugeln (Test gegen einen unabhängigen Kreis durch zwei oder drei Punkte); bei den Ellipsen zählt statt der Blöcke der Rang des Dualblocks (allgemeine Ellipse: 3 + 2 = 5), und die Zählung im Code folgt dieser Regel nicht.
- **Rechengrenze:** Pfadpunkte bei t über etwa 10¹⁰ bis 10¹¹ sind numerisch schlecht (Hesse-Matrix); die Tabelle meldet nicht zentrierte Punkte.
- **Die Fremdbarriere** nur im LP und nur `Σ s^(−p)` mit p = 1 und 2; andere nicht selbstkonkordante Barrieren habe ich nicht gemessen.
- **Zeiten** hängen vom Rechner ab und werden nicht getestet.
- **`ok = True` und `converged = True` im Kern heißen nicht „zentriert“.** `center` gilt auch nach 100 erschöpften Schritten, bei negativem Dekrement-Quadrat und im Rundungsstopp als ok; die Schranke `ν/t` gilt streng nur für einen zentrierten Punkt, und bei großem t ist das Dekrement durch Rundung nicht verlässlich bestimmbar. Das Ergebnis des Kerns trägt seit der Härtung das Feld `far_from_path` für die gefährlichste Klasse (eine Zentrierung endete nach 100 Schritten weit vom Pfad, siehe `tests/test_kern.py`). In Experiment 4 (Schrittfaktor μ) meldet es nur bei sehr großen Sprüngen: auf der Standardwolke bei μ = 10⁵ für die allgemeine Ellipse (nicht für die Kugel der Hauptansicht), auf anderen Wolken und Formen ebenfalls bei μ = 10⁵. In diesen Läufen ist der Wert **falsch** (allgemeine Ellipse der Standardwolke bei μ = 10⁵: über 100 statt 78,52 bei gemeldeter Lücke unter 10⁻¹¹; die genaue Zahl hängt von der CPU ab: 100,96 auf dem Entwicklungsrechner, 102,41 auf einem GitHub-Runner): die Tabelle in Experiment 4 trägt dafür die Spalte „weit vom Pfad“, das Verdikt zählt solche Läufe nicht mit, und das Diagramm markiert sie. Eine strengere Härtung wurde gemessen und verworfen, weil sie veröffentlichte Läufe der Stücke 2 und 4 abbrechen ließe, die richtige Werte liefern. Keine Zahl dieser README ändert sich.

## Verifikation

- **Kern** (`tests/test_kern.py`, wortgleich aus Stück 2): Barrieren gegen finite Differenzen, Mini-Programme, singuläre Hesse-Matrix.
- **Pfad** (`tests/test_path.py`): der Pfad der drei Mini-Programme in **geschlossener Form** (`x(t) = 1 + 1/t`; `t₀ = (1 + √(1 + 25 t²))/t`; `x = (1 + √(1 + t²))/t`), Stationarität per finiter Differenz, Fortsetzung und Zentrierungs-Marke, Dekrement-Folge (gedämpft, dann voll, dann quadratisch), `t · Fehler` und aktive Bedingungen von Hand, verdoppeltes ν, Kurzschritt und Schrittfaktor-Reihe, die Formel der äußeren Runden (`⌈log_μ(ν/(tol·t₀))⌉ + 1`), die Fremdbarriere in einer Dimension von Hand (`x = 1 + 1/√t`, `1 + (2/t)^(1/3)`), Gradient und Hesse-Matrix der Fremdbarriere gegen finite Differenzen.
- **Orakel** (`tests/test_oracle_kegel_barriere.py`): der Optimalwert aller sechs Formen gegen cvxpy mit Clarabel (8 Wolken; dessen eigene Genauigkeit liegt bei bis zu 3·10⁻⁸); Zahl der aktiven Zeilen der Kugel gegen einen unabhängigen Kreis durch zwei oder drei Punkte (15 Wolken, `t · Fehler` gleich der Zahl der Stützpunkte); `t · Fehler` gleich der Zahl der aktiven Zeilen im LP; die Schranke `ν/t` deckt den Fehler an jedem Pfadpunkt jeder Form; Wert fällt monoton; Dual am Pfad (Stück 4); Skalierung.
- **Randfälle** (`tests/test_edge_cases.py`): zwei Punkte, ein Punkt, d = 1 und 3 in jeder Form, nicht zentrierte Pfadpunkte, Experimente mit kleinen Parametern, ungültige Eingaben.
- **Verdikte** (`tests/test_verdicts.py`), **Claims** (`tests/test_claims.py`: jede Zahl dieser README), **Oberfläche** (AppTest: jeder Button, jeder Regler an Minimum und Maximum, alle Regler zugleich, jede Form, die Eckfälle aus Stück 4, PDF), Presets, Permalink, Layout, Abbildungen, Footer, Laufzeit-Importe.
- **Umfang:** 126 Tests (1 davon, das cvxpy-Orakel über alle sechs Formen, überspringt sich ohne cvxpy), rund 1 bis 2,5 Minuten je nach Auslastung des Rechners (2026-11-03, Windows: 2:21 bei ausgelastetem Rechner).
- **Nicht gelaufen:** kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy und scipy sind Gegenproben im dev-Satz; ohne sie überspringen sich die jeweiligen Tests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2 und 4 |
| `kba_forms.py` | Die sechs Formen als Kegelprogramm samt Rückumformung (aus Stück 4 übernommen) |
| `kba_path.py` | Pfadpunkte, Dekrement-Folge, aktive Bedingungen, verdoppeltes ν, Schrittfaktor-Reihe, Fremdbarriere |
| `kba_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kba_evaluation.py` | Analyse, Experimente, Verdikte, Formatierung |
| `kba_layout.py` | Fester Kopf (Stylesheet) |
| `kba_pdf_export.py` | PDF-Export (fpdf2) |
| `kba_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kba_scenario.py` | Punktwolken (gestreckt, gedreht), Handinstanz |
| `kba_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Pfad, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Primal-Dual-Schritt, allgemeiner Start (Phase I), Einbettung:** Stück 6. **Exponential- und Potenzkegel:** Stück 7.
- **Selbstkonkordanz beweisen** und die √ν-Schranke herleiten: nur als Hintergrund genannt.
- **Adaptives μ** (Mehrotra-artig): nicht gebaut.
- **Andere Fremdbarrieren** (nicht glatte, andere Exponenten): nur `Σ s^(−p)` mit p = 1 und 2.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html).
