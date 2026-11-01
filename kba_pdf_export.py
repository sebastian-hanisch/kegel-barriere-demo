"""PDF-Export der aktuellen Wolke mit dem Pfad der gewählten Form (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from kba_evaluation import Analysis, de, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "ν": "nu", "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3",
            "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis, path: list[dict]) -> bytes:
    s, run = a.settings, a.selected
    from kba_forms import FORM_NAMES

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Kegel-Barriere: der zentrale Pfad"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Form: {FORM_NAMES[s.form]}; Wolke: {s.n} Punkte, Streckung {de(s.stretch, 1)}, um {s.angle} Grad gedreht, Seed {s.seed}. nu = {run.nu}, "
                            f"{run.result.rounds} Runden, {run.result.newton_steps} Newton-Schritte, bewiesene Lücke {sci(run.result.gap_bound)}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["t", "Wert", "wahrer Fehler", "Schranke nu/t", "t * Fehler", "Newton"], [34, 40, 40, 40, 34, 24]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for r in path:
        for v, w in zip([sci(r["t"], 0), de(r["value"], 5), sci(r["error"]), sci(r["bound"]), de(r["m"], 3), r["steps"]], widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Barrierenmethode: x(t) minimiert t c^T x + phi(b - A x); die Schranke nu/t gilt auf dem Pfad, der wahre Fehler ist etwa m/t. "
                            "Quelle: kegel-barriere-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
