"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from ajm_evaluation import Analysis, de, set_text, signed, verdict_instance

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "€": "EUR", "≥": ">=", "≤": "<=", "„": '"', "“": '"', "×": "x", "**": ""}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s, c, names = a.settings, a.chosen, a.names
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Welche Menge? Gültig heißt nicht gleich genau"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    hidden = f", unbeobachtet: {set_text(names, s.hidden)}" if s.hidden else ""
    pdf.multi_cell(0, 5, _s(f"Welt: {s.world}. Behandlung {names[s.x]}, Wirkung {names[s.y]}{hidden}. n = {s.n}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Gewählte Menge: {set_text(names, c.z)}.",
        "Kriterium: " + ("Backdoor und Adjustierung gültig" if c.backdoor else ("nur nach dem Adjustierungskriterium gültig" if c.adjustment else "ungültig")) + ".",
        f"Wahre Wirkung: {de(a.truth, 3)}; Schätzung {de(a.estimate, 3)} (+-{de(1.96 * a.se, 3)}).",
        f"Standardfehler nach Formel bei n = {s.n}: {de(a.sd_formula, 4)}."
        + (f" Das {de(c.ratio, 2)}fache der genauesten gültigen Menge (Rang {c.rank})." if c.ratio is not None else ""),
    ]
    if a.best_sets:
        lines.append("Genaueste gültige Menge: " + " | ".join(set_text(names, z) for z in a.best_sets[:4]) + ".")
    if a.o is not None:
        lines.append(f"O-Menge: {set_text(names, a.o)}" + ("." if a.o_observed else " (nicht vollständig beobachtet)."))
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_instance(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: adjustierungsmenge-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
