"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import ajm_constants as C
import ajm_sets as S
from bdr_backdoor import directed_paths
from ajm_evaluation import Analysis, de, set_text, signed

NODE_SIZE = 64


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _arrow(fig, p0, p1, color, width=2, shrink=0.075):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / length, dy / length
    fig.add_annotation(x=x1 - ux * shrink, y=y1 - uy * shrink, ax=x0 + ux * shrink, ay=y0 + uy * shrink, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=width, arrowcolor=color, text="")


def build_diagram(a: Analysis) -> go.Figure:
    """Diagramm mit Behandlung x und Wirkung y (schwarzer Rand), gewählter Menge (blau), O-Menge (grüner Rand), unbeobachteten Knoten (grau) und der Rolle jedes Knotens."""
    s = a.settings
    pos = C.WORLDS[s.world]["pos"]
    fig = go.Figure()
    causal = set()
    for path in directed_paths(a.parents, s.x, s.y):
        causal |= set(zip(path, path[1:]))
    for j, ps in enumerate(a.parents):
        for i in ps:
            on = (i, j) in causal
            _arrow(fig, pos[i], pos[j], C.COLOR_CAUSAL if on else C.COLOR_EDGE, 3 if on else 2)
    o = set(a.o or ())
    fills, lines, widths, texts = [], [], [], []
    for v, name in enumerate(a.names):
        hidden, in_z = v in s.hidden, v in s.z
        fills.append(C.COLOR_HIDDEN if hidden else ("rgba(76,120,168,0.40)" if in_z else "rgba(47,107,101,0.08)"))
        if v in (s.x, s.y):
            lines.append("#1b1b1b")
            widths.append(4)
        elif v in o:
            lines.append(C.COLOR_BEST)
            widths.append(4)
        else:
            lines.append(C.COLOR_SET if in_z else C.COLOR_NEUTRAL)
            widths.append(1.5)
        tag = "Behandlung" if v == s.x else ("Wirkung" if v == s.y else S.role(a.parents, s.x, s.y, v))
        texts.append(name + "<br>(" + ("unbeobachtet, " if hidden else "") + tag + ")")
    fig.add_trace(go.Scatter(x=[p[0] for p in pos], y=[p[1] for p in pos], mode="markers+text", text=texts, textposition="middle center",
                             textfont=dict(size=10), marker=dict(size=NODE_SIZE, color=fills, line=dict(width=widths, color=lines)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.2, 1.2])
    fig.update_yaxes(visible=False, range=[-0.2, 1.2])
    return _fixed(fig, 400, showlegend=False)


def build_sets_table(a: Analysis) -> pd.DataFrame:
    out = []
    for r in a.rows:
        crit = "Backdoor und Adjustierung" if r.backdoor else ("nur Adjustierung" if r.adjustment else "ungültig")
        tags = [t for t, flag in (("minimal", r.minimal), ("O-Menge", r.is_o), ("genaueste", r.best)) if flag]
        out.append({
            "Menge": set_text(a.names, r.z),
            "Kriterium": crit,
            "Standardfehler": f"×{de(r.ratio, 2)}" if r.ratio is not None else "–",
            "Rang": str(r.rank) if r.rank is not None else "–",
            "Fehler im Grenzfall": signed(r.bias, 2) if not r.adjustment else "0,00",
            "Rolle": ", ".join(tags),
        })
    return pd.DataFrame(out)


def style_sets_table(df: pd.DataFrame, current: str):
    def row_style(row):
        if row["Menge"] == current:
            return ["background-color: rgba(76,120,168,0.25); font-weight: 600"] * len(row)
        if "genaueste" in row["Rolle"]:
            return ["background-color: rgba(47,107,101,0.15)"] * len(row)
        if row["Kriterium"] == "ungültig":
            return ["color: #999999"] * len(row)
        return [""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_sets_chart(res: dict) -> go.Figure:
    rows = sorted((r for r in res["rows"] if r["valid"]), key=lambda r: (r["sd_formula"], len(r["z"])))
    labels = [set_text(res["names"], r["z"]) for r in rows]
    colors = [C.COLOR_BEST if r["best"] else (C.COLOR_SET if r["minimal"] else C.COLOR_NEUTRAL) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[r["sd_formula"] for r in rows], orientation="h", marker_color=colors, name="Formel (grün: genaueste, blau: minimal)"))
    fig.add_trace(go.Scatter(y=labels, x=[r["sd_measured"] for r in rows], mode="markers", marker=dict(color="#1b1b1b", size=9, symbol="diamond"),
                             name=f"Streuung in {res['reps']} Wiederholungen"))
    fig.update_xaxes(title=f"Standardfehler des Koeffizienten bei n = {res['n']}", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return _fixed(fig, 120 + 38 * len(rows))


def build_rules_chart(res: dict) -> go.Figure:
    names = [n for n in res["ratios"] if len(res["ratios"][n])]
    med = [float(np.median(res["ratios"][n])) for n in names]
    q90 = [float(np.quantile(res["ratios"][n], 0.9)) for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=med, name="Median", marker_color=C.COLOR_SET, text=[f"×{de(v, 2)}" for v in med], textposition="outside"))
    fig.add_trace(go.Scatter(x=names, y=q90, mode="markers", name="90 %-Quantil", marker=dict(color=C.COLOR_WORSE, size=11, symbol="diamond")))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_BEST, dash="dot"), annotation_text="O-Menge", annotation_position="top left")
    fig.update_yaxes(title="Standardfehler im Verhältnis zur O-Menge", rangemode="tozero")
    fig.update_xaxes(tickangle=-15)
    return _fixed(fig, 420)


def build_roles_chart(res: dict) -> go.Figure:
    rows = [r for r in res["rows"] if r["n"]]
    labels = [r["role"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * r["down"] for r in rows], name="Varianz sinkt", marker_color=C.COLOR_BEST))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["same"] for r in rows], name="gleich", marker_color=C.COLOR_NEUTRAL))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["up"] for r in rows], name="Varianz steigt", marker_color=C.COLOR_WORSE))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Anteil der Hinzufügungen (%)", range=[0, 100])
    return _fixed(fig, 380)


def build_hidden_chart(res: dict) -> go.Figure:
    labels = ["gültige beobachtete Menge existiert", "O-Menge vollständig beobachtet"]
    vals = [100 * res["have_valid"] / res["diagrams"], 100 * res["o_available"] / res["diagrams"]]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[C.COLOR_SET, C.COLOR_BEST], text=[de(v, 1) + " %" for v in vals], textposition="outside"))
    fig.update_yaxes(title="Anteil der Diagramme (%)", range=[0, 112])
    return _fixed(fig, 340, showlegend=False)
