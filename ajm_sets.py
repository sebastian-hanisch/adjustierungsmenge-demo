"""Kern: Varianz der Schätzung je Adjustierungsmenge, minimale Mengen, O-Menge (Henckel, Perković, Maathuis) und die Rolle einer Variable.

Nur Standardbibliothek, numpy, `dsep_graph` und `bdr_backdoor` (wortgleich aus backdoor-demo), kein Streamlit. Diagramm = Tupel von Elternmengen, Kovarianz Σ
des linearen Modells mit Einheitsrauschen. Jede Einheit eine reine Funktion.
"""

from __future__ import annotations

import numpy as np

import bdr_backdoor as B
import dsep_graph as G

REL_TOL = 1e-7        # Varianzen gelten als gleich, wenn sie sich relativ um weniger als das unterscheiden


def variance_factor(cov: np.ndarray, x: int, y: int, z=()) -> float:
    """Asymptotische Varianz des Koeffizienten von x in y ~ x + z, mal n: Restvarianz von y gegeben (x, z) durch Restvarianz von x gegeben z."""
    z = list(z)
    idx = [x] + z
    c = cov[idx, y]
    beta = np.linalg.solve(cov[np.ix_(idx, idx)], c)
    res_y = cov[y, y] - c @ beta
    if z:
        bx = np.linalg.solve(cov[np.ix_(z, z)], cov[z, x])
        res_x = cov[x, x] - cov[z, x] @ bx
    else:
        res_x = cov[x, x]
    return float(res_y / res_x)


def same(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def causal_nodes(parents, x: int, y: int) -> set:
    """Die Knoten auf kausalen Pfaden von x nach y, ohne x (einschließlich y)."""
    return {v for p in B.directed_paths(parents, x, y) for v in p[1:]}


def o_set(parents, x: int, y: int) -> tuple | None:
    """O(x, y, G) = pa(cn) ohne die verbotenen Knoten (cn und ihre Nachfahren), ohne x und y; None, wenn es keinen kausalen Pfad gibt."""
    cn = causal_nodes(parents, x, y)
    if not cn:
        return None
    forb = B.forbidden_nodes(parents, x, y)
    pa = {p for c in cn for p in parents[c]}
    return tuple(sorted(v for v in pa - forb if v not in (x, y)))


def valid_sets(parents, x: int, y: int, observed) -> list[tuple]:
    """Alle nach dem Adjustierungskriterium gültigen Mengen unter den beobachteten Knoten (ohne x und y)."""
    return B.valid_sets(parents, x, y, [v for v in observed if v not in (x, y)], "adjustment")


def best_sets(cov: np.ndarray, x: int, y: int, valid) -> tuple[float, list[tuple]]:
    """Kleinste Varianz unter den gültigen Mengen und alle Mengen, die sie erreichen (Gleichstände)."""
    scores = {z: variance_factor(cov, x, y, z) for z in valid}
    best = min(scores.values())
    return best, [z for z in valid if same(scores[z], best)]


def dense_rank(values: list[float]) -> list[int]:
    """Rang nach aufsteigendem Wert (1 = kleinster); gleiche Werte (innerhalb von REL_TOL) teilen den Rang."""
    uniq: list[float] = []
    for v in sorted(values):
        if not uniq or not same(uniq[-1], v):
            uniq.append(v)
    return [1 + next(i for i, u in enumerate(uniq) if same(u, v)) for v in values]


def role(parents, x: int, y: int, v: int) -> str:
    """Rolle des Knotens v (nicht x, y) für die Genauigkeit: Präzision (nur über y wirksam), Instrument (nur über x), gemeinsame Ursache, Nachfahre von x, sonst weder noch."""
    cut = tuple(tuple(p for p in ps if p != x) for ps in parents)
    ancestor_x = v in G.ancestors(parents, {x}) - {x}
    ancestor_y = v in G.ancestors(cut, {y}) - {y}      # Vorfahre von y auf Wegen, die nicht über x laufen
    if ancestor_x and ancestor_y:
        return "gemeinsame Ursache"
    if ancestor_y:
        return "Präzision"
    if ancestor_x:
        return "Instrument"
    if v in G.descendants(parents, x):
        return "Nachfahre von x"
    return "weder noch"


ROLES = ("Präzision", "Instrument", "gemeinsame Ursache", "Nachfahre von x", "weder noch")
