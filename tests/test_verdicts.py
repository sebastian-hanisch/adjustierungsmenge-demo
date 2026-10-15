"""Verdikte an handgesetzten Zeilen: jede Schwelle und jede Zahl im Text (Vorlage DEMO-PLAYBOOK)."""

import numpy as np
import pytest

import ajm_evaluation as ev


def row(z, factor, ratio, valid=True, minimal=False, best=False, backdoor=True, is_o=False):
    return {"z": z, "valid": valid, "backdoor": backdoor, "minimal": minimal, "is_o": is_o, "best": best, "factor": factor, "ratio": ratio}


def test_verdict_sets_names_minimal_best_and_worst():
    res = {"names": ("A", "B", "C", "D"), "o": (1,), "rows": [
        row((), 9.0, None, valid=False),
        row((0,), 4.0, 2.0, minimal=True),
        row((1,), 1.0, 1.0, best=True, is_o=True),
        row((0, 1), 1.0, 1.0, best=True),
        row((0, 2), 9.0, 3.0),
    ]}
    kind, text = ev.verdict_sets(res)
    assert kind == "info"
    assert text == ("Von 5 Mengen sind 4 gültig. Die kleinste ({A}) hat den 2,00fachen Standardfehler der genauesten ({B}); die ungünstigste gültige Menge den 3,00fachen.")


def test_verdict_sets_omits_the_worst_when_it_is_the_minimal_set():
    res = {"names": ("A", "B"), "o": (1,), "rows": [row((0,), 4.0, 2.0, minimal=True), row((1,), 1.0, 1.0, best=True, is_o=True)]}
    assert ev.verdict_sets(res)[1] == "Von 2 Mengen sind 2 gültig. Die kleinste ({A}) hat den 2,00fachen Standardfehler der genauesten ({B})."


def test_verdict_rules_numbers():
    ratios = {name: np.array([1.0]) for name in ev.RULE_NAMES}
    ratios["Eltern von x"] = np.array([1.0, 1.5, 2.0])
    res = {"evaluated": 200, "violations": 0, "minimal_worse": 113, "ratios": ratios}
    kind, text = ev.verdict_rules(res)
    assert kind == "info" and "In 200 Diagrammen ist keine gültige Menge genauer als O (0 Ausnahmen)." in text
    assert "In 56,5 % hat mindestens eine minimale Menge" in text and "im Median beim 1,50fachen" in text


def test_verdict_roles_numbers():
    rows = [{"role": "Präzision", "n": 10, "down": 0.91, "up": 0.01, "same": 0.08, "median_se": 0.87},
            {"role": "Instrument", "n": 10, "down": 0.0, "up": 0.95, "same": 0.05, "median_se": 1.19},
            {"role": "Nachfahre von x", "n": 10, "down": 0.0, "up": 0.92, "same": 0.08, "median_se": 1.16}]
    text = ev.verdict_roles({"rows": rows})[1]
    assert "senkt die Varianz in 91 % der Fälle" in text and "hebt sie in 95 %" in text and "Nachfahre von x in 92 %" in text


def test_verdict_hidden_numbers():
    res = {"diagrams": 2000, "have_valid": 1888, "o_available": 1543, "loss_median": 1.19}
    text = ev.verdict_hidden(res)[1]
    assert "in 94,4 % der Diagramme" in text and "in 77,2 % erreichbar" in text and "im Median das 1,19fache" in text


def test_verdict_instance_rank_and_ties_wording():
    a = ev.analyse(ev.Settings("Präzisions-Welt", 4, 5, (1, 3), ()))
    kind, text = ev.verdict_instance(a)
    assert kind == "warning" and "das 1,56fache" in text and "{Schwankung, Wetter}" in text and "Rang 5 von 7" in text
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings("Präzisions-Welt", 4, 5, (1,), ())))
    assert "das 1,28fache" in text and "Rang 3 von 7" in text and "(und 1 weitere mit gleichem Standardfehler)" in text
