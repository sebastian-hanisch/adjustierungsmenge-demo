"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Zufallsdiagramme kommen aus SplitMix64 (ganzzahlig, plattformstabil): ihre Zählungen werden exakt geprüft. Stichprobenwerte (Streuung, Deckung) mit Toleranz, damit
Rundung und Bibliotheksversionen (CI installiert ungepinnt) nichts kippen.
"""

import numpy as np
import pytest

import ajm_evaluation as ev
import ajm_sets as S
import bdr_backdoor as B


@pytest.fixture(scope="module")
def six():
    return ev.sets_experiment("6-Knoten-Beispiel")


@pytest.fixture(scope="module")
def precision():
    return ev.sets_experiment("Präzisions-Welt")


@pytest.fixture(scope="module")
def rules():
    return ev.rules_experiment()


@pytest.fixture(scope="module")
def roles():
    return {r["role"]: r for r in ev.roles_experiment()["rows"]}


@pytest.fixture(scope="module")
def hidden():
    return ev.hidden_experiment()


def by_set(res):
    return {r["z"]: r for r in res["rows"]}


# ----------------------------------------------------------------------------- Standard-Stichprobe

def test_default_sample():
    a = ev.analyse(ev.Settings())
    assert a.estimate == pytest.approx(0.48, abs=0.005) and a.covers and a.lo == pytest.approx(0.34, abs=0.005) and a.hi == pytest.approx(0.61, abs=0.005)
    assert a.chosen.ratio == pytest.approx(1.46, abs=0.005) and a.chosen.rank == 5 and a.n_valid == 8


# ----------------------------------------------------------------------------- Experiment 1

def test_experiment_1_six_node_world(six):
    rows = by_set(six)
    ratio = {z: rows[z]["ratio"] for z in rows if rows[z]["valid"]}
    assert ratio[(2,)] == pytest.approx(1.46, abs=0.005)
    assert ratio[(1, 2)] == pytest.approx(1.35, abs=0.005) and ratio[(0, 2)] == pytest.approx(1.40, abs=0.005) and ratio[(0, 1, 2)] == pytest.approx(1.28, abs=0.005)
    for z in ((2, 4), (0, 2, 4), (1, 2, 4), (0, 1, 2, 4)):
        assert ratio[z] == pytest.approx(1.0)
    assert six["o"] == (2, 4) and len(ratio) == 8 and sum(r["best"] for r in rows.values()) == 4
    assert [w for w, r in rows.items() if r["minimal"]] == [(2,)]


def test_experiment_1_formula_and_sample(six, precision):
    for res in (six, precision):
        for r in res["rows"]:
            if r["valid"]:
                assert r["sd_measured"] == pytest.approx(r["sd_formula"], rel=0.14)
    rows = by_set(six)
    assert rows[(2,)]["sd_measured"] == pytest.approx(0.059, abs=0.001) and rows[(2,)]["sd_formula"] == pytest.approx(0.066, abs=0.001)
    assert rows[(2, 4)]["sd_measured"] == pytest.approx(0.043, abs=0.001) and rows[(2, 4)]["sd_formula"] == pytest.approx(0.045, abs=0.001)
    invalid = [r for r in six["rows"] if not r["valid"]]
    assert len(invalid) == 8 and all(r["cover"] == 0.0 for r in invalid)
    assert min(abs(r["bias"]) for r in invalid) == pytest.approx(0.30, abs=0.005) and max(abs(r["bias"]) for r in invalid) == pytest.approx(0.55, abs=0.005)


def test_experiment_1_precision_world(precision):
    rows = by_set(precision)
    assert len([r for r in rows.values() if r["valid"]]) == 16 and len(rows) == 32 and sum(r["backdoor"] for r in rows.values()) == 8
    assert rows[(1,)]["ratio"] == pytest.approx(1.28, abs=0.005) and rows[(1, 3)]["ratio"] == pytest.approx(1.56, abs=0.005)
    assert rows[(1, 2)]["ratio"] == pytest.approx(1.0) and precision["o"] == (1, 2)
    assert rows[(1, 6)]["ratio"] == pytest.approx(1.79, abs=0.005) and not rows[(1, 6)]["backdoor"]
    assert max(r["ratio"] for r in rows.values() if r["valid"]) == pytest.approx(2.0, abs=0.005)


def test_hand_instance(precision_instance):
    parents, b = precision_instance
    cov, _ = ev.population(b)
    assert [S.variance_factor(cov, 3, 4, z) for z in ((0,), (0, 1), (0, 2), (0, 1, 2))] == pytest.approx([1.0, 0.5, 2.0, 1.0])
    assert S.o_set(parents, 3, 4) == (0, 1)


# ----------------------------------------------------------------------------- Experiment 2

def test_experiment_2_o_is_never_beaten(rules):
    assert rules["diagrams"] == 3000 and rules["evaluated"] == 3000 and rules["o_invalid"] == 0 and rules["violations"] == 0
    assert rules["mean_valid"] == pytest.approx(11.7, abs=0.05) and rules["median_valid"] == 8 and rules["mean_minimal"] == pytest.approx(1.09, abs=0.005)
    assert rules["ties"] == 6925


def test_experiment_2_rules_against_o(rules):
    assert rules["minimal_worse"] == 1695 and round(100 * 1695 / 3000, 1) == 56.5
    r = rules["ratios"]
    small = r["kleinste minimale Menge"]
    assert np.median(small) == pytest.approx(1.015, abs=0.001) and small.mean() == pytest.approx(1.19, abs=0.005) and np.quantile(small, 0.9) == pytest.approx(1.54, abs=0.005)
    assert small.max() == pytest.approx(6.4, abs=0.05) and np.mean(small < 1 + 1e-7) == pytest.approx(0.49, abs=0.005)
    px = r["Eltern von x"]
    assert np.median(px) == pytest.approx(1.17, abs=0.005) and px.mean() == pytest.approx(1.31, abs=0.005) and np.quantile(px, 0.9) == pytest.approx(1.80, abs=0.005)
    assert np.mean(px < 1 + 1e-7) == pytest.approx(0.34, abs=0.01)
    nd = r["alle Nicht-Nachfahren von x"]
    assert np.median(nd) == pytest.approx(1.0, abs=1e-6) and nd.mean() == pytest.approx(1.09, abs=0.005) and np.mean(nd < 1 + 1e-7) == pytest.approx(0.76, abs=0.005)
    empty = r["leere Menge"]
    assert len(empty) == 2098 and np.median(empty) == pytest.approx(1.04, abs=0.005)


# ----------------------------------------------------------------------------- Experiment 3

def test_experiment_3_roles(roles):
    p, i, d, g = roles["Präzision"], roles["Instrument"], roles["Nachfahre von x"], roles["gemeinsame Ursache"]
    assert p["n"] == 9116 and round(100 * p["down"]) == 91 and p["median_se"] == pytest.approx(0.87, abs=0.005)
    assert i["n"] == 3563 and round(100 * i["up"]) == 95 and i["median_se"] == pytest.approx(1.19, abs=0.005)
    assert d["n"] == 6733 and round(100 * d["up"]) == 92 and d["median_se"] == pytest.approx(1.16, abs=0.005)
    assert g["n"] == 1195 and round(100 * g["down"]) == 40 and round(100 * g["up"]) == 44
    assert roles["weder noch"]["n"] == 15020 and round(100 * roles["weder noch"]["same"]) == 77


# ----------------------------------------------------------------------------- Experiment 4

def test_experiment_4_hidden(hidden):
    assert hidden["diagrams"] == 2000 and hidden["have_valid"] == 1888 and hidden["o_available"] == 1543 and hidden["unavailable_with_valid"] == 345
    assert round(100 * 1888 / 2000, 1) == 94.4 and round(100 * 1543 / 2000, 1) == 77.2
    assert hidden["loss_median"] == pytest.approx(1.19, abs=0.005) and hidden["loss_mean"] == pytest.approx(1.24, abs=0.005) and hidden["loss_max"] == pytest.approx(2.08, abs=0.005)


def test_unavailable_o_example_in_the_precision_world():
    a = ev.analyse(ev.Settings("Präzisions-Welt", 4, 5, (1,), (2,)))
    assert a.o == (1, 2) and not a.o_observed
    assert a.best_factor / a.o_factor == pytest.approx(1.28 ** 2, rel=0.01)


def test_backdoor_sets_of_the_precision_world_are_the_ones_without_the_descendant():
    parents = ev.world_scm("Präzisions-Welt").parents
    assert len(B.valid_sets(parents, 4, 5, [0, 1, 2, 3, 6], "backdoor")) == 8
