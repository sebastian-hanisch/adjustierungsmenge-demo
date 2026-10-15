"""Kern: Varianz je Menge, O-Menge, gültige Mengen, Gleichstände, Rang, Rolle; an den Mini-Instanzen von Hand."""

import numpy as np
import pytest

import ajm_sets as S
import bdr_backdoor as B
import dsep_graph as G


def cov_of(b):
    inv = np.linalg.inv(np.eye(len(b)) - b)
    return inv @ inv.T


def test_variance_factor_by_hand(precision_instance):
    parents, b = precision_instance
    cov = cov_of(b)
    assert S.variance_factor(cov, 3, 4, (0,)) == pytest.approx(1.0)
    assert S.variance_factor(cov, 3, 4, (0, 1)) == pytest.approx(0.5)
    assert S.variance_factor(cov, 3, 4, (0, 2)) == pytest.approx(2.0)
    assert S.variance_factor(cov, 3, 4, (0, 1, 2)) == pytest.approx(1.0)


def test_variance_factor_of_the_empty_set():
    """X -> Y mit Koeffizient 1: Var(X) = 1, Restvarianz von Y gegeben X = 1: Faktor 1. Mit Rauschen 2 in Y wäre es 4."""
    b = np.zeros((2, 2))
    b[1, 0] = 1.0
    assert S.variance_factor(cov_of(b), 0, 1, ()) == pytest.approx(1.0)
    b3 = np.zeros((3, 3))
    b3[2, 0] = 1.0
    b3[2, 1] = 1.0                                   # Y = X + P + e
    assert S.variance_factor(cov_of(b3), 0, 2, ()) == pytest.approx(2.0)
    assert S.variance_factor(cov_of(b3), 0, 2, (1,)) == pytest.approx(1.0)


def test_o_set_of_the_instances(precision_instance, mediator_instance, collider_instance, ancestor_instance):
    assert S.o_set(precision_instance[0], 3, 4) == (0, 1)
    assert S.o_set(mediator_instance[0], 1, 3) == (0,)
    assert S.o_set(collider_instance[0], 0, 1) == ()
    assert S.o_set(ancestor_instance[0], 2, 3) == (1,)
    assert S.o_set(precision_instance[0], 4, 3) is None            # y ist kein Nachfahre von x


def test_causal_nodes(mediator_instance):
    parents, _ = mediator_instance
    assert S.causal_nodes(parents, 1, 3) == {2, 3}
    assert S.causal_nodes(parents, 3, 1) == set()


def test_valid_sets_of_the_precision_instance(precision_instance):
    parents, _ = precision_instance
    assert S.valid_sets(parents, 3, 4, [0, 1, 2]) == [(0,), (0, 1), (0, 2), (0, 1, 2)]
    assert S.valid_sets(parents, 3, 4, [1, 2]) == []                 # Z unbeobachtet: kein Rückweg blockierbar


def test_valid_sets_allow_a_harmless_descendant_of_x():
    """Z -> X, Z -> Y, X -> Y, X -> D: D ist Nachfahre von X, aber auf keinem kausalen Pfad; Adjustierungskriterium erlaubt {Z, D}, Backdoor nicht."""
    parents = ((), (0,), (0, 1), (1,))      # 0 = Z, 1 = X, 2 = Y, 3 = D
    assert S.valid_sets(parents, 1, 2, [0, 3]) == [(0,), (0, 3)]
    assert B.valid_sets(parents, 1, 2, [0, 3], "backdoor") == [(0,)]


def test_best_sets_and_ties(ancestor_instance):
    parents, b = ancestor_instance
    cov = cov_of(b)
    valid = S.valid_sets(parents, 2, 3, [0, 1])
    assert valid == [(1,), (0, 1)]
    best, sets = S.best_sets(cov, 2, 3, valid)
    assert sets == [(1,), (0, 1)] and best == pytest.approx(S.variance_factor(cov, 2, 3, (1,)))


def test_best_sets_picks_the_smaller_variance(precision_instance):
    parents, b = precision_instance
    best, sets = S.best_sets(cov_of(b), 3, 4, S.valid_sets(parents, 3, 4, [0, 1, 2]))
    assert sets == [(0, 1)] and best == pytest.approx(0.5)


def test_same_and_dense_rank():
    assert S.same(1.0, 1.0 + 1e-9) and not S.same(1.0, 1.0 + 1e-5) and S.same(0.0, 5e-8)
    assert S.dense_rank([1.0, 0.5, 1.0, 2.0]) == [2, 1, 2, 3]
    assert S.dense_rank([3.0]) == [1] and S.dense_rank([1.0, 1.0 + 1e-9]) == [1, 1]


def test_roles(precision_instance):
    parents, _ = precision_instance
    assert S.role(parents, 3, 4, 0) == "gemeinsame Ursache"
    assert S.role(parents, 3, 4, 1) == "Präzision"
    assert S.role(parents, 3, 4, 2) == "Instrument"
    child = ((), (0,), (0, 1), (1,))          # 0 = Z, 1 = X, 2 = Y, 3 = D (Nachfahre von X)
    assert S.role(child, 1, 2, 3) == "Nachfahre von x"
    lone = ((), (), (0,), (1, 2))             # 0 = X, 1 = W (frei), 2 = ?, Y = 3 hängt an W und 2
    assert S.role(lone, 0, 3, 1) == "Präzision"
    free = ((), (), (0,))                     # 0 = X, 1 = W (isoliert), 2 = Y
    assert S.role(free, 0, 2, 1) == "weder noch"


def test_roles_constant_is_complete():
    assert set(S.ROLES) == {"Präzision", "Instrument", "gemeinsame Ursache", "Nachfahre von x", "weder noch"}


def test_o_set_is_a_valid_set_and_the_forbidden_nodes_are_excluded(mediator_instance):
    parents, _ = mediator_instance
    o = S.o_set(parents, 1, 3)
    assert B.adjustment_valid(parents, 1, 3, o) and not set(o) & B.forbidden_nodes(parents, 1, 3)
    assert G.descendants(parents, 1) >= {2, 3}


def test_same_is_inclusive_at_the_tolerance():
    assert S.same(0.0, S.REL_TOL) and S.same(S.REL_TOL, 0.0)
