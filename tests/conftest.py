"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def precision_instance():
    """Mini-Instanz 1 (0 = Z, 1 = P, 2 = I, 3 = X, 4 = Y): Z -> X, I -> X, Z -> Y, P -> Y, X -> Y, alle Koeffizienten 1, Einheitsrauschen.
    Handrechnung (Restvarianz von Y gegeben X und Z: P und Rauschen = 2, mit P: 1; Restvarianz von X gegeben Z: I und Rauschen = 2, mit I: 1):
    Varianz mal n der Menge {Z} = 2/2 = 1, {Z, P} = 1/2, {Z, I} = 2/1 = 2, {Z, P, I} = 1/1 = 1. Wahre Wirkung 1; gültig sind genau die Mengen mit Z.
    O-Menge = Eltern von Y ohne X = {Z, P}."""
    parents = ((), (), (), (0, 2), (0, 1, 3))
    b = np.zeros((5, 5))
    b[3, 0] = b[3, 2] = b[4, 0] = b[4, 1] = b[4, 3] = 1.0
    return parents, b


@pytest.fixture
def mediator_instance():
    """Mini-Instanz 2 (0 = Z, 1 = X, 2 = M, 3 = Y): Z -> X, Z -> Y, X -> M, M -> Y, X -> Y, alle 1. cn = {M, Y}; pa(cn) = {X, M, Z}; verboten {M, Y}: O = {Z}."""
    parents = ((), (0,), (1,), (0, 1, 2))
    b = np.zeros((4, 4))
    b[1, 0] = b[3, 0] = b[2, 1] = b[3, 2] = b[3, 1] = 1.0
    return parents, b


@pytest.fixture
def collider_instance():
    """Mini-Instanz 3 (0 = X, 1 = Y, 2 = C): X -> Y, X -> C, Y -> C. C ist verboten (Nachfahre von Y): O = {} (leer)."""
    parents = ((), (0,), (0, 1))
    b = np.zeros((3, 3))
    b[1, 0] = b[2, 0] = b[2, 1] = 1.0
    return parents, b


@pytest.fixture
def ancestor_instance():
    """Mini-Instanz 4 (0 = R, 1 = Z, 2 = X, 3 = Y): R -> Z, Z -> X, Z -> Y, X -> Y, alle 1. R ist nur Vorfahre von Z: {Z} und {R, Z} gleich genau."""
    parents = ((), (0,), (1,), (1, 2))
    b = np.zeros((4, 4))
    b[1, 0] = b[2, 1] = b[3, 1] = b[3, 2] = 1.0
    return parents, b
