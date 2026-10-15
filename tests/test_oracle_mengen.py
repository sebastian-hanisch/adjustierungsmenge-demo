"""Orakel: Varianzformel gegen den Kehrwert der Kovarianz und gegen Stichproben, O-Menge gegen Brute Force, Kriterium gegen die Wahrheit der Population."""

import itertools

import numpy as np
import pytest

import ajm_evaluation as ev
import ajm_sets as S
import bdr_backdoor as B
from kausal_generator import SplitMix64, coefficient_matrix, sample


def populations(count, seed, nodes=6):
    yield from ev.random_populations(count, seed, nodes)


def test_variance_formula_against_the_inverse_covariance():
    """Zweiter Rechenweg: Var·n = Restvarianz(y | x, z) · [Σ(x,z)⁻¹]₀₀ (Restvarianz von x gegeben z ist 1 / [Σ(x,z)⁻¹]₀₀)."""
    checked = 0
    for parents, x, y, cov, _ in populations(60, 4, 6):
        cands = [v for v in range(6) if v not in (x, y)]
        for z in B.all_sets(cands):
            idx = [x] + list(z)
            sub = cov[np.ix_(idx, idx)]
            c = cov[idx, y]
            res_y = cov[y, y] - c @ np.linalg.solve(sub, c)
            assert S.variance_factor(cov, x, y, z) == pytest.approx(res_y * np.linalg.inv(sub)[0, 0], rel=1e-9)
            checked += 1
    assert checked > 500


def test_o_set_is_valid_and_never_beaten_by_brute_force():
    ties = better = 0
    for parents, x, y, cov, _ in populations(300, 12, 6):
        cands = [v for v in range(6) if v not in (x, y)]
        vs = S.valid_sets(parents, x, y, cands)
        o = S.o_set(parents, x, y)
        assert o in vs
        vo = S.variance_factor(cov, x, y, o)
        for z in vs:
            vz = S.variance_factor(cov, x, y, z)
            assert vz >= vo - 1e-9 * max(1.0, vo)
            ties += z != o and S.same(vz, vo)
            better += vz > vo * (1 + 1e-7)
    assert ties > 100 and better > 500        # die Prüfung ist nicht leer: es gibt Gleichstände und strikt schlechtere Mengen


def test_adjustment_criterion_agrees_with_the_population_for_generic_coefficients():
    valid = invalid = 0
    for parents, x, y, cov, truth in populations(120, 21, 6):
        for z in B.all_sets([v for v in range(6) if v not in (x, y)]):
            correct = abs(ev.population_coefficient(cov, x, y, z) - truth) < 1e-8
            assert B.adjustment_valid(parents, x, y, z) == correct
            valid += correct
            invalid += not correct
    assert valid > 300 and invalid > 300


def test_formula_against_repeated_samples():
    """n = 200, 3 000 Wiederholungen: Streuung der Schätzungen gegen sqrt(Faktor / n), Abweichung unter 6 %."""
    scm = ev.world_scm("Präzisions-Welt")
    cov, _ = ev.population(coefficient_matrix(scm))
    d = sample(scm, 200, SplitMix64(3), reps=3000)
    nm = scm.names
    for z in ((1,), (1, 2), (1, 3), (1, 6)):
        est, _ = ev.ols_coefficient(d[nm[5]], d[nm[4]], [d[nm[v]] for v in z])
        assert est.std() == pytest.approx(np.sqrt(S.variance_factor(cov, 4, 5, z) / 200), rel=0.06), z


def test_every_valid_set_has_the_same_population_coefficient():
    scm = ev.world_scm("Präzisions-Welt")
    cov, total = ev.population(coefficient_matrix(scm))
    truth = float(total[5, 4])
    vs = S.valid_sets(scm.parents, 4, 5, [0, 1, 2, 3, 6])
    assert len(vs) == 16
    for z in vs:
        assert ev.population_coefficient(cov, 4, 5, z) == pytest.approx(truth, abs=1e-9)


def test_o_set_on_all_small_diagrams_is_best():
    """Alle Diagramme mit 4 Knoten (alle Elternmengen), mit der Koeffizientenmatrix mit 0,7 auf allen Kanten."""
    n = 4
    pairs = [(i, j) for j in range(n) for i in range(j)]
    count = 0
    for bits in itertools.product((0, 1), repeat=len(pairs)):
        edges = [p for p, b in zip(pairs, bits) if b]
        parents = tuple(tuple(i for i, j in edges if j == c) for c in range(n))
        b = np.zeros((n, n))
        for i, j in edges:
            b[j, i] = 0.7 + 0.1 * ((i + 2 * j) % 3)
        cov, _ = ev.population(b)
        for x, y in itertools.permutations(range(n), 2):
            o = S.o_set(parents, x, y)
            if o is None:
                continue
            cands = [v for v in range(n) if v not in (x, y)]
            vs = S.valid_sets(parents, x, y, cands)
            vo = S.variance_factor(cov, x, y, o)
            assert o in vs and all(S.variance_factor(cov, x, y, z) >= vo - 1e-9 for z in vs)
            count += 1
    assert count == 223
