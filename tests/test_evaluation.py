"""Auswertung: Welten, Analyse, Experimente (feste Ströme), Verdikte, Formatierung."""

import numpy as np
import pytest

import ajm_constants as C
import ajm_evaluation as ev
import ajm_sets as S
from kausal_generator import SplitMix64, sample


def test_number_format_and_set_text():
    assert ev.de(1.5, 1) == "1,5" and ev.de(-0.25) == "−0,25"
    assert ev.signed(0.3) == "+0,30" and ev.signed(-0.3) == "−0,30"
    assert ev.set_text(("a", "b", "c"), (0, 2)) == "{a, c}" and ev.set_text(("a",), ()) == "{} (leer)"


def test_population_and_coefficient_by_hand(precision_instance):
    _, b = precision_instance
    cov, total = ev.population(b)
    assert total[4, 3] == pytest.approx(1.0) and total[0, 0] == 0.0 and total[3, 0] == pytest.approx(1.0)
    assert ev.population_coefficient(cov, 3, 4, (0,)) == pytest.approx(1.0)
    assert ev.population_coefficient(cov, 3, 4, ()) != pytest.approx(1.0, abs=0.1)


def test_ols_against_numpy_lstsq():
    scm = ev.world_scm("6-Knoten-Beispiel")
    d = sample(scm, 400, SplitMix64(8), reps=3)
    nm = scm.names
    est, se = ev.ols_coefficient(d[nm[5]], d[nm[3]], [d[nm[2]], d[nm[4]]])
    for r in range(3):
        design = np.column_stack([np.ones(400), d[nm[3]][r], d[nm[2]][r], d[nm[4]][r]])
        coef, res, *_ = np.linalg.lstsq(design, d[nm[5]][r], rcond=None)
        assert est[r] == pytest.approx(coef[1], rel=1e-9)
        assert se[r] == pytest.approx(np.sqrt(res[0] / (400 - 4) * np.linalg.inv(design.T @ design)[1, 1]), rel=1e-9)


def test_random_dag_and_populations_are_reproducible():
    a, b = ev.random_dag(7, 0.4, SplitMix64(3)), ev.random_dag(7, 0.4, SplitMix64(3))
    assert a == b and all(p < j for j, ps in enumerate(a) for p in ps)
    assert sum(len(p) for p in ev.random_dag(7, 1.0, SplitMix64(1))) == 21 and sum(len(p) for p in ev.random_dag(7, 0.0, SplitMix64(1))) == 0
    first = [(p, x, y) for p, x, y, _, _ in ev.random_populations(5, 9)]
    assert first == [(p, x, y) for p, x, y, _, _ in ev.random_populations(5, 9)]
    assert all(x < y for _, x, y in first)


def test_default_analysis_of_the_six_node_world():
    a = ev.analyse(ev.Settings())
    c = a.chosen
    assert c.z == (2,) and c.backdoor and c.adjustment and c.minimal and not c.best and not c.is_o
    assert c.ratio == pytest.approx(1.465, abs=0.001) and c.rank == 5
    assert a.o == (2, 4) and a.o_observed and a.n_valid == 8 and len(a.rows) == 16
    assert a.truth == pytest.approx(0.5) and len(a.best_sets) == 4 and (2, 4) in a.best_sets
    assert a.sd_formula == pytest.approx(0.0655, abs=0.0005)
    assert a.covers == (abs(a.estimate - a.truth) <= C.Z_CRIT * a.se)


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b = ev.analyse(ev.Settings()), ev.analyse(ev.Settings())
    assert (a.estimate, a.se) == (b.estimate, b.se)
    assert ev.analyse(ev.Settings(seed=36)).estimate != a.estimate


def test_ratio_is_one_exactly_for_the_best_sets_and_rank_one():
    a = ev.analyse(ev.Settings())
    best = [r for r in a.rows if r.best]
    assert {r.z for r in best} == set(a.best_sets)
    assert all(r.rank == 1 and r.ratio == pytest.approx(1.0) for r in best)
    assert all(r.ratio is None and r.rank is None for r in a.rows if not r.adjustment)
    assert all(r.ratio >= 1 - 1e-12 for r in a.rows if r.adjustment)


def test_invalid_set_has_a_bias_and_no_rank():
    a = ev.analyse(ev.Settings(z=(4,)))
    c = a.chosen
    assert not c.adjustment and c.ratio is None and c.rank is None and abs(c.bias) > 0.1
    kind, text = ev.verdict_instance(a)
    assert kind == "error" and "**ungültig**" in text and "Rückweg" in text


def test_a_descendant_of_x_is_valid_only_by_the_adjustment_criterion_and_costs_precision():
    a = ev.analyse(ev.Settings("Präzisions-Welt", 4, 5, (1,), ()))
    b = ev.analyse(ev.Settings("Präzisions-Welt", 4, 5, (1, 6), ()))
    assert b.chosen.adjustment and not b.chosen.backdoor
    assert b.chosen.ratio > a.chosen.ratio
    assert "nur nach dem Adjustierungskriterium" in ev.verdict_instance(b)[1]


def test_verdict_of_the_best_set_and_ties():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(z=(2, 4))))
    assert kind == "success" and "am genauesten" in text and "gleichauf mit 3 weiteren Mengen" in text
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings("Präzisions-Welt", 4, 5, (1, 2), ())))
    assert kind == "success" and "gleichauf mit 1 weiteren Menge" in text


def test_verdict_of_a_worse_valid_set():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings()))
    assert kind == "warning" and "das 1,46fache" in text and "Rang 5 von 5" in text and "{Schwankung, Lieferverzug}" in text and "3 weitere" in text


def test_hidden_node_makes_the_optimal_set_unavailable():
    a = ev.analyse(ev.Settings("Präzisions-Welt", 4, 5, (1,), (2,)))
    assert a.o == (1, 2) and not a.o_observed and a.n_valid > 0
    kind, text = ev.verdict_unavailable_o(a)
    assert kind == "info" and "nicht erreichbar" in text and "1,28fachen" in text
    assert ev.verdict_unavailable_o(ev.analyse(ev.Settings())) is None


def test_no_valid_set_when_the_confounder_is_hidden():
    a = ev.analyse(ev.Settings(z=(4,), hidden=(2,)))
    assert a.n_valid == 0 and a.best_factor is None and a.best_sets == []
    assert ev.verdict_no_valid_set(a)[0] == "error" and ev.verdict_no_valid_set(ev.analyse(ev.Settings())) is None


def test_analysis_every_world_every_x_at_the_slider_edges():
    for world, w in C.WORLDS.items():
        for n in (C.N_MIN, C.N_MAX):
            a = ev.analyse(ev.Settings(world, *w["query"][:2], w["query"][2], (), n, 1))
            assert np.isfinite([a.estimate, a.se, a.sd_formula]).all()


def test_sets_experiment_structure():
    res = ev.sets_experiment("6-Knoten-Beispiel", reps=60)
    valid = [r for r in res["rows"] if r["valid"]]
    assert len(res["rows"]) == 16 and len(valid) == 8
    assert sum(r["best"] for r in valid) == 4 and sum(r["minimal"] for r in valid) == 1 and sum(r["is_o"] for r in valid) == 1
    assert all(abs(r["mean"] - res["truth"]) < 0.08 for r in valid)
    assert all(r["cover"] >= 0.85 for r in valid) and all(r["cover"] <= 1.0 for r in valid)
    assert all(r["sd_measured"] == pytest.approx(r["sd_formula"], rel=0.35) for r in valid)
    assert [r for r in res["rows"] if r["is_o"]][0]["z"] == (2, 4)


def test_sets_experiment_second_world():
    res = ev.sets_experiment("Präzisions-Welt", reps=40)
    assert len([r for r in res["rows"] if r["valid"]]) == 16 and res["o"] == (1, 2)
    assert sum(r["backdoor"] for r in res["rows"]) == 8            # die acht ohne Rückfragen


def test_rules_experiment_structure():
    res = ev.rules_experiment(diagrams=150)
    assert res["evaluated"] == 150 and res["o_invalid"] == 0 and res["violations"] == 0
    assert len(res["ratios"]["kleinste minimale Menge"]) == 150 and len(res["ratios"]["ungünstigste gültige Menge"]) == 150
    assert all((v >= 1 - 1e-7).all() for v in res["ratios"].values())
    assert ev.verdict_rules(res)[0] == "info"


def test_roles_experiment_structure():
    res = ev.roles_experiment(diagrams=80)
    by = {r["role"]: r for r in res["rows"]}
    assert set(by) == set(S.ROLES)
    assert by["Präzision"]["down"] > 0.6 and by["Instrument"]["up"] > 0.6
    for r in res["rows"]:
        assert r["down"] + r["same"] + r["up"] == pytest.approx(1.0) or r["n"] == 0
    assert ev.verdict_roles(res)[0] == "info"


def test_hidden_experiment_structure():
    res = ev.hidden_experiment(diagrams=200)
    assert res["diagrams"] == 200 and 0 < res["o_available"] < 200 and res["have_valid"] > 150
    assert res["loss_median"] >= 1.0 and res["unavailable_with_valid"] > 0
    assert "unbeobachteten Knoten" in ev.verdict_hidden(res)[1]


def test_preferred_picks_o_then_the_smallest():
    assert ev.preferred([(0, 1, 2), (0, 1), (1, 2)], (1, 2)) == (1, 2)
    assert ev.preferred([(0, 1, 2), (1, 2), (0, 2)], None) == (0, 2)
    assert ev.preferred([(0, 1, 2), (0, 2)], (5,)) == (0, 2)


def test_o_is_observed_when_it_is_exactly_the_set_of_candidates():
    a = ev.analyse(ev.Settings("Präzisions-Welt", 4, 5, (1,), (0, 3, 6)))
    assert a.candidates == (1, 2) and a.o == (1, 2) and a.o_observed
    assert ev.verdict_unavailable_o(a) is None


def test_the_worst_valid_set_is_never_better_than_the_smallest_minimal_set():
    res = ev.rules_experiment(diagrams=100)
    worst, small = res["ratios"]["ungünstigste gültige Menge"], res["ratios"]["kleinste minimale Menge"]
    assert (worst >= small - 1e-12).all() and (worst > small + 1e-9).any()


def test_random_populations_have_a_causal_path_and_x_before_y():
    from ajm_sets import causal_nodes

    for parents, x, y, cov, truth in ev.random_populations(40, 5):
        assert x < y and y in causal_nodes(parents, x, y) and cov.shape == (7, 7) and np.isfinite(truth)
