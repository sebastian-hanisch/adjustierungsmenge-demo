"""Presets, Reglergrenzen, Konstanten und die Auswahllisten der Oberfläche."""

import ajm_constants as C
import ajm_evaluation as ev
import ajm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == {"world", "x", "y", "z", "hidden", "n", "seed"}, name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid():
    for name, p in C.PRESETS.items():
        n = len(C.WORLDS[p["world"]]["nodes"])
        nodes = {p["x"], p["y"], *p["z"], *p["hidden"]}
        assert p["x"] in P.x_options(p["world"]) and p["y"] in P.y_options(p["world"], p["x"]), name
        assert all(0 <= v < n for v in nodes) and p["x"] not in p["z"] and p["y"] not in p["z"] and set(p["z"]).isdisjoint(p["hidden"]), name
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0 and 0 <= p["seed"] <= C.SEED_MAX, name


def test_first_preset_is_the_default_setting():
    p = C.PRESETS["Minimale Menge (Voreinstellung)"]
    assert ev.Settings() == ev.Settings(p["world"], p["x"], p["y"], tuple(p["z"]), tuple(p["hidden"]), p["n"], p["seed"])


def test_presets_show_different_verdicts():
    out = {}
    for name, p in C.PRESETS.items():
        a = ev.analyse(ev.Settings(p["world"], p["x"], p["y"], tuple(p["z"]), tuple(p["hidden"]), p["n"], p["seed"]))
        out[name] = (a.chosen.adjustment, a.chosen.backdoor, a.chosen.best, a.o_observed)
    assert out["Minimale Menge (Voreinstellung)"] == (True, True, False, True)
    assert out["Die genaueste Menge"] == (True, True, True, True)
    assert out["Eltern von x"] == (True, True, False, True)
    assert out["Mit Nachfahre von x"] == (True, False, False, True)
    assert out["Präzision dazu"] == (True, True, True, True)
    assert out["Eine Variable fehlt"] == (True, True, True, False)


def test_options_follow_the_world_and_x():
    assert P.x_options("6-Knoten-Beispiel") == [0, 1, 2, 3, 4]               # Servicegrad hat keinen Nachfahren
    assert P.y_options("6-Knoten-Beispiel", 3) == [5] and P.y_options("6-Knoten-Beispiel", 0) == [2, 3, 4, 5]
    assert 5 not in P.x_options("Präzisions-Welt") and 6 not in P.x_options("Präzisions-Welt")
    assert P.n_nodes("Präzisions-Welt") == 7


def test_bounds_and_step_grid():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert C.DEFAULT_N in range(C.N_MIN, C.N_MAX + 1, C.N_STEP)


def test_experiment_constants_are_consistent():
    assert C.RANDOM_P_LO + C.RANDOM_P_WIDTH <= 1.0 and C.COEF_LO < C.COEF_HI and C.EXP_SETS_REPS >= 200 and C.RANDOM_NODES == 7
    assert len({C.EXP_RULES_SEED, C.EXP_ROLES_SEED, C.EXP_HIDDEN_SEED, C.EXP_HIDDEN_PICK_SEED, C.EXP_SETS_SEED}) == 5


def test_worlds_are_topologically_ordered():
    for w in C.WORLDS.values():
        assert all(i < j for i, j in w["coefs"]) and len(w["pos"]) == len(w["nodes"])
        x, y, z, hidden = w["query"]
        assert x < y and set(z).isdisjoint({x, y}) and not hidden
