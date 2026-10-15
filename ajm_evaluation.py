"""Auswertung: Welten, Schätzung mit gewählter Menge, Varianz je Menge, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import ajm_constants as C
import ajm_sets as S
import bdr_backdoor as B
import dsep_graph as G
from kausal_generator import Scm, SplitMix64, coefficient_matrix, sample, total_effect


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def set_text(names, z) -> str:
    return "{" + ", ".join(names[v] for v in z) + "}" if z else "{} (leer)"


# ----------------------------------------------------------------------------- Welten, Population, Schätzer

def world_scm(world: str) -> Scm:
    """Lineares Strukturmodell der Welt (Rauschen N(0,1) in jedem Knoten)."""
    w = C.WORLDS[world]
    n = len(w["nodes"])
    parents = G.parents_from_edges(n, w["coefs"])
    return Scm(names=w["nodes"], parents=parents, coefs=tuple(tuple(w["coefs"][(p, j)] for p in parents[j]) for j in range(n)),
               intercepts=(0.0,) * n, noise_sd=(1.0,) * n, binary=(False,) * n)


def population(b: np.ndarray):
    """Kovarianz Σ = (I − B)⁻¹(I − B)⁻ᵀ und Gesamtwirkungen T[j, i] = (I − B)⁻¹ − I bei Einheitsrauschen."""
    inv = np.linalg.inv(np.eye(len(b)) - b)
    return inv @ inv.T, inv - np.eye(len(b))


def population_coefficient(cov: np.ndarray, x: int, y: int, z=()) -> float:
    """Koeffizient von x in der Regression von y auf x und z in der Population."""
    idx = [x] + list(z)
    return float(np.linalg.solve(cov[np.ix_(idx, idx)], cov[idx, y])[0])


def ols_coefficient(y: np.ndarray, x: np.ndarray, zs=()) -> tuple[np.ndarray, np.ndarray]:
    """Koeffizient von x und Standardfehler in y ~ 1 + x + zs (kleinste Quadrate), je Wiederholung (Zeilen). Eingaben (reps, n)."""
    reps, n = y.shape
    design = np.stack([np.ones_like(y), x] + list(zs), axis=2)
    xtx = np.einsum("rni,rnj->rij", design, design)
    xty = np.einsum("rni,rn->ri", design, y)
    beta = np.linalg.solve(xtx, xty[..., None])[..., 0]
    resid = y - np.einsum("rni,ri->rn", design, beta)
    s2 = (resid ** 2).sum(axis=1) / (n - design.shape[2])
    se = np.sqrt(s2 * np.linalg.inv(xtx)[:, 1, 1])
    return beta[:, 1], se


def random_dag(n: int, p: float, rng: SplitMix64) -> tuple:
    """Zufälliges Diagramm in der Knotenreihenfolge 0..n-1 mit Kantenwahrscheinlichkeit p (ein Block Gleichverteilte, feste Reihenfolge)."""
    u = rng.uniform(n * (n - 1) // 2)
    k = 0
    parents = []
    for j in range(n):
        ps = []
        for i in range(j):
            if u[k] < p:
                ps.append(i)
            k += 1
        parents.append(tuple(ps))
    return tuple(parents)


def random_coefficients(parents, rng: SplitMix64) -> np.ndarray:
    n = len(parents)
    u = rng.uniform(2 * n * n)
    b = np.zeros((n, n))
    k = 0
    for c, ps in enumerate(parents):
        for p in ps:
            sign = 1.0 if u[k] < 0.5 else -1.0
            b[c, p] = sign * (C.COEF_LO + (C.COEF_HI - C.COEF_LO) * u[k + 1])
            k += 2
    return b


def random_populations(count: int, seed: int, nodes: int = C.RANDOM_NODES):
    """Zufallsdiagramme mit einem kausalen Pfad von x nach y (x < y) und zufälligen Koeffizienten: (Eltern, x, y, Σ, wahre Wirkung)."""
    rng = SplitMix64(seed)
    got = attempts = 0
    while got < count:
        attempts += 1
        if attempts > 100 * count:
            raise RuntimeError("zu wenige Zufallsdiagramme mit einem kausalen Pfad von x nach y")
        parents = random_dag(nodes, C.RANDOM_P_LO + C.RANDOM_P_WIDTH * float(rng.uniform(1)[0]), rng)
        u = rng.uniform(2)
        x, y = sorted((int(u[0] * nodes), int(u[1] * nodes)))
        if x == y or not B.directed_paths(parents, x, y):
            continue
        cov, total = population(random_coefficients(parents, rng))
        got += 1
        yield parents, x, y, cov, float(total[y, x])


# ----------------------------------------------------------------------------- Auswertung der Einstellungen

@dataclass(frozen=True)
class Settings:
    world: str = C.DEFAULT_WORLD
    x: int = 3
    y: int = 5
    z: tuple = (2,)
    hidden: tuple = ()
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED


@dataclass
class SetRow:
    z: tuple
    backdoor: bool
    adjustment: bool
    factor: float                 # asymptotische Varianz mal n
    bias: float                   # Koeffizient der Population minus Wahrheit
    rank: int | None = None       # Rang unter den gültigen Mengen (1 = genaueste, Gleichstände teilen den Rang)
    ratio: float | None = None    # Standardfehler / Standardfehler der genauesten gültigen Menge (nur für gültige Mengen)
    minimal: bool = False         # inklusionsminimal unter den gültigen Mengen
    is_o: bool = False
    best: bool = False


@dataclass
class Analysis:
    settings: Settings
    names: tuple
    parents: tuple
    candidates: tuple
    rows: list
    chosen: SetRow
    truth: float
    o: tuple | None
    o_observed: bool
    o_factor: float | None
    n_valid: int
    best_factor: float | None
    best_sets: list
    estimate: float
    se: float
    lo: float
    hi: float
    covers: bool
    sd_formula: float
    check: B.BackdoorResult


def observed_candidates(n_nodes: int, x: int, y: int, hidden) -> tuple:
    return tuple(v for v in range(n_nodes) if v not in (x, y) and v not in set(hidden))


def analyse(s: Settings) -> Analysis:
    scm = world_scm(s.world)
    parents = scm.parents
    z = tuple(sorted(s.z))
    cov, total = population(coefficient_matrix(scm))
    truth = total_effect(scm, scm.names[s.x], scm.names[s.y])
    cands = observed_candidates(len(parents), s.x, s.y, s.hidden)
    rows = [SetRow(zz, B.backdoor_valid(parents, s.x, s.y, zz), B.adjustment_valid(parents, s.x, s.y, zz), S.variance_factor(cov, s.x, s.y, zz),
                   population_coefficient(cov, s.x, s.y, zz) - truth) for zz in B.all_sets(cands)]
    valid = [r for r in rows if r.adjustment]
    o = S.o_set(parents, s.x, s.y)
    best_factor, best = (None, [])
    if valid:
        best_factor = min(r.factor for r in valid)
        best = [r.z for r in valid if S.same(r.factor, best_factor)]
        ranks = S.dense_rank([r.factor for r in valid])
        mins = set(B.minimal_sets([r.z for r in valid]))
        for r, rank in zip(valid, ranks):
            r.rank, r.ratio, r.minimal = rank, math.sqrt(r.factor / best_factor), r.z in mins
            r.best = r.z in best
    for r in rows:
        r.is_o = o is not None and r.z == o
    chosen = next(r for r in rows if r.z == z)
    d = sample(scm, s.n, SplitMix64(s.seed))
    est, se = ols_coefficient(d[scm.names[s.y]], d[scm.names[s.x]], [d[scm.names[v]] for v in z])
    est, se = float(est[0]), float(se[0])
    return Analysis(
        settings=s, names=scm.names, parents=parents, candidates=cands, rows=rows, chosen=chosen, truth=truth, o=o,
        o_observed=o is not None and set(o) <= set(cands), o_factor=None if o is None else S.variance_factor(cov, s.x, s.y, o), n_valid=len(valid), best_factor=best_factor, best_sets=best, estimate=est, se=se,
        lo=est - C.Z_CRIT * se, hi=est + C.Z_CRIT * se, covers=bool(abs(est - truth) <= C.Z_CRIT * se), sd_formula=math.sqrt(chosen.factor / s.n),
        check=B.backdoor_check(parents, s.x, s.y, z),
    )


# ----------------------------------------------------------------------------- Experiment 1: alle Mengen einer Welt

def sets_experiment(world: str = C.DEFAULT_WORLD, n: int = C.EXP_SETS_N, reps: int = C.EXP_SETS_REPS) -> dict:
    """Alle Mengen der Welt (Frage der Voreinstellung): Varianz nach Formel gegen die Streuung von `reps` festen Wiederholungen."""
    scm = world_scm(world)
    x, y, _, _ = C.WORLDS[world]["query"]
    parents = scm.parents
    cov, total = population(coefficient_matrix(scm))
    truth = float(total[y, x])
    cands = observed_candidates(len(parents), x, y, ())
    d = sample(scm, n, SplitMix64(C.EXP_SETS_SEED), reps)
    sets = B.all_sets(cands)
    valid = [z for z in sets if B.adjustment_valid(parents, x, y, z)]
    best_factor, best = S.best_sets(cov, x, y, valid)
    mins = set(B.minimal_sets(valid))
    o = S.o_set(parents, x, y)
    rows = []
    for z in sets:
        est, se = ols_coefficient(d[scm.names[y]], d[scm.names[x]], [d[scm.names[v]] for v in z])
        factor = S.variance_factor(cov, x, y, z)
        ok = z in valid
        rows.append({"z": z, "valid": ok, "backdoor": B.backdoor_valid(parents, x, y, z), "minimal": z in mins, "is_o": z == o, "best": z in best,
                     "factor": factor, "ratio": math.sqrt(factor / best_factor) if ok else None, "sd_formula": math.sqrt(factor / n),
                     "sd_measured": float(est.std()), "mean": float(est.mean()), "bias": population_coefficient(cov, x, y, z) - truth,
                     "cover": float(np.mean(np.abs(est - truth) <= C.Z_CRIT * se))})
    return {"world": world, "names": scm.names, "x": x, "y": y, "n": n, "reps": reps, "truth": truth, "o": o, "rows": rows}


# ----------------------------------------------------------------------------- Experiment 2: Regeln gegen die O-Menge

RULE_NAMES = ("kleinste minimale Menge", "Eltern von x", "alle Nicht-Nachfahren von x", "leere Menge", "ungünstigste gültige Menge")


def rules_experiment(diagrams: int = C.EXP_RULES_DIAGRAMS) -> dict:
    """In Zufallsdiagrammen: Standardfehler-Verhältnis jeder Regel zur O-Menge, wo die Regel gültig ist; Verletzungen der Optimalität; minimale Mengen schlechter als O."""
    ratios = {name: [] for name in RULE_NAMES}
    o_invalid = violations = minimal_worse = ties = 0
    n_valid, n_min = [], []
    done = 0
    for parents, x, y, cov, _ in random_populations(diagrams, C.EXP_RULES_SEED):
        cands = [v for v in range(len(parents)) if v not in (x, y)]
        vs = S.valid_sets(parents, x, y, cands)
        o = S.o_set(parents, x, y)
        if o not in vs:
            o_invalid += 1
            continue
        done += 1
        sc = {z: S.variance_factor(cov, x, y, z) for z in vs}
        vo = sc[o]
        if any(v < vo and not S.same(v, vo) for v in sc.values()):
            violations += 1
        ties += sum(1 for z, v in sc.items() if z != o and S.same(v, vo))
        mins = B.minimal_sets(vs)
        n_valid.append(len(vs))
        n_min.append(len(mins))
        smallest = min(len(z) for z in mins)
        ratios["kleinste minimale Menge"].append(math.sqrt(min(sc[z] for z in mins if len(z) == smallest) / vo))
        minimal_worse += any(sc[z] > vo and not S.same(sc[z], vo) for z in mins)
        ratios["ungünstigste gültige Menge"].append(math.sqrt(max(sc.values()) / vo))
        pax = tuple(sorted(p for p in parents[x] if p != y))
        if pax in sc:
            ratios["Eltern von x"].append(math.sqrt(sc[pax] / vo))
        desc = G.descendants(parents, x)
        nd = tuple(v for v in cands if v not in desc)
        if nd in sc:
            ratios["alle Nicht-Nachfahren von x"].append(math.sqrt(sc[nd] / vo))
        if () in sc:
            ratios["leere Menge"].append(math.sqrt(sc[()] / vo))
    return {"diagrams": diagrams, "evaluated": done, "o_invalid": o_invalid, "violations": violations, "ties": ties, "minimal_worse": minimal_worse,
            "mean_valid": float(np.mean(n_valid)), "median_valid": float(np.median(n_valid)), "mean_minimal": float(np.mean(n_min)),
            "ratios": {k: np.asarray(v) for k, v in ratios.items()}}


# ----------------------------------------------------------------------------- Experiment 3: eine Variable hinzufügen

def roles_experiment(diagrams: int = C.EXP_ROLES_DIAGRAMS) -> dict:
    """Je gültiger Menge und je Knoten außerhalb, dessen Hinzunahme gültig bleibt: ändert sich die Varianz? Gruppiert nach der Rolle des Knotens."""
    found = {r: [] for r in S.ROLES}
    for parents, x, y, cov, _ in random_populations(diagrams, C.EXP_ROLES_SEED):
        cands = [v for v in range(len(parents)) if v not in (x, y)]
        vs = set(S.valid_sets(parents, x, y, cands))
        roles = {v: S.role(parents, x, y, v) for v in cands}
        for z in vs:
            base = S.variance_factor(cov, x, y, z)
            for v in cands:
                z2 = tuple(sorted(set(z) | {v}))
                if v in z or z2 not in vs:
                    continue
                found[roles[v]].append(S.variance_factor(cov, x, y, z2) / base)
    rows = []
    for r in S.ROLES:
        a = np.asarray(found[r])
        if not len(a):
            rows.append({"role": r, "n": 0, "down": 0.0, "up": 0.0, "same": 0.0, "median_se": 1.0})
            continue
        down = float(np.mean(a < 1 - 1e-7))
        up = float(np.mean(a > 1 + 1e-7))
        rows.append({"role": r, "n": len(a), "down": down, "up": up, "same": 1 - down - up, "median_se": math.sqrt(float(np.median(a)))})
    return {"diagrams": diagrams, "rows": rows}


# ----------------------------------------------------------------------------- Experiment 4: ein Knoten unbeobachtet

def hidden_experiment(diagrams: int = C.EXP_HIDDEN_DIAGRAMS) -> dict:
    """Ein zufälliger Knoten (nicht x, y) ist unbeobachtet: Wie oft gibt es eine gültige Menge, wie oft ist O erreichbar, und wie viel ungenauer ist sonst die beste?"""
    rng = SplitMix64(C.EXP_HIDDEN_PICK_SEED)
    have_valid = o_available = 0
    loss = []
    for parents, x, y, cov, _ in random_populations(diagrams, C.EXP_HIDDEN_SEED):
        cands = [v for v in range(len(parents)) if v not in (x, y)]
        h = cands[int(rng.uniform(1)[0] * len(cands))]
        obs = [v for v in cands if v != h]
        vs = S.valid_sets(parents, x, y, obs)
        o = S.o_set(parents, x, y)
        have_valid += bool(vs)
        available = h not in o
        o_available += available
        if vs and not available:
            loss.append(math.sqrt(min(S.variance_factor(cov, x, y, z) for z in vs) / S.variance_factor(cov, x, y, o)))
    la = np.asarray(loss)
    return {"diagrams": diagrams, "have_valid": have_valid, "o_available": o_available, "unavailable_with_valid": len(la),
            "loss_median": float(np.median(la)), "loss_mean": float(la.mean()), "loss_max": float(la.max())}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def preferred(sets, o=None) -> tuple:
    """Die Menge, die man als genaueste nennt: O, wenn sie unter den besten ist, sonst die kleinste (nach Größe und Nummer)."""
    return o if o in sets else min(sets, key=lambda z: (len(z), z))


def verdict_instance(a: Analysis) -> tuple[str, str]:
    c, names = a.chosen, a.names
    head = f"Menge {set_text(names, c.z)}: "
    if not c.adjustment:
        why = "enthält einen verbotenen Knoten" if any(v in B.forbidden_nodes(a.parents, a.settings.x, a.settings.y) for v in c.z) else "lässt einen Rückweg offen"
        return "error", head + f"**ungültig**: Sie {why}; der Koeffizient liegt im Grenzfall um {signed(c.bias)} daneben (Wahrheit {de(a.truth)})."
    kind = "" if c.backdoor else " (nur nach dem Adjustierungskriterium)"
    if c.best:
        others = len(a.best_sets) - 1
        extra = "" if not others else f", gleichauf mit {others} weiteren {'Menge' if others == 1 else 'Mengen'}"
        return "success", head + f"**gültig{kind} und unter allen gültigen Mengen am genauesten**{extra}."
    best = preferred(a.best_sets, a.o)
    more = len(a.best_sets) - 1
    tail = "" if not more else f" (und {more} weitere mit gleichem Standardfehler)"
    n_ranks = len({r.rank for r in a.rows if r.adjustment})
    return "warning", head + (f"**gültig{kind}**, aber der Standardfehler ist das {de(c.ratio, 2)}fache des kleinsten möglichen (Rang {c.rank} von {n_ranks}); "
                              f"am genauesten ist {set_text(names, best)}{tail}.")


def verdict_no_valid_set(a: Analysis) -> tuple[str, str] | None:
    if a.n_valid:
        return None
    return "error", "Unter den beobachteten Variablen gibt es keine gültige Menge: Ein Rückweg läuft nur über unbeobachtete Knoten."


def verdict_unavailable_o(a: Analysis) -> tuple[str, str] | None:
    if a.o is None or a.o_observed or not a.n_valid:
        return None
    ratio = math.sqrt(a.best_factor / a.o_factor)
    return "info", (f"Die genaueste Menge {set_text(a.names, a.o)} enthält unbeobachtete Knoten und ist nicht erreichbar; die beste beobachtete gültige Menge hat den "
                    f"{de(ratio, 2)}fachen Standardfehler.")


def verdict_sets(res: dict) -> tuple[str, str]:
    valid = [r for r in res["rows"] if r["valid"]]
    minimal = [r for r in valid if r["minimal"]]
    best = [r["z"] for r in valid if r["best"]]
    worst = max(valid, key=lambda r: r["factor"])
    text = (f"Von {len(res['rows'])} Mengen sind {len(valid)} gültig. Die kleinste ({set_text(res['names'], minimal[0]['z'])}) hat den {de(minimal[0]['ratio'], 2)}fachen "
            f"Standardfehler der genauesten ({set_text(res['names'], preferred(best, res['o']))})")
    if not S.same(worst["factor"], minimal[0]["factor"]):
        text += f"; die ungünstigste gültige Menge den {de(worst['ratio'], 2)}fachen"
    return "info", text + "."


def verdict_rules(res: dict) -> tuple[str, str]:
    r = res["ratios"]
    return "info", (f"In {res['evaluated']} Diagrammen ist keine gültige Menge genauer als O ({res['violations']} Ausnahmen). In {de(100 * res['minimal_worse'] / res['evaluated'], 1)} % "
                    f"hat mindestens eine minimale Menge einen größeren Standardfehler; die Eltern von x liegen im Median beim {de(float(np.median(r['Eltern von x'])), 2)}fachen.")


def verdict_roles(res: dict) -> tuple[str, str]:
    by = {r["role"]: r for r in res["rows"]}
    return "info", (f"Ein Knoten, der nur über y wirkt (Präzision), senkt die Varianz in {de(100 * by['Präzision']['down'], 0)} % der Fälle; ein Knoten, der nur über x wirkt "
                    f"(Instrument), hebt sie in {de(100 * by['Instrument']['up'], 0)} %, ein Nachfahre von x in {de(100 * by['Nachfahre von x']['up'], 0)} %.")


def verdict_hidden(res: dict) -> tuple[str, str]:
    return "info", (f"Mit einem unbeobachteten Knoten gibt es in {de(100 * res['have_valid'] / res['diagrams'], 1)} % der Diagramme noch eine gültige Menge; die genaueste Menge ist in "
                    f"{de(100 * res['o_available'] / res['diagrams'], 1)} % erreichbar, sonst ist die beste beobachtete im Median das {de(res['loss_median'], 2)}fache.")
