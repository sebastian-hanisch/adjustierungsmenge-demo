"""Konstanten: Welten (feste Diagramme mit Koeffizienten), Regler-Grenzen, Experimente, Farben und Presets."""

# Rechenaufwand am Maximum: eine Stichprobe mit n = 5 000 und 7 Knoten, Regression mit höchstens 5 Kontrollvariablen, unter 20 ms.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 5000, 100, 500
SEED_MAX, DEFAULT_SEED = 999999, 35
Z_CRIT = 1.959964                                  # 95-%-Intervall, Normalnäherung (n >= 100)

# Welten: Knoten in topologischer Reihenfolge (Pfeile nur von kleinem zu großem Index). Die Daten kommen aus diesem Diagramm; Behandlung x, Wirkung y,
# die Menge und unbeobachtete Knoten wählt man in der App.
WORLDS = {
    "6-Knoten-Beispiel": {
        "nodes": ("Region", "Wetter", "Schwankung", "Faktor", "Lieferverzug", "Servicegrad"),
        "coefs": {(0, 2): 0.8, (2, 3): 0.9, (0, 4): 0.7, (1, 4): 0.7, (2, 5): -0.6, (3, 5): 0.5, (4, 5): -0.8},
        "pos": ((0.0, 0.5), (0.0, 0.0), (0.33, 0.75), (0.66, 0.9), (0.4, 0.15), (1.0, 0.4)),
        "query": (3, 5, (2,), ()),
        "help": "Die Schwankung allein schließt beide Rückwege. Mit dem Lieferverzug, der nur den Servicegrad bewegt, wird die Schätzung deutlich genauer.",
    },
    "Präzisions-Welt": {
        "nodes": ("Region", "Schwankung", "Wetter", "Zuteilungsregel", "Faktor", "Servicegrad", "Rückfragen"),
        "coefs": {(0, 1): 0.8, (1, 4): 0.9, (1, 5): -0.6, (2, 5): 0.8, (3, 4): 0.7, (4, 5): 0.5, (4, 6): 0.8},
        "pos": ((0.0, 0.85), (0.33, 0.85), (0.75, 1.0), (0.0, 0.1), (0.4, 0.4), (1.0, 0.55), (0.8, 0.0)),
        "query": (4, 5, (1,), ()),
        "help": "Das Wetter bewegt nur den Servicegrad, die Zuteilungsregel nur den Faktor, die Rückfragen sind eine Folge des Faktors.",
    },
}
WORLD_ORDER = tuple(WORLDS)
DEFAULT_WORLD = "6-Knoten-Beispiel"

# Experimente: feste Ströme (Seeds), nie ein einzelner Seed als Befund.
EXP_SETS_N, EXP_SETS_REPS, EXP_SETS_SEED = 500, 300, 7          # Experiment 1: alle Mengen einer Welt, feste Wiederholungen
EXP_RULES_DIAGRAMS, EXP_RULES_SEED = 3000, 41                   # Experiment 2: Zufallsdiagramme, Regeln gegen O
EXP_ROLES_DIAGRAMS, EXP_ROLES_SEED = 1500, 51                   # Experiment 3: eine Variable hinzufügen, nach Rolle
EXP_HIDDEN_DIAGRAMS, EXP_HIDDEN_SEED, EXP_HIDDEN_PICK_SEED = 2000, 43, 77   # Experiment 4: ein Knoten unbeobachtet
RANDOM_NODES = 7
RANDOM_P_LO, RANDOM_P_WIDTH = 0.25, 0.25                        # Kantenwahrscheinlichkeit 0,25 bis 0,5
COEF_LO, COEF_HI = 0.4, 1.2

COLOR_CAUSAL = "#2F6B65"
COLOR_EDGE = "#6b6b6b"
COLOR_SET = "#4c78a8"
COLOR_BEST = "#2F6B65"
COLOR_WORSE = "#D68A2E"
COLOR_INVALID = "#c0392b"
COLOR_HIDDEN = "#cfcfcf"
COLOR_NEUTRAL = "#8a8a8a"


def preset(world: str, z: tuple, hidden: tuple = (), n: int = DEFAULT_N) -> dict:
    x, y, _, _ = WORLDS[world]["query"]
    return {"world": world, "x": x, "y": y, "z": z, "hidden": hidden, "n": n, "seed": DEFAULT_SEED}


PRESETS = {
    "Minimale Menge (Voreinstellung)": preset("6-Knoten-Beispiel", (2,)),
    "Die genaueste Menge": preset("6-Knoten-Beispiel", (2, 4)),
    "Eltern von x": preset("Präzisions-Welt", (1, 3)),
    "Mit Nachfahre von x": preset("Präzisions-Welt", (1, 6)),
    "Präzision dazu": preset("Präzisions-Welt", (1, 2)),
    "Eine Variable fehlt": preset("Präzisions-Welt", (1,), hidden=(2,)),
}
PRESET_HELP = {
    "Minimale Menge (Voreinstellung)": "Nur die Schwankung: gültig und die kleinste Menge, aber nicht die genaueste.",
    "Die genaueste Menge": "Schwankung und Lieferverzug: Der Lieferverzug erklärt einen Teil der Streuung des Servicegrads, ohne etwas zu verzerren.",
    "Eltern von x": "Die Eltern des Faktors (Standard beim Neigungswert): gültig, aber die Zuteilungsregel bindet nur Streuung des Faktors.",
    "Mit Nachfahre von x": "Die Rückfragen sind eine Folge des Faktors: nach dem Adjustierungskriterium erlaubt, aber ungenauer; das Backdoor-Kriterium verbietet sie.",
    "Präzision dazu": "Schwankung und Wetter: Das Wetter bewegt nur den Servicegrad und senkt die Streuung der Schätzung.",
    "Eine Variable fehlt": "Das Wetter ist nicht beobachtet: Die genaueste Menge ist nicht erreichbar, die beste beobachtete Menge ist die Schwankung, allein oder mit der Region.",
}
