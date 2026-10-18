# Welche Menge? Gültig heißt nicht gleich genau – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-adjustierungsmenge-demo.streamlit.app/)**

Fünftes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [schlechte-kontrollen-demo](https://github.com/sebastian-hanisch/schlechte-kontrollen-demo) und [backdoor-demo](https://github.com/sebastian-hanisch/backdoor-demo): Dort stand, welche Mengen gültig sind und welche Variablen man nicht aufnimmt. Hier steht, **welche der gültigen Mengen** man nimmt, weil ihre Schätzungen verschieden stark streuen.

## Kernfrage

Alle nach dem Adjustierungskriterium gültigen Mengen liefern denselben Koeffizienten in der Population, aber nicht dieselbe Genauigkeit. Ist die kleinste (minimale) Menge die genaueste? Gibt es eine beste Menge, die man aus dem Diagramm allein abliest? Und was bleibt, wenn ein Knoten dieser Menge nicht beobachtet ist?

## Modell

- **Welten** (feste Diagramme, Knoten in topologischer Reihenfolge): das 6-Knoten-Beispiel aus dem dritten Stück (Region, Wetter, Schwankung, Faktor, Lieferverzug, Servicegrad) und eine Präzisions-Welt mit sieben Knoten (Region, Schwankung, Wetter, Zuteilungsregel, Faktor, Servicegrad, Rückfragen), in der das Wetter nur den Servicegrad, die Zuteilungsregel nur den Faktor und die Rückfragen nur eine Folge des Faktors bewegen.
- **Daten:** lineares Normalmodell mit Einheitsrauschen aus dem Diagramm (Generator `kausal_generator.py`, wortgleich in allen Stücken der Linie). Die **wahre Wirkung** ist die Summe der Pfadprodukte.
- **Frage:** Behandlung x (ein Knoten mit Nachfahren), Wirkung y (ein Nachfahre von x), Menge Z und unbeobachtete Knoten wählt man frei.

## Methodik

- **Gültigkeit:** Adjustierungskriterium (Shpitser, VanderWeele, Robins) über `bdr_backdoor.py` (wortgleich aus backdoor-demo); das Backdoor-Kriterium wird mitgeführt.
- **Genauigkeit:** asymptotische Varianz des Koeffizienten von x in y ~ x + Z, mal n: Restvarianz von y gegeben x und Z durch Restvarianz von x gegeben Z, aus der Kovarianz der Population.
- **O-Menge** (Henckel, Perković, Maathuis): Eltern der Knoten auf kausalen Pfaden, ohne die verbotenen Knoten; im linearen Modell nicht zu übertreffen. Die Demo prüft sie gegen Brute Force über alle gültigen Mengen.
- **Experimente** laufen über feste Zufallsströme (SplitMix64, ganzzahlig, plattformstabil).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| 6-Knoten-Welt: Wie unterscheiden sich die 8 gültigen Mengen? | Standardfehler gegen die genaueste Menge: {Schwankung} **×1,46**, {Wetter, Schwankung} ×1,35, {Region, Schwankung} ×1,40, {Region, Wetter, Schwankung} ×1,28; die vier Mengen mit dem Lieferverzug haben alle **×1,00** (Gleichstand). Die O-Menge ist {Schwankung, Lieferverzug}. | `test_experiment_1_six_node_world` |
| Stimmt die Formel? (n = 500, 300 feste Wiederholungen) | Gemessene Streuung der Schätzungen und Formel weichen bei 300 Wiederholungen um höchstens 14 % ab (0,059 gegen 0,066 bei {Schwankung}; 0,043 gegen 0,045 bei O; Stichprobenrauschen, mit 4 000 Wiederholungen unter 5 %); die 8 ungültigen Mengen decken die Wirkung in **0 %**, mit Fehlern von 0,30 bis 0,55. | `test_experiment_1_formula_and_sample` |
| Präzisions-Welt (16 von 32 Mengen gültig) | {Schwankung} **×1,28**, Eltern von x {Schwankung, Zuteilungsregel} **×1,56**, O = {Schwankung, Wetter} **×1,00**; {Schwankung, Rückfragen} ×1,79 (nur nach dem Adjustierungskriterium gültig), ungünstigste gültige Menge ×2,00. | `test_experiment_1_precision_world` |
| Hand-Instanz Z → X, Z → Y, P → Y, I → X, X → Y (alle 1) | Varianz·n: {Z} 1, {Z, P} 0,5, {Z, I} 2, {Z, P, I} 1; O = {Z, P}. | `test_hand_instance` |
| Ist O nie zu schlagen? (3 000 Zufallsdiagramme, 7 Knoten) | O ist in **3 000 von 3 000** gültig, es gibt **0 Fälle**, in denen eine gültige Menge kleinere Varianz hat. Im Mittel 11,7 gültige Mengen (Median 8) und 1,09 minimale; 6 925 Gleichstände mit anderen Mengen. | `test_experiment_2_o_is_never_beaten` |
| Ist die kleinste Menge die genaueste? | In **56,5 %** der Diagramme (1 695) hat mindestens eine minimale Menge einen größeren Standardfehler als O. Kleinste minimale Menge: Median ×1,01, Mittel ×1,19, 90 %-Quantil ×1,54, Höchstwert ×6,4; gleich O in 49 %. | `test_experiment_2_rules_against_o` |
| Eltern von x und andere Regeln | Eltern von x: Median **×1,17**, Mittel ×1,31, 90 %-Quantil ×1,80, gleich O in 34 %. Alle Nicht-Nachfahren: Median ×1,00, Mittel ×1,09, gleich O in 76 %. Leere Menge (in 2 098 Diagrammen gültig): Median ×1,04. | `test_experiment_2_rules_against_o` |
| Was ändert eine zusätzliche Variable? | Nur über y wirksam (Präzision, 9 116 Hinzufügungen): Varianz sinkt in **91 %**, Median ×0,87. Nur über x (Instrument, 3 563): steigt in **95 %**, ×1,19. Nachfahre von x (6 733): steigt in **92 %**, ×1,16. Gemeinsame Ursache (1 195): sinkt in 40 %, steigt in 44 %. | `test_experiment_3_roles` |
| Ein Knoten unbeobachtet (2 000 Diagramme) | Gültige beobachtete Menge in 94,4 %, O vollständig beobachtet in **77,2 %**; in den übrigen 345 Fällen mit gültiger Menge ist die beste beobachtete im Median **×1,19**, höchstens ×2,08 so ungenau wie O. | `test_experiment_4_hidden` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Diagramm ist richtig** | Die O-Menge folgt aus dem gezeichneten Diagramm. Fehlt ein Pfeil, kann eine „gültige“ Menge verzerrt sein, auch die als genaueste ausgewählte (nicht untersucht, ob stärker). | Wie falsch darf das Diagramm sein? |
| **Das Modell ist linear, der Schätzer die Regression** | Die Optimalität gilt für die Varianz des Koeffizienten in linearen Modellen. Bei nichtlinearen Zusammenhängen entscheiden die Lernverfahren. | Doppelt robust und Double ML |
| **Alle Knoten der O-Menge sind beobachtet** | Bei einem zufällig verdeckten Knoten (7-Knoten-Diagramme) ist sie in 77 % der Diagramme erreichbar. | Frontdoor, Instrumente |
| **n ist groß genug** | Jede Variable kostet einen Freiheitsgrad; die Formel ist asymptotisch. Gemessen wurde der Unterschied nicht. | Schätzen |
| **Messen kostet nichts** | Die Demo zählt nur die Varianz, nicht den Aufwand je Variable. | nicht gebaut |
| **Es gibt Vergleichsstandorte mit und ohne hohen Faktor** | Die Formel sagt nichts über die Überlappung. | Schätzen |

Die Zahlen gelten für diese erfundene Welt und für Zufallsdiagramme dieser Generatorfamilie (7 Knoten, Kantenwahrscheinlichkeit 0,25 bis 0,5, Koeffizienten ±0,4 bis ±1,2); „Wahrheit“ ist das simulierte Modell, nicht ein Befund über echte Standorte.

## Bezug zu den Nachbarn im Portfolio

- [backdoor-demo](https://github.com/sebastian-hanisch/backdoor-demo): liefert das Adjustierungskriterium und die gültigen Mengen; `bdr_backdoor.py` ist wortgleich übernommen. Dort steht „gültig“, hier die Genauigkeit.
- [schlechte-kontrollen-demo](https://github.com/sebastian-hanisch/schlechte-kontrollen-demo): misst den Standardfehler je einzelner Kontrolle (Präzisionsvariable, Instrument); hier steht die Wahl unter mehreren Variablen.
- [cuped-demo](https://github.com/sebastian-hanisch/cuped-demo): Varianzreduktion durch eine Vorperiode bei zufälliger Zuteilung; hier ist die Zuteilung beobachtet und die Frage, welche gültige Menge am genauesten ist.

## Tests

Mini-Instanzen von Hand (Varianz je Menge {Z} 1, {Z, P} 0,5, {Z, I} 2; O-Mengen von Mediator, Collider, Vorfahre), Varianzformel gegen den Kehrwert der Kovarianz und gegen 3 000 feste Wiederholungen, O gegen Brute Force auf 300 Zufallsdiagrammen und allen 223 Diagrammen mit 4 Knoten, Adjustierungskriterium gegen die Population, Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Welt, Behandlung, unbeobachtete Knoten, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Strukturmodell, Wirkung) |
| `dsep_graph.py` | d-Separation, Pfade, Aussagenmengen (wortgleich aus d-separation-demo) |
| `bdr_backdoor.py` | Rückwege, Backdoor- und Adjustierungskriterium, gültige Mengen (wortgleich aus backdoor-demo) |
| `ajm_sets.py` | Varianz je Menge, O-Menge, Gleichstände, Rang, Rolle einer Variable |
| `ajm_evaluation.py` | Welten, Population, Analyse, Experimente, Verdikte |
| `ajm_visualization.py`, `ajm_pdf_export.py` | Figuren, Tabellen, PDF |
| `ajm_constants.py`, `ajm_presets.py` | Welten, Regler, Presets, Permalink, Zustandsbereinigung |
| `tests/` | Mini-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Kostenoptimale Mengen:** Messkosten je Variable würden die Frage verschieben; hier zählt nur die Varianz.
- **Optimale Menge bei verborgenen Variablen:** Die Demo sucht dann unter den beobachteten gültigen Mengen per Brute Force; die verallgemeinerte Menge ist nicht umgesetzt.
- **Schätzer jenseits der Regression, Stratifizieren, Matching, Neigungswert:** eigene Stücke der Linie.
- **Nichtlineare Modelle, Zeitreihen, Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html).
