"""Welche Menge? Gültig heißt nicht gleich genau - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Linie Kausale Diagramme und Adjustierung. Mehrere Mengen sind nach dem Adjustierungskriterium gültig; ihre Schätzungen streuen nicht alle gleich stark.
Kern: ajm_sets.py, bdr_backdoor.py und dsep_graph.py (wortgleich aus backdoor-demo und d-separation-demo), Generator: kausal_generator.py.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import ajm_constants as C
from ajm_evaluation import (
    RULE_NAMES,
    Settings,
    analyse,
    de,
    hidden_experiment,
    roles_experiment,
    rules_experiment,
    set_text,
    sets_experiment,
    signed,
    verdict_hidden,
    verdict_instance,
    verdict_no_valid_set,
    verdict_roles,
    verdict_rules,
    verdict_sets,
    verdict_unavailable_o,
)
from ajm_pdf_export import build_pdf
from ajm_presets import (
    apply_preset,
    bounds,
    clean_state,
    init_session_state_defaults,
    load_permalink_settings,
    on_world_change,
    on_x_change,
    randomize_seed,
    sync_query_params,
    x_options,
    y_options,
)
from ajm_visualization import (
    build_diagram,
    build_hidden_chart,
    build_roles_chart,
    build_rules_chart,
    build_sets_chart,
    build_sets_table,
    style_sets_table,
)

st.set_page_config(page_title="Welche Menge? – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _sets(world):
    return sets_experiment(world)


@st.cache_data(show_spinner=SPINNER)
def _rules():
    return rules_experiment()


@st.cache_data(show_spinner=SPINNER)
def _roles():
    return roles_experiment()


@st.cache_data(show_spinner=SPINNER)
def _hidden():
    return hidden_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎯 Welche Menge? Gültig heißt nicht gleich genau")
st.markdown(
    """
**Unter den gültigen Mengen ist eine die genaueste, und die kleinste minimale Menge ist es nur in rund der Hälfte der Zufallsdiagramme.** Dies ist das fünfte Stück der Linie **Kausale Diagramme und Adjustierung**. Das
dritte Stück hat gezeigt, welche Mengen die Hintertüren schließen; das vierte, welche Variablen man nicht aufnimmt. Hier geht es um die Wahl **unter** den gültigen Mengen:
Sie liefern alle denselben Mittelwert, aber ihre Schätzungen streuen nicht alle gleich stark. In linearen Modellen gibt es eine **optimale Menge** (O-Menge), die man aus dem
Diagramm allein abliest.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man die genaueste Menge", expanded=True):
    st.markdown(
        """
1. **Gültig zuerst.** Nur Mengen, die das Adjustierungskriterium erfüllen, kommen in Frage; ungültige haben (bei allgemeinen Koeffizienten) einen Fehler, der auch mit mehr Daten bleibt.
2. **Variablen, die nur y bewegen, helfen meist.** Sie erklären Streuung des Servicegrads, ohne etwas zu verzerren: Der Standardfehler sinkt meist (Experiment 3: in 91 % der Hinzufügungen).
3. **Variablen, die nur x bewegen, schaden meist.** Sie binden harmlose Streuung des Faktors, die man braucht, um die Wirkung zu sehen: Der Standardfehler steigt meist (95 %). Ebenso bei Nachfahren von x (92 %).
4. **Die O-Menge** sind die Eltern der Knoten auf den kausalen Pfaden, ohne die verbotenen Knoten. Keine gültige Menge ist genauer; es kann Gleichstände geben.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
clean_state()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt (liefert Diagramm und Daten)**")
    world = st.selectbox("Diagramm", C.WORLD_ORDER, key="world_select", on_change=on_world_change,
                         help="Das wahre Ursachen-Diagramm. Beim Wechsel setzt die Demo Behandlung, Wirkung, Menge und unbeobachtete Knoten auf die Voreinstellung der Welt.")
    names = C.WORLDS[world]["nodes"]
    nn = len(names)
    st.markdown("**Frage**")
    x = st.selectbox("Behandlung x", x_options(world), key="x_select", format_func=lambda v: names[v], on_change=on_x_change,
                     help="Die Variable, deren Wirkung man wissen will. Nur Knoten mit mindestens einem Nachfahren.")
    y = st.selectbox("Wirkung y", y_options(world, x), key="y_select", format_func=lambda v: names[v], help="Ein Nachfahre von x: sonst gibt es keinen kausalen Pfad.")
    hidden = st.multiselect("Nicht beobachtet", [v for v in range(nn) if v not in (x, y)], key="hidden_select", format_func=lambda v: names[v],
                            help="Diese Knoten lassen sich nicht in die Menge aufnehmen. Liegt die genaueste Menge teils darin, ist sie nicht erreichbar.")
    st.markdown("**Menge**")
    z = st.multiselect("Festgehaltene Variablen Z", [v for v in range(nn) if v not in (x, y) and v not in hidden], key="z_select", format_func=lambda v: names[v],
                       help="Die Variablen, die man in die Regression aufnimmt.")
    st.markdown("**Daten**")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Stichprobengröße der simulierten Daten.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"w": world, "x": x, "y": y, "z": ",".join(str(v) for v in z), "h": ",".join(str(v) for v in hidden), "n": int(n), "seed": int(seed)})

settings = Settings(world, int(x), int(y), tuple(sorted(z)), tuple(sorted(hidden)), int(n), int(seed))
a = _analysis(settings)
c = a.chosen

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
c1.metric("Kriterium der Menge", "beide Kriterien" if c.backdoor else ("nur Adjustierung" if c.adjustment else "ungültig"),
          delta=("Fehler im Grenzfall " + signed(c.bias) if not c.adjustment else "Fehler im Grenzfall 0"), delta_color="off",
          help="Das Adjustierungskriterium ist allgemeiner: Es erlaubt Nachfahren von x, die auf keinem kausalen Pfad liegen und keine Verzerrung öffnen.")
c2.metric("Standardfehler gegen die genaueste Menge", f"×{de(c.ratio, 2)}" if c.ratio is not None else "–",
          delta=(f"Rang {c.rank} von {len({r.rank for r in a.rows if r.adjustment})}" if c.rank is not None else "nur für gültige Mengen"), delta_color="off",
          help="Verhältnis des asymptotischen Standardfehlers dieser Menge zu dem der genauesten gültigen Menge (×1,00 heißt: nicht zu übertreffen).")
c3.metric("Schätzung (Koeffizient von x)", signed(a.estimate, 3), delta=f"Wahrheit {de(a.truth, 2)}, 95-%-Intervall {de(a.lo, 2)} bis {de(a.hi, 2)}", delta_color="off")
c4.metric("Gültige Mengen", f"{a.n_valid} von {len(a.rows)}", delta="unter den beobachteten Variablen", delta_color="off")

left, right = st.columns([1.1, 1])
with left:
    st.plotly_chart(build_diagram(a), width="stretch", key="diagram_chart")
    st.caption("Grüne Pfeile: kausale Pfade. Blau gefüllt: festgehaltene Variablen. Grüner Rand: O-Menge. Schwarzer Rand: Behandlung und Wirkung. Grauer Knoten: unbeobachtet. "
               "Unter dem Namen steht die Rolle für die Genauigkeit.")
with right:
    st.markdown("**Alle Mengen unter den beobachteten Variablen**")
    cur = set_text(a.names, settings.z)
    st.dataframe(style_sets_table(build_sets_table(a), cur), hide_index=True, width="stretch", height=min(60 + 35 * len(a.rows), 480))
    st.caption("Blau: die aktuelle Menge. Grün hinterlegt: unter den gültigen am genauesten. „Standardfehler“ gegen die genaueste gültige Menge.")
verdict(verdict_instance(a))
for extra in (verdict_no_valid_set(a), verdict_unavailable_o(a)):
    if extra:
        verdict(extra)
st.caption(f"Standardfehler nach Formel bei n = {settings.n}: {de(a.sd_formula, 3)}; aus der Stichprobe geschätzt: {de(a.se, 3)}.")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="welche-menge.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Alle gültigen Mengen der Welt")
res1 = _sets(world)
st.caption(f"Welt „{world}“, Behandlung {res1['names'][res1['x']]}, Wirkung {res1['names'][res1['y']]}, n = {res1['n']}, {res1['reps']} feste Wiederholungen. Balken: "
           "Standardfehler nach Formel; Raute: gemessene Streuung der Schätzungen.")
st.plotly_chart(build_sets_chart(res1), width="stretch", key="sets_chart")
verdict(verdict_sets(res1))
st.dataframe(
    [{"Menge": set_text(res1["names"], r["z"]), "Kriterium": ("Backdoor und Adjustierung" if r["backdoor"] else "nur Adjustierung") if r["valid"] else "ungültig",
      "Standardfehler (Formel)": de(r["sd_formula"], 4), "Streuung gemessen": de(r["sd_measured"], 4), "Mittel": de(r["mean"], 3),
      "Deckung": f"{de(100 * r['cover'], 0)} %", "Rolle": ", ".join(t for t, f in (("minimal", r["minimal"]), ("O-Menge", r["is_o"]), ("genaueste", r["best"])) if f)}
     for r in res1["rows"] if r["valid"]],
    hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Welche Regel trifft die genaueste Menge?")
res2 = _rules()
st.caption(f"{res2['diagrams']} Zufallsdiagramme mit {C.RANDOM_NODES} Knoten, zufällige Koeffizienten, alle Teilmengen. Gezeigt ist der Standardfehler jeder Regel im Verhältnis "
           "zur O-Menge, wo die Regel eine gültige Menge liefert.")
st.plotly_chart(build_rules_chart(res2), width="stretch", key="rules_chart")
verdict(verdict_rules(res2))
ties = f"{res2['ties']:,}".replace(",", ".")
st.caption(f"Je Diagramm gibt es im Mittel {de(res2['mean_valid'], 1)} gültige Mengen (Median {de(res2['median_valid'], 0)}) und {de(res2['mean_minimal'], 2)} minimale. "
           f"Die O-Menge war in {res2['diagrams'] - res2['o_invalid']} von {res2['diagrams']} Diagrammen gültig; {ties} Paare aus O und einer anderen "
           "Menge haben denselben Standardfehler (Gleichstände).")
st.dataframe([{"Regel": n_, "Diagramme, in denen sie gültig ist": len(res2["ratios"][n_]),
               "Median": f"×{de(float(np.median(res2['ratios'][n_])), 2)}", "90 %-Quantil": f"×{de(float(np.quantile(res2['ratios'][n_], 0.9)), 2)}",
               "Höchstwert": f"×{de(float(res2['ratios'][n_].max()), 1)}",
               "Gleich O": f"{de(100 * float(np.mean(res2['ratios'][n_] < 1 + 1e-7)), 0)} %"} for n_ in RULE_NAMES], hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Was ändert eine zusätzliche Variable?")
res3 = _roles()
st.caption(f"{res3['diagrams']} Zufallsdiagramme. Zu jeder gültigen Menge wird jeder Knoten außerhalb einzeln hinzugefügt, soweit die Menge gültig bleibt; gezählt wird, ob die "
           "Varianz der Schätzung sinkt oder steigt, nach der Rolle des Knotens.")
st.plotly_chart(build_roles_chart(res3), width="stretch", key="roles_chart")
verdict(verdict_roles(res3))
st.dataframe([{"Rolle": r["role"], "Hinzufügungen": f"{r['n']:,}".replace(",", "."), "Varianz sinkt": f"{de(100 * r['down'], 0)} %", "gleich": f"{de(100 * r['same'], 0)} %",
               "Varianz steigt": f"{de(100 * r['up'], 0)} %", "Median Standardfehler": f"×{de(r['median_se'], 2)}"} for r in res3["rows"] if r["n"]],
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Wenn ein Knoten unbeobachtet ist")
res4 = _hidden()
st.caption(f"{res4['diagrams']} Zufallsdiagramme mit {C.RANDOM_NODES} Knoten, ein zufälliger Knoten (nicht x, y) ist unbeobachtet.")
st.plotly_chart(build_hidden_chart(res4), width="stretch", key="hidden_chart")
verdict(verdict_hidden(res4))
st.caption(f"In {res4['unavailable_with_valid']} Diagrammen gibt es eine gültige Menge, aber die O-Menge ist nicht erreichbar; dort hat die beste beobachtete Menge im Mittel das "
           f"{de(res4['loss_mean'], 2)}fache, höchstens das {de(res4['loss_max'], 1)}fache des Standardfehlers der O-Menge.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Diagramm ist richtig** | Die O-Menge folgt aus dem gezeichneten Diagramm. Fehlt ein Pfeil, kann eine „gültige“ Menge verzerrt sein, auch die als genaueste ausgewählte (nicht untersucht, ob stärker). | Wie falsch darf das Diagramm sein? |
| **Das Modell ist linear, der Schätzer die Regression** | Die Optimalität gilt für die Varianz des Koeffizienten in linearen Modellen. Bei nichtlinearen Zusammenhängen entscheiden die Lernverfahren. | Doppelt robust und Double ML |
| **Alle Knoten der O-Menge sind beobachtet** | Bei einem zufällig verdeckten Knoten (7-Knoten-Diagramme) ist sie in {de(100 * res4['o_available'] / res4['diagrams'], 0)} % der Diagramme erreichbar (Experiment 4). | Frontdoor, Instrumente |
| **n ist groß genug** | Jede Variable kostet einen Freiheitsgrad; die Formel ist asymptotisch. Bei n ≥ 100 und höchstens fünf Variablen ist der Unterschied nach einer Überschlagsrechnung klein gegenüber den Unterschieden zwischen den Mengen, aber nicht null; gemessen wurde er nicht. | Schätzen |
| **Messen kostet nichts** | Die Demo zählt nur die Varianz, nicht den Aufwand je Variable. Kostenoptimale Mengen sind ein eigenes Problem. | nicht gebaut |
| **Es gibt Vergleichsstandorte mit und ohne hohen Faktor** | Die Formel sagt nichts über die Überlappung; bei fast deterministischer Zuteilung ist die Varianz groß. | Schätzen |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Ein linearer Graph $V = BV + \varepsilon$ mit Einheitsrauschen, $\Sigma = (I-B)^{-1}(I-B)^{-\top}$. Die **wahre Wirkung** von $x$ auf $y$ ist $T_{yx}$, die Summe
der Pfadprodukte.

**Zulässige Menge** (Adjustierungskriterium, Shpitser, VanderWeele, Robins 2010). $Z \cap \mathrm{forb}(x,y) = \emptyset$ und $x \perp_d y \mid Z$ im Graphen ohne die ersten
Kanten der kausalen Pfade. Alle zulässigen Mengen liefern in der Population denselben Koeffizienten $T_{yx}$.

**Varianz.** Der Koeffizient von $x$ in $y \sim x + Z$ hat die asymptotische Varianz
$$\operatorname{Var}\big(\hat\beta_{x\cdot Z}\big) \approx \frac{1}{n}\,\frac{\sigma^2_{y \mid x,Z}}{\sigma^2_{x \mid Z}},$$
Restvarianz von $y$ gegeben $x$ und $Z$ durch Restvarianz von $x$ gegeben $Z$. Eine Variable, die nur $y$ erklärt, verkleinert den Zähler; eine, die nur $x$ erklärt, verkleinert
den Nenner und vergrößert die Varianz.

**O-Menge** (Henckel, Perković, Maathuis 2022). Mit den Knoten $\mathrm{cn}(x,y)$ auf kausalen Pfaden (ohne $x$) und den verbotenen Knoten $\mathrm{forb}(x,y)$
(= $\mathrm{cn}$ und ihre Nachfahren):
$$O(x,y,G) = \operatorname{pa}\big(\mathrm{cn}(x,y)\big) \setminus \mathrm{forb}(x,y).$$
Im linearen Modell hat $O$ unter allen zulässigen Mengen die kleinste asymptotische Varianz; andere Mengen können gleichauf liegen.

**Beobachtungsgrenze.** Ist ein Knoten von $O$ nicht beobachtet, ist $O$ nicht erreichbar; die Demo sucht dann unter den beobachteten zulässigen Mengen die genaueste
(vollständig, nicht über die verallgemeinerte Menge).

**Literatur.** E. Henckel, E. Perković, M. Maathuis, *Graphical criteria for efficient total effect estimation via adjustment in causal linear models*, JRSS-B 84, 2022;
I. Shpitser, T. VanderWeele, J. Robins, *On the validity of covariate adjustment for estimating causal effects*, UAI 2010 (beide aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html)."
)
