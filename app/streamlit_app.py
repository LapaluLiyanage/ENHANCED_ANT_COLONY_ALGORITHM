"""Streamlit demo: multi-objective transportation planner.

Run from the repository root:

    pip install -r requirements.txt
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import os
import sys
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from motp import MOTP, aco, exact, fixedprob  # noqa: E402
from motp.aco import ACOParams  # noqa: E402
from motp.combiners import COMBINERS  # noqa: E402

# Categorical colours, fixed order (validated default palette).
C_ACO, C_FRONT, C_IDEAL = "#eb6834", "#8f8e88", "#0b0b0b"
HEURISTIC_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

st.set_page_config(page_title="MOTP Planner", page_icon="🚚", layout="wide")


# --------------------------------------------------------------------- inputs
def sample_problem(kind: str):
    if kind == "4×4 sample (2 objectives)":
        files, names = ["objective1.csv", "objective2.csv"], ["Cost", "Time"]
    else:
        files, names = ["objective1_new.csv", "objective2_new.csv", "objective3_new.csv"], ["Cost", "Time", "Distance"]
    from motp.problem import read_objective_csv

    mats = []
    for f in files:
        c, s, d = read_objective_csv(os.path.join(ROOT, "data", f))
        mats.append(c)
    return np.array(mats), s, d, names


def parse_uploads(files):
    from motp.problem import read_objective_csv

    mats, supply, demand = [], None, None
    for f in files:
        c, s, d = read_objective_csv(io.BytesIO(f.getvalue()))
        if supply is None:
            supply, demand = s, d
        elif c.shape != mats[0].shape:
            raise ValueError(f"{f.name}: table size {c.shape} differs from {mats[0].shape}")
        mats.append(c)
    return np.array(mats), supply, demand


def labels(n, prefix):
    return [f"{prefix}{i + 1}" for i in range(n)]


with st.sidebar:
    st.header("1 · Problem")
    source = st.radio(
        "Data source",
        ["4×4 sample (2 objectives)", "3×4 sample (3 objectives)", "Random instance", "Upload CSVs"],
    )
    names = None
    try:
        if source.endswith("objectives)"):
            costs, supply, demand, names = sample_problem(source)
        elif source == "Random instance":
            c1, c2 = st.columns(2)
            m = c1.number_input("Sources", 2, 30, 6)
            n = c2.number_input("Destinations", 2, 30, 8)
            k = c1.number_input("Objectives", 2, 3, 2)
            seed = c2.number_input("Seed", 0, 10_000, 7)
            corr = st.slider("Objective correlation", 0.0, 0.9, 0.0, 0.1,
                             help="0 = objectives conflict freely; higher = they tend to agree")
            p0 = MOTP.random(int(m), int(n), int(k), correlation=corr, seed=int(seed))
            costs, supply, demand = p0.costs, p0.supply, p0.demand
        else:
            st.caption("One CSV per objective, same layout as `data/objective1.csv`: "
                       "row labels, a `Supply` column and a `Demand` row.")
            ups = st.file_uploader("Objective CSVs", type="csv", accept_multiple_files=True)
            if not ups:
                st.info("Upload at least one CSV to continue.")
                st.stop()
            costs, supply, demand = parse_uploads(ups)
            names = [os.path.splitext(u.name)[0] for u in ups]
    except Exception as e:  # show input problems instead of a traceback
        st.error(f"Could not read the data: {e}")
        st.stop()

    K = costs.shape[0]
    names = names or [f"Objective {i + 1}" for i in range(K)]
    names = [st.text_input(f"Name of objective {i + 1}", v, key=f"name{i}-{source}") for i, v in enumerate(names)]

    st.header("2 · Methods")
    combiner = st.selectbox("Heuristic combiner", list(COMBINERS), index=list(COMBINERS).index("geometric"),
                            help="How the paper's fixed-probability method merges objectives. "
                                 "'geometric' is the method from the source paper.")
    compare_all = st.checkbox("Also run every other combiner and VAM", True)
    run_aco = st.checkbox("Run ant colony (MOACO + local search)", True)
    if run_aco:
        c1, c2 = st.columns(2)
        n_iter = c1.slider("Iterations", 10, 300, 80, 10)
        n_ants = c2.slider("Ants", 5, 60, 20, 5)
        ls_prob = st.slider("Local-search probability", 0.0, 1.0, 0.1, 0.05)
        aco_seed = st.number_input("ACO seed", 0, 10_000, 0)
    run_exact = st.checkbox("Compute exact LP reference", True,
                            help="Ideal point and supported Pareto front via linear programming.")

# ------------------------------------------------------------ build problem
m, n = costs.shape[1:]
rows, cols = labels(m, "S"), labels(n, "D")
problem = MOTP.balanced(costs, supply, demand, names)
dummy = problem.shape != (m, n)
if dummy:
    extra = "destination" if problem.shape[1] > n else "source"
    rows = rows + (["Dummy"] if extra == "source" else [])
    cols = cols + (["Dummy"] if extra == "destination" else [])

st.title("🚚 Multi-Objective Transportation Planner")
st.caption("Enhanced Ant Colony Algorithm research demo: compare the geometric-mean heuristic, "
           "classic VAM, an ant colony, and the exact LP optimum, then pick a shipping plan.")

if dummy:
    st.warning(f"Total supply ({supply.sum():g}) ≠ total demand ({demand.sum():g}). "
               f"A zero-cost dummy {extra} was added to balance the problem.")

with st.expander("Problem data", expanded=False):
    tabs = st.tabs(names)
    for t, name, mat in zip(tabs, names, problem.costs):
        df = pd.DataFrame(mat, index=rows, columns=cols)
        df["Supply"] = problem.supply
        df.loc["Demand"] = list(problem.demand) + [np.nan]
        t.dataframe(df.style.format(precision=2, na_rep=""), width="stretch")


# -------------------------------------------------------------------- solve
@st.cache_data(show_spinner=False)
def solve(costs, supply, demand, combiner, compare_all, run_aco, aco_cfg, run_exact):
    p = MOTP(costs, supply, demand)
    out = {"heuristics": {}, "aco": None, "ideal": None, "front": None, "times": {}}
    combs = list(COMBINERS) if compare_all else [combiner]
    for c in combs:
        t = time.perf_counter()
        r = fixedprob.run(p, c)
        out["times"][f"FP-{c}"] = time.perf_counter() - t
        out["heuristics"][f"FP-{c}"] = (r.allocation.astype(float), r.objective_totals)
    if compare_all:
        r = fixedprob.vam(p, "arithmetic", normalize="max")
        out["heuristics"]["VAM (normalised)"] = (r.allocation.astype(float), r.objective_totals)
    if run_exact:
        t = time.perf_counter()
        ideal, nadir, _ = exact.ideal_and_nadir(p)
        front = exact.weighted_sum_front(p, steps=40 if p.n_objectives == 2 else 10)
        out["times"]["Exact LP"] = time.perf_counter() - t
        out["ideal"], out["front"] = ideal, front
    if run_aco:
        t = time.perf_counter()
        seeds = [v[0] for v in out["heuristics"].values()]
        res = aco.run(p, ACOParams(**aco_cfg), seed_allocations=seeds)
        out["times"]["MOACO"] = time.perf_counter() - t
        out["aco"] = (res.archive_allocations, res.archive_objectives, res.history)
    return out


aco_cfg = dict(n_iter=n_iter, n_ants=n_ants, ls_prob=ls_prob, seed=int(aco_seed)) if run_aco else None
with st.spinner("Solving…"):
    try:
        R = solve(problem.costs, problem.supply, problem.demand, combiner, compare_all, run_aco, aco_cfg, run_exact)
    except Exception as e:
        st.error(f"Solver error: {e}")
        st.stop()

ideal = R["ideal"] if R["ideal"] is not None else None
if ideal is None:  # fall back to the best values any method found
    allF = [f for _, f in R["heuristics"].values()]
    if R["aco"]:
        allF += list(R["aco"][1])
    ideal = np.min(allF, axis=0)


def gap(f):
    return float(np.mean((np.asarray(f) - ideal) / np.where(ideal == 0, 1, ideal)))


# ---------------------------------------------------------------- candidates
cands = []  # (label, method, allocation, objectives)
for name, (x, f) in R["heuristics"].items():
    cands.append((name, name, x, f))
if R["aco"]:
    for i, (x, f) in enumerate(zip(R["aco"][0], R["aco"][1]), 1):
        cands.append((f"MOACO plan {i}", "MOACO", x, f))

# ------------------------------------------------------------------ summary
best_heur = min(R["heuristics"].items(), key=lambda kv: gap(kv[1][1]))
c = st.columns(4)
c[0].metric("Problem size", f"{m} × {n}", f"{K} objectives", delta_color="off")
paper_key = "FP-geometric" if "FP-geometric" in R["heuristics"] else f"FP-{combiner}"
paper_gap = gap(R["heuristics"][paper_key][1])
c[1].metric("Paper method gap" if paper_key == "FP-geometric" else f"{paper_key} gap", f"{paper_gap:.1%}",
            "above ideal", delta_color="off", help="Average % above the best value of each objective")
if R["aco"]:
    aco_best = min(gap(f) for f in R["aco"][1])
    c[2].metric("Ant colony gap", f"{aco_best:.1%}", f"{(aco_best - paper_gap) * 100:+.1f} pts vs paper method",
                delta_color="inverse")
    c[3].metric("Trade-off plans found", f"{len(R['aco'][1])}", "by the ant colony", delta_color="off")
st.divider()

# ------------------------------------------------------------ trade-off view
left, right = st.columns([3, 2])
with left:
    st.subheader("Trade-offs")
    if K >= 2:
        xi, yi = 0, 1
        if K == 3:
            a, b = st.columns(2)
            xi = names.index(a.selectbox("X axis", names, 0))
            yi = names.index(b.selectbox("Y axis", names, 1))
        recs = []
        if R["front"] is not None:
            for f in R["front"]:
                recs.append(dict(series="Exact LP front", x=f[xi], y=f[yi], label="LP"))
        if R["aco"]:
            for i, f in enumerate(R["aco"][1], 1):
                recs.append(dict(series="Ant colony plans", x=f[xi], y=f[yi], label=f"MOACO plan {i}"))
        for name, (_, f) in R["heuristics"].items():
            recs.append(dict(series=name, x=f[xi], y=f[yi], label=name))
        recs.append(dict(series="Ideal point", x=ideal[xi], y=ideal[yi], label="Ideal"))
        df = pd.DataFrame(recs)

        heur_names = list(R["heuristics"])
        domain = ["Exact LP front", "Ant colony plans"] + heur_names + ["Ideal point"]
        palette = [C_FRONT, C_ACO] + [HEURISTIC_COLORS[i % len(HEURISTIC_COLORS)] for i in range(len(heur_names))] + [C_IDEAL]
        present = [d for d in domain if d in set(df.series)]
        color = alt.Color("series:N", title=None,
                          scale=alt.Scale(domain=present, range=[palette[domain.index(d)] for d in present]),
                          legend=alt.Legend(orient="bottom", columns=3, symbolSize=120, labelLimit=220))

        def padded(v):
            lo, hi = float(np.min(v)), float(np.max(v))
            pad = (hi - lo) * 0.06 or max(abs(hi), 1) * 0.02
            return [lo - pad, hi + pad]

        X = alt.X("x:Q", title=names[xi], scale=alt.Scale(domain=padded(df.x), nice=False))
        Y = alt.Y("y:Q", title=names[yi], scale=alt.Scale(domain=padded(df.y), nice=False))
        tip = [alt.Tooltip("label:N", title="Plan"),
               alt.Tooltip("x:Q", title=names[xi], format=",.2f"),
               alt.Tooltip("y:Q", title=names[yi], format=",.2f")]
        base = alt.Chart(df).encode(x=X, y=Y, color=color, tooltip=tip)
        layers = []
        if R["front"] is not None and K == 2:
            layers.append(base.transform_filter(alt.datum.series == "Exact LP front")
                          .mark_line(strokeWidth=2, color=C_FRONT))
        layers += [
            base.transform_filter(alt.datum.series == "Exact LP front")
                .mark_circle(size=45, opacity=0.9),
            base.transform_filter(alt.datum.series == "Ant colony plans")
                .mark_circle(size=90, opacity=0.9, stroke="white", strokeWidth=1),
            base.transform_filter(alt.FieldOneOfPredicate(field="series", oneOf=heur_names))
                .mark_square(size=200, opacity=0.95, stroke="white", strokeWidth=1.5),
            base.transform_filter(alt.datum.series == "Ideal point")
                .mark_point(shape="cross", filled=True, size=320),
        ]
        st.altair_chart(alt.layer(*layers).properties(height=420).interactive(), width="stretch")
        st.caption("Lower-left is better on both axes. Squares are single-answer heuristics; "
                   "orange dots are the ant colony's non-dominated plans; the line joins exact LP trade-offs.")

with right:
    st.subheader("Compare methods")
    table = []
    for name, (_, f) in R["heuristics"].items():
        table.append({"Method": name, **{nm: f[i] for i, nm in enumerate(names)}, "Gap to ideal": gap(f)})
    if R["aco"]:
        F = R["aco"][1]
        b = int(np.argmin([gap(f) for f in F]))
        table.append({"Method": "MOACO (best compromise)", **{nm: F[b][i] for i, nm in enumerate(names)},
                      "Gap to ideal": gap(F[b])})
    if R["ideal"] is not None:
        table.append({"Method": "Ideal (not one plan)", **{nm: ideal[i] for i, nm in enumerate(names)},
                      "Gap to ideal": 0.0})
    tdf = pd.DataFrame(table).sort_values("Gap to ideal")
    st.dataframe(
        tdf.style.format({**{nm: "{:,.0f}" for nm in names}, "Gap to ideal": "{:.2%}"}),
        hide_index=True, width="stretch",
    )
    st.caption("Gap to ideal = average % above the best achievable value of each objective.")

st.divider()

# ------------------------------------------------------------- plan picker
st.subheader("Choose a shipping plan")
mode = st.radio("How to choose", ["By priorities", "Pick a specific plan"], horizontal=True,
                help="By priorities: move the sliders and the plan with the smallest importance-weighted "
                     "shortfall from the ideal (relative to the worst plan found) is chosen.")
if mode == "By priorities":
    cols_w = st.columns(K)
    w = np.array([cols_w[i].slider(f"Importance of {nm}", 0, 10, 5, key=f"w{i}") for i, nm in enumerate(names)],
                 dtype=float)
    w = w / w.sum() if w.sum() else np.full(K, 1 / K)
    # Weighted Tchebycheff on the ideal->worst range: picks the plan whose worst
    # (importance-weighted) shortfall is smallest, so equal weights give a balanced plan.
    allF = np.array([f for *_, f in cands])
    worst = allF.max(axis=0)
    span = np.where(worst - ideal > 0, worst - ideal, 1.0)
    scores = [float(np.max(w * (np.asarray(f) - ideal) / span)) for f in allF]
    choice = int(np.argmin(scores))
else:
    choice = st.selectbox("Plan", range(len(cands)), format_func=lambda i: cands[i][0])

label, method, X, F = cands[choice]
c = st.columns(K + 1)
c[0].metric("Selected plan", label)
for i, nm in enumerate(names):
    c[i + 1].metric(f"Total {nm}", f"{F[i]:,.0f}", f"+{(F[i] - ideal[i]) / max(ideal[i], 1e-9):.1%} vs ideal",
                    delta_color="inverse")

plan = pd.DataFrame(X, index=rows, columns=cols)
p1, p2 = st.columns([3, 2])
with p1:
    st.markdown("**Units to ship** (rows = sources, columns = destinations)")
    st.dataframe(plan.style.format("{:,.0f}").background_gradient(cmap="Blues", axis=None),
                 width="stretch")
with p2:
    st.markdown("**Shipment list**")
    ship = [{"From": rows[i], "To": cols[j], "Units": X[i, j],
             **{nm: X[i, j] * problem.costs[k, i, j] for k, nm in enumerate(names)}}
            for i, j in zip(*np.nonzero(X))]
    sdf = pd.DataFrame(ship)
    st.dataframe(sdf.style.format({"Units": "{:,.0f}", **{nm: "{:,.0f}" for nm in names}}),
                 hide_index=True, width="stretch")

d1, d2 = st.columns(2)
d1.download_button("Download plan matrix (CSV)", plan.to_csv().encode(), "shipping_plan_matrix.csv", "text/csv")
d2.download_button("Download shipment list (CSV)", sdf.to_csv(index=False).encode(), "shipment_list.csv", "text/csv")

# ------------------------------------------------------------- convergence
if R["aco"]:
    with st.expander("Ant colony details"):
        hist = pd.DataFrame({"Iteration": np.arange(1, len(R["aco"][2]) + 1), "Plans in archive": R["aco"][2]})
        st.altair_chart(alt.Chart(hist).mark_line(color=C_ACO, strokeWidth=2).encode(
            x="Iteration:Q", y="Plans in archive:Q",
            tooltip=["Iteration", "Plans in archive"]).properties(height=220), width="stretch")
        st.caption("Number of non-dominated plans kept by the colony. "
                   "The heuristics' solutions seed the colony, so it never does worse than them.")
        st.write({k: f"{v:.3f}s" for k, v in R["times"].items()})

with st.expander("How it works"):
    st.markdown(
        """
* **FP-\\<combiner\\>**: the repository's fixed-probability heuristic. Objectives are merged
  (geometric mean in the source paper), turned into a "probability" matrix, and allocated
  greedily with Vogel-style penalties that are computed once.
* **VAM (normalised)**: classic Vogel's Approximation Method on the normalised arithmetic mean.
* **Ant colony (MOACO)**: pheromone per objective, stochastic ants with different priorities,
  stepping-stone local search, and an archive of non-dominated plans.
* **Exact LP**: linear programming gives the ideal point (the best value of each objective on its
  own) and exact trade-off plans. For plain linear problems like this one, LP is the gold standard;
  the heuristics matter for speed, very large problems, or non-linear cost rules.
"""
    )
