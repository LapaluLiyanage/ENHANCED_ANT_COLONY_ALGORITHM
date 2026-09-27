"""Reproducible benchmark: fixed-probability variants vs VAM vs a real MOACO.

    python experiments/run_benchmark.py            # full run (~10-20 min on 2 cores)
    python experiments/run_benchmark.py --quick    # smoke run (~1 min)

Outputs (in results/):
    raw.csv          one row per (instance, method, run)
    summary.md       mean/median metrics, average Friedman ranks, Wilcoxon tests
    gap_by_method.png, front_example.png

Metrics (objectives minimised):
    gap   - compromise gap: mean relative distance to the exact ideal point of the
            best solution a method returns (lower is better; 0 = ideal).
    hv    - hypervolume ratio vs the reference front (higher is better; 1 = as
            good as the best known front). Single-solution methods get the HV of
            their one point, which is why the MOACO dominates this metric.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motp import MOTP, aco, exact, fixedprob, metrics  # noqa: E402
from motp.aco import ACOParams  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

BASELINE = "FP-geometric"  # the method as presented in the source paper


def heuristic_methods():
    """name -> callable(problem) -> objective vector (deterministic)."""
    ms = {}
    for c in ["geometric", "arithmetic", "harmonic", "minimum"]:
        ms[f"FP-{c}"] = lambda p, c=c: fixedprob.run(p, c).objective_totals
    for c in ["arithmetic", "harmonic", "minimum"]:
        ms[f"FP-{c}-norm"] = lambda p, c=c: fixedprob.run(p, c, normalize="max").objective_totals
    ms["VAM-arithmetic-norm"] = lambda p: fixedprob.vam(p, "arithmetic", normalize="max").objective_totals
    return ms


def aco_methods(n_iter):
    base = dict(n_ants=20, n_iter=n_iter)
    return {
        "MOACO": lambda p, s: aco.run(p, ACOParams(seed=s, **base)),
        "MOACO+LS": lambda p, s: aco.run(p, ACOParams(seed=s, ls_prob=0.1, **base)),
        "MOACO+LS (FP-seeded)": lambda p, s: aco.run(
            p, ACOParams(seed=s, ls_prob=0.1, **base),
            seed_allocations=[fixedprob.run(p, c).allocation for c in ["geometric", "arithmetic", "harmonic", "minimum"]],
        ),
    }


def run_instance(args):
    inst_id, m, n, k, corr, seed, n_iter, n_runs = args
    p = MOTP.random(m, n, k, cost_range=(1, 100), correlation=corr, seed=seed)
    ideal, nadir, _ = exact.ideal_and_nadir(p)
    rows, found = [], []

    for name, f in heuristic_methods().items():
        t = time.perf_counter()
        pt = np.atleast_2d(f(p))
        rows.append(dict(method=name, run=0, points=pt, time=time.perf_counter() - t))
        found.append(pt)

    for name, f in aco_methods(n_iter).items():
        for r in range(n_runs):
            t = time.perf_counter()
            res = f(p, 1000 * seed + r)
            rows.append(dict(method=name, run=r, points=res.archive_objectives,
                             time=time.perf_counter() - t, archive=len(res.archive_objectives)))
            found.append(res.archive_objectives)

    t = time.perf_counter()
    supported = exact.weighted_sum_front(p, steps=50 if k == 2 else 12)
    rows.append(dict(method="Exact LP (weighted sums)", run=0, points=supported,
                     time=time.perf_counter() - t, archive=len(supported)))
    found.append(supported)

    front, is_exact = exact.reference_front(p, extra_points=np.vstack(found))
    out = []
    for row in rows:
        pts = row.pop("points")
        out.append(dict(
            instance=inst_id, m=m, n=n, K=k, correlation=corr, seed=seed, **row,
            gap=metrics.compromise_gap(pts, ideal),
            hv=metrics.hv_ratio(pts, front, ideal, nadir),
            front_exact=is_exact,
        ))
    return out


def summarize(df: pd.DataFrame) -> str:
    # average stochastic runs per instance
    per = df.groupby(["instance", "method"], as_index=False)[["gap", "hv", "time"]].mean()
    methods = list(dict.fromkeys(df["method"]))
    lines = ["# Benchmark summary", ""]
    lines.append(f"{per['instance'].nunique()} instances, sizes "
                 f"{sorted({(int(a), int(b)) for a, b in zip(df.m, df.n)})}, "
                 f"objectives {sorted(int(k) for k in df.K.unique())}, "
                 f"correlations {sorted(float(c) for c in df.correlation.unique())}; "
                 f"{int(df.run.max()) + 1} runs per stochastic method, averaged per instance.")
    lines.append("")

    for metric, better in [("gap", "lower"), ("hv", "higher")]:
        wide = per.pivot(index="instance", columns="method", values=metric)[methods]
        ranks = wide.rank(axis=1, ascending=(better == "lower")).mean()
        fr = stats.friedmanchisquare(*[wide[c] for c in methods])
        lines += [f"## {metric} ({better} is better)", "",
                  f"Friedman chi2 = {fr.statistic:.1f}, p = {fr.pvalue:.2e}", "",
                  f"| method | mean | median | avg rank | vs {BASELINE}: better/worse | Wilcoxon p (Holm) |",
                  "|---|---|---|---|---|---|"]
        pvals = {}
        for c in methods:
            if c == BASELINE:
                continue
            d = wide[c] - wide[BASELINE]
            d = d[d != 0]
            pvals[c] = stats.wilcoxon(d).pvalue if len(d) >= 5 else float("nan")
        holm = _holm(pvals)
        for c in sorted(methods, key=lambda c: ranks[c]):
            if c == BASELINE:
                bw, pv = "-", "-"
            else:
                sign = -1 if better == "lower" else 1
                diff = sign * (wide[c] - wide[BASELINE])
                bw = f"{(diff > 1e-12).sum()}/{(diff < -1e-12).sum()}"
                pv = f"{holm[c]:.1e}" if not np.isnan(holm[c]) else "n/a"
            lines.append(f"| {c} | {wide[c].mean():.4f} | {wide[c].median():.4f} | {ranks[c]:.2f} | {bw} | {pv} |")
        lines.append("")

    t = per.groupby("method")["time"].mean().reindex(methods)
    lines += ["## Mean runtime per instance (s)", "", "| method | seconds |", "|---|---|"]
    lines += [f"| {c} | {t[c]:.4f} |" for c in methods]
    lines.append("")
    return "\n".join(lines)


def _holm(pvals: dict) -> dict:
    items = sorted((p, c) for c, p in pvals.items() if not np.isnan(p))
    out, running = {c: float("nan") for c in pvals}, 0.0
    for rank, (p, c) in enumerate(items):
        running = max(running, min(1.0, (len(items) - rank) * p))
        out[c] = running
    return out


def plots(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = df.groupby(["instance", "method"], as_index=False)["gap"].mean()
    order = per.groupby("method")["gap"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.2))
    data = [per.loc[per.method == m, "gap"].values for m in order]
    bp = ax.boxplot(data, vert=False, labels=order, patch_artist=True, widths=0.55,
                    medianprops=dict(color="#0b0b0b", linewidth=1.5), flierprops=dict(markersize=3))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor("#eb6834" if m.startswith("MOACO") else "#2a78d6" if m.startswith("FP") else "#1baf7a")
        patch.set_alpha(0.8)
        patch.set_edgecolor("#fcfcfb")
    ax.invert_yaxis()
    ax.set_xlabel("Compromise gap to exact ideal point (lower is better)")
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "gap_by_method.png"), dpi=160)
    plt.close(fig)


def front_example(seed: int = 7) -> None:
    """Objective-space picture for one 2-objective instance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = MOTP.random(10, 12, 2, seed=seed)
    ideal, _, _ = exact.ideal_and_nadir(p)
    sup = exact.weighted_sum_front(p, steps=60)
    res = aco.run(p, ACOParams(seed=seed, ls_prob=0.1, n_iter=100))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sup[:, 0], sup[:, 1], "-", color="#8f8e88", linewidth=2, label="Exact supported Pareto front (LP)")
    ax.plot(res.archive_objectives[:, 0], res.archive_objectives[:, 1], "o", color="#eb6834", markersize=5,
            markeredgecolor="#fcfcfb", label="MOACO+LS archive")
    colors = ["#2a78d6", "#1baf7a", "#4a3aa7", "#e87ba4"]
    for c, col in zip(["geometric", "arithmetic", "harmonic", "minimum"], colors):
        f = fixedprob.run(p, c).objective_totals
        ax.plot(f[0], f[1], "s", color=col, markersize=9, markeredgecolor="#fcfcfb", label=f"FP-{c}")
    ax.plot(*ideal, "*", color="#0b0b0b", markersize=12, label="Ideal point")
    ax.set_xlabel("Objective 1 total")
    ax.set_ylabel("Objective 2 total")
    ax.grid(color="#e6e5e0", linewidth=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Random 10x12 instance (seed {seed})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "front_example.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3, help="runs per stochastic method per instance")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    sizes = [(5, 5), (10, 10), (15, 20)]
    ks, corrs, seeds, runs, iters = [2, 3], [0.0, 0.5], range(a.seeds), a.runs, a.iters
    if a.quick:
        sizes, ks, corrs, seeds, runs, iters = [(5, 5), (8, 10)], [2, 3], [0.0], range(3), 2, 20

    jobs = [(f"{m}x{n}-K{k}-c{c}-s{s}", m, n, k, c, s, iters, runs)
            for (m, n), k, c, s in itertools.product(sizes, ks, corrs, seeds)]
    os.makedirs(RESULTS, exist_ok=True)
    print(f"{len(jobs)} instances on {a.workers} workers...")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, out in enumerate(ex.map(run_instance, jobs), 1):
            rows += out
            print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "raw.csv"), index=False)
    summary = summarize(df)
    open(os.path.join(RESULTS, "summary.md"), "w").write(summary)
    plots(df)
    front_example()
    print(summary)


if __name__ == "__main__":
    main()
