r"""
Sample-size control (V5.2): separates the effect of the ENVIRONMENT from that of the SAMPLE SIZE in
the RSSI-only vs. COMMON comparison.

Question: does COMMON help outdoors and hurt indoors because the environment differs, or because the
indoor datasets are small (572-2,436 samples versus 56-130 thousand)?

Test: Antwerp 2019 and Bonn are subsampled to the sizes of the indoor scenarios (572 = Brno iUG,
930 = Bucharest iAG, 2,436 = Brno iAG) and run with exactly the V5 protocol (same grids,
preprocessing, 20 seeds, 80/10/10 random split and 20-block spatial holdout), comparing only
RSSI_ONLY and COMMON_FULL.

- Nothing from V5 is rerun: the full results in results_equal_footing_v5 are only read, as the
  reference (full outdoor and indoor).
- By default each seed draws a different subsample (the same one for RSSI_ONLY and COMMON_FULL and
  for both splits, so the comparison stays paired). With --fixed-subset, a single subsample per size
  is used for every seed.
- Cases are checkpointed as in V5: if a run stops, simply run it again.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Same models, grids, preprocessing, metrics, and checkpoints as the audited V5 benchmark.
from . import core as base
from .data import FeatureStore

EXT_PROTOCOL_VERSION = "MULTIDATASET_EQUAL_FOOTING_SUBSAMPLE_CONFOUND_V5_2"
METHODS = list(base.METHODS)
SEEDS_20 = list(base.SEEDS_20)
VARIANTS = ["RSSI_ONLY", "COMMON_FULL"]
SPLITS = ["random", "spatial"]
OUTDOOR = ["antwerp2019", "bonn"]
INDOOR = ["brno_iAG_BUT", "brno_iUG_BUT", "bucharest_iAG_UPB"]
DEFAULT_SIZES = [572, 930, 2436]
SUBSET_SALT = 20261022
LABELS = {"antwerp2019": "Antwerp 2019", "bonn": "Bonn", "brno_iAG_BUT": "Brno iAG",
          "brno_iUG_BUT": "Brno iUG", "bucharest_iAG_UPB": "Bucharest iAG"}


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def sub_name(dataset: str, n: int) -> str:
    return f"{dataset}_n{n}"


def draw_subset(n_total: int, n: int, seed: int, fixed: bool) -> np.ndarray:
    if n >= n_total:
        raise ValueError(f"Size {n} >= dataset size ({n_total}).")
    rng = np.random.default_rng([SUBSET_SALT, n] if fixed else [SUBSET_SALT, n, seed])
    return np.sort(rng.choice(n_total, size=n, replace=False)).astype(np.int64)


def checkpoint(root: Path, dataset: str, split: str, seed: int, variant: str, method: str) -> Path:
    return base.checkpoint_path(root, dataset, split, seed, variant, method)


def load_complete(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if d.get("status") == "complete" else None
    except Exception:
        return None


# --------------------------------------------------------------------------- execution
def plan(datasets, sizes, seeds, out: Path):
    jobs = [(d, n, st, s, v, m) for d in datasets for n in sizes for s in seeds
            for st in SPLITS for v in VARIANTS for m in METHODS]
    done = sum(load_complete(checkpoint(out, sub_name(d, n), st, s, v, m)) is not None
               for d, n, st, s, v, m in jobs)
    return jobs, done


def run_cases(datasets, sizes, seeds, profile, threads, cache_root: Path, out: Path,
              fixed: bool, force: bool) -> None:
    inference_reps = 2 if profile == "quick" else 5
    jobs, done = plan(datasets, sizes, seeds, out)
    total, todo = len(jobs), len(jobs) - (0 if force else done)
    log(f"Planned cases: {total}; already complete: {done}; to run now: {todo}")
    t0 = time.time(); executed = 0
    for d in datasets:
        store = FeatureStore(cache_root / d)
        groups = np.asarray(store.spatial_groups)
        for n in sizes:
            name = sub_name(d, n)
            for seed in seeds:
                idx = draw_subset(len(store.y), n, int(seed), fixed)
                splits = {"random": base.split_random(idx, int(seed)),
                          "spatial": base.split_spatial(idx, groups, int(seed))}
                for st in SPLITS:
                    base.save_split(out, name, st, int(seed), splits[st])
                    for v in VARIANTS:
                        for m in METHODS:
                            already = (not force) and load_complete(checkpoint(out, name, st, int(seed), v, m)) is not None
                            base.run_tuned_case(store, out, name, st, int(seed), v, m, splits[st],
                                                profile, threads, inference_reps, force,
                                                protocol_version=EXT_PROTOCOL_VERSION)
                            if not already:
                                executed += 1
                                el = time.time() - t0
                                eta = el / executed * (todo - executed)
                                log(f"PROGRESS {executed}/{todo} | elapsed {el/3600:.2f} h | "
                                    f"estimated remaining {eta/3600:.2f} h")
    log("Run complete.")


# --------------------------------------------------------------------------- aggregation
def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * p[k]); adj[k] = min(1.0, running)
    return adj


def collect(out: Path, baseline: Path, datasets, sizes, seeds) -> pd.DataFrame:
    rows, missing = [], []
    def add(root, ds_folder, scenario, n_label, env, st, seed, v, m):
        d = load_complete(checkpoint(root, ds_folder, st, seed, v, m))
        if d is None:
            missing.append(f"{ds_folder}/{st}/seed_{seed}/{v}/{m}")
            return
        rows.append({"scenario": scenario, "size": n_label, "environment": env, "split": st,
                     "seed": seed, "variant": v, "method": m, "mean_error_m": d["mean_error_m"],
                     "median_error_m": d.get("median_error_m"), "n_train": d.get("n_train"),
                     "n_test": d.get("n_test")})
    for st in SPLITS:
        for s in seeds:
            for v in VARIANTS:
                for m in METHODS:
                    for d in datasets:
                        add(baseline, d, d, "full", "outdoor", st, s, v, m)
                        for n in sizes:
                            add(out, sub_name(d, n), d, str(n), "outdoor", st, s, v, m)
                    for d in INDOOR:
                        add(baseline, d, d, "full", "indoor", st, s, v, m)
    if missing:
        log(f"WARNING: {len(missing)} missing cases (first: {missing[:5]})")
    return pd.DataFrame(rows)


def effects(cases: pd.DataFrame) -> pd.DataFrame:
    """Relative improvement of COMMON over RSSI-only per scenario and seed (positive = COMMON better)."""
    piv = cases.pivot_table(index=["scenario", "size", "environment", "split", "seed", "method"],
                            columns="variant", values="mean_error_m").dropna().reset_index()
    piv["delta_pct"] = 100 * (piv["RSSI_ONLY"] - piv["COMMON_FULL"]) / piv["RSSI_ONLY"]
    return piv


def condition_of(row) -> str:
    if row["environment"] == "indoor":
        return "indoor"
    return "outdoor_full" if row["size"] == "full" else f"outdoor_n{row['size']}"


def summarize(eff: pd.DataFrame) -> pd.DataFrame:
    eff = eff.copy(); eff["condition"] = eff.apply(condition_of, axis=1)
    # Equal weight per scenario within the condition, per seed (as in the article's environment classes).
    seedlvl = eff.groupby(["condition", "split", "method", "seed"]).delta_pct.mean().reset_index()
    out = []
    for (c, st), z in seedlvl.groupby(["condition", "split"]):
        rows = []
        for m in METHODS:
            x = z[z.method == m].delta_pct.to_numpy()
            if len(x) == 0:
                continue
            try:
                p = float(wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan
            except ValueError:
                p = np.nan
            rows.append({"condition": c, "split": st, "method": m, "n_seeds": len(x),
                         "mean_delta_pct": x.mean(), "median_delta_pct": float(np.median(x)),
                         "common_better_fraction": float(np.mean(x > 0)), "p_raw": p})
        r = pd.DataFrame(rows)
        ok = r.p_raw.notna()
        r["p_holm_6methods"] = np.nan
        if ok.any():
            r.loc[ok, "p_holm_6methods"] = holm(r.loc[ok, "p_raw"].to_numpy())
        r["significant"] = r.p_holm_6methods < 0.05
        out.append(r)
    return pd.concat(out, ignore_index=True)


def per_scenario(eff: pd.DataFrame) -> pd.DataFrame:
    g = eff.groupby(["scenario", "size", "split", "method"]).delta_pct
    return g.agg(n_seeds="count", mean_delta_pct="mean", median_delta_pct="median").reset_index()


def latex_table(summ: pd.DataFrame, sizes: Sequence[int]) -> str:
    cols = ["outdoor_full"] + [f"outdoor_n{n}" for n in sorted(sizes, reverse=True)] + ["indoor"]
    heads = ["Outdoor (full)"] + [f"$n={n:,}$".replace(",", "{,}") for n in sorted(sizes, reverse=True)] + ["Indoor"]
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{Mean relative improvement (\%) of COMMON over RSSI-only for the outdoor scenarios "
             r"(Antwerp 2019 and Bonn) at full size and subsampled to the sizes of the indoor scenarios, "
             r"compared with the indoor class. Positive values favor COMMON. $^{*}$Holm-adjusted $p<0.05$ "
             r"across the six methods within each column and split.}",
             r"\label{tab:subsample_confound}", r"\scriptsize",
             r"\begin{tabular}{l" + "r" * len(cols) + "r" * len(cols) + "}", r"\toprule",
             r" & \multicolumn{%d}{c}{Random split} & \multicolumn{%d}{c}{Spatial holdout} \\" % (len(cols), len(cols)),
             r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (len(cols) + 1, len(cols) + 2, 2 * len(cols) + 1),
             "Method & " + " & ".join(heads + heads) + r" \\", r"\midrule"]
    for m in METHODS:
        cells = []
        for st in SPLITS:
            for c in cols:
                z = summ[(summ.condition == c) & (summ.split == st) & (summ.method == m)]
                if z.empty:
                    cells.append("--")
                else:
                    r = z.iloc[0]
                    cells.append(f"${r.mean_delta_pct:+.2f}{'^{*}' if bool(r.significant) else ''}$")
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def verdict(summ: pd.DataFrame, sizes: Sequence[int]) -> List[str]:
    txt = []
    for st in SPLITS:
        txt.append(f"\n[{'random split' if st == 'random' else 'spatial holdout'}]")
        for c in ["outdoor_full"] + [f"outdoor_n{n}" for n in sorted(sizes, reverse=True)] + ["indoor"]:
            z = summ[(summ.condition == c) & (summ.split == st)]
            if z.empty:
                continue
            better = int(((z.mean_delta_pct > 0) & z.significant).sum())
            worse = int(((z.mean_delta_pct < 0) & z.significant).sum())
            ns = len(z) - better - worse
            txt.append(f"  {c:>15}: COMMON better in {better}, worse in {worse}, no significant difference in {ns} "
                       f"(mean over methods {z.mean_delta_pct.mean():+.2f}%)")
        small = summ[(summ.split == st) & summ.condition.isin([f"outdoor_n{n}" for n in sizes])]
        if not small.empty:
            frac_pos = float((small.mean_delta_pct > 0).mean())
            frac_neg_sig = float(((small.mean_delta_pct < 0) & small.significant).mean())
            if frac_pos >= 0.8:
                txt.append("  Reading: at the indoor sizes, COMMON still helps outdoors. "
                           "Sample size alone does not explain the indoor loss; the result supports an environment effect.")
            elif frac_neg_sig >= 0.5:
                txt.append("  Reading: with few samples, COMMON also hurts outdoors. "
                           "Sample size explains at least part of the indoor-outdoor difference.")
            else:
                txt.append("  Reading: mixed result. The sample-size effect depends on the method; "
                           "inspect the per-method and per-scenario tables before concluding.")
    return txt


def plot(summ: pd.DataFrame, sizes: Sequence[int], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    conds = [f"outdoor_n{n}" for n in sorted(sizes)] + ["outdoor_full", "indoor"]
    xl = [f"{n:,}" for n in sorted(sizes)] + ["Outdoor\n(full)", "Indoor\n(full)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, st in zip(axes, SPLITS):
        for i, m in enumerate(METHODS):
            ys = []
            for c in conds:
                z = summ[(summ.condition == c) & (summ.split == st) & (summ.method == m)]
                ys.append(float(z.mean_delta_pct.iloc[0]) if not z.empty else np.nan)
            ax.plot(range(len(conds) - 1), ys[:-1], marker="o", label=m)
            ax.plot([len(conds) - 1], [ys[-1]], marker="s", linestyle="none", color=ax.lines[-1].get_color())
        ax.axhline(0, color="0.4", lw=0.8)
        ax.axvline(len(conds) - 1.5, color="0.7", lw=0.8, ls="--")
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(xl)
        ax.set_title("Random split" if st == "random" else "Spatial holdout")
        ax.set_xlabel("Outdoor sample size (subsampled)")
    axes[0].set_ylabel("Relative improvement of COMMON over RSSI-only (%)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return True


def aggregate(out: Path, baseline: Path, datasets, sizes, seeds) -> None:
    agg = out / "aggregate_subsample"; agg.mkdir(parents=True, exist_ok=True)
    cases = collect(out, baseline, datasets, sizes, seeds)
    if cases.empty:
        raise RuntimeError("Nenhum caso encontrado para agregar.")
    cases.to_csv(agg / "cases.csv", index=False)
    eff = effects(cases); eff.to_csv(agg / "effects_by_seed.csv", index=False)
    summ = summarize(eff); summ.to_csv(agg / "summary_by_condition.csv", index=False)
    per_scenario(eff).to_csv(agg / "summary_by_scenario.csv", index=False)
    abs_err = (cases.groupby(["scenario", "size", "split", "variant", "method"]).mean_error_m
               .agg(["mean", "std", "count"]).reset_index())
    abs_err.to_csv(agg / "absolute_errors.csv", index=False)
    (agg / "table_subsample_by_condition.tex").write_text(latex_table(summ, sizes), encoding="utf-8")
    has_fig = plot(summ, sizes, agg / "figure_subsample.png")
    v = verdict(summ, sizes)
    readme = [
        "OUTDOOR SUBSAMPLING EXPERIMENT (V5.2)",
        "",
        "Effect = 100 x (RSSI_ONLY error - COMMON_FULL error) / RSSI_ONLY error, per seed.",
        "Positive = COMMON better. Outdoor and indoor conditions give equal weight to each scenario within",
        "the condition. Two-sided Wilcoxon over the seeds, Holm across the 6 methods.",
        "",
        f"Sizes: {list(sizes)}; seeds: {len(seeds)}; outdoor scenarios: {list(datasets)}.",
        f"Cases read: {len(cases)}.",
        "",
        "SUMMARY", *v, "",
        "FILES",
        "  summary_by_condition.csv  mean effect, median, fraction of seeds with COMMON better, p and Holm p",
        "  summary_by_scenario.csv   the same per scenario (Antwerp and Bonn separately)",
        "  effects_by_seed.csv       paired values used in the tests",
        "  absolute_errors.csv       mean error (m) per scenario, size, split, representation, and method",
        "  table_subsample_by_condition.tex  LaTeX table by condition",
        "  figure_subsample.png      " + ("written" if has_fig else "NOT written (install matplotlib)"),
        "  (the per-scenario table used in the article is written by `python -m lorabench paper`)",
        "",
        "INTERPRETATION CAVEAT",
        "  Subsampling equalizes the number of samples, not the spatial density: 572 points spread over",
        "  a city still cover a much larger area than a building floor. The test separates sample size from",
        "  the environment, but not the scale or density of the environment.",
    ]
    (agg / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    for line in v:
        print(line)
    log(f"Aggregates saved to {agg}")


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--task", choices=["check", "run", "aggregate"], default="run")
    ap.add_argument("--datasets", nargs="*", default=OUTDOOR)
    ap.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS_20)
    ap.add_argument("--profile", choices=["quick", "desktop"], default="desktop")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cache-dir", default="cache_equal_footing_v5")
    ap.add_argument("--baseline-results", default="results_equal_footing_v5")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--fixed-subset", action="store_true",
                    help="a single subsample per size for every seed")
    ap.add_argument("--force", action="store_true")


def run(args) -> None:
    cache_root = Path(args.cache_dir); baseline = Path(args.baseline_results)
    out = Path(args.output_dir or ("results_equal_footing_v5_subsample" + ("_fixed" if args.fixed_subset else "")))
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds]; sizes = sorted(set(int(n) for n in args.sizes))

    # ---- checks
    for d in args.datasets:  # the indoor references are read from the baseline results, not from the cache
        if not (cache_root / d / "metadata.json").exists():
            raise FileNotFoundError(f"Missing cache: {cache_root / d}. Run `python -m lorabench prepare` first.")
    for d in args.datasets:
        n_total = len(FeatureStore(cache_root / d).y)
        bad = [n for n in sizes if n >= n_total]
        if bad:
            raise ValueError(f"{d} has {n_total} samples; invalid sizes: {bad}")
    jobs, done = plan(args.datasets, sizes, seeds, out)
    log(f"Output: {out.resolve()}")
    log(f"Subsample {'fixed per size' if args.fixed_subset else 'drawn per seed'}; "
        f"sizes {sizes}; {len(seeds)} seeds; profile {args.profile}; threads {args.threads}")
    log(f"Planned new cases: {len(jobs)} (complete: {done})")
    if args.task == "check":
        log("CHECK: no model was trained.")
        return
    if args.task == "run":
        (out / "system_metadata.json").write_text(json.dumps(base.system_metadata(args.threads), indent=2),
                                                  encoding="utf-8")
        run_cases(args.datasets, sizes, seeds, args.profile, args.threads, cache_root, out,
                  args.fixed_subset, args.force)
    aggregate(out, baseline, args.datasets, sizes, seeds)
    log("FINISHED SUBSAMPLE CONFOUND EXTENSION")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
