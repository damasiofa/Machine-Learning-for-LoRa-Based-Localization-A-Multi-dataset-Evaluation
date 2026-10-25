"""Tables and figures of the article, computed from the checkpoints (environment-class analysis).

Environment classes: Outdoor = Antwerp 2019 and Bonn; Indoor = Brno iAG, Brno iUG, and Bucharest iAG
(Bucharest Outdoor is excluded). Unless stated otherwise:
  * a paired effect is computed per scenario and seed;
  * the class value of a seed is the unweighted mean over the scenarios of the class;
  * the 20 seed values are tested with a two-sided Wilcoxon signed-rank test, with Holm correction
    within each family described below.

Outputs (written to --out, default paper_outputs/; numbering of the IEEE Access manuscript):
  table_errors.tex/.csv          Table 3  mean, median, P90 error per scenario, split, representation, method
  table_sample_size.tex/.csv     Table 4  COMMON effect in the full and subsampled outdoor datasets and indoors
  table_costs.tex/.csv           Table 5  offline/online costs, scenario-balanced, 3 significant figures
  table_overhead.tex/.csv        Table 6  relative cost overhead of COMMON with Wilcoxon-Holm markers
  fig_environment_effects.*      Fig. 4   COMMON vs RSSI-only by environment class (Holm over 6 methods)
  fig_spatial_gap.*              Fig. 5   increase in mean error from random split to spatial holdout
  fig_rssi_snr.*                 Fig. 6   adding RSSI to SNR-only and SNR to RSSI-only (Holm over 6 methods)
  fig_ablation.*                 Fig. 7   feature-group ablation (Holm over 6 methods per class, split, and group)
  fig_tradeoff.*                 Fig. 8   mean error vs. inference time, marker area ~ model size
  summary.txt                    numbers quoted in the text
With --check, every value is compared with the values published in the manuscript
(lorabench/reference_values.json) and the differences are listed in check_report.txt.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .core import METHODS, holm_adjust
from .data import PAPER_INDOOR, PAPER_OUTDOOR, PAPER_SCENARIOS, SCENARIO_LABELS, ENVIRONMENT_BY_SCENARIO

SPLITS = ["random", "spatial"]
ENVS = ["Outdoor", "Indoor"]
ABLATIONS = ["DROP_SNR", "DROP_ESP", "DROP_SNR_ESP", "DROP_RX_MASK", "DROP_AGGREGATES"]
ABLATION_LABELS = {"DROP_SNR": "SNR", "DROP_ESP": "ESP", "DROP_SNR_ESP": "SNR+ESP", "DROP_RX_MASK": "RX mask",
                   "DROP_AGGREGATES": "Aggregates"}
ISOLATION = {"ADD_RSSI_TO_SNR": ("SNR_ONLY", "RSSI_SNR", "Adding RSSI to SNR-only"),
             "ADD_SNR_TO_RSSI": ("RSSI_ONLY", "RSSI_SNR", "Adding SNR to RSSI-only")}
FIELDS = ["mean_error_m", "median_error_m", "p90_error_m", "inference_ms_per_sample", "model_size_mb",
          "final_fit_time_s", "search_time_s", "n_features", "n_train", "n_val", "n_test"]
COST_METRICS = ["final_fit_time_s", "search_time_s", "inference_ms_per_sample", "model_size_mb"]
REFERENCE = Path(__file__).resolve().parent / "reference_values.json"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- loading
def load_cases(root: Path, datasets: Sequence[str], variants: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Reads every complete checkpoint of the given result folders into one table."""
    rows = []
    root = Path(root)
    for d in datasets:
        base = root / d / "checkpoints"
        if not base.exists():
            continue
        for p in base.glob("*/seed_*/*/*.json"):
            variant = p.parent.name
            if variants and variant not in variants:
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("status") != "complete":
                continue
            rows.append({"scenario": d, "split": p.parents[2].name, "seed": int(rec["seed"]), "variant": variant,
                         "method": rec["method"], **{f: rec.get(f, np.nan) for f in FIELDS}})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["scenario", "split", "seed", "variant", "method"] + FIELDS)
    for f in FIELDS:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    return df


def subsample_folders(root: Path) -> Dict[str, tuple]:
    out = {}
    for p in sorted(Path(root).glob("*_n*")):
        if p.is_dir() and (p / "checkpoints").exists():
            scen, _, n = p.name.rpartition("_n")
            if n.isdigit():
                out[p.name] = (scen, int(n))
    return out


# --------------------------------------------------------------------------- statistics
def wilcoxon_p(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 2 or not np.any(x != 0):
        return np.nan
    try:
        return float(wilcoxon(x, alternative="two-sided", method="auto").pvalue)
    except Exception:
        return np.nan


def add_holm(df: pd.DataFrame, family: List[str]) -> pd.DataFrame:
    df = df.copy(); df["p_holm"] = np.nan
    if df.empty:
        df["significant"] = pd.Series(dtype=bool)
        return df
    for _, idx in df.groupby(family).groups.items():
        ix = [i for i in idx if np.isfinite(df.loc[i, "p_raw"])]
        if ix:
            df.loc[ix, "p_holm"] = holm_adjust(df.loc[ix, "p_raw"].to_numpy())
    df["significant"] = df["p_holm"] < 0.05
    return df


def class_seed_values(effects: pd.DataFrame, keys: List[str], value: str = "effect") -> pd.DataFrame:
    """Unweighted mean over the scenarios of each environment class, per seed."""
    e = effects.copy(); e["env"] = e["scenario"].map(ENVIRONMENT_BY_SCENARIO)
    e = e[e["env"].notna()]
    return e.groupby(["env"] + keys + ["seed"], as_index=False)[value].mean()


def tested(seed_vals: pd.DataFrame, keys: List[str], family: List[str], value: str = "effect") -> pd.DataFrame:
    rows = []
    for k, z in seed_vals.groupby(["env"] + keys):
        x = z[value].to_numpy(dtype=float)
        rows.append({**dict(zip(["env"] + keys, k if isinstance(k, tuple) else (k,))), "n_seeds": len(x),
                     "mean": float(np.mean(x)), "median": float(np.median(x)), "p_raw": wilcoxon_p(x)})
    return add_holm(pd.DataFrame(rows), family)


def paired(cases: pd.DataFrame, ref: str, cmp: str, ref_split: Optional[str] = None,
           cmp_split: Optional[str] = None, field: str = "mean_error_m") -> pd.DataFrame:
    """Paired table of (reference, comparison) errors per scenario, seed, method, and split."""
    a = cases[cases.variant == ref]
    b = cases[cases.variant == cmp]
    if ref_split is not None:
        a = a[a.split == ref_split]
    if cmp_split is not None:
        b = b[b.split == cmp_split]
    keys = ["scenario", "seed", "method"] + ([] if (ref_split or cmp_split) else ["split"])
    m = a[keys + [field]].merge(b[keys + [field]], on=keys, suffixes=("_ref", "_cmp"))
    return m


# --------------------------------------------------------------------------- analyses
def environment_effects(main: pd.DataFrame) -> pd.DataFrame:
    """Fig. 4: 100 (E_RSSI - E_COMMON) / E_RSSI; Holm across the six methods per class and split."""
    m = paired(main[main.split.isin(SPLITS)], "RSSI_ONLY", "COMMON_FULL")
    m["effect"] = 100 * (m.mean_error_m_ref - m.mean_error_m_cmp) / m.mean_error_m_ref
    return tested(class_seed_values(m, ["split", "method"]), ["split", "method"], ["env", "split"])


def spatial_gap(main: pd.DataFrame) -> pd.DataFrame:
    """Fig. 5: 100 (E_spatial - E_random) / E_random per scenario and seed, class mean, mean over seeds."""
    rows = []
    for v in ["RSSI_ONLY", "COMMON_FULL"]:
        sub = main[main.variant == v]
        r = sub[sub.split == "random"][["scenario", "seed", "method", "mean_error_m"]]
        s = sub[sub.split == "spatial"][["scenario", "seed", "method", "mean_error_m"]]
        m = r.merge(s, on=["scenario", "seed", "method"], suffixes=("_random", "_spatial"))
        m["effect"] = 100 * (m.mean_error_m_spatial - m.mean_error_m_random) / m.mean_error_m_random
        m["variant"] = v
        rows.append(m)
    if not rows:
        return pd.DataFrame()
    cs = class_seed_values(pd.concat(rows), ["variant", "method"])
    return cs.groupby(["env", "variant", "method"], as_index=False).effect.mean()


def isolation_effects(main: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Fig. 6: 100 (E_ref - E_cmp) / E_ref; Holm across the six methods per class, split, and comparison."""
    allc = pd.concat([main[(main.variant == "RSSI_ONLY") & main.split.isin(SPLITS)], ext], ignore_index=True)
    out = []
    for cid, (ref, cmp, _) in ISOLATION.items():
        m = paired(allc, ref, cmp)
        if m.empty:
            continue
        m["effect"] = 100 * (m.mean_error_m_ref - m.mean_error_m_cmp) / m.mean_error_m_ref
        m["comparison"] = cid
        out.append(m)
    if not out:
        return pd.DataFrame()
    cs = class_seed_values(pd.concat(out), ["comparison", "split", "method"])
    return tested(cs, ["comparison", "split", "method"], ["env", "comparison", "split"])


def ablation_effects(main: pd.DataFrame) -> pd.DataFrame:
    """Fig. 7: 100 (E_drop - E_COMMON) / E_COMMON; Holm across the six methods per class, split, and dropped group
    (this family reproduces every significance marker of the published figure)."""
    out = []
    for base in SPLITS:
        for v in ABLATIONS:
            m = paired(main, "COMMON_FULL", v, ref_split=base, cmp_split=f"ablation_{base}")
            if m.empty:
                continue
            m["effect"] = 100 * (m.mean_error_m_cmp - m.mean_error_m_ref) / m.mean_error_m_ref
            m["split"] = base; m["ablation"] = v
            out.append(m)
    if not out:
        return pd.DataFrame()
    cs = class_seed_values(pd.concat(out), ["split", "method", "ablation"])
    return tested(cs, ["split", "method", "ablation"], ["env", "split", "ablation"])


def sample_size_effects(main: pd.DataFrame, sub: pd.DataFrame, sizes_of: Dict[str, int]) -> pd.DataFrame:
    """Table 4: per-scenario COMMON effect; Holm across the six methods within each column and split."""
    frames = []
    full = main[main.split.isin(SPLITS)].copy(); full["size"] = "full"
    frames.append(full)
    if not sub.empty:
        s = sub.copy()
        s["size"] = s["scenario"].map(lambda f: str(sizes_of[f][1]))
        s["scenario"] = s["scenario"].map(lambda f: sizes_of[f][0])
        frames.append(s)
    allc = pd.concat(frames, ignore_index=True)
    a = allc[allc.variant == "RSSI_ONLY"]; b = allc[allc.variant == "COMMON_FULL"]
    keys = ["scenario", "size", "split", "seed", "method"]
    m = a[keys + ["mean_error_m"]].merge(b[keys + ["mean_error_m"]], on=keys, suffixes=("_ref", "_cmp"))
    m["effect"] = 100 * (m.mean_error_m_ref - m.mean_error_m_cmp) / m.mean_error_m_ref
    rows = []
    for (sc, sz, st, me), z in m.groupby(["scenario", "size", "split", "method"]):
        x = z.sort_values("seed").effect.to_numpy(dtype=float)
        rows.append({"scenario": sc, "size": sz, "split": st, "method": me, "n_seeds": len(x),
                     "mean": float(np.mean(x)), "p_raw": wilcoxon_p(x)})
    return add_holm(pd.DataFrame(rows), ["scenario", "size", "split"])


def error_table(main: pd.DataFrame) -> pd.DataFrame:
    """Table 3: mean over the seeds of the per-seed mean, median, and P90 errors."""
    sub = main[main.split.isin(SPLITS) & main.variant.isin(["RSSI_ONLY", "COMMON_FULL"])]
    g = sub.groupby(["scenario", "split", "variant", "method"])[["mean_error_m", "median_error_m", "p90_error_m"]]
    return g.mean().reset_index().assign(n_seeds=g.size().to_numpy())


def cost_table(main: pd.DataFrame) -> pd.DataFrame:
    """Table 5: median over seeds per scenario/split; mean over scenarios of the class; mean over splits."""
    sub = main[main.split.isin(SPLITS) & main.variant.isin(["RSSI_ONLY", "COMMON_FULL"])].copy()
    med = sub.groupby(["scenario", "split", "variant", "method"])[COST_METRICS].median().reset_index()
    med["env"] = med.scenario.map(ENVIRONMENT_BY_SCENARIO)
    med = med[med.env.notna()]
    by_env = med.groupby(["env", "split", "variant", "method"])[COST_METRICS].mean().reset_index()
    return by_env.groupby(["env", "variant", "method"])[COST_METRICS].mean().reset_index()


def overhead_table(main: pd.DataFrame) -> pd.DataFrame:
    """Table 6: per seed, mean over the scenarios of the class of the paired ratio COMMON/RSSI-only; median over
    seeds per split, averaged over the two splits; Wilcoxon on (ratio - 1) with Holm across the six methods
    within each class, split, and metric."""
    sub = main[main.split.isin(SPLITS)]
    rows = []
    for metric in COST_METRICS:
        m = paired(sub, "RSSI_ONLY", "COMMON_FULL", field=metric)
        m["effect"] = m[f"{metric}_cmp"] / m[f"{metric}_ref"]
        m = m[np.isfinite(m.effect)]
        cs = class_seed_values(m, ["split", "method"])
        for (env, st, me), z in cs.groupby(["env", "split", "method"]):
            x = z.effect.to_numpy(dtype=float)
            rows.append({"env": env, "split": st, "method": me, "metric": metric, "n_seeds": len(x),
                         "median_ratio": float(np.median(x)), "p_raw": wilcoxon_p(x - 1.0)})
    cols = ["env", "method", "metric", "overhead_pct", "mark", "sig_random", "sig_spatial"]
    if not rows:
        return pd.DataFrame(columns=cols)
    t = add_holm(pd.DataFrame(rows), ["env", "split", "metric"])
    out = []
    for (env, me, metric), z in t.groupby(["env", "method", "metric"]):
        sig = dict(zip(z.split, z.significant))
        ratio = float(z.median_ratio.mean())
        mark = "*" if all(sig.get(s, False) for s in SPLITS) else ("r" if sig.get("random") else ("s" if sig.get("spatial") else ""))
        out.append({"env": env, "method": me, "metric": metric, "overhead_pct": 100 * (ratio - 1), "mark": mark,
                    "sig_random": bool(sig.get("random", False)), "sig_spatial": bool(sig.get("spatial", False))})
    return pd.DataFrame(out, columns=cols)


def tradeoff_table(main: pd.DataFrame) -> pd.DataFrame:
    """Fig. 8 (random split): class mean of the scenario mean errors and of the per-scenario medians of
    inference time and model size."""
    sub = main[(main.split == "random") & main.variant.isin(["RSSI_ONLY", "COMMON_FULL"])]
    g = sub.groupby(["scenario", "variant", "method"])
    per = pd.concat([g.mean_error_m.mean(), g.inference_ms_per_sample.median(), g.model_size_mb.median()], axis=1).reset_index()
    per["env"] = per.scenario.map(ENVIRONMENT_BY_SCENARIO)
    per = per[per.env.notna()]
    return per.groupby(["env", "variant", "method"], as_index=False)[["mean_error_m", "inference_ms_per_sample", "model_size_mb"]].mean()


# --------------------------------------------------------------------------- LaTeX
def sig3(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    r = float(f"{x:.3g}")
    dec = max(0, 2 - math.floor(math.log10(abs(r))))
    return f"{r:.{dec}f}"


def latex_errors(t: pd.DataFrame) -> str:
    metrics = [("mean_error_m", "Mean"), ("median_error_m", "Median"), ("p90_error_m", "P90")]
    variants = ["RSSI_ONLY", "COMMON_FULL"]
    cols = [(st, k, v) for st in SPLITS for k, _ in metrics for v in variants]
    idx = t.set_index(["scenario", "split", "variant", "method"])
    best = {}
    for (sc, st, v), z in t.groupby(["scenario", "split", "variant"]):
        for k, _ in metrics:
            if z[k].notna().any():
                best[(sc, st, k, v)] = z.loc[z[k].idxmin(), "method"]
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Mean, median, and 90th-percentile (P90) localization errors for each machine-learning method and "
         r"retained scenario. Each statistic is computed on the test partition of every seed and then averaged over the "
         r"20 seeds. Errors are in meters, rounded to the meter for the outdoor scenarios. R: RSSI-only; C: COMMON. Bold "
         r"values indicate the lowest error within each scenario, split protocol, statistic, and representation.}",
         r"\label{tab:absolute_performance_all_methods}",
         r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
         r" & & \multicolumn{6}{c}{Random split} & \multicolumn{6}{c}{Spatial holdout} \\",
         r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}",
         r" & & \multicolumn{2}{c}{Mean} & \multicolumn{2}{c}{Median} & \multicolumn{2}{c}{P90}"
         r" & \multicolumn{2}{c}{Mean} & \multicolumn{2}{c}{Median} & \multicolumn{2}{c}{P90} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}",
         r"Scenario & Method & R & C & R & C & R & C & R & C & R & C & R & C \\", r"\midrule"]
    prev = None
    for i, sc in enumerate(PAPER_SCENARIOS):
        env = ENVIRONMENT_BY_SCENARIO[sc]
        if env != prev:
            if prev is not None:
                L.append(r"\midrule")
            L.append(r"\multicolumn{14}{l}{\textit{" + env + r" scenarios}} \\")
            prev = env
        elif i > 0:
            L.append(r"\addlinespace[2pt]")
        for j, m in enumerate(METHODS):
            cells = []
            for st, k, v in cols:
                try:
                    x = float(idx.loc[(sc, st, v, m), k])
                except KeyError:
                    x = float("nan")
                if not np.isfinite(x):
                    cells.append("--"); continue
                s = f"{x:.0f}" if env == "Outdoor" else f"{x:.2f}"
                cells.append(r"\textbf{" + s + "}" if best.get((sc, st, k, v)) == m else s)
            L.append((SCENARIO_LABELS[sc] if j == 0 else "") + f" & {m} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def sample_size_columns(t: pd.DataFrame, indoor_sizes: Dict[str, int]) -> List[tuple]:
    sizes = sorted({int(s) for s in t["size"] if s != "full"}, reverse=True)
    cols = [("Full outdoor", sc, "full") for sc in PAPER_OUTDOOR if ((t.scenario == sc) & (t["size"] == "full")).any()]
    for n in sizes:
        group = f"$n={n:,}$".replace(",", "{,}")
        cols += [(group, sc, str(n)) for sc in PAPER_OUTDOOR if ((t.scenario == sc) & (t["size"] == str(n))).any()]
        cols += [(group, sc, "full") for sc in PAPER_INDOOR if indoor_sizes.get(sc) == n]
    return cols


def latex_sample_size(t: pd.DataFrame, cols: List[tuple], dims: Dict[str, int], outdoor_n: Dict[str, int]) -> str:
    groups = []
    for g, _, _ in cols:
        if not groups or groups[-1][0] != g:
            groups.append([g, 0])
        groups[-1][1] += 1
    head1 = " & " + " & ".join(r"\multicolumn{%d}{c}{%s}" % (k, g) for g, k in groups) + r" \\"
    rules, c0 = [], 2
    for _, k in groups:
        rules.append(r"\cmidrule(lr){%d-%d}" % (c0, c0 + k - 1)); c0 += k
    short = {"antwerp2019": "Antwerp", "bonn": "Bonn", "brno_iAG_BUT": "Brno iAG", "brno_iUG_BUT": "Brno iUG",
             "bucharest_iAG_UPB": "Bucharest iAG"}
    head2 = " & " + " & ".join(short.get(sc, sc) for _, sc, _ in cols) + r" \\"
    dimrow = "COMMON dim. & " + " & ".join(str(dims.get(sc, "--")) for _, sc, _ in cols) + r" \\"
    idx = t.set_index(["scenario", "size", "split", "method"])
    present = [s for s in PAPER_OUTDOOR if s in outdoor_n]
    full_txt = " and ".join(f"{outdoor_n[s]:,} {'samples ' if i == 0 else ''}in {SCENARIO_LABELS[s]}"
                            for i, s in enumerate(present))
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Sample-size control: mean relative improvement (\%) of COMMON over RSSI-only in the full outdoor "
         r"datasets (" + full_txt + r") and when both are subsampled to the sizes of the indoor scenarios, shown "
         r"next to the indoor scenario of the same size. Positive values favor COMMON. $^{*}$Holm-adjusted $p<0.05$ across "
         r"the six methods within each column and split (two-sided Wilcoxon signed-rank test over 20 seeds).}",
         r"\label{tab:sample_size_control}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", head1, "".join(rules), head2, dimrow, r"\midrule"]
    for st, name in [("random", "Random split"), ("spatial", "Spatial holdout")]:
        if st == "spatial":
            L.append(r"\addlinespace")
        L.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(cols) + 1, name))
        for m in METHODS:
            cells = []
            for _, sc, sz in cols:
                try:
                    r = idx.loc[(sc, sz, st, m)]
                except KeyError:
                    cells.append("--"); continue
                v = f"{r['mean']:+.1f}".replace("-0.0", "+0.0")
                cells.append("$" + v + (r"^{*}" if bool(r["significant"]) else "") + "$")
            L.append(m + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def latex_costs(t: pd.DataFrame) -> str:
    cols = [("final_fit_time_s", "Outdoor"), ("search_time_s", "Outdoor"), ("final_fit_time_s", "Indoor"),
            ("search_time_s", "Indoor"), ("inference_ms_per_sample", "Outdoor"), ("model_size_mb", "Outdoor"),
            ("inference_ms_per_sample", "Indoor"), ("model_size_mb", "Indoor")]
    idx = t.set_index(["env", "variant", "method"])
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Offline and online computational costs for the RSSI-only and COMMON",
         r"representations. Values are scenario-balanced means of the condition-level",
         r"medians across 20 seeds. Random and spatial evaluations receive equal weight.",
         r"Values are rounded to three significant figures.}",
         r"\label{tab:computational_cost_env}", r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"& & \multicolumn{4}{c}{Offline} & \multicolumn{4}{c}{Online} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"& & \multicolumn{2}{c}{Outdoor} & \multicolumn{2}{c}{Indoor} & \multicolumn{2}{c}{Outdoor} & \multicolumn{2}{c}{Indoor} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Method & Representation &",
         r"\begin{tabular}[c]{@{}c@{}}Final fit\\(s)\end{tabular} &", r"\begin{tabular}[c]{@{}c@{}}Search\\(s)\end{tabular} &",
         r"\begin{tabular}[c]{@{}c@{}}Final fit\\(s)\end{tabular} &", r"\begin{tabular}[c]{@{}c@{}}Search\\(s)\end{tabular} &",
         r"\begin{tabular}[c]{@{}c@{}}Inference\\(ms/sample)\end{tabular} &", r"\begin{tabular}[c]{@{}c@{}}Model size\\(MB)\end{tabular} &",
         r"\begin{tabular}[c]{@{}c@{}}Inference\\(ms/sample)\end{tabular} &", r"\begin{tabular}[c]{@{}c@{}}Model size\\(MB)\end{tabular} \\",
         r"\midrule"]
    for i, m in enumerate(METHODS):
        if i:
            L.append(r"\addlinespace")
        for j, (v, lab) in enumerate([("RSSI_ONLY", "RSSI"), ("COMMON_FULL", "COMMON")]):
            cells = []
            for metric, env in cols:
                try:
                    cells.append(sig3(float(idx.loc[(env, v, m), metric])))
                except KeyError:
                    cells.append("--")
            L.append((m if j == 0 else "") + f" & {lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def latex_overhead(t: pd.DataFrame) -> str:
    cols = [("final_fit_time_s", "Outdoor"), ("search_time_s", "Outdoor"), ("final_fit_time_s", "Indoor"),
            ("search_time_s", "Indoor"), ("inference_ms_per_sample", "Outdoor"), ("model_size_mb", "Outdoor"),
            ("inference_ms_per_sample", "Indoor"), ("model_size_mb", "Indoor")]
    idx = t.set_index(["env", "method", "metric"])
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Relative computational overhead (\%) of COMMON over RSSI-only: the median over the 20 seeds of the "
         r"scenario-balanced paired ratio, averaged over the random and spatial protocols. Being a median of paired ratios, "
         r"it is not the ratio of the medians reported in Table~\ref{tab:computational_cost_env}. Significance from two-sided "
         r"paired Wilcoxon signed-rank tests with Holm correction across the six methods within each "
         r"environment--split--metric family: $^{*}$~both protocols and $^{r}$ indicates significance only in the random protocol.}",
         r"\label{tab:relative_computational_cost}", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"& \multicolumn{4}{c}{Offline computational cost} & \multicolumn{4}{c}{Online computational cost} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"& \multicolumn{2}{c}{Outdoor} & \multicolumn{2}{c}{Indoor} & \multicolumn{2}{c}{Outdoor} & \multicolumn{2}{c}{Indoor} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Method & Final fit & Search & Final fit & Search & Inference & Model size & Inference & Model size \\", r"\midrule"]
    for m in METHODS:
        cells = []
        for metric, env in cols:
            try:
                r = idx.loc[(env, m, metric)]
            except KeyError:
                cells.append("--"); continue
            mark = {"*": r"^{*}", "r": r"^{r}", "s": r"^{s}"}.get(r["mark"], "")
            cells.append(f"${r['overhead_pct']:+.1f}{mark}$")
        L.append(m + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- figures
def _heatmap(ax, values, sig, row_labels, col_labels, norm, cmap):
    import matplotlib
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    im = ax.imshow(values, cmap=cm, norm=norm, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                r, g, b, _ = cm(norm(v))
                ink = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "black"
                ax.text(j, i, f"{v:+.1f}{'*' if sig[i, j] else ''}", ha="center", va="center", fontsize=9, color=ink)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True); ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", length=0)
    return im


def _norm(values):
    from matplotlib.colors import TwoSlopeNorm
    v = np.asarray(values, dtype=float); v = v[np.isfinite(v)]
    lo, hi = (float(v.min()), float(v.max())) if len(v) else (-1.0, 1.0)
    lo = min(lo, -1e-6); hi = max(hi, 1e-6)
    return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)


def _save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")


def fig_environment_effects(t: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    conds = [("Outdoor", "random"), ("Indoor", "random"), ("Outdoor", "spatial"), ("Indoor", "spatial")]
    idx = t.set_index(["env", "split", "method"])
    V = np.full((6, 4), np.nan); S = np.zeros((6, 4), bool)
    for i, m in enumerate(METHODS):
        for j, (e, s) in enumerate(conds):
            if (e, s, m) in idx.index:
                V[i, j] = idx.loc[(e, s, m), "mean"]; S[i, j] = bool(idx.loc[(e, s, m), "significant"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = _heatmap(ax, V, S, METHODS, [f"{e} - {s}" for e, s in conds], _norm(V), "RdBu_r")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03).set_label("Relative improvement of COMMON over RSSI-only (%)")
    ax.set_title("Environment-stratified representation effect")
    _save(fig, out, "fig_environment_effects"); plt.close(fig)


def fig_spatial_gap(t: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(METHODS)); w = 0.36
    idx = t.set_index(["env", "variant", "method"])
    for ax, env in zip(axes, ENVS):
        for k, (v, lab, col) in enumerate([("RSSI_ONLY", "RSSI-only", "tab:blue"), ("COMMON_FULL", "COMMON", "tab:orange")]):
            vals = [idx.loc[(env, v, m), "effect"] if (env, v, m) in idx.index else np.nan for m in METHODS]
            bars = ax.bar(x + (k - 0.5) * w, vals, w, label=lab, color=col)
            for b, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{val:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=25, ha="right")
        ax.set_title(f"{env} scenarios")
    axes[0].set_ylabel("Mean error increase: spatial vs random (%)"); axes[0].legend(loc="upper left", frameon=False, ncol=2)
    fig.suptitle("Spatial generalization penalty by environment")
    fig.tight_layout(); _save(fig, out, "fig_spatial_gap"); plt.close(fig)


def fig_rssi_snr(t: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    conds = [("Outdoor", "random"), ("Indoor", "random"), ("Outdoor", "spatial"), ("Indoor", "spatial")]
    idx = t.set_index(["env", "comparison", "split", "method"])
    mats = {}
    for cid in ISOLATION:
        V = np.full((6, 4), np.nan); S = np.zeros((6, 4), bool)
        for i, m in enumerate(METHODS):
            for j, (e, s) in enumerate(conds):
                if (e, cid, s, m) in idx.index:
                    V[i, j] = idx.loc[(e, cid, s, m), "mean"]; S[i, j] = bool(idx.loc[(e, cid, s, m), "significant"])
        mats[cid] = (V, S)
    norm = _norm(np.concatenate([v.ravel() for v, _ in mats.values()]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, cid in zip(axes, ISOLATION):
        V, S = mats[cid]
        im = _heatmap(ax, V, S, METHODS, [f"{e}\n{s}" for e, s in conds], norm, "RdBu_r")
        ax.set_title(ISOLATION[cid][2])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label("Relative error reduction (%)")
    fig.suptitle("Incremental value of RSSI and SNR by environment")
    _save(fig, out, "fig_rssi_snr"); plt.close(fig)


def fig_ablation(t: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    idx = t.set_index(["env", "split", "method", "ablation"])
    panels = [("Outdoor", "random"), ("Indoor", "random"), ("Outdoor", "spatial"), ("Indoor", "spatial")]
    mats = []
    for e, s in panels:
        V = np.full((6, 5), np.nan); S = np.zeros((6, 5), bool)
        for i, m in enumerate(METHODS):
            for j, a in enumerate(ABLATIONS):
                if (e, s, m, a) in idx.index:
                    V[i, j] = idx.loc[(e, s, m, a), "mean"]; S[i, j] = bool(idx.loc[(e, s, m, a), "significant"])
        mats.append((V, S))
    norm = _norm(np.concatenate([v.ravel() for v, _ in mats]))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for ax, (e, s), (V, S) in zip(axes.ravel(), panels, mats):
        im = _heatmap(ax, V, S, METHODS, [ABLATION_LABELS[a] for a in ABLATIONS], norm, "RdBu_r")
        ax.set_title(f"{e} - {s}")
    fig.colorbar(im, ax=axes, shrink=0.6).set_label("Impact of dropping the group (%)")
    fig.suptitle("Feature-group ablation by environment (positive = dropping worsens accuracy)")
    _save(fig, out, "fig_ablation"); plt.close(fig)


def fig_tradeoff(t: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    t = t[np.isfinite(t.inference_ms_per_sample) & np.isfinite(t.model_size_mb) & (t.inference_ms_per_sample > 0)]
    if t.empty:
        log("Fig. 8 skipped: no inference-time or model-size values in the checkpoints")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    smax = max(float(t.model_size_mb.max()), 1e-9)
    area = lambda mb: 2500 * np.sqrt(np.maximum(mb, 0) / smax) + 10
    for ax, env in zip(axes, ENVS):
        z = t[t.env == env]
        for v, lab, mk, col in [("RSSI_ONLY", "RSSI-only", "o", "tab:blue"), ("COMMON_FULL", "COMMON", "s", "tab:orange")]:
            q = z[z.variant == v]
            ax.scatter(q.inference_ms_per_sample, q.mean_error_m, s=area(q.model_size_mb), marker=mk, color=col,
                       alpha=0.75, edgecolor="white", label=lab)
            for _, r in q.iterrows():
                # label just above the marker edge (marker area in points^2)
                off = float(np.sqrt(area(r.model_size_mb))) / 2 + 2
                ax.annotate(r["method"], (r.inference_ms_per_sample, r.mean_error_m), textcoords="offset points",
                            xytext=(0, off), ha="center", va="bottom", fontsize=8, annotation_clip=False)
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3); ax.margins(x=0.12, y=0.18)
        ax.set_xlabel("Median inference time (ms/sample, logarithmic scale)"); ax.set_title(env)
        leg = ax.legend(loc="upper left")
        for h in leg.legend_handles:
            h.set_sizes([60])
    axes[0].set_ylabel("Mean localization error (m)")
    # size key in its own invisible axes below the panels, so that large markers cannot cover the labels
    refs = [x for x in (0.05, 1, 10, 100) if x < 0.7 * smax] + [float(f"{smax:.2g}")]
    fig.tight_layout()
    key = fig.add_axes([0.2, -0.2, 0.6, 0.16])
    key.set_xlim(-0.5, len(refs) - 0.5); key.set_ylim(0, 1); key.axis("off")
    for i, x in enumerate(refs):
        key.scatter([i], [0.5], s=area(x), color="0.6", edgecolor="white", clip_on=False)
        key.annotate(f"{x:g} MB", (i, 0.5), textcoords="offset points", xytext=(np.sqrt(area(x)) / 2 + 4, 0),
                     ha="left", va="center", fontsize=9)
    key.set_title("Model size", fontsize=10)
    _save(fig, out, "fig_tradeoff"); plt.close(fig)


# --------------------------------------------------------------------------- check against the manuscript
def check(results: Dict[str, pd.DataFrame], out: Path) -> List[str]:
    if not REFERENCE.exists():
        return ["reference_values.json not found; nothing checked."]
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    lines = [f"Reference: {ref.get('source', '?')}", ""]

    def compare(name, pairs, decimals):
        if not pairs:
            lines.append(f"{name}: not computed (missing inputs)"); return
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        bad = [(k, a, b) for k, a, b in pairs if a is None or not np.isfinite(a) or abs(round(a, decimals) - b) > tol]
        mx = max((abs(a - b) for _, a, b in pairs if a is not None and np.isfinite(a)), default=float("nan"))
        lines.append(f"{name}: {len(pairs) - len(bad)}/{len(pairs)} values equal after rounding (max |diff| {mx:.3g})")
        for k, a, b in bad[:20]:
            lines.append(f"    {k}: computed {a if a is None else round(a, decimals + 2)}, published {b}")

    def sig_compare(name, pairs):
        if not pairs:
            return
        bad = [k for k, a, b in pairs if bool(a) != bool(b)]
        lines.append(f"{name} significance: {len(pairs) - len(bad)}/{len(pairs)} markers equal")
        for k in bad[:20]:
            lines.append(f"    {k}")

    t = results.get("environment_effects")
    if t is not None and not t.empty and "fig_environment_effects" in ref:
        idx = t.set_index(["env", "split", "method"]); pv, ps = [], []
        for key, (val, s) in ref["fig_environment_effects"].items():
            e, sp, m = key.split("|")
            got = idx.loc[(e, sp, m)] if (e, sp, m) in idx.index else None
            pv.append((key, None if got is None else float(got["mean"]), val)); ps.append((key, None if got is None else got["significant"], s))
        compare("Fig. 4 environment effects", pv, 1); sig_compare("Fig. 4", ps)
    t = results.get("spatial_gap")
    if t is not None and not t.empty and "fig_spatial_gap" in ref:
        idx = t.set_index(["env", "variant", "method"])
        pv = [(k, float(idx.loc[tuple(k.split("|")), "effect"]) if tuple(k.split("|")) in idx.index else None, v)
              for k, v in ref["fig_spatial_gap"].items()]
        compare("Fig. 5 spatial gap", pv, 0)
    t = results.get("isolation_effects")
    if t is not None and not t.empty and "fig_rssi_snr" in ref:
        idx = t.set_index(["env", "comparison", "split", "method"]); pv, ps = [], []
        for key, (val, s) in ref["fig_rssi_snr"].items():
            k = tuple(key.split("|"))
            got = idx.loc[k] if k in idx.index else None
            pv.append((key, None if got is None else float(got["mean"]), val)); ps.append((key, None if got is None else got["significant"], s))
        compare("Fig. 6 RSSI/SNR isolation", pv, 1); sig_compare("Fig. 6", ps)
    t = results.get("ablation_effects")
    if t is not None and not t.empty and "fig_ablation" in ref:
        idx = t.set_index(["env", "split", "method", "ablation"]); pv, ps = [], []
        for key, (val, s) in ref["fig_ablation"].items():
            k = tuple(key.split("|"))
            got = idx.loc[k] if k in idx.index else None
            pv.append((key, None if got is None else float(got["mean"]), val)); ps.append((key, None if got is None else got["significant"], s))
        compare("Fig. 7 ablation", pv, 1); sig_compare("Fig. 7", ps)
    t = results.get("error_table")
    if t is not None and not t.empty and "table_errors" in ref:
        idx = t.set_index(["scenario", "split", "variant", "method"])
        for stat, dec_out, dec_in in [("mean_error_m", 0, 2), ("median_error_m", 0, 2), ("p90_error_m", 0, 2)]:
            for env, dec in [("Outdoor", dec_out), ("Indoor", dec_in)]:
                pv = []
                for key, val in ref["table_errors"][stat].items():
                    sc, sp, v, m = key.split("|")
                    if ENVIRONMENT_BY_SCENARIO.get(sc) != env:
                        continue
                    pv.append((key, float(idx.loc[(sc, sp, v, m), stat]) if (sc, sp, v, m) in idx.index else None, val))
                compare(f"Table 3 {stat} ({env.lower()})", pv, dec)
    t = results.get("sample_size")
    if t is not None and not t.empty and "table_sample_size" in ref:
        idx = t.set_index(["scenario", "size", "split", "method"]); pv, ps = [], []
        for key, (val, s) in ref["table_sample_size"].items():
            k = tuple(key.split("|"))
            got = idx.loc[k] if k in idx.index else None
            pv.append((key, None if got is None else float(got["mean"]), val)); ps.append((key, None if got is None else got["significant"], s))
        compare("Table 4 sample-size control", pv, 1); sig_compare("Table 4", ps)
    t = results.get("cost_table")
    if t is not None and not t.empty and "table_costs" in ref:
        idx = t.set_index(["env", "variant", "method"]); bad = []; n = 0
        for key, val in ref["table_costs"].items():
            e, v, m, metric = key.split("|"); n += 1
            got = sig3(float(idx.loc[(e, v, m), metric])) if (e, v, m) in idx.index else None
            if got != val:
                bad.append(f"    {key}: computed {got}, published {val}")
        lines.append(f"Table 5 costs: {n - len(bad)}/{n} values equal at three significant figures"); lines += bad[:20]
    t = results.get("overhead_table")
    if t is not None and not t.empty and "table_overhead" in ref:
        idx = t.set_index(["env", "method", "metric"]); pv, ps = [], []
        for key, (val, mark) in ref["table_overhead"].items():
            k = tuple(key.split("|"))
            got = idx.loc[k] if k in idx.index else None
            pv.append((key, None if got is None else float(got["overhead_pct"]), val))
            ps.append((key, None if got is None else got["mark"], mark))
        compare("Table 6 overhead", pv, 1)
        bad = [k for k, a, b in ps if (a or "") != (b or "")]
        lines.append(f"Table 6 markers: {len(ps) - len(bad)}/{len(ps)} equal"); lines += [f"    {k}" for k in bad[:20]]
    (out / "check_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


# --------------------------------------------------------------------------- driver
def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--results", default="results_equal_footing_v5")
    ap.add_argument("--rssi-results", default="results_equal_footing_v5_rssi_impact")
    ap.add_argument("--subsample-results", default="results_equal_footing_v5_subsample")
    ap.add_argument("--cache-dir", default="cache_equal_footing_v5")
    ap.add_argument("--out", default="paper_outputs")
    ap.add_argument("--check", action="store_true", help="compare every value with the published manuscript")
    ap.add_argument("--no-figures", action="store_true")


def summary_lines(res: Dict[str, pd.DataFrame]) -> List[str]:
    L = []
    t = res.get("environment_effects")
    if t is not None and not t.empty:
        for env in ENVS:
            for st in SPLITS:
                z = t[(t.env == env) & (t.split == st)]
                if len(z):
                    L.append(f"COMMON vs RSSI-only, {env.lower()} {st}: {z['mean'].min():+.1f} to {z['mean'].max():+.1f}% "
                             f"(significant for {int(z.significant.sum())}/{len(z)} methods)")
    t = res.get("spatial_gap")
    if t is not None and not t.empty:
        L.append(f"Spatial-holdout increase in mean error: {t.effect.min():.1f}% to {t.effect.max():.1f}%")
    t = res.get("error_table")
    if t is not None and not t.empty:
        for env in ENVS:
            z = t[t.scenario.map(ENVIRONMENT_BY_SCENARIO) == env]
            wins = z.loc[z.groupby(["scenario", "split", "variant"]).mean_error_m.idxmin(), "method"].value_counts()
            L.append(f"Lowest mean error, {env.lower()} conditions: " + ", ".join(f"{m} {n}" for m, n in wins.items()))
    return L


def run(args) -> None:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log(f"Reading checkpoints from {args.results} ...")
    main_cases = load_cases(Path(args.results), PAPER_SCENARIOS)
    if main_cases.empty:
        raise SystemExit(f"No complete checkpoints found under {args.results}.")
    ext = load_cases(Path(args.rssi_results), PAPER_SCENARIOS, variants=["SNR_ONLY", "RSSI_SNR"])
    folders = subsample_folders(Path(args.subsample_results))
    sub = load_cases(Path(args.subsample_results), list(folders), variants=["RSSI_ONLY", "COMMON_FULL"])
    log(f"Cases: main {len(main_cases)}, RSSI/SNR extension {len(ext)}, subsample {len(sub)}")

    res: Dict[str, pd.DataFrame] = {}
    res["error_table"] = error_table(main_cases)
    res["error_table"].to_csv(out / "table_errors.csv", index=False)
    (out / "table_errors.tex").write_text(latex_errors(res["error_table"]), encoding="utf-8")

    res["environment_effects"] = environment_effects(main_cases)
    res["environment_effects"].to_csv(out / "fig_environment_effects.csv", index=False)
    res["spatial_gap"] = spatial_gap(main_cases)
    res["spatial_gap"].to_csv(out / "fig_spatial_gap.csv", index=False)

    if not ext.empty:
        res["isolation_effects"] = isolation_effects(main_cases, ext)
        res["isolation_effects"].to_csv(out / "fig_rssi_snr.csv", index=False)
    else:
        log(f"RSSI/SNR isolation skipped: no SNR_ONLY/RSSI_SNR checkpoints under {args.rssi_results}")
    if main_cases.split.str.startswith("ablation_").any():
        res["ablation_effects"] = ablation_effects(main_cases)
        res["ablation_effects"].to_csv(out / "fig_ablation.csv", index=False)
    else:
        log("Ablation figure skipped: no ablation checkpoints")

    sizes_n = main_cases[(main_cases.split == "random")].groupby("scenario")[["n_train", "n_val", "n_test"]].first().sum(axis=1)
    res["sample_size"] = sample_size_effects(main_cases, sub, folders)
    res["sample_size"].to_csv(out / "table_sample_size.csv", index=False)
    cols = sample_size_columns(res["sample_size"], {s: int(sizes_n[s]) for s in PAPER_INDOOR if s in sizes_n})
    dims = main_cases[(main_cases.variant == "COMMON_FULL") & (main_cases.split == "random")].groupby("scenario").n_features.first()
    (out / "table_sample_size.tex").write_text(
        latex_sample_size(res["sample_size"], cols, {k: int(v) for k, v in dims.items()},
                          {s: int(sizes_n[s]) for s in PAPER_OUTDOOR if s in sizes_n}), encoding="utf-8")

    res["cost_table"] = cost_table(main_cases)
    res["cost_table"].to_csv(out / "table_costs.csv", index=False)
    (out / "table_costs.tex").write_text(latex_costs(res["cost_table"]), encoding="utf-8")
    res["overhead_table"] = overhead_table(main_cases)
    res["overhead_table"].to_csv(out / "table_overhead.csv", index=False)
    (out / "table_overhead.tex").write_text(latex_overhead(res["overhead_table"]), encoding="utf-8")
    res["tradeoff"] = tradeoff_table(main_cases)
    res["tradeoff"].to_csv(out / "fig_tradeoff.csv", index=False)

    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        fig_environment_effects(res["environment_effects"], out)
        fig_spatial_gap(res["spatial_gap"], out)
        if "isolation_effects" in res:
            fig_rssi_snr(res["isolation_effects"], out)
        if "ablation_effects" in res:
            fig_ablation(res["ablation_effects"], out)
        fig_tradeoff(res["tradeoff"], out)

    lines = summary_lines(res)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        log(line)
    if args.check:
        log("")
        for line in check(res, out):
            log(line)
    log(f"Paper outputs written to {out.resolve()}")
