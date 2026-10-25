"""Computational-cost aggregation (former aggregate_computational_costs_v5.py): scenario medians,
paired COMMON/RSSI-only ratios, and the 1:1:1:1 deployment synthesis, plus ablation cost ratios.

Reads <output-dir>/<scenario>/aggregate/all_case_summaries.csv written by `run --task aggregate`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SCENARIO_TO_DEPLOYMENT = {
    "antwerp2019": "antwerp",
    "bonn": "bonn",
    "brno_iAG_BUT": "brno",
    "brno_iUG_BUT": "brno",
    "bucharest_iAG_UPB": "bucharest",
    "bucharest_o_UPB": "bucharest",
}

DEPLOYMENTS = ["antwerp", "bonn", "brno", "bucharest"]
METHODS = ["kNN", "WkNN", "RF", "XGBoost", "MLP", "ELM"]
MAIN_VARIANTS = ["RSSI_ONLY", "COMMON_FULL"]

COST_METRICS = [
    "inference_ms_per_sample",
    "end_to_end_inference_ms_per_sample",
    "model_size_mb",
    "final_fit_time_s",
    "search_time_s",
    "total_case_runtime_s",
]

ABLATION_COST_METRICS = [
    "inference_ms_per_sample",
    "end_to_end_inference_ms_per_sample",
    "model_size_mb",
    "final_fit_time_s",
    "total_case_runtime_s",
]


def q25(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    return float(np.percentile(x, 25)) if len(x) else np.nan


def q75(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    return float(np.percentile(x, 75)) if len(x) else np.nan


def safe_ratio(num, den):
    try:
        num = float(num)
        den = float(den)
    except Exception:
        return np.nan
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return np.nan
    return num / den


def collect_all_cases(output_dir: Path) -> pd.DataFrame:
    frames = []
    missing = []
    for scenario, deployment in SCENARIO_TO_DEPLOYMENT.items():
        p = output_dir / scenario / "aggregate" / "all_case_summaries.csv"
        if not p.exists():
            missing.append(str(p))
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        if "dataset_id" not in df.columns:
            df["dataset_id"] = scenario
        df["scenario_id"] = scenario
        df["deployment_id"] = deployment
        frames.append(df)
    if missing:
        print("WARNING: missing files:")
        for p in missing:
            print("  -", p)
    if not frames:
        raise SystemExit("No all_case_summaries.csv found. Run `python -m lorabench run --task aggregate` first.")
    return pd.concat(frames, ignore_index=True, sort=False)


def absolute_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in COST_METRICS if m in df.columns]
    keys = ["deployment_id", "scenario_id", "split_type", "variant", "method"]
    rows = []
    for key, z in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row["n_seeds"] = int(z["seed"].nunique()) if "seed" in z.columns else len(z)
        for m in metrics:
            vals = pd.to_numeric(z[m], errors="coerce").dropna()
            row[f"{m}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{m}_q25"] = q25(vals)
            row[f"{m}_q75"] = q75(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def main_ratios(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in COST_METRICS if m in df.columns]
    rows = []
    main = df[df["variant"].isin(MAIN_VARIANTS) & df["split_type"].isin(["random", "spatial"])].copy()
    for (dep, scenario, split_type, method, seed), z in main.groupby(
        ["deployment_id", "scenario_id", "split_type", "method", "seed"], dropna=False
    ):
        a = z[z.variant == "RSSI_ONLY"]
        b = z[z.variant == "COMMON_FULL"]
        if len(a) != 1 or len(b) != 1:
            continue
        ar, br = a.iloc[0], b.iloc[0]
        row = {
            "deployment_id": dep,
            "scenario_id": scenario,
            "split_type": split_type,
            "method": method,
            "seed": int(seed),
        }
        for m in metrics:
            r = safe_ratio(br.get(m), ar.get(m))
            row[f"{m}_ratio_common_over_rssi"] = r
            row[f"{m}_overhead_pct"] = 100.0 * (r - 1.0) if np.isfinite(r) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def deployment_seed_average(seed_ratios: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    if seed_ratios.empty:
        return pd.DataFrame()
    ratio_cols = [c for c in seed_ratios.columns if c.endswith("_ratio_common_over_rssi") or c.endswith("_overhead_pct") or c.endswith("_ratio_drop_over_full") or c.endswith("_change_pct")]
    group_cols = ["deployment_id", *id_cols, "seed"]
    out = seed_ratios.groupby(group_cols, as_index=False)[ratio_cols].mean()
    counts = seed_ratios.groupby(group_cols, as_index=False)["scenario_id"].nunique().rename(columns={"scenario_id": "n_scenarios"})
    return counts.merge(out, on=group_cols, how="left")


def summarize_across_seeds(depseed: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    if depseed.empty:
        return pd.DataFrame()
    metric_cols = [c for c in depseed.columns if c.endswith("_ratio_common_over_rssi") or c.endswith("_overhead_pct") or c.endswith("_ratio_drop_over_full") or c.endswith("_change_pct")]
    group_cols = ["deployment_id", *id_cols]
    rows = []
    for key, z in depseed.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = int(z.seed.nunique())
        row["n_scenarios"] = int(z.n_scenarios.max()) if "n_scenarios" in z.columns else np.nan
        for c in metric_cols:
            vals = pd.to_numeric(z[c], errors="coerce").dropna()
            row[f"{c}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{c}_q25"] = q25(vals)
            row[f"{c}_q75"] = q75(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def cross_deployment_seed(depseed: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    if depseed.empty:
        return pd.DataFrame()
    metric_cols = [c for c in depseed.columns if c.endswith("_ratio_common_over_rssi") or c.endswith("_overhead_pct") or c.endswith("_ratio_drop_over_full") or c.endswith("_change_pct")]
    group_cols = [*id_cols, "seed"]
    out = depseed.groupby(group_cols, as_index=False)[metric_cols].mean()
    counts = depseed.groupby(group_cols, as_index=False)["deployment_id"].nunique().rename(columns={"deployment_id": "n_deployments"})
    return counts.merge(out, on=group_cols, how="left")


def cross_summary(crossseed: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    if crossseed.empty:
        return pd.DataFrame()
    metric_cols = [c for c in crossseed.columns if c.endswith("_ratio_common_over_rssi") or c.endswith("_overhead_pct") or c.endswith("_ratio_drop_over_full") or c.endswith("_change_pct")]
    rows = []
    for key, z in crossseed.groupby(id_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(id_cols, key_tuple))
        row["n_seeds"] = int(z.seed.nunique())
        row["n_deployments"] = int(z.n_deployments.min()) if "n_deployments" in z.columns else np.nan
        for c in metric_cols:
            vals = pd.to_numeric(z[c], errors="coerce").dropna()
            row[f"{c}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{c}_q25"] = q25(vals)
            row[f"{c}_q75"] = q75(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_ratios(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in ABLATION_COST_METRICS if m in df.columns]
    rows = []
    for base in ["random", "spatial"]:
        anchors = df[(df.split_type == base) & (df.variant == "COMMON_FULL")].copy()
        drops = df[df.split_type == f"ablation_{base}"].copy()
        if anchors.empty or drops.empty:
            continue
        anchor_idx = anchors.set_index(["scenario_id", "method", "seed"])
        for _, dr in drops.iterrows():
            k = (dr.scenario_id, dr.method, dr.seed)
            if k not in anchor_idx.index:
                continue
            ar = anchor_idx.loc[k]
            if isinstance(ar, pd.DataFrame):
                ar = ar.iloc[0]
            row = {
                "deployment_id": dr.deployment_id,
                "scenario_id": dr.scenario_id,
                "base_split_type": base,
                "method": dr.method,
                "variant": dr.variant,
                "seed": int(dr.seed),
            }
            for m in metrics:
                r = safe_ratio(dr.get(m), ar.get(m))
                row[f"{m}_ratio_drop_over_full"] = r
                row[f"{m}_change_pct"] = 100.0 * (r - 1.0) if np.isfinite(r) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--output-dir", default="results_equal_footing_v5")


def main():
    ap = argparse.ArgumentParser(description="Aggregates the computational costs of the V5 equal-footing benchmark.")
    add_arguments(ap)
    run(ap.parse_args())


def run(args):
    output = Path(args.output_dir)
    target = output / "aggregate_equal_weight" / "computational"
    target.mkdir(parents=True, exist_ok=True)

    df = collect_all_cases(output)

    abs_summary = absolute_summary(df)
    abs_summary.to_csv(target / "scenario_computational_summary_across_seeds.csv", index=False)

    sr = main_ratios(df)
    sr.to_csv(target / "scenario_seed_main_computational_ratios.csv", index=False)

    ds = deployment_seed_average(sr, ["split_type", "method"])
    ds.to_csv(target / "deployment_seed_main_computational_ratios.csv", index=False)

    dsum = summarize_across_seeds(ds, ["split_type", "method"])
    dsum.to_csv(target / "deployment_main_computational_summary.csv", index=False)

    cs = cross_deployment_seed(ds, ["split_type", "method"])
    cs.to_csv(target / "cross_deployment_seed_main_computational_ratios.csv", index=False)

    csum = cross_summary(cs, ["split_type", "method"])
    csum.to_csv(target / "cross_deployment_main_computational_summary.csv", index=False)

    ab = ablation_ratios(df)
    ab.to_csv(target / "scenario_seed_ablation_computational_ratios.csv", index=False)

    abds = deployment_seed_average(ab, ["base_split_type", "method", "variant"])
    abds.to_csv(target / "deployment_seed_ablation_computational_ratios.csv", index=False)

    absum = summarize_across_seeds(abds, ["base_split_type", "method", "variant"])
    absum.to_csv(target / "deployment_ablation_computational_summary.csv", index=False)

    abcs = cross_deployment_seed(abds, ["base_split_type", "method", "variant"])
    abcs.to_csv(target / "cross_deployment_seed_ablation_computational_ratios.csv", index=False)

    abcsum = cross_summary(abcs, ["base_split_type", "method", "variant"])
    abcsum.to_csv(target / "cross_deployment_ablation_computational_summary.csv", index=False)

    readme = f"""Computational-cost aggregation — V5 Equal Footing

Source: per-scenario aggregate/all_case_summaries.csv files generated by `run --task aggregate`.

Cross-deployment synthesis uses dimensionless paired ratios. Brno and Bucharest subscenarios are averaged within deployment for the same seed; Antwerp, Bonn, Brno and Bucharest then receive equal 1:1:1:1 weight.

For main comparisons, ratio = COMMON_FULL / RSSI_ONLY. A ratio > 1 means COMMON_FULL is more computationally expensive for time/storage metrics. overhead_pct = 100*(ratio-1).

For ablation, ratio = DROP / COMMON_FULL. A ratio < 1 means the ablated representation costs less than COMMON_FULL for that metric.

Absolute raw times are summarized only at scenario level because dataset sizes differ. Cross-deployment conclusions use paired ratios rather than pooling raw seconds or model sizes.

Cases loaded: {len(df)}
Seeds found: {sorted(pd.to_numeric(df['seed'], errors='coerce').dropna().astype(int).unique().tolist()) if 'seed' in df.columns else []}
"""
    (target / "README_COMPUTATIONAL.txt").write_text(readme, encoding="utf-8")

    print(f"OK: computational aggregation saved to: {target}")
    print(f"Cases loaded: {len(df)}")
    print("Main files:")
    print(" - scenario_computational_summary_across_seeds.csv")
    print(" - deployment_main_computational_summary.csv")
    print(" - cross_deployment_main_computational_summary.csv")
    print(" - deployment_ablation_computational_summary.csv")
    print(" - cross_deployment_ablation_computational_summary.csv")


if __name__ == "__main__":
    main()
