"""RSSI/SNR isolation (V5.1): trains SNR_ONLY and RSSI_SNR on the exact V5 splits and compares them
with the existing RSSI_ONLY results, without rerunning any V5 case.

Unchanged from run_rssi_impact_extension_v5.py except for the package imports, the protocol version
passed explicitly to core.run_tuned_case, a system_metadata.json with the thread pools of the run, and
run(args) in place of main().
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

# Reuse exactly the same model implementations, hyperparameter grids, preprocessing,
# error metrics, and checkpoint writer from the audited V5 benchmark.
from . import core as base
from .data import (
    FeatureStore,
    PRIMARY_SCENARIOS,
    expand_dataset_tokens,
    deployment_for_scenario,
)

EXT_PROTOCOL_VERSION = "MULTIDATASET_EQUAL_FOOTING_RSSI_IMPACT_EXTENSION_V5_1"
NEW_VARIANTS = ["SNR_ONLY", "RSSI_SNR"]
SPLITS = ["random", "spatial"]
METHODS = list(base.METHODS)
SEEDS_20 = list(base.SEEDS_20)

# The physical source count is three, even though the third source contains two cities.
SOURCE_DATASET_BY_SCENARIO = {
    "antwerp2019": "antwerp2019_dataset",
    "bonn": "bonn_dataset",
    "brno_iAG_BUT": "brno_bucharest_dataset",
    "brno_iUG_BUT": "brno_bucharest_dataset",
    "bucharest_iAG_UPB": "brno_bucharest_dataset",
    "bucharest_o_UPB": "brno_bucharest_dataset",
}
SOURCE_DATASET_LABELS = {
    "antwerp2019_dataset": "Antwerp 2019",
    "bonn_dataset": "Bonn",
    "brno_bucharest_dataset": "Brno-Bucharest",
}

COMPARISONS = {
    # Which standalone link measurement is better?
    "SNR_ONLY_VS_RSSI_ONLY": {
        "reference": "RSSI_ONLY",
        "comparison": "SNR_ONLY",
        "positive_meaning": "SNR_ONLY has lower error than RSSI_ONLY",
    },
    # Direct incremental RSSI effect requested by the user.
    "ADD_RSSI_TO_SNR": {
        "reference": "SNR_ONLY",
        "comparison": "RSSI_SNR",
        "positive_meaning": "adding RSSI to SNR lowers localization error",
    },
    # Symmetric complement: what does SNR add when RSSI is already present?
    "ADD_SNR_TO_RSSI": {
        "reference": "RSSI_ONLY",
        "comparison": "RSSI_SNR",
        "positive_meaning": "adding SNR to RSSI lowers localization error",
    },
}


class ImpactFeatureStore(FeatureStore):
    """FeatureStore extended only with the two new representations needed here."""

    def groups_for_variant(self, variant: str) -> List[str]:
        if variant == "SNR_ONLY":
            return ["snr"]
        if variant == "RSSI_SNR":
            return ["rssi", "snr"]
        return super().groups_for_variant(variant)


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def checkpoint_path(root: Path, dataset: str, split_type: str, seed: int, variant: str, method: str) -> Path:
    return root / dataset / "checkpoints" / split_type / f"seed_{seed}" / variant / f"{method}.json"


def load_complete_checkpoint(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return d if d.get("status") == "complete" else None


def load_split(baseline_results: Path, dataset: str, split_type: str, seed: int) -> Dict[str, np.ndarray]:
    p = baseline_results / dataset / "splits" / split_type / f"seed_{seed}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing audited V5 split: {p}")
    with np.load(p, allow_pickle=False) as z:
        out = {k: np.asarray(z[k]) for k in z.files}
    for k in ("train", "val", "test"):
        if k not in out:
            raise RuntimeError(f"Split {p} does not contain {k!r}")
    return out


def validate_existing_v5(
    datasets: Sequence[str],
    seeds: Sequence[int],
    cache_root: Path,
    baseline_results: Path,
) -> None:
    """Fail instead of silently rerunning any old V5 experiment."""
    missing: List[str] = []
    for d in datasets:
        cdir = cache_root / d
        if not (cdir / "metadata.json").exists():
            missing.append(f"cache:{cdir / 'metadata.json'}")
            continue
        for st in SPLITS:
            for seed in seeds:
                sp = baseline_results / d / "splits" / st / f"seed_{seed}.npz"
                if not sp.exists():
                    missing.append(f"split:{sp}")
                for m in METHODS:
                    cp = checkpoint_path(baseline_results, d, st, seed, "RSSI_ONLY", m)
                    if load_complete_checkpoint(cp) is None:
                        missing.append(f"baseline:{cp}")
    if missing:
        preview = "\n".join(f"  - {x}" for x in missing[:30])
        extra = "" if len(missing) <= 30 else f"\n  ... and {len(missing)-30} more"
        raise RuntimeError(
            "The incremental RSSI-impact extension requires the completed V5 baseline.\n"
            "No old experiment will be rerun automatically. Missing items:\n"
            f"{preview}{extra}\n"
            "Restore/copy the existing cache_equal_footing_v5 and results_equal_footing_v5 folders first."
        )
    log("Baseline check passed: existing RSSI_ONLY results and exact random/spatial splits are available for every requested seed.")


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def rank_biserial(values: Sequence[float]) -> float:
    d = np.asarray(values, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    pos = float(r[d > 0].sum())
    neg = float(r[d < 0].sum())
    den = pos + neg
    return (pos - neg) / den if den else 0.0


def bootstrap_ci(values: Sequence[float], nboot: int = 5000, seed: int = 20260913) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(nboot, dtype=float)
    for i in range(nboot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _case_for_variant(
    baseline_results: Path,
    extension_results: Path,
    dataset: str,
    split_type: str,
    seed: int,
    variant: str,
    method: str,
) -> Dict[str, Any]:
    root = baseline_results if variant == "RSSI_ONLY" else extension_results
    p = checkpoint_path(root, dataset, split_type, seed, variant, method)
    d = load_complete_checkpoint(p)
    if d is None:
        raise RuntimeError(f"Missing completed case: {p}")
    return d


def build_scenario_seed_effects(
    datasets: Sequence[str],
    seeds: Sequence[int],
    baseline_results: Path,
    extension_results: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for d in datasets:
        deployment = deployment_for_scenario(d)
        source_dataset = SOURCE_DATASET_BY_SCENARIO[d]
        for st in SPLITS:
            for m in METHODS:
                for seed in seeds:
                    cases = {
                        v: _case_for_variant(baseline_results, extension_results, d, st, seed, v, m)
                        for v in ["RSSI_ONLY", "SNR_ONLY", "RSSI_SNR"]
                    }
                    errors = {v: float(cases[v]["mean_error_m"]) for v in cases}
                    for comparison_id, spec in COMPARISONS.items():
                        ref_v = spec["reference"]
                        cmp_v = spec["comparison"]
                        ref_e = errors[ref_v]
                        cmp_e = errors[cmp_v]
                        effect_m = ref_e - cmp_e
                        effect_pct = 100.0 * effect_m / ref_e if ref_e != 0 else np.nan
                        rows.append({
                            "source_dataset_id": source_dataset,
                            "source_dataset_label": SOURCE_DATASET_LABELS[source_dataset],
                            "deployment_id": deployment,
                            "scenario_id": d,
                            "split_type": st,
                            "comparison_id": comparison_id,
                            "reference_variant": ref_v,
                            "comparison_variant": cmp_v,
                            "positive_effect_means": spec["positive_meaning"],
                            "method": m,
                            "seed": int(seed),
                            "reference_error_m": ref_e,
                            "comparison_error_m": cmp_e,
                            "effect_reference_minus_comparison_m": effect_m,
                            "relative_effect_pct": effect_pct,
                        })
    return pd.DataFrame(rows)


def aggregate_seed_units(scenario_effects: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    keys = [unit_col, "split_type", "comparison_id", "method", "seed"]
    out = (
        scenario_effects.groupby(keys, as_index=False)
        .agg(
            relative_effect_pct=("relative_effect_pct", "mean"),
            n_scenarios=("scenario_id", "nunique"),
        )
    )
    return out


def unit_statistics(unit_seed: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = [unit_col, "split_type", "comparison_id", "method"]
    for key, z in unit_seed.groupby(group_cols, sort=True):
        unit_id, split_type, comparison_id, method = key
        d = z["relative_effect_pct"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if len(d) >= 2 and np.any(d != 0):
            try:
                p = float(wilcoxon(d, alternative="two-sided", method="auto").pvalue)
            except Exception:
                p = np.nan
        else:
            p = np.nan
        lo, hi = bootstrap_ci(d)
        rows.append({
            unit_col: unit_id,
            "split_type": split_type,
            "comparison_id": comparison_id,
            "method": method,
            "n_paired_seeds": int(len(d)),
            "n_scenarios": int(z["n_scenarios"].max()) if len(z) else 0,
            "mean_relative_effect_pct": float(np.mean(d)) if len(d) else np.nan,
            "median_relative_effect_pct": float(np.median(d)) if len(d) else np.nan,
            "relative_ci95_low_pct": lo,
            "relative_ci95_high_pct": hi,
            "positive_effect_fraction": float(np.mean(d > 0)) if len(d) else np.nan,
            "rank_biserial": rank_biserial(d),
            "wilcoxon_p_raw": p,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["wilcoxon_p_holm_6methods"] = 1.0
    # Separate multiplicity family for each unit x split x scientific comparison.
    for (unit_id, st, comp), idx in out.groupby([unit_col, "split_type", "comparison_id"]).groups.items():
        ix = list(idx)
        valid = [i for i in ix if np.isfinite(out.loc[i, "wilcoxon_p_raw"])]
        if valid:
            out.loc[valid, "wilcoxon_p_holm_6methods"] = holm_adjust(out.loc[valid, "wilcoxon_p_raw"].to_numpy())
    out["significant_holm_0_05"] = out["wilcoxon_p_holm_6methods"] < 0.05
    out["significant_comparison_better"] = out["significant_holm_0_05"] & (out["mean_relative_effect_pct"] > 0)
    out["significant_reference_better"] = out["significant_holm_0_05"] & (out["mean_relative_effect_pct"] < 0)
    return out


def cross_unit_summary(unit_stats: pd.DataFrame, unit_col: str, expected_units: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (st, comp, m), z in unit_stats.groupby(["split_type", "comparison_id", "method"], sort=True):
        vals = z["mean_relative_effect_pct"].to_numpy(dtype=float)
        rows.append({
            "split_type": st,
            "comparison_id": comp,
            "method": m,
            "n_units": int(z[unit_col].nunique()),
            "expected_units": expected_units,
            "mean_equal_weight_relative_effect_pct": float(np.mean(vals)),
            "median_unit_relative_effect_pct": float(np.median(vals)),
            "units_with_positive_mean_effect": int(np.sum(vals > 0)),
            "units_significant_comparison_better": int(z["significant_comparison_better"].sum()),
            "units_significant_reference_better": int(z["significant_reference_better"].sum()),
            "units_not_significant": int((~z["significant_holm_0_05"]).sum()),
        })
    return pd.DataFrame(rows)


def build_computational_effects(
    datasets: Sequence[str],
    seeds: Sequence[int],
    baseline_results: Path,
    extension_results: Path,
) -> pd.DataFrame:
    metrics = [
        "end_to_end_inference_ms_per_sample",
        "model_size_mb",
        "final_fit_time_s",
        "search_time_s",
        "total_case_runtime_s",
    ]
    rows: List[Dict[str, Any]] = []
    for d in datasets:
        deployment = deployment_for_scenario(d)
        source_dataset = SOURCE_DATASET_BY_SCENARIO[d]
        for st in SPLITS:
            for m in METHODS:
                for seed in seeds:
                    cases = {
                        v: _case_for_variant(baseline_results, extension_results, d, st, seed, v, m)
                        for v in ["RSSI_ONLY", "SNR_ONLY", "RSSI_SNR"]
                    }
                    for comparison_id, spec in COMPARISONS.items():
                        ref = cases[spec["reference"]]
                        cmp = cases[spec["comparison"]]
                        row: Dict[str, Any] = {
                            "source_dataset_id": source_dataset,
                            "deployment_id": deployment,
                            "scenario_id": d,
                            "split_type": st,
                            "comparison_id": comparison_id,
                            "method": m,
                            "seed": int(seed),
                            "reference_variant": spec["reference"],
                            "comparison_variant": spec["comparison"],
                        }
                        for metric in metrics:
                            a = float(ref.get(metric, np.nan))
                            b = float(cmp.get(metric, np.nan))
                            ratio = b / a if np.isfinite(a) and np.isfinite(b) and a != 0 else np.nan
                            row[f"{metric}_ratio_comparison_over_reference"] = ratio
                            row[f"{metric}_overhead_pct"] = 100.0 * (ratio - 1.0) if np.isfinite(ratio) else np.nan
                        rows.append(row)
    return pd.DataFrame(rows)


def computational_unit_summary(comp: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    ratio_cols = [c for c in comp.columns if c.endswith("_ratio_comparison_over_reference")]
    # First average scenario ratios within each unit and seed, preserving equal scenario weight.
    unit_seed = comp.groupby([unit_col, "split_type", "comparison_id", "method", "seed"], as_index=False)[ratio_cols].mean()
    rows: List[Dict[str, Any]] = []
    for key, z in unit_seed.groupby([unit_col, "split_type", "comparison_id", "method"], sort=True):
        unit_id, st, comp_id, method = key
        row: Dict[str, Any] = {unit_col: unit_id, "split_type": st, "comparison_id": comp_id, "method": method, "n_seeds": int(z.seed.nunique())}
        for col in ratio_cols:
            arr = z[col].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            stem = col.removesuffix("_ratio_comparison_over_reference")
            med = float(np.median(arr)) if len(arr) else np.nan
            row[f"{stem}_ratio_median"] = med
            row[f"{stem}_overhead_pct_median"] = 100.0 * (med - 1.0) if np.isfinite(med) else np.nan
            row[f"{stem}_ratio_q25"] = float(np.percentile(arr, 25)) if len(arr) else np.nan
            row[f"{stem}_ratio_q75"] = float(np.percentile(arr, 75)) if len(arr) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_aggregates(
    datasets: Sequence[str],
    seeds: Sequence[int],
    baseline_results: Path,
    extension_results: Path,
) -> None:
    agg = extension_results / "aggregate_rssi_impact"
    agg.mkdir(parents=True, exist_ok=True)

    scenario = build_scenario_seed_effects(datasets, seeds, baseline_results, extension_results)
    scenario.to_csv(agg / "scenario_seed_rssi_impact_relative_effects.csv", index=False)

    deployment_seed = aggregate_seed_units(scenario, "deployment_id")
    deployment_stats = unit_statistics(deployment_seed, "deployment_id")
    deployment_cross = cross_unit_summary(deployment_stats, "deployment_id", expected_units=4)
    deployment_seed.to_csv(agg / "deployment_seed_rssi_impact_relative_effects.csv", index=False)
    deployment_stats.to_csv(agg / "deployment_rssi_impact_statistical_comparison.csv", index=False)
    deployment_cross.to_csv(agg / "cross_deployment_rssi_impact_summary.csv", index=False)

    source_seed = aggregate_seed_units(scenario, "source_dataset_id")
    source_stats = unit_statistics(source_seed, "source_dataset_id")
    source_cross = cross_unit_summary(source_stats, "source_dataset_id", expected_units=3)
    source_seed.to_csv(agg / "source_dataset_seed_rssi_impact_relative_effects.csv", index=False)
    source_stats.to_csv(agg / "source_dataset_rssi_impact_statistical_comparison.csv", index=False)
    source_cross.to_csv(agg / "cross_source_dataset_rssi_impact_summary.csv", index=False)

    comp = build_computational_effects(datasets, seeds, baseline_results, extension_results)
    comp.to_csv(agg / "scenario_seed_rssi_impact_computational_ratios.csv", index=False)
    computational_unit_summary(comp, "deployment_id").to_csv(agg / "deployment_rssi_impact_computational_summary.csv", index=False)
    computational_unit_summary(comp, "source_dataset_id").to_csv(agg / "source_dataset_rssi_impact_computational_summary.csv", index=False)

    metadata = {
        "protocol_version": EXT_PROTOCOL_VERSION,
        "baseline_protocol": "MULTIDATASET_EQUAL_FOOTING_20SEEDS_V5",
        "purpose": "Incremental feature-isolation study focused on RSSI impact without rerunning completed V5 experiments.",
        "new_variants_tuned": NEW_VARIANTS,
        "reused_variant": "RSSI_ONLY",
        "reused_splits": ["random", "spatial"],
        "comparisons": COMPARISONS,
        "primary_source_dataset_weighting": "1:1:1 for Antwerp 2019, Bonn, and Brno-Bucharest",
        "deployment_sensitivity_weighting": "1:1:1:1 for Antwerp, Bonn, Brno, and Bucharest",
        "important_note": "ESP is intentionally excluded from the RSSI-removal comparison because ESP is mathematically derived from RSSI and SNR; including it would not constitute an RSSI-free representation.",
        "statistical_unit": "seed",
        "paired_seeds": list(map(int, seeds)),
        "test": "two-sided Wilcoxon signed-rank; Holm correction across the six methods separately for each comparison, split, and unit",
    }
    (agg / "protocol_metadata_rssi_impact.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""RSSI-impact extension for V5 Equal Footing

This extension DOES NOT rerun the completed V5 CORE, spatial holdout, or ablation experiments.
It reuses the exact existing random/spatial split files and the existing RSSI_ONLY test results.
Only two new validation-tuned representations are trained:
  - SNR_ONLY = per-gateway SNR only
  - RSSI_SNR = per-gateway RSSI + per-gateway SNR

Three paired questions are answered:
  1. SNR_ONLY_VS_RSSI_ONLY: standalone SNR versus standalone RSSI.
  2. ADD_RSSI_TO_SNR: SNR_ONLY versus RSSI_SNR. Positive effect = RSSI adds value when SNR is already present.
  3. ADD_SNR_TO_RSSI: RSSI_ONLY versus RSSI_SNR. Positive effect = SNR adds value when RSSI is already present.

Why ESP is not used as a no-RSSI control:
ESP = f(RSSI,SNR), so ESP carries RSSI information by construction. Using ESP in a purported no-RSSI representation would contaminate the test.

Primary synthesis uses three equal-weight SOURCE DATASETS:
  Antwerp 2019 : Bonn : Brno-Bucharest = 1:1:1.
The four-deployment 1:1:1:1 analysis is also saved as a sensitivity analysis.

Requested seeds: {list(map(int, seeds))}
New tuned cases expected: {len(datasets) * len(seeds) * len(SPLITS) * len(NEW_VARIANTS) * len(METHODS)}
No old case is automatically regenerated.
"""
    (agg / "README_RSSI_IMPACT.txt").write_text(readme, encoding="utf-8")
    log(f"RSSI-impact aggregation written to {agg}")


def run_new_cases(
    datasets: Sequence[str],
    seeds: Sequence[int],
    profile: str,
    threads: int,
    cache_root: Path,
    baseline_results: Path,
    extension_results: Path,
    force: bool,
) -> None:
    # New checkpoints are tagged with the extension protocol version.
    inference_reps = 2 if profile == "quick" else 5

    for d in datasets:
        store = ImpactFeatureStore(cache_root / d)
        log(f"DATASET {d}: n={len(store.y)} metric={store.meta['target_metric']} RSSI={store.meta['feature_groups']['rssi']} SNR={store.meta['feature_groups']['snr']}")
        for st in SPLITS:
            for seed in seeds:
                split = load_split(baseline_results, d, st, seed)
                # Copy the exact split into the extension folder for audit/provenance.
                base.save_split(extension_results, d, st, seed, split)
                for variant in NEW_VARIANTS:
                    for method in METHODS:
                        base.run_tuned_case(
                            store,
                            extension_results,
                            d,
                            st,
                            int(seed),
                            variant,
                            method,
                            split,
                            profile,
                            threads,
                            inference_reps,
                            force,
                            protocol_version=EXT_PROTOCOL_VERSION,
                        )


def check_extension_complete(
    datasets: Sequence[str], seeds: Sequence[int], extension_results: Path
) -> None:
    missing: List[str] = []
    for d in datasets:
        for st in SPLITS:
            for seed in seeds:
                for v in NEW_VARIANTS:
                    for m in METHODS:
                        p = checkpoint_path(extension_results, d, st, seed, v, m)
                        if load_complete_checkpoint(p) is None:
                            missing.append(str(p))
    if missing:
        preview = "\n".join(f"  - {x}" for x in missing[:30])
        extra = "" if len(missing) <= 30 else f"\n  ... and {len(missing)-30} more"
        raise RuntimeError(f"RSSI-impact extension incomplete. Missing:\n{preview}{extra}")
    log("Extension completeness check passed.")


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--task", choices=["check", "run", "aggregate"], default="run")
    ap.add_argument("--datasets", default="primary")
    ap.add_argument("--profile", choices=["quick", "desktop"], default="desktop")
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS_20)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cache-dir", default="cache_equal_footing_v5")
    ap.add_argument("--baseline-results", default="results_equal_footing_v5")
    ap.add_argument("--output-dir", default="results_equal_footing_v5_rssi_impact")
    ap.add_argument("--force", action="store_true")


def run(args) -> None:
    datasets = expand_dataset_tokens(args.datasets)
    seeds = list(map(int, args.seeds))
    cache_root = Path(args.cache_dir)
    baseline_results = Path(args.baseline_results)
    extension_results = Path(args.output_dir)
    extension_results.mkdir(parents=True, exist_ok=True)

    validate_existing_v5(datasets, seeds, cache_root, baseline_results)
    if args.task == "check":
        log("CHECK ONLY: no model was trained.")
        return

    if args.task == "run":
        # records the machine and the thread pools of the run (not written by the original V5.1 script)
        (extension_results / "system_metadata.json").write_text(
            json.dumps(base.system_metadata(args.threads, EXT_PROTOCOL_VERSION), indent=2), encoding="utf-8")
        run_new_cases(datasets, seeds, args.profile, args.threads, cache_root, baseline_results, extension_results, args.force)

    check_extension_complete(datasets, seeds, extension_results)
    write_aggregates(datasets, seeds, baseline_results, extension_results)
    log("FINISHED RSSI IMPACT EXTENSION")


def main() -> None:
    ap = argparse.ArgumentParser(description="Incremental V5 experiment for isolating the impact of RSSI without repeating completed analyses.")
    add_arguments(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
