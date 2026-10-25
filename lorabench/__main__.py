"""Command-line interface: python -m lorabench <command> [options]

Commands (former PowerShell scripts in brackets):
  check-system   Python, CPU, RAM, and installed packages                 [00]
  prepare        build the dataset caches and audit the common protocol     [02]
  smoke          quick functional test: seed 7, quick grids, <=1000 samples [03]
  run            main benchmark: --task core|spatial|ablation_random|
                 ablation_spatial|ablation|aggregate|cache|all              [04-07]
  costs          per-scenario aggregation + computational-cost synthesis    [07B]
  rssi-impact    RSSI/SNR isolation (SNR_ONLY, RSSI_SNR)                    [08, 08A]
  subsample      sample-size control (outdoor subsampled to indoor sizes)   [09, 09A]
  paper          article tables and figures (environment classes)
  env            reproducibility report (versions, hardware, checkpoints)
  reproduce      every stage above, in order, resuming from checkpoints     [08]
  selftest       whole pipeline on small synthetic data (about 5 minutes)

Every command accepts --threads (default 6): the whole process, including the BLAS library used by MLP
and ELM and the OpenMP kernels of scikit-learn and XGBoost, then runs with at most that many threads.
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from . import threads
from .threads import DEFAULT_THREADS


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        # encoding, errors, fileno, ... come from the console stream
        return getattr(self.streams[0], name)


@contextlib.contextmanager
def tee(log_path):
    """Copies console output to log_path (appending), like Tee-Object -Append in the old scripts."""
    if not log_path:
        yield
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, f), _Tee(old_err, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _core_args(**kw):
    from .core import SEEDS_20
    base = dict(task="all", datasets="primary", profile="desktop", seeds=list(SEEDS_20), threads=DEFAULT_THREADS,
                data_dir="datasets", cache_dir="cache_equal_footing_v5", output_dir="results_equal_footing_v5",
                force=False, force_cache=False, max_samples=0)
    base.update(kw)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- commands
def cmd_check_system(args):
    from . import env
    env.print_system()


def cmd_prepare(args):
    from .data import audit, expand_dataset_tokens
    ok = audit(Path(args.data_dir), Path(args.cache_dir), args.force_cache, expand_dataset_tokens(args.datasets))
    if not ok:
        raise SystemExit(2)


def cmd_smoke(args):
    from . import core
    core.run(_core_args(task="all", profile="quick", seeds=args.seeds, threads=args.threads, data_dir=args.data_dir,
                        cache_dir=args.cache_dir, output_dir=args.output_dir, max_samples=args.max_samples,
                        datasets=args.datasets))


def cmd_run(args):
    from . import core
    core.run(args)


def cmd_costs(args):
    from . import core, costs
    if not args.skip_aggregate:
        core.run(_core_args(task="aggregate", datasets=args.datasets, data_dir=args.data_dir, threads=args.threads,
                            cache_dir=args.cache_dir, output_dir=args.output_dir))
    costs.run(SimpleNamespace(output_dir=args.output_dir))


def cmd_rssi_impact(args):
    from . import rssi_impact
    rssi_impact.run(args)


def cmd_subsample(args):
    from . import subsample
    subsample.run(args)


def cmd_paper(args):
    from . import paper
    paper.run(args)


def cmd_env(args):
    from . import env
    env.run(args.root, args.out)


def cmd_reproduce(args):
    """All stages in the order of the original scripts; every stage resumes from its checkpoints."""
    from . import core, costs, rssi_impact, subsample, paper, env
    from .data import audit, expand_dataset_tokens
    logs = Path(args.log_dir)
    common = dict(datasets=args.datasets, threads=args.threads, data_dir=args.data_dir, cache_dir=args.cache_dir,
                  output_dir=args.output_dir, seeds=args.seeds, profile=args.profile)
    def prepare():
        if not audit(Path(args.data_dir), Path(args.cache_dir), False, expand_dataset_tokens(args.datasets)):
            raise SystemExit("Dataset audit failed: the common protocol is not satisfied.")

    def run_subsample():
        chosen = set(expand_dataset_tokens(args.datasets))
        outdoor = [d for d in args.subsample_datasets if d in chosen]
        if not outdoor:
            print("Sample-size control skipped: none of --subsample-datasets is in --datasets.")
            return
        subsample.run(SimpleNamespace(
            task="run", datasets=outdoor, sizes=list(args.sizes), seeds=args.seeds, profile=args.profile,
            threads=args.threads, cache_dir=args.cache_dir, baseline_results=args.output_dir,
            output_dir=args.subsample_output_dir, fixed_subset=False, force=False))

    stages = [
        ("02_prepare", prepare),
        ("04_core", lambda: core.run(_core_args(task="core", **common))),
        ("05_spatial", lambda: core.run(_core_args(task="spatial", **common))),
        ("06A_ablation_random", lambda: core.run(_core_args(task="ablation_random", **common))),
        ("06B_ablation_spatial", lambda: core.run(_core_args(task="ablation_spatial", **common))),
        ("07_aggregate", lambda: core.run(_core_args(task="aggregate", **common))),
        ("07B_costs", lambda: costs.run(SimpleNamespace(output_dir=args.output_dir))),
        ("08_rssi_impact", lambda: rssi_impact.run(SimpleNamespace(
            task="run", datasets=args.datasets, profile=args.profile, seeds=args.seeds, threads=args.threads,
            cache_dir=args.cache_dir, baseline_results=args.output_dir, output_dir=args.rssi_output_dir, force=False))),
        ("09_subsample", run_subsample),
        ("10_paper", lambda: paper.run(SimpleNamespace(
            results=args.output_dir, rssi_results=args.rssi_output_dir, subsample_results=args.subsample_output_dir,
            cache_dir=args.cache_dir, out=args.paper_dir, check=False, no_figures=False))),
        ("11_env", lambda: env.run(".", "reproducibility")),
    ]
    names = [s for s, _ in stages]
    start = names.index(args.start_at) if args.start_at else 0
    for name, fn in stages[start:]:
        t0 = time.time()
        with tee(logs / f"{name}.log"):
            print(f"===== {name} =====", flush=True)
            fn()
            print(f"===== {name} done in {(time.time() - t0) / 3600:.2f} h =====", flush=True)


def cmd_selftest(args):
    """Runs every command on synthetic data in a scratch folder and checks that the outputs exist."""
    import os
    from . import synthetic
    work = Path(args.workdir).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    old = Path.cwd()
    os.chdir(work)
    t_start = time.time()
    try:
        synthetic.write_all(Path("datasets"))
        # all six scenarios go through the adapters; training uses one outdoor and one indoor scenario and
        # two seeds to stay short
        seeds = [7, 13]
        dsets = "antwerp2019,brno_iUG_BUT"
        thr = args.threads
        print("Self-test on synthetic data (about 5 minutes). The numbers are meaningless.")
        print("== selftest: prepare"); cmd_prepare(SimpleNamespace(data_dir="datasets", cache_dir="cache_equal_footing_v5",
                                                                  force_cache=False, datasets="primary"))
        print("== selftest: run --task all (quick)")
        from . import core
        core.run(_core_args(task="all", profile="quick", seeds=seeds, threads=thr, datasets=dsets))
        print("== selftest: costs"); cmd_costs(SimpleNamespace(skip_aggregate=False, datasets=dsets, data_dir="datasets", threads=thr,
                                                              cache_dir="cache_equal_footing_v5",
                                                              output_dir="results_equal_footing_v5"))
        print("== selftest: rssi-impact")
        cmd_rssi_impact(SimpleNamespace(task="run", datasets=dsets, profile="quick", seeds=seeds, threads=thr,
                                        cache_dir="cache_equal_footing_v5", baseline_results="results_equal_footing_v5",
                                        output_dir="results_equal_footing_v5_rssi_impact", force=False))
        print("== selftest: subsample")
        cmd_subsample(SimpleNamespace(task="run", datasets=["antwerp2019"], sizes=[140, 260], seeds=seeds,
                                      profile="quick", threads=thr, cache_dir="cache_equal_footing_v5",
                                      baseline_results="results_equal_footing_v5", output_dir=None,
                                      fixed_subset=False, force=False))
        print("== selftest: paper")
        cmd_paper(SimpleNamespace(results="results_equal_footing_v5", rssi_results="results_equal_footing_v5_rssi_impact",
                                  subsample_results="results_equal_footing_v5_subsample", cache_dir="cache_equal_footing_v5",
                                  out="paper_outputs", check=False, no_figures=False))
        print("== selftest: env"); cmd_env(SimpleNamespace(root=".", out="reproducibility"))
        expected = [
            "results_equal_footing_v5/aggregate_equal_weight/cross_deployment_random_method_summary.csv",
            "results_equal_footing_v5/aggregate_equal_weight/computational/cross_deployment_main_computational_summary.csv",
            "results_equal_footing_v5/brno_iUG_BUT/aggregate/VALIDATION_REPORT.txt",
            "results_equal_footing_v5/aggregate_equal_weight/deployment_ablation_spatial_statistical_comparison.csv",
            "results_equal_footing_v5_rssi_impact/aggregate_rssi_impact/cross_source_dataset_rssi_impact_summary.csv",
            "results_equal_footing_v5_subsample/aggregate_subsample/summary_by_condition.csv",
            "paper_outputs/table_errors.tex", "paper_outputs/table_sample_size.tex", "paper_outputs/table_costs.tex",
            "paper_outputs/table_overhead.tex", "paper_outputs/fig_environment_effects.png",
            "paper_outputs/fig_spatial_gap.png", "paper_outputs/fig_rssi_snr.png", "paper_outputs/fig_ablation.png",
            "paper_outputs/fig_tradeoff.png", "reproducibility/report.md",
        ]
        missing = [p for p in expected if not Path(p).exists()]
        print("\nSELFTEST", "PASSED" if not missing else "FAILED",
              f"({len(expected) - len(missing)}/{len(expected)} outputs, {(time.time() - t_start) / 60:.1f} min)")
        for p in missing:
            print("  missing:", p)
        if missing:
            raise SystemExit(1)
    finally:
        os.chdir(old)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    from . import core, rssi_impact, subsample, paper
    from .core import SEEDS_20
    ap = argparse.ArgumentParser(prog="python -m lorabench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help="also append the console output to this file")
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")
    threads_help = f"maximum number of threads in every library (default {DEFAULT_THREADS})"

    p = sub.add_parser("check-system", help="Python, CPU, RAM, installed packages, and thread limit")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=threads_help); p.set_defaults(func=cmd_check_system)

    p = sub.add_parser("prepare", help="build the dataset caches and audit the common protocol")
    p.add_argument("--data-dir", default="datasets"); p.add_argument("--cache-dir", default="cache_equal_footing_v5")
    p.add_argument("--datasets", default="primary"); p.add_argument("--force-cache", action="store_true")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=threads_help)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("smoke", help="quick functional test of the main benchmark (seed 7, quick grids, <=1000 samples)")
    p.add_argument("--datasets", default="primary"); p.add_argument("--seeds", nargs="*", type=int, default=[7])
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=threads_help)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--data-dir", default="datasets"); p.add_argument("--cache-dir", default="cache_equal_footing_v5")
    p.add_argument("--output-dir", default="results_smoke_equal_footing_v5")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("run", help="main benchmark (core, spatial holdout, ablations, aggregation)")
    core.add_arguments(p); p.set_defaults(func=cmd_run)

    p = sub.add_parser("costs", help="per-scenario aggregation followed by the computational-cost synthesis")
    p.add_argument("--output-dir", default="results_equal_footing_v5"); p.add_argument("--datasets", default="primary")
    p.add_argument("--data-dir", default="datasets"); p.add_argument("--cache-dir", default="cache_equal_footing_v5")
    p.add_argument("--skip-aggregate", action="store_true", help="reuse the existing aggregate/all_case_summaries.csv files")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=threads_help)
    p.set_defaults(func=cmd_costs)

    p = sub.add_parser("rssi-impact", help="RSSI/SNR isolation (SNR_ONLY and RSSI_SNR)")
    rssi_impact.add_arguments(p); p.set_defaults(func=cmd_rssi_impact)

    p = sub.add_parser("subsample", help="sample-size control (outdoor datasets subsampled to the indoor sizes)")
    subsample.add_arguments(p); p.set_defaults(func=cmd_subsample)

    p = sub.add_parser("paper", help="article tables and figures by environment class")
    paper.add_arguments(p); p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=threads_help)
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("env", help="reproducibility report")
    p.add_argument("--root", default="."); p.add_argument("--out", default="reproducibility")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=threads_help); p.set_defaults(func=cmd_env)

    p = sub.add_parser("reproduce", help="every stage in order (resumes from checkpoints)")
    p.add_argument("--datasets", default="primary"); p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS_20))
    p.add_argument("--profile", choices=["quick", "desktop"], default="desktop")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=threads_help)
    p.add_argument("--data-dir", default="datasets"); p.add_argument("--cache-dir", default="cache_equal_footing_v5")
    p.add_argument("--output-dir", default="results_equal_footing_v5")
    p.add_argument("--rssi-output-dir", default="results_equal_footing_v5_rssi_impact")
    p.add_argument("--subsample-output-dir", default="results_equal_footing_v5_subsample")
    p.add_argument("--subsample-datasets", nargs="*", default=["antwerp2019", "bonn"])
    p.add_argument("--sizes", nargs="*", type=int, default=[572, 930, 2436])
    p.add_argument("--paper-dir", default="paper_outputs"); p.add_argument("--log-dir", default="logs")
    p.add_argument("--start-at", choices=["02_prepare", "04_core", "05_spatial", "06A_ablation_random", "06B_ablation_spatial",
                                          "07_aggregate", "07B_costs", "08_rssi_impact", "09_subsample", "10_paper", "11_env"])
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("selftest", help="whole pipeline on small synthetic data")
    p.add_argument("--workdir", default="selftest_run")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=threads_help)
    p.add_argument("--keep", action="store_true", help="keep the scratch folder"); p.set_defaults(func=cmd_selftest)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # The limit must be in the environment before NumPy, SciPy, scikit-learn, and XGBoost are loaded
    # (build_parser imports them); threadpoolctl then enforces it on the libraries already in memory.
    threads.set_environment(threads.from_argv(argv))
    args = build_parser().parse_args(argv)
    pools = threads.limit(args.threads)
    with tee(args.log):
        print(f"Thread limit: {args.threads} (" + (", ".join(f"{p['api']} {p['num_threads']}" for p in pools)
                                                    or "no BLAS/OpenMP library loaded") + ")", flush=True)
        args.func(args)


if __name__ == "__main__":
    main()
