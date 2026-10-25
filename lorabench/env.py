# -*- coding: utf-8 -*-
"""Reproducibility report (former coletar_reprodutibilidade.py).

Collects the hardware, operating system, package versions, BLAS threading, pip freeze, the
metadata recorded by the runs, the protocol fields stored in the checkpoints, the selected
hyperparameters, and a few facts checked directly in the code. Only reads files.

Writes to <root>/<out>/:
    report.md                     human-readable summary
    environment.json              the same data, structured
    pip_freeze.txt                exact package versions of the running environment
    selected_hyperparameters.csv  configuration selected in every case
    paper_text.tex                Implementation-environment paragraph and Data Availability section
"""
from __future__ import annotations

import collections
import datetime as dt
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from . import threads as thread_limit

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib",
            "threadpoolctl", "psutil", "openpyxl", "matplotlib"]
RESULT_DIRS = ["results_equal_footing_v5", "results_equal_footing_v5_rssi_impact",
               "results_equal_footing_v5_subsample"]
THREAD_VARS = list(thread_limit.ENV_VARS)
PACKAGE_DIR = Path(__file__).resolve().parent
CORE_SOURCE = PACKAGE_DIR / "core.py"


def msg(t=""):
    print(t, flush=True)


def powershell(cmd, timeout=30):
    """Runs a PowerShell command and returns its output (empty outside Windows)."""
    if os.name != "nt":
        return ""
    for exe in ("powershell", "pwsh"):
        try:
            r = subprocess.run([exe, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            continue
    return ""


# --------------------------------------------------------------------------- 1. system
def collect_system():
    s = {"platform": platform.platform(), "machine": platform.machine(),
         "python_version": platform.python_version(), "python_full": sys.version,
         "python_executable": sys.executable,
         "inside_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix)}
    if os.name == "nt":
        rel, ver, csd, _ = platform.win32_ver()
        s["win32_ver"] = {"release": rel, "version": ver, "csd": csd}
        try:
            s["win32_edition"] = platform.win32_edition()
        except Exception:
            pass
        # Python < 3.12 reports Windows 11 as "Windows 10"; the correct name comes from CIM.
        os_info = powershell("$o=Get-CimInstance Win32_OperatingSystem; "
                             "\"$($o.Caption)|$($o.Version)|$($o.BuildNumber)|$($o.OSArchitecture)\"")
        if os_info:
            cap, ver, build, arch = (os_info.split("|") + ["", "", "", ""])[:4]
            s["windows"] = {"caption": cap.strip(), "version": ver, "build": build, "arch": arch}
        disp = powershell("(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion').DisplayVersion")
        if disp:
            s.setdefault("windows", {})["display_version"] = disp
        cpu = powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        if cpu:
            s["cpu_name"] = cpu.strip()
        mem = powershell("Get-CimInstance Win32_PhysicalMemory | ForEach-Object "
                         "{ \"$($_.Capacity)|$($_.Speed)|$($_.ConfiguredClockSpeed)|$($_.Manufacturer)|$($_.PartNumber)\" }")
        if mem:
            mods = []
            for line in mem.splitlines():
                p = (line.split("|") + [""] * 5)[:5]
                try:
                    cap_gb = round(int(p[0]) / 1024 ** 3, 1)
                except Exception:
                    cap_gb = None
                mods.append({"capacity_gb": cap_gb, "speed_mt_s": p[1], "configured_mt_s": p[2],
                             "manufacturer": p[3].strip(), "part_number": p[4].strip()})
            s["ram_modules"] = mods
        gpu = powershell("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }")
        if gpu:
            s["gpu"] = [g.strip() for g in gpu.splitlines() if g.strip()]
    else:
        s["cpu_name"] = platform.processor() or platform.machine()
    try:
        import psutil
        s["physical_cpus"] = psutil.cpu_count(logical=False)
        s["logical_cpus"] = psutil.cpu_count(logical=True)
        s["ram_total_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        s["logical_cpus"] = os.cpu_count()
    s["thread_variables_now"] = {v: os.environ.get(v) for v in THREAD_VARS}
    return s


def print_system():
    """Short system check (former 00_VERIFICAR_SISTEMA.ps1)."""
    s = collect_system()
    msg("=== System ===")
    msg(f"Python {s['python_version']} ({s['python_executable']})")
    msg(f"Platform: {s['platform']}")
    msg(f"CPU: {s.get('cpu_name', '?')} | physical cores: {s.get('physical_cpus', '?')} | logical: {s.get('logical_cpus', '?')}")
    msg(f"RAM: {s.get('ram_total_gb', '?')} GB")
    missing = []
    for name in PACKAGES:
        try:
            msg(f"  {name:<14} {md.version(name)}")
        except md.PackageNotFoundError:
            missing.append(name)
            msg(f"  {name:<14} NOT INSTALLED")
    if missing:
        msg("Install the requirements: python -m pip install -r requirements.txt")
        return
    import numpy, scipy.linalg, sklearn.neighbors, xgboost  # noqa: F401,E401  (load BLAS and OpenMP)
    pools = thread_limit.pools()
    msg("Thread pools (limited by --threads): " +
        (", ".join(f"{p['api']} {p['version'] or ''} = {p['num_threads']}".replace("  ", " ") for p in pools) or "none"))


# --------------------------------------------------------------------------- 2. packages
def install_date(dist):
    try:
        for f in dist.files or []:
            if str(f).endswith(("METADATA", "PKG-INFO")):
                p = Path(dist.locate_file(f))
                if p.exists():
                    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    return None


def collect_packages():
    out = {}
    for name in PACKAGES:
        try:
            d = md.distribution(name)
            out[name] = {"version": d.version, "installed_at": install_date(d)}
        except md.PackageNotFoundError:
            out[name] = {"version": None, "installed_at": None}
    return out


def collect_blas():
    info = {"threadpoolctl": None, "numpy_config": None}
    try:
        import numpy  # noqa: F401  (loads NumPy's BLAS)
        import sklearn  # noqa: F401
        from threadpoolctl import threadpool_info
        info["threadpoolctl"] = [{k: v for k, v in d.items() if k in
                                  ("user_api", "internal_api", "version", "num_threads",
                                   "threading_layer", "architecture", "prefix")}
                                 for d in threadpool_info()]
    except Exception as e:
        info["threadpoolctl"] = f"unavailable: {e}"
    try:
        import numpy as np
        try:
            cfg = np.show_config(mode="dicts")
            info["numpy_config"] = cfg.get("Build Dependencies", cfg)
        except TypeError:
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                np.show_config()
            info["numpy_config"] = buf.getvalue()[:3000]
    except Exception as e:
        info["numpy_config"] = f"unavailable: {e}"
    return info


def pip_freeze():
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        return r.stdout
    except Exception as e:
        return f"# pip freeze failed: {e}\n"


# --------------------------------------------------------------------------- 3. runs
def read_text(p):
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return ""


def collect_recorded_metadata(root):
    out = {}
    for name in RESULT_DIRS + [p.name for p in root.glob("results_*") if p.is_dir()]:
        p = root / name / "system_metadata.json"
        if p.exists() and name not in out:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                d["_written_at"] = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                out[name] = d
            except Exception as e:
                out[name] = {"error": str(e)}
    return out


def collect_checkpoints(root):
    summary = {"n_cases": 0, "by_split": collections.Counter(), "protocol_version": collections.Counter(),
               "selection_mode": collections.Counter(), "refit_after_selection": collections.Counter(),
               "train_validation_refit": collections.Counter(), "final_model_training_scope": collections.Counter(),
               "inference_repetitions": collections.Counter(), "test_used_after_selection_only": collections.Counter(),
               "first": None, "last": None}
    sizes = {}    # (dataset, split) -> {n_train: [min, max], ...}
    nfeat = {}    # (dataset, variant) -> set
    hyper = collections.Counter()  # (dataset, split, variant, method, params) -> n
    dirs = [root / n for n in RESULT_DIRS if (root / n).is_dir()]
    for base in dirs:
        for p in base.glob("*/checkpoints/*/seed_*/*/*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("status") != "complete":
                continue
            summary["n_cases"] += 1
            st = d.get("split_type", "?")
            summary["by_split"][st] += 1
            for k in ("protocol_version", "selection_mode", "refit_after_selection", "train_validation_refit",
                      "final_model_training_scope", "inference_repetitions", "test_used_after_selection_only"):
                summary[k][str(d.get(k))] += 1
            fin = d.get("finished_at")
            if fin:
                summary["first"] = fin if summary["first"] is None or fin < summary["first"] else summary["first"]
                summary["last"] = fin if summary["last"] is None or fin > summary["last"] else summary["last"]
            ds = d.get("dataset_id", "?")
            if st in ("random", "spatial"):
                t = sizes.setdefault((ds, st), {})
                for k in ("n_train", "n_val", "n_test"):
                    v = d.get(k)
                    if isinstance(v, int):
                        lo, hi = t.get(k, [v, v])
                        t[k] = [min(lo, v), max(hi, v)]
                nfeat.setdefault((ds, d.get("variant")), set()).add(d.get("n_features"))
                hyper[(ds, st, d.get("variant"), d.get("method"),
                       json.dumps(d.get("best_params"), sort_keys=True))] += 1
    for k in list(summary):
        if isinstance(summary[k], collections.Counter):
            summary[k] = dict(summary[k])
    summary["dirs_read"] = [str(p) for p in dirs]
    return summary, sizes, nfeat, hyper


def check_code():
    """Checks in the source the facts that the article states."""
    if not CORE_SOURCE.exists():
        return {"error": f"{CORE_SOURCE} not found"}
    t = read_text(CORE_SOURCE)
    facts = [
        ("no_refit", r'"refit_after_selection"\s*:\s*False',
         "The configuration selected on validation goes straight to the test partition, without refitting on train+validation."),
        ("pickle_size", r"pickle\.dumps\(\(selected\[\"model\"\],selected\.get\(\"x_scaler\"\),selected\.get\(\"y_scaler\"\)\)",
         "RF, XGBoost, MLP and ELM: model size = pickle of (model, input scaler, target scaler)."),
        ("knn_size", r"reference_bytes\":int\(best_nn\._fit_X\.nbytes\+ytr\.nbytes\)",
         "kNN/WkNN: model size = bytes of the reference matrix (standardized training set) + coordinates + scaler parameters; not a pickle."),
        ("inference_5_reps", r'inference_reps=2 if args\.profile=="quick" else 5',
         "Inference: desktop profile = 5 timed repetitions (quick = 2)."),
        ("warm_up", r"_=pred_fn\(\); times=\[\]",
         "One warm-up call before the timed repetitions; the median is used."),
        ("ms_per_sample", r'"inference_ms_per_sample":1000\*model_s/len\(ytest\)',
         "ms/sample = median time to predict the whole test partition / number of test samples."),
        ("rf_njobs", r"RandomForestRegressor\(\*\*params,n_jobs=threads",
         "RF uses n_jobs = --threads."),
        ("xgb_njobs", r"XGBRegressor\(.*tree_method=\"hist\",n_jobs=threads",
         "XGBoost: tree_method=hist, n_jobs = --threads."),
        ("xgb_multioutput", r"MultiOutputRegressor\(base,n_jobs=1\)",
         "XGBoost: one regressor per coordinate (MultiOutputRegressor), trained sequentially."),
        ("knn_njobs", r"NearestNeighbors\(n_neighbors=maxk,metric=metric,algorithm=\"brute\",n_jobs=threads\)",
         "kNN/WkNN: brute-force neighbor search with n_jobs = --threads (for the Euclidean and Manhattan metrics, "
         "scikit-learn runs the search in OpenMP, which follows the thread limit instead of n_jobs)."),
        ("mlp_no_njobs", r"MLPRegressor\(\*\*params,max_iter=300,early_stopping=True,validation_fraction=0\.1",
         "MLP: no n_jobs (parallelism only through BLAS); early stopping on 10% of the training partition."),
        ("elm_ridge", r"Ridge\(alpha=self\.alpha,fit_intercept=True,solver=\"lsqr\"",
         "ELM: output weights by ridge regression (lsqr solver); parallelism only through BLAS."),
    ]
    out = {}
    for key, pattern, text in facts:
        out[key] = {"found": bool(re.search(pattern, t)), "meaning": text}
    main_src = read_text(PACKAGE_DIR / "__main__.py") if (PACKAGE_DIR / "__main__.py").exists() else ""
    lim_src = read_text(PACKAGE_DIR / "threads.py") if (PACKAGE_DIR / "threads.py").exists() else ""
    out["thread_limit"] = {
        "found": bool(re.search(r"threads\.limit\(args\.threads\)", main_src)
                      and re.search(r"threadpool_limits\(limits=", lim_src) and re.search(r"OMP_NUM_THREADS", lim_src)),
        "meaning": "Every command limits the whole process (joblib, BLAS, OpenMP) to --threads threads (lorabench/threads.py)."}
    out["thread_record"] = {"found": bool(re.search(r'"thread_pools":thread_pools\(\)', t)),
                            "meaning": "Each run records its BLAS and OpenMP thread pools in system_metadata.json."}
    return out


def recorded_pools(recorded):
    """Thread pools stored by the runs, per result folder (None when the run did not record them)."""
    return {name: (d.get("thread_pools") if isinstance(d, dict) else None) for name, d in recorded.items()}


# --------------------------------------------------------------------------- 4. output
def windows_version(system, recorded):
    w = system.get("windows", {})
    name = w.get("caption")
    if not name:
        raw = (recorded or {}).get("platform") or system.get("platform") or ""
        m = re.search(r"Windows-(\d+)-10\.0\.(\d+)", raw)
        if m:  # Python < 3.12 calls Windows 11 "10"; build >= 22000 is Windows 11
            name = f"Windows {'11' if int(m.group(2)) >= 22000 else '10'}"
            w = dict(w, build=w.get("build") or m.group(2))
        else:
            name = raw
    name = (name or "").replace("Microsoft ", "")
    extra = ", ".join(x for x in [w.get("display_version"), f"build {w['build']}" if w.get("build") else None] if x)
    return f"{name} ({extra})" if extra else name


def run(root: str = ".", out: str = "reproducibility") -> None:
    root = Path(root).resolve()
    outdir = root / out
    outdir.mkdir(exist_ok=True)

    msg(f"Benchmark folder: {root}")
    msg("1/5 System and hardware...")
    system = collect_system()
    msg("2/5 Packages, BLAS, and pip freeze...")
    packages = collect_packages()
    blas = collect_blas()
    (outdir / "pip_freeze.txt").write_text(pip_freeze(), encoding="utf-8")
    msg("3/5 Metadata recorded by the runs...")
    recorded = collect_recorded_metadata(root)
    msg("4/5 Checkpoints (may take a few minutes)...")
    ck, sizes, nfeat, hyper = collect_checkpoints(root)
    msg("5/5 Checking the code...")
    code = check_code()

    # ---- warnings
    warnings = []
    if not system.get("inside_venv"):
        warnings.append("Not running inside a virtual environment; package versions may differ from those of the benchmark runs.")
    g = recorded.get("results_equal_footing_v5") or next(iter(recorded.values()), None)
    if g and "numpy_version" in g:
        if g.get("numpy_version") != packages["numpy"]["version"]:
            warnings.append(f"NumPy changed since the runs: recorded {g.get('numpy_version')}, now {packages['numpy']['version']}.")
        if g.get("pandas_version") != packages["pandas"]["version"]:
            warnings.append(f"pandas changed since the runs: recorded {g.get('pandas_version')}, now {packages['pandas']['version']}.")
        pv = (g.get("python_version") or "").split()[0]
        if pv and pv != system["python_version"]:
            warnings.append(f"Python changed since the runs: recorded {pv}, now {system['python_version']}.")
    if ck["first"]:
        for name, info in packages.items():
            if info["installed_at"] and info["installed_at"] > ck["first"][:16]:
                warnings.append(f"{name} {info['version']} was installed on {info['installed_at']}, after the first run "
                                f"({ck['first']}); the version used in the experiments may have been different.")
    if ck["n_cases"] == 0:
        warnings.append("No checkpoints found. Check that results_equal_footing_v5 is inside the benchmark folder.")
    pools_by_dir = recorded_pools(recorded)
    unrecorded = [name for name, p in pools_by_dir.items() if p is None]
    if unrecorded:
        warnings.append(f"{', '.join(unrecorded)}: the runs did not record their thread pools, so they were made "
                        "without the thread limit; the BLAS library (MLP, ELM) and the OpenMP neighbor search (kNN, "
                        f"WkNN) then used their defaults, usually all {system.get('logical_cpus', '?')} logical threads.")
    for name, plist in pools_by_dir.items():
        req = (recorded.get(name) or {}).get("threads_requested")
        over = [p for p in (plist or []) if req and (p.get("num_threads") or 0) > req]
        if over:
            warnings.append(f"{name}: thread pools above --threads={req}: "
                            + ", ".join(f"{p.get('api')} {p.get('num_threads')}" for p in over))

    # ---- values for the paper text
    thr = sorted({str(d.get("threads_requested")) for d in recorded.values() if isinstance(d, dict) and d.get("threads_requested")})
    threads_txt = thr[0] if len(thr) == 1 else "/".join(thr) or "?"
    blas_libs = [b for b in (blas["threadpoolctl"] or []) if isinstance(b, dict) and b.get("user_api") == "blas"]
    # library and version only: the thread counts of this process reflect the --threads of the env command
    blas_txt = ", ".join(f"{b.get('internal_api')} {b.get('version') or ''}".strip() for b in blas_libs) or "?"
    blas_lib = blas_libs[0].get("internal_api", "?") if blas_libs else "?"
    os_name = windows_version(system, g) if os.name == "nt" or "Windows" in str((g or {}).get("platform", "")) else system["platform"]
    blas_names = {"openblas": "OpenBLAS", "mkl": "Intel MKL", "blis": "BLIS", "accelerate": "Apple Accelerate"}
    blas_lib = blas_names.get(str(blas_lib).lower(), blas_lib)
    in_words = {"1": "one", "2": "two", "4": "four", "6": "six", "8": "eight", "12": "twelve"}
    threads_txt = in_words.get(str(threads_txt), threads_txt)
    v = lambda name: packages[name]["version"] or "?"
    py = ((g or {}).get("python_version") or system["python_version"]).split()[0]
    npv = (g or {}).get("numpy_version") or packages["numpy"]["version"]
    pdv = (g or {}).get("pandas_version") or packages["pandas"]["version"]
    ram = (g or {}).get("ram_gb") or system.get("ram_total_gb")
    ram_txt = f"{round(float(ram))}" if ram else "?"
    speeds = sorted({m.get("configured_mt_s") or m.get("speed_mt_s") for m in system.get("ram_modules", [])} - {None, ""})
    cpu = system.get("cpu_name", "?")
    reps = ck["inference_repetitions"]
    reps_txt = max(reps, key=reps.get) if reps else "5"
    period = f"{ck['first'][:10]} to {ck['last'][:10]}" if ck["first"] else "?"
    limited = bool(pools_by_dir) and not unrecorded
    th = "thread" if threads_txt in ("one", "1") else "threads"
    if limited:
        parallel_txt = (f"All computations were limited to {threads_txt} {th}: RF, XGBoost, and the neighbor\n"
                        f"searches of kNN and WkNN used {threads_txt} {th}, and so did the {blas_lib} linear-algebra\n"
                        f"library through which MLP and ELM are parallelized.")
    else:
        parallel_txt = (f"RF and XGBoost used {threads_txt} worker {th}, whereas the {blas_lib} linear-algebra library\n"
                        f"used by MLP and ELM and the OpenMP neighbor searches of kNN and WkNN kept their default\n"
                        f"of {system.get('logical_cpus', '?')}? threads.")

    tex = rf"""% ---------------------------------------------------------------------------
% Written by `python -m lorabench env` on {dt.datetime.now():%Y-%m-%d %H:%M}
% Check the values marked with ? and the report before pasting into the article.
% ---------------------------------------------------------------------------

% Implementation environment:
The experiments were implemented in Python and executed on a personal computer
running {os_name}, equipped with an {cpu} processor and
{ram_txt}~GB of {"/".join(speeds) + "~MT/s " if speeds else ""}RAM; no GPU was used.
The software stack comprised Python {py}, NumPy {npv}, pandas {pdv},
SciPy {v('scipy')}, scikit-learn {v('scikit-learn')}, and XGBoost {v('xgboost')}.
{parallel_txt} For each case, the configuration selected on the validation partition
was applied directly to the test partition, without refitting on the combined training and
validation data. Inference time is the median of {reps_txt} timed predictions of the whole test
partition, after one warm-up call, divided by the number of test samples. Model size is the
size of the serialized model together with its input and target scalers; for kNN and WkNN,
it is the size of the stored reference matrix, the reference coordinates, and the scaler parameters.

% Data Availability (in ieeeaccess.cls use \section*{{}}):
\section*{{Data Availability}}
The Antwerp, Bonn, and Brno--Bucharest datasets are publicly available from \cite{{aernouts_sigfox_2019}},
\cite{{noauthor_mclab-hbrslora-bonn_2024}}, and \cite{{svertoka_lorawan_2022}}, respectively.
The benchmark code, the list of random seeds, the random and spatial partitions, the selected
hyperparameters, and the per-case results are available at
\url{{https://github.com/<user>/<repository>}} and archived on Zenodo
(DOI: \href{{https://doi.org/10.5281/zenodo.XXXXXXX}}{{10.5281/zenodo.XXXXXXX}}).
"""
    (outdir / "paper_text.tex").write_text(tex, encoding="utf-8")

    # ---- hyperparameter CSV
    lines = ["dataset;split;representation;method;hyperparameters;n_seeds"]
    for (ds, st, var, met, prm), n in sorted(hyper.items(), key=lambda x: (x[0][:4], -x[1])):
        lines.append(f"{ds};{st};{var};{met};{prm};{n}")
    (outdir / "selected_hyperparameters.csv").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    # ---- JSON
    data = {"written_at": dt.datetime.now().isoformat(timespec="seconds"), "root": str(root),
            "current_system": system, "current_packages": packages, "blas": blas,
            "metadata_recorded_by_runs": recorded, "checkpoints": ck,
            "partition_sizes": {f"{d}|{s}": v for (d, s), v in sorted(sizes.items())},
            "n_features": {f"{d}|{v}": sorted(x for x in s if x is not None) for (d, v), s in sorted(nfeat.items())},
            "code_facts": code, "warnings": warnings}
    (outdir / "environment.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ---- report
    R = []
    R.append("# Reproducibility of the V5 equal-footing benchmark\n")
    R.append(f"Written on {dt.datetime.now():%Y-%m-%d %H:%M} from `{root}`.\n")
    if warnings:
        R.append("## Warnings\n")
        R += [f"- {a}" for a in warnings]
        R.append("")
    R.append("## Values for the article\n")
    R.append("| Item | Value |\n|---|---|")
    R.append(f"| Operating system | {os_name} |")
    R.append(f"| CPU | {cpu} ({system.get('physical_cpus', '?')} cores, {system.get('logical_cpus', '?')} threads) |")
    R.append(f"| RAM | {ram_txt} GB{' at ' + '/'.join(speeds) + ' MT/s' if speeds else ''} |")
    R.append(f"| GPU (present, unused) | {', '.join(system.get('gpu', [])) or '-'} |")
    R.append(f"| Python | {py} |")
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "threadpoolctl"):
        info = packages[name]
        R.append(f"| {name} | {info['version']} (installed {info['installed_at'] or '?'}) |")
    R.append(f"| BLAS library (NumPy/SciPy) | {blas_txt} |")
    R.append(f"| --threads recorded by the runs | {threads_txt} |")
    R.append("| Thread pools recorded by the runs | " + "; ".join(
        f"{name}: " + (", ".join(f"{p.get('api')} {p.get('num_threads')}" for p in plist) if plist is not None
                       else "not recorded (run without the thread limit)")
        for name, plist in pools_by_dir.items()) + " |")
    R.append(f"| Inference repetitions | {reps_txt} (plus 1 warm-up) |")
    R.append(f"| Run period (finished_at) | {period} |")
    R.append(f"| Complete cases read | {ck['n_cases']} |")
    R.append("")
    R.append("## Metadata recorded by the benchmark (system_metadata.json)\n")
    if recorded:
        for name, d in recorded.items():
            R.append(f"- **{name}** (file from {d.get('_written_at', '?')}): " +
                     ", ".join(f"{k} = {val}" for k, val in d.items() if not k.startswith("_") and k != "thread_pools"))
    else:
        R.append("- None found.")
    R.append("\nNote: this file is rewritten at every run and does not store the scikit-learn and XGBoost versions; "
             "for those, compare the installation dates above with the run period.\n")
    R.append("## Protocol recorded in the checkpoints\n")
    for k in ("by_split", "protocol_version", "selection_mode", "refit_after_selection", "train_validation_refit",
              "final_model_training_scope", "inference_repetitions", "test_used_after_selection_only"):
        R.append(f"- {k}: {ck[k]}")
    R.append("\n## Partition sizes (min-max across seeds)\n")
    R.append("| Scenario | Split | Train | Validation | Test |\n|---|---|---|---|---|")
    for (ds, st), t in sorted(sizes.items()):
        f = lambda k: "-".join(str(x) for x in sorted(set(t.get(k, ["?"]))))
        R.append(f"| {ds} | {st} | {f('n_train')} | {f('n_val')} | {f('n_test')} |")
    R.append("\n## Facts checked in the code (lorabench/core.py)\n")
    if "error" in code:
        R.append(f"- {code['error']}")
    else:
        for k, val in code.items():
            mark = "[FOUND]" if val["found"] else "[NOT FOUND]"
            R.append(f"- {mark} {val['meaning']}")
    R.append("\n## Thread variables in the current environment\n")
    R += [f"- {k} = {val}" for k, val in system["thread_variables_now"].items()]
    R.append("\n## Files written\n")
    R.append("- `paper_text.tex`: implementation-environment paragraph and Data Availability section.")
    R.append("- `pip_freeze.txt`: exact package versions (publish with the code).")
    R.append("- `selected_hyperparameters.csv`: selected configuration per scenario, split, representation, and method.")
    R.append("- `environment.json`: everything collected, in structured form.")
    (outdir / "report.md").write_text("\n".join(R) + "\n", encoding="utf-8")

    msg("")
    msg("\n".join(R[R.index("## Values for the article\n"):R.index("## Metadata recorded by the benchmark (system_metadata.json)\n")]))
    if warnings:
        msg("WARNINGS:")
        for a in warnings:
            msg(f"  - {a}")
    msg(f"\nFiles saved to: {outdir}")
