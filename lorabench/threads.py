"""Thread limit for the whole process: joblib, BLAS, and OpenMP.

Every command runs with at most --threads threads (default 6, as in the article). Without this limit,
NumPy and SciPy (OpenBLAS or MKL) and the OpenMP kernels of scikit-learn use every logical core:
MLP and ELM do their matrix products in BLAS, and the brute-force neighbor search of kNN and WkNN runs
in OpenMP for the Euclidean and Manhattan metrics whatever n_jobs is. RF (joblib threads) and XGBoost
(OpenMP) receive n_jobs = --threads.

The limit is applied in two ways: environment variables, set before NumPy is imported, which every
library and child process reads when it starts; and threadpoolctl, for the libraries already loaded.
When the package is used from Python instead of `python -m lorabench`, call `limit(6)` first.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

DEFAULT_THREADS = 6
ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "LOKY_MAX_CPU_COUNT")


def from_argv(argv: Sequence[str], default: int = DEFAULT_THREADS) -> int:
    """Value of --threads in a command line, read before argparse runs and before NumPy is loaded."""
    for i, arg in enumerate(argv):
        if arg == "--threads" and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith("--threads="):
            value = arg.split("=", 1)[1]
        else:
            continue
        try:
            return max(1, int(value))
        except ValueError:
            return default
    return default


def set_environment(n: int) -> None:
    """Sets the variables that BLAS, OpenMP, and joblib read when they start."""
    for name in ENV_VARS:
        os.environ[name] = str(int(n))


def limit(n: int) -> List[Dict]:
    """Limits the whole process to n threads and returns the thread pools of the loaded libraries."""
    set_environment(n)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=int(n))  # stays in force for the rest of the process
    return pools()


def pools() -> List[Dict]:
    """BLAS and OpenMP libraries loaded in the process, with their current number of threads."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    return [{"api": d.get("internal_api"), "library": d.get("prefix"), "version": d.get("version"),
             "num_threads": d.get("num_threads")} for d in threadpool_info()]
