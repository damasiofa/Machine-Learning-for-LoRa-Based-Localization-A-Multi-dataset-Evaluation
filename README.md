# LoRaWAN fingerprinting benchmark (lorabench)

Code for the article

> A. D. F. Araújo, J. J. P. C. Rodrigues, F. R. Barbosa, and M. A. Al-Betar, "Machine Learning for LoRaWAN
> Fingerprinting-Based Localization: An Equal-Footing Multi-Deployment Comparison of RSSI and Multi-Feature
> Representations," submitted to *IEEE Access*, 2026.

It compares six machine-learning methods (kNN, WkNN, Random Forest, XGBoost, MLP, and ELM) for LoRaWAN
fingerprinting-based localization on three public datasets (Antwerp 2019, Bonn, and Brno-Bucharest), under the
same feature semantics, the same 20 random seeds, the same validation-only model selection, and the same
spatial-holdout protocol in every scenario. Two representations are compared: RSSI-only and COMMON
(RSSI, SNR, ESP, reception mask, and 15 aggregate descriptors).

Everything runs on a CPU from one command-line program, `python -m lorabench`. The article's experiments ran on
Windows 11; the package was also tested on Linux.

## Contents

```
lorabench/              the Python package
  data.py               dataset adapters, common representation, spatial blocks, dataset audit
  core.py               main benchmark: random split, spatial holdout, ablations, aggregation
  costs.py              computational-cost synthesis
  rssi_impact.py        RSSI/SNR isolation (SNR-only and RSSI+SNR representations)
  subsample.py          sample-size control (outdoor datasets subsampled to the indoor sizes)
  paper.py              tables and figures of the article (environment classes)
  env.py                reproducibility report
  threads.py            thread limit for the whole process (BLAS, OpenMP, joblib)
  synthetic.py          small synthetic datasets for the self-test
  reference_values.json values published in the article, used by `paper --check`
datasets/README.md      where to download the three datasets
docs/PROTOCOL.md        the experimental protocol in detail
requirements.txt        dependencies (minimum versions)
requirements-paper.txt  versions used for the article
```

## Installation

Python 3.10 or newer (tested with Python 3.12 and 3.13; the article used Python 3.12).

Windows (PowerShell), from the folder of this README:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run every command from the folder of this README, or install the package with `python -m pip install -e .`
to run it from any folder (the `lorabench` command then replaces `python -m lorabench`).

Check the installation before downloading the datasets (about 5 minutes; it runs every command on small synthetic
data, so the numbers are meaningless):

```bash
python -m lorabench selftest
```

## Data

The datasets are not redistributed here. Download the three files listed in [datasets/README.md](datasets/README.md)
into `datasets/`, then build the caches and audit the common protocol:

```bash
python -m lorabench prepare
```

The audit must end with `COMMON PROTOCOL COMPATIBLE: True`.

## Reproducing the article

One command runs every stage in order and writes one log per stage to `logs/`:

```bash
python -m lorabench reproduce --threads 6
```

All stages resume from their checkpoints: if the run stops, run the same command again (completed cases are
reported as `SKIP`). Use `--start-at <stage>` to begin at a later stage. The article's runs took about 5.5 days
on a six-core desktop CPU with six threads (Windows 11, AMD Ryzen 5 3600X): about 3.5 days for the main benchmark,
2 days for the RSSI/SNR extension, and 5 hours for the sample-size control.

The stages can also be run one at a time. The former PowerShell script of each stage is given for reference.

| Stage | Command | Former script |
|---|---|---|
| system check | `python -m lorabench check-system` | `00_VERIFICAR_SISTEMA.ps1` |
| dependencies | `python -m pip install -r requirements.txt` | `01_INSTALAR_DEPENDENCIAS.ps1` |
| caches and audit | `python -m lorabench prepare` | `02_AUDITAR_E_PREPARAR_DATASETS.ps1` |
| quick test (seed 7, quick grids, at most 1,000 samples) | `python -m lorabench smoke` | `03_TESTE_RAPIDO_1_SEMENTE.ps1` |
| random split, 20 seeds | `python -m lorabench run --task core` | `04_CORE_20_SEMENTES_CPU.ps1` |
| spatial holdout | `python -m lorabench run --task spatial` | `05_SPATIAL_HOLDOUT_20_SEMENTES_CPU.ps1` |
| ablation, random split | `python -m lorabench run --task ablation_random` | `06A_ABLACAO_RANDOM_20_SEMENTES_CPU.ps1` |
| ablation, spatial holdout | `python -m lorabench run --task ablation_spatial` | `06B_ABLACAO_SPATIAL_20_SEMENTES_CPU.ps1` |
| per-scenario and per-deployment aggregation | `python -m lorabench run --task aggregate` | `07_AGREGAR_RESULTADOS.ps1` |
| computational costs | `python -m lorabench costs` | `07B_AGREGAR_CUSTO_COMPUTACIONAL.ps1` |
| everything in the main benchmark | `python -m lorabench run --task all` | `08_EXECUTAR_TUDO_20_SEMENTES_CPU.ps1` |
| RSSI/SNR isolation | `python -m lorabench rssi-impact` | `08_EXECUTAR_TUDO_20_SEMENTES_CPU_RSSI_IMPACTO.ps1` |
| RSSI/SNR quick test | `python -m lorabench rssi-impact --profile quick --seeds 7 --output-dir results_equal_footing_v5_rssi_impact_smoke` | `08A_TESTE_RAPIDO_RSSI_IMPACTO_1_SEMENTE.ps1` |
| sample-size control | `python -m lorabench subsample` | `09_SUBAMOSTRAGEM_OUTDOOR_20_SEMENTES.ps1` |
| sample-size quick test | `python -m lorabench subsample --sizes 572 --seeds 7 --profile quick --output-dir results_equal_footing_v5_subsample_smoke` | `09A_TESTE_RAPIDO_SUBAMOSTRAGEM.ps1` |
| tables and figures | `python -m lorabench paper` | (new) |
| reproducibility report | `python -m lorabench env` | `coletar_reprodutibilidade.py` |

`python -m lorabench <command> --help` lists the options of each command (seeds, threads, folders, datasets).
`--datasets` accepts `primary` (all six scenarios), deployment names (`antwerp,bonn,brno,bucharest`), or scenario
identifiers (`antwerp2019,bonn,brno_iAG_BUT,brno_iUG_BUT,bucharest_iAG_UPB,bucharest_o_UPB`).
To keep a log of a single command, put `--log FILE` before the command name, for example
`python -m lorabench --log logs/04_core.log run --task core`.

The default folder names are the same as in the original scripts, so results produced by them are read and
resumed without changes.

## Outputs

| Folder | Content |
|---|---|
| `cache_equal_footing_v5/<scenario>/` | feature groups (`.npy`), targets, spatial blocks, `metadata.json` |
| `results_equal_footing_v5/<scenario>/checkpoints/` | one JSON file per case (split, seed, representation, method) with errors, selected hyperparameters, and costs, plus test errors, predictions, and validation results |
| `results_equal_footing_v5/<scenario>/aggregate/` | per-scenario statistics (paired Wilcoxon tests, ablations, random-vs-spatial gap, `VALIDATION_REPORT.txt`) |
| `results_equal_footing_v5/aggregate_equal_weight/` | synthesis with equal weight per deployment; `computational/` holds the cost synthesis |
| `results_equal_footing_v5_rssi_impact/` | SNR-only and RSSI+SNR checkpoints; `aggregate_rssi_impact/` |
| `results_equal_footing_v5_subsample/` | subsampled outdoor checkpoints; `aggregate_subsample/` |
| `paper_outputs/` | tables and figures of the article (below) |
| `reproducibility/` | `report.md`, `environment.json`, `pip_freeze.txt`, `selected_hyperparameters.csv`, `paper_text.tex` |

`python -m lorabench paper` reproduces the results section of the article from the checkpoints:

| File | Article |
|---|---|
| `table_errors.tex` / `.csv` | Table 3: mean, median, and P90 error per scenario, split, representation, and method |
| `fig_environment_effects.png` / `.pdf` / `.csv` | Fig. 4: COMMON vs. RSSI-only by environment class |
| `fig_spatial_gap.*` | Fig. 5: error increase from the random split to the spatial holdout |
| `fig_rssi_snr.*` | Fig. 6: adding RSSI to SNR-only and SNR to RSSI-only |
| `fig_ablation.*` | Fig. 7: feature-group ablation |
| `table_sample_size.tex` / `.csv` | Table 4: sample-size control |
| `table_costs.tex` / `.csv` | Table 5: offline and online computational costs |
| `table_overhead.tex` / `.csv` | Table 6: relative computational overhead of COMMON |
| `fig_tradeoff.*` | Fig. 8: mean error vs. inference time and model size |
| `summary.txt` | numbers quoted in the text |

With `--check`, every value is compared with the published one and the differences are written to
`paper_outputs/check_report.txt`. Accuracy values and significance markers are deterministic for a given software
environment (`requirements-paper.txt` lists the versions used for the article); timing values (Tables 5 and 6,
Fig. 8) depend on the hardware and on the load of the computer, so only the article's own checkpoints reproduce them
exactly.

## Notes

* Models are selected on the validation partition only, and the selected model is applied to the test partition
  without refitting on training plus validation data. See [docs/PROTOCOL.md](docs/PROTOCOL.md).
* `ConvergenceWarning` messages from the MLP are expected and do not stop the run.
* **Threads.** Every command runs with at most `--threads` threads (default 6). The limit
  covers the whole process: RF and XGBoost (`n_jobs`), the BLAS library through which MLP and ELM are
  parallelized, and the OpenMP kernels that scikit-learn uses for the neighbor search of kNN and WkNN (these
  ignore `n_jobs`). It is set through the environment variables `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, and similar before NumPy is loaded, and enforced with `threadpoolctl`. The first line of
  every command shows the limit in force, and each run records its thread pools in `system_metadata.json`.
  Keep the same value for a whole run: it affects the measured times and can slightly change the ELM results,
  because ELM solves its output layer iteratively (LSQR) in single precision (see
  [docs/PROTOCOL.md](docs/PROTOCOL.md)). When the package is used from Python rather than from the command line,
  call `lorabench.threads.limit(6)` first.
* Do not mix caches or results of different protocol versions. Each checkpoint records its `protocol_version`.
* The analysis of the article excludes the Bucharest Outdoor scenario; the benchmark still processes it.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the three datasets (references in
[datasets/README.md](datasets/README.md)).

## License

[MIT](LICENSE). The datasets keep their own licenses.
