# Experimental protocol

This is the protocol implemented by `lorabench` (protocol version `MULTIDATASET_EQUAL_FOOTING_20SEEDS_V5`, with the
extensions `..._RSSI_IMPACT_EXTENSION_V5_1` and `..._SUBSAMPLE_CONFOUND_V5_2`). Every checkpoint records the
version that produced it.

## Scenarios

| Scenario | Deployment | Environment class (article) | Target | Error metric |
|---|---|---|---|---|
| `antwerp2019` | Antwerp | Outdoor | latitude, longitude | haversine distance (m) |
| `bonn` | Bonn | Outdoor | latitude, longitude | haversine distance (m) |
| `brno_iAG_BUT` | Brno | Indoor | local x, y (m) | Euclidean distance (m) |
| `brno_iUG_BUT` | Brno | Indoor | local x, y (m) | Euclidean distance (m) |
| `bucharest_iAG_UPB` | Bucharest | Indoor | local x, y (m) | Euclidean distance (m) |
| `bucharest_o_UPB` | Bucharest | not used in the article | local x, y (m) | Euclidean distance (m) |

The scenarios of Brno and Bucharest are run separately, so that different local coordinate systems are never mixed.
The number of gateways stays native to each deployment: "equal footing" means the same feature semantics and the
same experimental protocol, not the same number of gateways.

## Feature groups and representations

Every scenario is converted into the same five feature groups:

* `rssi`: RSSI per gateway (missing reception: -200 dBm);
* `snr`: SNR per gateway (missing reception: 0 dB);
* `esp`: estimated signal power, recomputed in every dataset as ESP = RSSI - 10 log10(1 + 10^(-SNR/10))
  (missing reception: -200 dBm);
* `rx_mask`: 1 if the gateway received the transmission, 0 otherwise;
* `aggregates`: 15 statistics of each transmission computed in the same way everywhere: number of receiving
  gateways; mean, standard deviation, minimum, and maximum of RSSI, SNR, and ESP over the receiving gateways; and the
  RSSI and SNR ranges.

Gateway coordinates and packet metadata (for example, the spreading factor) are not used, because they are not
available in an equivalent form in all datasets.

| Representation | Groups |
|---|---|
| `RSSI_ONLY` | rssi |
| `COMMON_FULL` (COMMON in the article) | rssi, snr, esp, rx_mask, aggregates |
| `SNR_ONLY` (RSSI/SNR extension) | snr |
| `RSSI_SNR` (RSSI/SNR extension) | rssi, snr |
| `DROP_SNR`, `DROP_ESP`, `DROP_SNR_ESP`, `DROP_RX_MASK`, `DROP_AGGREGATES` (ablations) | COMMON without the named group(s) |

`DROP_SNR_ESP` is included because ESP is a deterministic function of RSSI and SNR: dropping SNR alone leaves part of
its information in ESP.

## Partitions

The same 20 seeds are used everywhere: 7, 13, 42, 73, 101, 256, 314, 512, 1001, 1729, 2026, 2718, 4096, 8191, 12345,
16381, 32749, 65537, 99991, 104729. Within a scenario and seed, all representations and methods use the same indices.

* **Random split**: 80% training, 10% validation, 10% test.
* **Spatial holdout**: the target positions are projected to a metric plane when needed, ordered by a 2-D Morton
  (Z-order) code, and cut into 20 contiguous blocks with about the same number of samples; repeated measurements at
  the same position never fall in different blocks. For each seed, 16 blocks are used for training, 2 for
  validation, and 2 for test, so the proportions are again about 80/10/10 and whole blocks are held out.

## Methods and model selection

| Method | Search grid (`desktop` profile, used in the article) | Preprocessing |
|---|---|---|
| kNN, WkNN | metric (Manhattan, Euclidean, cosine) x k (1-20, 25, 30, 35, 40, 50); WkNN weights by inverse distance | standardized inputs |
| Random Forest | 5 configurations (200-500 trees, depth 15, 20, 25, or unlimited) | none |
| XGBoost | 13 configurations (200-700 trees, depth 4-10, learning rate 0.10-0.02), subsample and column subsample 0.9 | none |
| MLP | 4 architectures (two or three hidden layers, early stopping) | standardized inputs and targets |
| ELM | 17 configurations (200-3,000 hidden units; tanh, ReLU, or sigmoid; ridge output layer) | standardized inputs and targets |

The exact grids are in `lorabench/core.py` (function `grid`). The `quick` profile uses smaller grids for tests only.

* Every configuration is trained on the training partition and scored by its mean error on the validation partition.
* The configuration with the lowest validation mean error is selected, and **the model already trained** is applied to
  the test partition: there is no refit on training plus validation data.
* All scalers are fitted on the training partition only.
* Ablations do not search again: each `DROP_*` variant reuses the hyperparameters selected for `COMMON_FULL` in the
  same scenario, seed, and split, and is trained on the same training partition and tested on the same test partition.

## Measured quantities

For every case (scenario, split, seed, representation, method) the checkpoint stores the test errors and
predictions, the validation results of every configuration, and:

* mean, median, P75, P90, P95, P99, and maximum error, and the share of errors below 5 m to 1,000 m;
* search time (the whole validation search), final fit time (fit of the selected configuration plus preprocessing);
* inference time per test sample: median of 5 timed repetitions after one warm-up call (`quick`: 2 repetitions);
* model size: size of the pickled model and scalers, or, for kNN and WkNN, the stored reference data and scaler.

Timing values depend on the computer, on its load, and on the number of threads. Every thread pool of the process
(joblib, BLAS, and OpenMP) is limited to `--threads` threads (default 6); see `lorabench/threads.py`. In tests with one
and two threads, kNN, WkNN, XGBoost, and MLP gave identical errors. RF sums the predictions of its trees in
parallel, so its errors can differ in the last digits (about 1e-12 relative). ELM solves its ridge output layer
with LSQR (tolerance 1e-4) in single precision; when BLAS splits the matrix-vector products among threads, their
rounding depends on the number of threads, and the ELM errors can change slightly. The limit removes the
dependence on the number of cores of the computer; ELM results can still differ slightly across processor types
and library versions.

## Statistics computed by the benchmark (`run --task aggregate`)

Within each scenario: paired differences over the 20 seeds per method, two-sided Wilcoxon signed-rank tests, Holm
correction across the six methods for RSSI-only vs. COMMON, ablation effects, and the random-vs-spatial gap.
Across deployments (`aggregate_equal_weight/`): relative effects per scenario and seed, averaged within Brno and within
Bucharest at the seed level, so that Antwerp, Bonn, Brno, and Bucharest receive equal weight. Absolute errors in meters
are never pooled across datasets.

## Analysis of the article (`paper`)

The article groups the scenarios into two environment classes, Outdoor (Antwerp 2019 and Bonn) and Indoor (Brno iAG,
Brno iUG, and Bucharest iAG). For each comparison, the paired effect is computed per scenario and seed; the class
value of a seed is the unweighted mean over the scenarios of the class; the 20 seed values are tested with a two-sided
Wilcoxon signed-rank test, and the p-values are corrected with the Holm method.

| Output | Effect per scenario and seed | Reported value | Holm family |
|---|---|---|---|
| Fig. 4 | 100 (E_RSSI - E_COMMON) / E_RSSI | mean over seeds | six methods, per class and split |
| Fig. 5 | 100 (E_spatial - E_random) / E_random | mean over seeds | (no test) |
| Fig. 6 | 100 (E_ref - E_cmp) / E_ref, with SNR-only -> RSSI+SNR and RSSI-only -> RSSI+SNR | mean over seeds | six methods, per class, split, and comparison |
| Fig. 7 | 100 (E_drop - E_COMMON) / E_COMMON | mean over seeds | six methods, per class, split, and dropped group |
| Table 4 | as Fig. 4, per scenario and size (no class mean) | mean over seeds | six methods, per column and split |
| Table 5 | median over seeds per scenario and split | mean over the scenarios of the class, then over the two splits | (no test) |
| Table 6 | ratio COMMON / RSSI-only of each cost | median over seeds of the class mean, averaged over the two splits | six methods, per class, split, and cost; test on ratio - 1 |
| Fig. 8 | random split: class mean of the scenario mean errors and of the per-scenario medians of inference time and model size | | (no test) |

E is the mean test error of the case. In Table 6, `*` marks significance in both splits and `r` in the random split
only.

## RSSI/SNR extension (`rssi-impact`)

`SNR_ONLY` and `RSSI_SNR` are trained with the same splits, grids, and selection rule, and compared with the existing
`RSSI_ONLY` checkpoints. No case of the main benchmark is rerun.

## Sample-size control (`subsample`)

Antwerp 2019 and Bonn are subsampled to the sizes of the indoor scenarios (572 = Brno iUG, 930 = Bucharest iAG,
2,436 = Brno iAG) and run with the same protocol (20 seeds, random split and 20-block spatial holdout), for RSSI-only and
COMMON only. Each seed draws its own subsample, which is shared by both representations and both splits, so the
comparison stays paired (`--fixed-subset` uses one subsample per size for all seeds). The full-size results are only
read, as the reference.
