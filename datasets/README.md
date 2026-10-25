# Datasets

The three public datasets are not redistributed with this code. Download them into this folder; the file names may
carry a suffix such as `(1)`, because the adapters find each file by a fragment of its name.

| Deployment | File placed in `datasets/` | Source |
|---|---|---|
| Antwerp 2019 | `3904158.zip` (the ZIP of the Zenodo record), or the file `lorawan_antwerp_2019_dataset.json.txt` from it | Zenodo record 3904158, version 1.3: <https://doi.org/10.5281/zenodo.3904158> |
| Bonn | `lora-bonn-main.zip` or `lora-bonn.zip` (any ZIP whose name contains `lora-bonn`) | GitHub repository `mclab-hbrs/lora-bonn`, *Code > Download ZIP*; the article used commit `74d9e29`: <https://github.com/mclab-hbrs/lora-bonn> |
| Brno and Bucharest | `LoRaWAN datasets 2021-2022-20221021T204201Z-001.zip` (any ZIP whose name contains `LoRaWAN datasets 2021-2022`) | Zenodo record 7236698: <https://zenodo.org/records/7236698> |

## What is read from each file

* **Antwerp 2019**: `lorawan_antwerp_2019_dataset.json.txt` (one record per transmission with latitude, longitude,
  spreading factor, and the RSSI and SNR of every receiving gateway). All 130,430 valid transmissions and all
  spreading factors (7 to 12) are used, with the 44 gateways that appear in the data.
* **Bonn**: `survey/samples.csv` inside the repository ZIP. Receptions with the same date, GPS coordinates, and
  packet number are grouped into one fingerprint (56,540 fingerprints, 9 gateways).
* **Brno and Bucharest**: the `1. Raw option` spreadsheets of the four scenarios (`iAG_BUT`, `iUG_BUT` bottom floor,
  `iAG_UPB`, and `o_UPB`). Coordinates are in the local Cartesian system of the dataset, in meters.

## Expected audit

`python -m lorabench prepare` builds `cache_equal_footing_v5/` and prints, for every scenario, the number of
samples and gateways, the dimension of each representation, and the spatial blocks. With the files above:

| Scenario | Samples | Gateways | RSSI-only | COMMON |
|---|---:|---:|---:|---:|
| antwerp2019 | 130,430 | 44 | 44 | 191 |
| bonn | 56,540 | 9 | 9 | 51 |
| brno_iAG_BUT | 2,436 | 7 | 7 | 43 |
| brno_iUG_BUT | 572 | 6 | 6 | 39 |
| bucharest_iAG_UPB | 930 | 11 | 11 | 59 |
| bucharest_o_UPB | 930 | 11 | 11 | 59 |

Every scenario must have 20 spatial blocks, and the audit must end with `COMMON PROTOCOL COMPATIBLE: True`.

## References

* M. Aernouts, R. Berkvens, K. Van Vlaenderen, and M. Weyn, "Sigfox and LoRaWAN datasets for fingerprint
  localization in large urban and rural areas," *Data*, vol. 3, no. 2, p. 13, 2018; dataset version 1.3,
  Zenodo, 2019, doi:10.5281/zenodo.3904158.
* M. Rademacher, H. Linka, T. Horstmann, and M. Henze, "Path loss in urban LoRa networks: A large-scale
  measurement study," in *Proc. IEEE 94th Vehicular Technology Conference (VTC2021-Fall)*, 2021, pp. 1-6,
  doi:10.1109/VTC2021-Fall52928.2021.9625531.
* E. Svertoka, A. Rusu-Casandra, R. Burget, I. Marghescu, J. Hosek, and A. Ometov, "LoRaWAN: Lost for
  localization?," *IEEE Sensors Journal*, vol. 22, no. 23, pp. 23307-23319, 2022, doi:10.1109/JSEN.2022.3212319.
