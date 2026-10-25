"""Small synthetic stand-ins for the three raw datasets, in their original file formats.

Used by `python -m lorabench selftest` to exercise every stage of the suite in a few minutes without
downloading the real data. The numbers produced from synthetic data are meaningless.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np


def _rssi(dist_m: np.ndarray, rng: np.random.Generator, p0: float = -60.0, n_exp: float = 2.7) -> np.ndarray:
    return p0 - 10 * n_exp * np.log10(np.maximum(dist_m, 1.0)) + rng.normal(0, 4, size=dist_m.shape)


def _snr(rssi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(0.4 * (rssi + 112) + rng.normal(0, 1.5, size=rssi.shape), -20, 12)


def write_antwerp(data_dir: Path, n: int, rng: np.random.Generator) -> Path:
    gw_lat = 51.21 + rng.uniform(-0.03, 0.03, 8)
    gw_lon = 4.40 + rng.uniform(-0.05, 0.05, 8)
    # repeated transmissions at a limited set of positions, as in drive-test data
    pos_lat = 51.21 + rng.uniform(-0.03, 0.03, n // 2)
    pos_lon = 4.40 + rng.uniform(-0.05, 0.05, n // 2)
    recs = []
    for i in range(n):
        j = i % len(pos_lat)
        lat, lon = pos_lat[j] + rng.normal(0, 1e-5), pos_lon[j] + rng.normal(0, 1e-5)
        d = np.hypot((gw_lat - lat) * 111_000, (gw_lon - lon) * 70_000)
        r = _rssi(d, rng, p0=-40.0, n_exp=2.2)
        heard = np.where(r > -118)[0]
        if len(heard) == 0:
            heard = np.array([int(np.argmax(r))])
        s = _snr(r, rng)
        recs.append({"latitude": float(lat), "longitude": float(lon), "sf": int(rng.integers(7, 13)),
                     "gateways": [{"id": f"GW{k:02d}", "rssi": float(round(r[k])), "snr": float(round(s[k], 1))} for k in heard]})
    p = Path(data_dir) / "3904158.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lorawan_antwerp_2019_dataset.json.txt", json.dumps(recs))
    return p


def write_bonn(data_dir: Path, n_packets: int, rng: np.random.Generator) -> Path:
    gw_lat = 50.73 + rng.uniform(-0.02, 0.02, 5)
    gw_lon = 7.10 + rng.uniform(-0.03, 0.03, 5)
    lines = ["transceived_at,latitude,longitude,pkt_number,gw,rssi,snr"]
    for i in range(n_packets):
        lat = 50.73 + rng.uniform(-0.02, 0.02); lon = 7.10 + rng.uniform(-0.03, 0.03)
        d = np.hypot((gw_lat - lat) * 111_000, (gw_lon - lon) * 70_000)
        r = _rssi(d, rng, p0=-40.0, n_exp=2.2); s = _snr(r, rng)
        heard = np.where(r > -120)[0]
        if len(heard) == 0:
            heard = np.array([int(np.argmax(r))])
        day = f"2021-03-{1 + i % 28:02d}"
        for k in heard:
            lines.append(f"{day},{lat:.6f},{lon:.6f},{i},gw{k},{r[k]:.1f},{s[k]:.1f}")
    p = Path(data_dir) / "lora-bonn-main.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lora-bonn-main/survey/samples.csv", "\n".join(lines) + "\n")
    return p


def _scenario_workbook(n: int, n_gw: int, size_m: float, rng: np.random.Generator) -> bytes:
    from openpyxl import Workbook
    gx = rng.uniform(0, size_m, n_gw); gy = rng.uniform(0, size_m / 2, n_gw)
    px = rng.uniform(0, size_m, max(25, n // 4)); py = rng.uniform(0, size_m / 2, max(25, n // 4))
    wb = Workbook(); ws = wb.active
    ws.append(["id", "x", "y", "message"])
    for i in range(n):
        j = i % len(px)
        d = np.hypot(gx - px[j], gy - py[j])
        r = _rssi(d, rng, p0=-55.0, n_exp=3.2); s = _snr(r, rng)
        heard = np.where(r > -110)[0]
        if len(heard) == 0:
            heard = np.array([int(np.argmax(r))])
        msg = {"uplink_message": {"rx_metadata": [
            {"gateway_ids": {"eui": f"A84041FFFF{k:06X}"}, "rssi": float(round(r[k])), "snr": float(round(s[k], 1))}
            for k in heard]}}
        ws.append([i, float(round(px[j], 2)), float(round(py[j], 2)), json.dumps(msg)])
    buf = io.BytesIO(); wb.save(buf)
    return buf.getvalue()


def write_brno_bucharest(data_dir: Path, sizes: dict, rng: np.random.Generator) -> Path:
    root = "LoRaWAN datasets 2021-2022"
    folders = {"brno_iAG_BUT": "iAG_BUT/1. Raw option", "brno_iUG_BUT": "iUG_BUT/Bottom floor/1. Raw option",
               "bucharest_iAG_UPB": "iAG_UPB/1. Raw option", "bucharest_o_UPB": "o_UPB/1. Raw option"}
    p = Path(data_dir) / "LoRaWAN datasets 2021-2022-20221021T204201Z-001.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for scen, folder in folders.items():
            n_gw = {"brno_iAG_BUT": 7, "brno_iUG_BUT": 6}.get(scen, 9)
            zf.writestr(f"{root}/{folder}/measurements.xlsx", _scenario_workbook(sizes[scen], n_gw, 60.0, rng))
    return p


def write_all(data_dir: Path, seed: int = 20261024) -> None:
    """Writes the three synthetic ZIP files with the names the adapters look for."""
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    write_antwerp(data_dir, 700, rng)
    write_bonn(data_dir, 500, rng)
    write_brno_bucharest(data_dir, {"brno_iAG_BUT": 260, "brno_iUG_BUT": 140,
                                    "bucharest_iAG_UPB": 180, "bucharest_o_UPB": 180}, rng)
