#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_pipeline_metrics.py
===========================
Pipeline-level metrics for protocol ablation.

Calibration metric (RMSE) uses the M11 moving observation vector.
Validation metrics (KS, worst-window) use the full PM-peak link set
after Rule C filtering, and operate on time-series samples.

Optional next-day transfer metrics compare against Day2 1h windows
with fixed Rule C thresholds.
"""

import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "calibration"))

from build_l2_sim_vector_traveltime import build_simulation_vector_tt, load_stop_seq_mapping
from metrics_v3 import (
    RULE_C_T_CRITICAL,
    RULE_C_SPEED_KMH,
    RULE_C_MAX_DIST_M,
    SUBWINDOW_DURATION_SEC,
    SUBWINDOW_STEP_SEC
)


L2_PROTOCOL = {
    "calib_obs_file": PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector_corridor_M11_moving_irn.csv",
    "dist_csv": PROJECT_ROOT / "data" / "processed" / "kmb_route_stop_dist.csv",
    "tt_mode": "moving",
    "t_min": 0.0,
    "t_max": 3600.0,
    "window_duration_sec": 3600,
    "subwindow_duration_sec": SUBWINDOW_DURATION_SEC,
    "subwindow_step_sec": SUBWINDOW_STEP_SEC,
    "tz_offset_hours": 8,
    "validation_stats_csv": PROJECT_ROOT / "data" / "processed" / "link_stats.csv",
    "validation_speeds_csv": PROJECT_ROOT / "data" / "processed" / "link_speeds.csv",
    "validation_start_hour_hkt": 17,
    "transfer_stats_csv": PROJECT_ROOT / "data2" / "processed" / "link_stats_offpeak.csv",
    "transfer_speeds_csv": PROJECT_ROOT / "data2" / "processed" / "link_speeds_offpeak.csv",
    "transfer_start_hour_hkt": 15
}

_printed_sample_sizes = False


def compute_rmse(sim_values: np.ndarray, obs_values: np.ndarray) -> Optional[float]:
    valid_mask = ~np.isnan(sim_values) & ~np.isnan(obs_values)
    if valid_mask.sum() < 3:
        return None
    return float(np.sqrt(np.mean((sim_values[valid_mask] - obs_values[valid_mask]) ** 2)))


def compute_ks_statistic(real_values: np.ndarray, sim_values: np.ndarray) -> Optional[float]:
    real_valid = real_values[~np.isnan(real_values)]
    sim_valid = sim_values[~np.isnan(sim_values)]
    if len(real_valid) < 3 or len(sim_valid) < 3:
        return None
    ks_stat, _ = ks_2samp(real_valid, sim_valid)
    return float(ks_stat)


def compute_w1_statistic(real_values: np.ndarray, sim_values: np.ndarray) -> Optional[float]:
    real_valid = real_values[~np.isnan(real_values)]
    sim_valid = sim_values[~np.isnan(sim_values)]
    if len(real_valid) < 3 or len(sim_valid) < 3:
        return None
    w1_stat = wasserstein_distance(real_valid, sim_valid)
    return float(w1_stat)


def format_window_time(start_sec: int) -> str:
    hours = start_sec // 3600
    minutes = (start_sec % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def compute_worst_window_ks_by_time(
    real_values: np.ndarray,
    real_times: np.ndarray,
    sim_values: np.ndarray,
    sim_times: np.ndarray,
    total_duration_sec: int,
    window_duration_sec: int,
    step_sec: int,
    min_samples: int = 5
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    if len(real_values) < min_samples or len(sim_values) < min_samples:
        return None, None, None

    worst_ks = None
    worst_start = None
    worst_end = None

    for start_sec in range(0, total_duration_sec - window_duration_sec + 1, step_sec):
        end_sec = start_sec + window_duration_sec

        real_mask = (real_times >= start_sec) & (real_times < end_sec)
        sim_mask = (sim_times >= start_sec) & (sim_times < end_sec)
        real_win = real_values[real_mask]
        sim_win = sim_values[sim_mask]

        if len(real_win) < min_samples or len(sim_win) < min_samples:
            continue

        ks_stat = compute_ks_statistic(real_win, sim_win)
        if ks_stat is None:
            continue

        if worst_ks is None or ks_stat > worst_ks:
            worst_ks = ks_stat
            worst_start = start_sec
            worst_end = end_sec

    if worst_ks is None:
        return None, None, None

    return worst_ks, format_window_time(worst_start), format_window_time(worst_end)


def compute_worst_window_w1_by_time(
    real_values: np.ndarray,
    real_times: np.ndarray,
    sim_values: np.ndarray,
    sim_times: np.ndarray,
    total_duration_sec: int,
    window_duration_sec: int,
    step_sec: int,
    min_samples: int = 5
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    if len(real_values) < min_samples or len(sim_values) < min_samples:
        return None, None, None

    worst_w1 = None
    worst_start = None
    worst_end = None

    for start_sec in range(0, total_duration_sec - window_duration_sec + 1, step_sec):
        end_sec = start_sec + window_duration_sec

        real_mask = (real_times >= start_sec) & (real_times < end_sec)
        sim_mask = (sim_times >= start_sec) & (sim_times < end_sec)
        real_win = real_values[real_mask]
        sim_win = sim_values[sim_mask]

        if len(real_win) < min_samples or len(sim_win) < min_samples:
            continue

        w1_stat = compute_w1_statistic(real_win, sim_win)
        if w1_stat is None:
            continue

        if worst_w1 is None or w1_stat > worst_w1:
            worst_w1 = w1_stat
            worst_start = start_sec
            worst_end = end_sec

    if worst_w1 is None:
        return None, None, None

    return worst_w1, format_window_time(worst_start), format_window_time(worst_end)


def apply_rule_c_filter(
    df_stats: pd.DataFrame,
    t_star: float = RULE_C_T_CRITICAL,
    v_star: float = RULE_C_SPEED_KMH,
    max_dist_m: float = RULE_C_MAX_DIST_M
) -> Tuple[pd.DataFrame, float]:
    cond_ghost = (
        (df_stats["tt_median"] >= t_star) &
        (df_stats["speed_median"] <= v_star) &
        (df_stats["dist_m"] <= max_dist_m)
    )
    flagged_fraction = cond_ghost.sum() / len(df_stats) if len(df_stats) > 0 else 0.0
    clean_df = df_stats.loc[~cond_ghost].copy()
    return clean_df, float(flagged_fraction)


def build_link_key_set(df: pd.DataFrame) -> set:
    keys = df[["route", "bound", "from_seq", "to_seq"]].copy()
    keys["route"] = keys["route"].astype(str)
    keys["bound"] = keys["bound"].astype(str)
    return set(tuple(row) for row in keys.to_numpy())


def load_real_timeseries(
    link_speeds_csv: Path,
    link_keys: set,
    window_start_hour_hkt: int,
    window_duration_sec: int,
    tz_offset_hours: int
) -> pd.DataFrame:
    if not link_speeds_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(link_speeds_csv)
    if df.empty:
        return df

    df["route"] = df["route"].astype(str)
    df["bound"] = df["bound"].astype(str)
    df = df[df[["route", "bound", "from_seq", "to_seq"]].apply(tuple, axis=1).isin(link_keys)]

    if df.empty:
        return df

    dep_ts = pd.to_datetime(df["departure_ts"], utc=True, errors="coerce")
    local_ts = dep_ts + pd.Timedelta(hours=tz_offset_hours)
    local_ts = local_ts.dt.tz_localize(None)
    df["local_departure"] = local_ts

    window_start = local_ts.dt.floor("D").iloc[0] + pd.Timedelta(hours=window_start_hour_hkt)
    window_end = window_start + pd.Timedelta(seconds=window_duration_sec)

    df = df[(df["local_departure"] >= window_start) & (df["local_departure"] < window_end)].copy()
    if df.empty:
        return df

    df["event_time_sec"] = (df["local_departure"] - window_start).dt.total_seconds()
    return df


def load_sim_timeseries(
    stopinfo_path: Path,
    route_stop_csv: Path,
    t_min: float,
    t_max: float,
    tt_mode: str = "moving",
    max_gap_seconds: float = 1800.0
) -> pd.DataFrame:
    if not stopinfo_path.exists():
        return pd.DataFrame()

    stop_mapping = load_stop_seq_mapping(str(route_stop_csv))
    if not stop_mapping:
        return pd.DataFrame()

    tree = ET.parse(stopinfo_path)
    root = tree.getroot()

    vehicle_stops = defaultdict(list)
    for stopinfo in root.findall("stopinfo"):
        vehicle_id = stopinfo.get("id")
        bus_stop = stopinfo.get("busStop")
        started = float(stopinfo.get("started"))
        ended = float(stopinfo.get("ended"))
        arrival_str = stopinfo.get("arrival")
        arrival = float(arrival_str) if arrival_str else started

        if bus_stop in stop_mapping:
            vehicle_stops[vehicle_id].append({
                "busStop": bus_stop,
                "started": started,
                "ended": ended,
                "arrival": arrival,
                "mappings": stop_mapping[bus_stop]
            })

    records = []

    for vehicle_id, stops in vehicle_stops.items():
        stops_sorted = sorted(stops, key=lambda x: x["started"])
        parts = vehicle_id.split("_")
        if len(parts) < 3:
            continue

        veh_route = parts[1]
        veh_bound = parts[2].split(".")[0]

        for i in range(len(stops_sorted) - 1):
            curr_stop = stops_sorted[i]
            next_stop = stops_sorted[i + 1]

            if tt_mode == "door":
                travel_time = next_stop["ended"] - curr_stop["ended"]
            else:
                travel_time = next_stop["started"] - curr_stop["ended"]

            event_time = next_stop["started"]
            if t_min is not None and event_time < t_min:
                continue
            if t_max is not None and event_time > t_max:
                continue

            if travel_time <= 0 or travel_time > max_gap_seconds:
                continue

            curr_matches = [m for m in curr_stop["mappings"] if m[0] == veh_route and m[1] == veh_bound]
            next_matches = [m for m in next_stop["mappings"] if m[0] == veh_route and m[1] == veh_bound]
            if not curr_matches or not next_matches:
                continue

            from_seq = curr_matches[0][2]
            cum_dist_curr = curr_matches[0][3]
            to_seq = next_matches[0][2]
            cum_dist_next = next_matches[0][3]
            if to_seq != from_seq + 1:
                continue

            dist_m = cum_dist_next - cum_dist_curr
            if dist_m <= 0:
                dist_m = 500.0

            speed_kmh = (dist_m / 1000.0) / (travel_time / 3600.0)

            records.append({
                "route": veh_route,
                "bound": veh_bound,
                "from_seq": from_seq,
                "to_seq": to_seq,
                "travel_time_s": travel_time,
                "speed_kmh": speed_kmh,
                "event_time_sec": event_time
            })

    return pd.DataFrame(records)


def extract_m11_rmse(
    stopinfo_path: Path,
    obs_csv: Path,
    dist_csv: Path,
    real_speeds_csv: Optional[Path],
    window_start_hour_hkt: int,
    window_duration_sec: int,
    tz_offset_hours: int
) -> Tuple[Optional[float], Optional[float]]:
    obs_df = pd.read_csv(obs_csv)
    obs_speeds = obs_df["mean_speed_kmh"].to_numpy()

    sim_df = build_simulation_vector_tt(
        stopinfo_path=str(stopinfo_path),
        observation_csv=str(obs_csv),
        route_stop_csv=str(dist_csv),
        output_csv=None,
        max_gap_seconds=1800.0,
        verbose=False,
        tmin=L2_PROTOCOL["t_min"],
        tmax=L2_PROTOCOL["t_max"],
        tt_mode=L2_PROTOCOL["tt_mode"]
    )

    sim_speeds = sim_df["sim_speed_kmh"].to_numpy() if "sim_speed_kmh" in sim_df.columns else np.full(len(obs_speeds), np.nan)
    rmse_speed = compute_rmse(sim_speeds, obs_speeds)

    rmse_tt = None
    if real_speeds_csv and real_speeds_csv.exists():
        m11_keys = build_link_key_set(obs_df)
        real_df = load_real_timeseries(
            real_speeds_csv,
            m11_keys,
            window_start_hour_hkt,
            window_duration_sec,
            tz_offset_hours
        )
        if not real_df.empty and "travel_time_sim_s" in sim_df.columns:
            real_tt = (
                real_df.groupby(["route", "bound", "from_seq", "to_seq"])["travel_time_s"]
                .mean()
                .reset_index()
            )
            merged = obs_df.merge(real_tt, on=["route", "bound", "from_seq", "to_seq"], how="left")
            obs_tt = merged["travel_time_s"].to_numpy()
            sim_tt = sim_df["travel_time_sim_s"].to_numpy()
            rmse_tt = compute_rmse(sim_tt, obs_tt)

    return rmse_speed, rmse_tt


def prepare_real_window(
    stats_csv: Path,
    speeds_csv: Path,
    start_hour_hkt: int,
    window_duration_sec: int,
    tz_offset_hours: int
) -> Dict:
    df_stats = pd.read_csv(stats_csv)
    clean_df, flagged_fraction = apply_rule_c_filter(df_stats)
    link_keys = build_link_key_set(clean_df)

    real_df = load_real_timeseries(
        speeds_csv,
        link_keys,
        start_hour_hkt,
        window_duration_sec,
        tz_offset_hours
    )

    return {
        "link_keys": link_keys,
        "real_speed": real_df["speed_kmh"].to_numpy() if not real_df.empty else np.array([]),
        "real_tt": real_df["travel_time_s"].to_numpy() if not real_df.empty else np.array([]),
        "real_time": real_df["event_time_sec"].to_numpy() if not real_df.empty else np.array([]),
        "n_real_links": len(clean_df),
        "flagged_fraction": flagged_fraction
    }


def compute_validation_metrics(
    stopinfo_path: Path,
    real_window: Dict,
    dist_csv: Path
) -> Dict:
    global _printed_sample_sizes

    sim_df = load_sim_timeseries(
        stopinfo_path,
        dist_csv,
        t_min=L2_PROTOCOL["t_min"],
        t_max=L2_PROTOCOL["t_max"],
        tt_mode=L2_PROTOCOL["tt_mode"]
    )
    if sim_df.empty:
        return {
            "ks_speed": None,
            "ks_tt": None,
            "W1_speed": None,
            "W1_TT": None,
            "worst_ks_speed": None,
            "worst_ks_tt": None,
            "worst_W1_speed": None,
            "worst_W1_TT": None,
            "worst_window_speed": None,
            "worst_window_tt": None,
            "n_real_samples": len(real_window["real_speed"]),
            "n_sim_samples": 0
        }

    sim_df["route"] = sim_df["route"].astype(str)
    sim_df["bound"] = sim_df["bound"].astype(str)
    sim_df = sim_df[sim_df[["route", "bound", "from_seq", "to_seq"]].apply(tuple, axis=1).isin(real_window["link_keys"])]

    sim_speed = sim_df["speed_kmh"].to_numpy() if not sim_df.empty else np.array([])
    sim_tt = sim_df["travel_time_s"].to_numpy() if not sim_df.empty else np.array([])
    sim_time = sim_df["event_time_sec"].to_numpy() if not sim_df.empty else np.array([])

    if not _printed_sample_sizes:
        print(
            f"Sanity check: len(real_speed_clean)={len(real_window['real_speed'])}, "
            f"len(sim_speed)={len(sim_speed)}"
        )
        _printed_sample_sizes = True

    ks_speed = compute_ks_statistic(real_window["real_speed"], sim_speed)
    ks_tt = compute_ks_statistic(real_window["real_tt"], sim_tt)
    w1_speed = compute_w1_statistic(real_window["real_speed"], sim_speed)
    w1_tt = compute_w1_statistic(real_window["real_tt"], sim_tt)

    worst_ks_speed, worst_start_speed, worst_end_speed = compute_worst_window_ks_by_time(
        real_window["real_speed"],
        real_window["real_time"],
        sim_speed,
        sim_time,
        L2_PROTOCOL["window_duration_sec"],
        L2_PROTOCOL["subwindow_duration_sec"],
        L2_PROTOCOL["subwindow_step_sec"]
    )
    worst_ks_tt, worst_start_tt, worst_end_tt = compute_worst_window_ks_by_time(
        real_window["real_tt"],
        real_window["real_time"],
        sim_tt,
        sim_time,
        L2_PROTOCOL["window_duration_sec"],
        L2_PROTOCOL["subwindow_duration_sec"],
        L2_PROTOCOL["subwindow_step_sec"]
    )
    worst_w1_speed, _, _ = compute_worst_window_w1_by_time(
        real_window["real_speed"],
        real_window["real_time"],
        sim_speed,
        sim_time,
        L2_PROTOCOL["window_duration_sec"],
        L2_PROTOCOL["subwindow_duration_sec"],
        L2_PROTOCOL["subwindow_step_sec"]
    )
    worst_w1_tt, _, _ = compute_worst_window_w1_by_time(
        real_window["real_tt"],
        real_window["real_time"],
        sim_tt,
        sim_time,
        L2_PROTOCOL["window_duration_sec"],
        L2_PROTOCOL["subwindow_duration_sec"],
        L2_PROTOCOL["subwindow_step_sec"]
    )

    return {
        "ks_speed": ks_speed,
        "ks_tt": ks_tt,
        "W1_speed": w1_speed,
        "W1_TT": w1_tt,
        "worst_ks_speed": worst_ks_speed,
        "worst_ks_tt": worst_ks_tt,
        "worst_W1_speed": worst_w1_speed,
        "worst_W1_TT": worst_w1_tt,
        "worst_window_speed": f"{worst_start_speed}-{worst_end_speed}" if worst_start_speed else None,
        "worst_window_tt": f"{worst_start_tt}-{worst_end_tt}" if worst_start_tt else None,
        "n_real_samples": len(real_window["real_speed"]),
        "n_sim_samples": len(sim_speed)
    }


def compute_metrics_for_run(
    stopinfo_path: Path,
    real_window: Dict,
    dist_csv: Path,
    transfer_window: Optional[Dict] = None,
    transfer_stopinfo_path: Optional[Path] = None
) -> Dict:
    rmse_speed, rmse_tt = extract_m11_rmse(
        stopinfo_path,
        L2_PROTOCOL["calib_obs_file"],
        dist_csv,
        L2_PROTOCOL["validation_speeds_csv"],
        L2_PROTOCOL["validation_start_hour_hkt"],
        L2_PROTOCOL["window_duration_sec"],
        L2_PROTOCOL["tz_offset_hours"]
    )

    validation_metrics = compute_validation_metrics(stopinfo_path, real_window, dist_csv)

    transfer_metrics = {
        "next_day_ks_speed": None,
        "next_day_ks_tt": None
    }
    if transfer_window and transfer_stopinfo_path and transfer_stopinfo_path.exists():
        transfer_validation = compute_validation_metrics(transfer_stopinfo_path, transfer_window, dist_csv)
        transfer_metrics["next_day_ks_speed"] = transfer_validation["ks_speed"]
        transfer_metrics["next_day_ks_tt"] = transfer_validation["ks_tt"]

    return {
        "rmse_speed": rmse_speed,
        "rmse_tt": rmse_tt,
        **validation_metrics,
        **transfer_metrics
    }


def compute_metrics_for_all_configs(
    base_dir: Path,
    config_ids: List[str],
    seeds: List[int],
    output_csv: Path,
    transfer_base_dir: Optional[Path] = None,
    transfer_window: Optional[Dict] = None
) -> pd.DataFrame:
    real_window = prepare_real_window(
        L2_PROTOCOL["validation_stats_csv"],
        L2_PROTOCOL["validation_speeds_csv"],
        L2_PROTOCOL["validation_start_hour_hkt"],
        L2_PROTOCOL["window_duration_sec"],
        L2_PROTOCOL["tz_offset_hours"]
    )

    results = []
    for config_id in config_ids:
        for seed in seeds:
            stopinfo_path = base_dir / config_id / f"seed{seed}" / "stopinfo.xml"
            if not stopinfo_path.exists():
                print(f"[{config_id}] seed={seed}: stopinfo.xml missing, skip")
                continue

            transfer_stopinfo_path = None
            if transfer_base_dir:
                transfer_stopinfo_path = transfer_base_dir / config_id / f"seed{seed}" / "stopinfo.xml"

            metrics = compute_metrics_for_run(
                stopinfo_path,
                real_window,
                L2_PROTOCOL["dist_csv"],
                transfer_window=transfer_window,
                transfer_stopinfo_path=transfer_stopinfo_path
            )

            results.append({
                "config_id": config_id,
                "seed": seed,
                **metrics
            })

            ks_speed = metrics.get("ks_speed")
            print(f"[{config_id}] seed={seed}: KS(speed)={ks_speed:.4f}" if ks_speed is not None else f"[{config_id}] seed={seed}: KS(speed)=N/A")

    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved metrics: {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Compute pipeline metrics (RMSE, KS, worst-window, transfer).")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "experiments_v4" / "unified_l2" / "protocol_ablation"),
        help="Protocol ablation output root"
    )
    parser.add_argument(
        "--config-ids",
        type=str,
        nargs="+",
        default=["A0", "A2", "A3", "A4"],
        help="Config IDs"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "experiments_v4" / "unified_l2" / "protocol_ablation" / "full_metrics.csv"),
        help="Output metrics CSV"
    )
    parser.add_argument(
        "--transfer-base-dir",
        type=str,
        default=None,
        help="Optional Day2 stopinfo root (same config_id/seed layout)"
    )
    parser.add_argument(
        "--transfer-stats-csv",
        type=str,
        default=str(L2_PROTOCOL["transfer_stats_csv"]),
        help="Day2 link_stats CSV for Rule C"
    )
    parser.add_argument(
        "--transfer-speeds-csv",
        type=str,
        default=str(L2_PROTOCOL["transfer_speeds_csv"]),
        help="Day2 link_speeds CSV"
    )
    parser.add_argument(
        "--transfer-start-hour-hkt",
        type=int,
        default=L2_PROTOCOL["transfer_start_hour_hkt"],
        help="Day2 window start hour (HKT)"
    )

    args = parser.parse_args()

    transfer_window = None
    if args.transfer_base_dir:
        transfer_window = prepare_real_window(
            Path(args.transfer_stats_csv),
            Path(args.transfer_speeds_csv),
            args.transfer_start_hour_hkt,
            L2_PROTOCOL["window_duration_sec"],
            L2_PROTOCOL["tz_offset_hours"]
        )

    compute_metrics_for_all_configs(
        base_dir=Path(args.base_dir),
        config_ids=args.config_ids,
        seeds=args.seeds,
        output_csv=Path(args.output_csv),
        transfer_base_dir=Path(args.transfer_base_dir) if args.transfer_base_dir else None,
        transfer_window=transfer_window
    )


if __name__ == "__main__":
    main()
