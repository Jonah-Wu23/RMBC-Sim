#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_dapper_l2_baselines.py
==========================
DAPPER-based L2 baselines with SUMO-backed observation operator.

Aligned L2 definition (paper):
  - State x = [capacityFactor, minGap, impatience]
  - Observation y = 11-link corridor moving speeds (traffic-only)
  - IES settings: Ne=10, K=3, damping beta=0.3
  - Rule C thresholds fixed (used only for KS/worst-window helpers)

Outputs:
  - results.csv: per-method per-seed metrics
  - summary.csv: method-level mean/std summary
  - tables/dapper_l2_baselines.md: paper-ready table

Note:
  This script uses real SUMO runs via the existing IESLoop pipeline,
  but the DA updates come from DAPPER (EnKF/EnKS/iEnKS).
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local paths
sys_paths = [
    PROJECT_ROOT,
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "scripts" / "calibration",
    PROJECT_ROOT / "scripts" / "eval",
    PROJECT_ROOT / "DAPPER-master",
]
import sys
for p in sys_paths:
    sys.path.insert(0, str(p))

import struct_tools

# DAPPER expects NicePrint.printopts["excluded"] to be a list (struct_tools>=0.2 uses a set).
_excluded = struct_tools.NicePrint.printopts.get("excluded", [])
if isinstance(_excluded, set):
    struct_tools.NicePrint.printopts["excluded"] = list(_excluded)

import dapper.da_methods as da
import dapper.mods as modelling
from dapper.tools.chronos import Chronology
from dapper.tools.matrices import CovMat
from dapper.tools.randvars import GaussRV
from dapper.tools.seeding import set_seed

from run_ies_loop import IESLoop
from metrics_v4 import (
    PROTOCOL_V4_CONFIG,
    compute_ks_with_critical,
    compute_worst_window_exhaustive,
)


# ============================================================================
# L2 fixed definition (paper)
# ============================================================================

L2_STATE_NAMES = ["capacityFactor", "minGap", "impatience"]
L2_OBS_FILE = PROJECT_ROOT / "data" / "calibration" / "l2_observation_vector_corridor_M11_moving_irn.csv"

L2_ENSEMBLE_SIZE = 10
L2_MAX_ITERS = 3
L2_DAMPING = 0.3

L2_TIME_MIN = 0.0
L2_TIME_MAX = 3600.0
L2_TT_MODE = "moving"

OBS_VAR_FLOOR = 1.0  # (km/h)^2 lower bound for obs variance

OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments_v4" / "a1_dapper_baselines"


# ============================================================================
# Utilities
# ============================================================================


def load_l2_priors() -> Tuple[List[str], np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    priors_path = PROJECT_ROOT / "config" / "calibration" / "l2_priors.json"
    with open(priors_path, "r", encoding="utf-8") as f:
        priors = json.load(f)

    names = []
    mu = []
    sigma = []
    bounds = []
    for p in priors["parameters"]:
        names.append(p["name"])
        mu.append(p["mu"])
        sigma.append(p["sigma"])
        bounds.append((p["min"], p["max"]))

    return names, np.array(mu, dtype=float), np.array(sigma, dtype=float), bounds


def compute_obs_variance(obs_df: pd.DataFrame, fallback_std: float) -> np.ndarray:
    obs_var = obs_df["std_speed_kmh"].values ** 2
    obs_var = np.where(
        (obs_var == 0) | np.isnan(obs_var),
        fallback_std ** 2,
        obs_var,
    )
    obs_var = np.maximum(obs_var, OBS_VAR_FLOOR ** 2)
    return obs_var


def compute_group_weights(obs_df: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(obs_df))
    if "route" not in obs_df.columns or "bound" not in obs_df.columns:
        return weights

    n_groups = obs_df.groupby(["route", "bound"]).ngroups
    weight_per_group = 1.0 / n_groups
    for (_, _), g in obs_df.groupby(["route", "bound"]):
        idx = g.index.tolist()
        w_per_link = weight_per_group / len(idx)
        for i in idx:
            weights[i] = w_per_link

    weights = weights * (len(obs_df) / weights.sum())
    return weights


def fill_missing_sim_values(
    Y_sim: np.ndarray,
    matched_masks: np.ndarray,
    Y_obs: np.ndarray,
) -> np.ndarray:
    Y_filled = Y_sim.copy()
    M = Y_sim.shape[1]
    for j in range(M):
        valid_mask = matched_masks[:, j] & np.isfinite(Y_sim[:, j])
        if valid_mask.any():
            fill_val = Y_sim[valid_mask, j].mean()
            Y_filled[~valid_mask, j] = fill_val
        else:
            Y_filled[:, j] = Y_obs[j]
    return Y_filled


def compute_tt_from_speed(speed_kmh: np.ndarray, dist_m: np.ndarray) -> np.ndarray:
    speed_kmh = np.asarray(speed_kmh, dtype=float)
    dist_m = np.asarray(dist_m, dtype=float)
    tt = np.full_like(speed_kmh, np.nan, dtype=float)
    valid = speed_kmh > 0
    tt[valid] = dist_m[valid] * 3.6 / speed_kmh[valid]
    return tt


def evaluate_metrics(
    obs_df: pd.DataFrame,
    sim_speed: np.ndarray,
    matched_mask: np.ndarray,
    scenario: str,
    min_valid_ratio: float = 0.7,
) -> Dict[str, object]:
    # Rule C is not re-applied here; the L2 observation vector is treated as fixed.
    obs_speed = obs_df["mean_speed_kmh"].values
    dist_m = obs_df["dist_m"].values
    M = len(obs_speed)

    valid_speed = matched_mask & np.isfinite(sim_speed)
    n_valid_speed = int(valid_speed.sum())
    min_valid = max(5, int(M * min_valid_ratio))

    metrics = {
        "n_valid_speed": n_valid_speed,
        "n_valid_tt": 0,
        "ks_speed": None,
        "ks_tt": None,
        "worst_speed": None,
        "worst_tt": None,
        "diverged": n_valid_speed < min_valid,
    }

    if metrics["diverged"]:
        return metrics

    ks_speed = compute_ks_with_critical(obs_speed[valid_speed], sim_speed[valid_speed])

    obs_tt = compute_tt_from_speed(obs_speed[valid_speed], dist_m[valid_speed])
    sim_tt = compute_tt_from_speed(sim_speed[valid_speed], dist_m[valid_speed])
    valid_tt = np.isfinite(obs_tt) & np.isfinite(sim_tt)
    n_valid_tt = int(valid_tt.sum())

    metrics["n_valid_tt"] = n_valid_tt
    metrics["ks_speed"] = ks_speed

    if n_valid_tt >= max(5, int(M * min_valid_ratio)):
        ks_tt = compute_ks_with_critical(obs_tt[valid_tt], sim_tt[valid_tt])
        metrics["ks_tt"] = ks_tt

    scenario_cfg = PROTOCOL_V4_CONFIG["scenarios"].get(scenario, {})
    duration_sec = scenario_cfg.get("duration_sec", 3600)
    base_time_sec = scenario_cfg.get("utc_start_sec", 0)

    metrics["worst_speed"] = compute_worst_window_exhaustive(
        obs_speed[valid_speed],
        sim_speed[valid_speed],
        total_duration_sec=duration_sec,
        base_time_sec=base_time_sec,
    )

    if n_valid_tt >= 5:
        metrics["worst_tt"] = compute_worst_window_exhaustive(
            obs_tt[valid_tt],
            sim_tt[valid_tt],
            total_duration_sec=duration_sec,
            base_time_sec=base_time_sec,
        )

    return metrics


# ============================================================================
# SUMO-backed observation operator for DAPPER
# ============================================================================


class L2SumoSimulator:
    def __init__(
        self,
        obs_file: Path,
        ensemble_size: int,
        seed: int,
        output_root: Path,
        run_prefix: str,
        use_baseline: bool = False,
        t_min: float = L2_TIME_MIN,
        t_max: float = L2_TIME_MAX,
        tt_mode: str = L2_TT_MODE,
    ):
        self.obs_file = obs_file
        self.obs_df = pd.read_csv(obs_file).reset_index(drop=True)
        self.output_root = output_root
        self.run_prefix = run_prefix
        self.call_index = 0

        self.ies = IESLoop(
            project_root=str(PROJECT_ROOT),
            label=run_prefix,
            ensemble_size=ensemble_size,
            max_iters=1,
            seed=seed,
            use_baseline=use_baseline,
            update_damping=L2_DAMPING,
            t_min=t_min,
            t_max=t_max,
            tt_mode=tt_mode,
        )

        self._sync_obs()

    def _sync_obs(self) -> None:
        self.ies.obs_df = self.obs_df
        self.ies.Y_obs = self.obs_df["mean_speed_kmh"].values
        self.ies.obs_csv_path = str(self.obs_file)
        self.ies.obs_variance = compute_obs_variance(self.obs_df, self.ies.obs_noise_std)
        self.ies._compute_group_weights()

    def _next_run_tag(self) -> str:
        self.call_index += 1
        return f"{self.run_prefix}_call{self.call_index:03d}"

    def simulate_ensemble(
        self,
        X_ensemble: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[float], List[Optional[str]]]:
        X_ensemble = np.atleast_2d(X_ensemble).astype(float)
        self.ies.ensemble_size = X_ensemble.shape[0]
        # Clip to L2 bounds to avoid invalid SUMO scale factors.
        for i, (low, high) in enumerate(self.ies.bounds):
            X_ensemble[:, i] = np.clip(X_ensemble[:, i], low, high)
        # Ensure capacityFactor stays positive for SUMO --scale.
        X_ensemble[:, 0] = np.maximum(X_ensemble[:, 0], 0.1)

        run_tag = self._next_run_tag()
        self.ies.output_dir = self.output_root / run_tag
        self.ies.output_dir.mkdir(parents=True, exist_ok=True)

        configs = self.ies.generate_sumo_configs(1, X_ensemble)
        edgedata_paths, insertion_rates = self.ies.run_parallel_simulations(configs)

        for i, ir in enumerate(insertion_rates):
            if ir < 0.50:
                edgedata_paths[i] = None

        Y_sim, matched_masks = self.ies.collect_simulation_results(edgedata_paths)
        Y_filled = fill_missing_sim_values(Y_sim, matched_masks, self.ies.Y_obs)
        return Y_filled, matched_masks, insertion_rates, edgedata_paths

    def simulate_single(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        Y_filled, matched_masks, _, _ = self.simulate_ensemble(np.atleast_2d(x))
        return Y_filled[0], matched_masks[0]


def build_hmm(
    simulator: L2SumoSimulator,
    param_mu: np.ndarray,
    param_sigma: np.ndarray,
    obs_variance: np.ndarray,
) -> modelling.HiddenMarkovModel:
    Nx = len(param_mu)
    Ny = len(simulator.obs_df)
    # DAPPER Chronology requires 3 parameters; use a single-step window (Ko=0).
    tseq = Chronology(dt=1.0, dko=1, T=1.0)

    def dyn(x, t, dt):
        return x

    def obs_model(x):
        Y_filled, _, _, _ = simulator.simulate_ensemble(x)
        return Y_filled

    Dyn = {"M": Nx, "model": dyn, "noise": 0}
    Obs = {
        "M": Ny,
        "model": obs_model,
        "noise": GaussRV(C=CovMat(obs_variance, "diag")),
    }

    X0 = GaussRV(mu=param_mu, C=CovMat(param_sigma ** 2, "diag"))
    sectors = {"param": np.arange(Nx)}

    return modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, sectors=sectors)


def prepare_truth_and_obs(hmm: modelling.HiddenMarkovModel, y_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = hmm.tseq.K
    Ko = hmm.tseq.Ko
    xx = np.tile(hmm.X0.mu, (K + 1, 1))
    yy = np.zeros((Ko + 1, len(y_obs)), dtype=float)
    yy[0] = y_obs
    return xx, yy


def clip_params(values: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    clipped = values.copy()
    for i, (lo, hi) in enumerate(bounds):
        clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


@dataclass
class RunResult:
    method_id: str
    method_name: str
    seed: int
    diverged: bool
    n_valid_speed: int
    n_valid_tt: int
    ks_speed: Optional[float]
    ks_speed_passed: Optional[bool]
    ks_tt: Optional[float]
    ks_tt_passed: Optional[bool]
    worst_speed: Optional[float]
    worst_speed_start: Optional[str]
    worst_tt: Optional[float]
    worst_tt_start: Optional[str]
    posterior_capacityFactor: Optional[float]
    posterior_minGap: Optional[float]
    posterior_impatience: Optional[float]
    compute_time_sec: float

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


# ============================================================================
# Main run
# ============================================================================


def run_dapper_method(
    method_id: str,
    method_name: str,
    xp_factory,
    seed: int,
    obs_df: pd.DataFrame,
    obs_variance: np.ndarray,
    param_mu: np.ndarray,
    param_sigma: np.ndarray,
    param_bounds: List[Tuple[float, float]],
    output_dir: Path,
    scenario: str,
) -> RunResult:
    start_time = time.time()
    dapper_seed = seed if seed != 0 else 1

    simulator = L2SumoSimulator(
        obs_file=L2_OBS_FILE,
        ensemble_size=L2_ENSEMBLE_SIZE,
        seed=dapper_seed,
        output_root=output_dir / "sumo_runs" / method_id / f"seed{seed}",
        run_prefix=f"{method_id}_seed{seed}",
        use_baseline=False,
    )

    weights = compute_group_weights(obs_df)
    obs_var_weighted = obs_variance / weights

    hmm = build_hmm(simulator, param_mu, param_sigma, obs_var_weighted)
    xx, yy = prepare_truth_and_obs(hmm, simulator.ies.Y_obs)

    xp = xp_factory()
    xp.seed = dapper_seed
    set_seed(dapper_seed)

    try:
        xp.assimilate(hmm, xx, yy, fail_gently=False)
        posterior = xp.stats.mu.a[0]
    except Exception:
        posterior = np.full_like(param_mu, np.nan, dtype=float)

    posterior = clip_params(posterior, param_bounds)
    if not np.isfinite(posterior).all():
        return RunResult(
            method_id=method_id,
            method_name=method_name,
            seed=seed,
            diverged=True,
            n_valid_speed=0,
            n_valid_tt=0,
            ks_speed=None,
            ks_speed_passed=None,
            ks_tt=None,
            ks_tt_passed=None,
            worst_speed=None,
            worst_speed_start=None,
            worst_tt=None,
            worst_tt_start=None,
            posterior_capacityFactor=None,
            posterior_minGap=None,
            posterior_impatience=None,
            compute_time_sec=time.time() - start_time,
        )

    sim_speed, matched_mask = simulator.simulate_single(posterior)
    metrics = evaluate_metrics(obs_df, sim_speed, matched_mask, scenario)

    ks_speed = metrics["ks_speed"].ks_stat if metrics.get("ks_speed") else None
    ks_speed_passed = metrics["ks_speed"].passed if metrics.get("ks_speed") else None
    ks_tt = metrics["ks_tt"].ks_stat if metrics.get("ks_tt") else None
    ks_tt_passed = metrics["ks_tt"].passed if metrics.get("ks_tt") else None
    worst_speed = metrics["worst_speed"].worst_ks if metrics.get("worst_speed") else None
    worst_speed_start = metrics["worst_speed"].window_start_time if metrics.get("worst_speed") else None
    worst_tt = metrics["worst_tt"].worst_ks if metrics.get("worst_tt") else None
    worst_tt_start = metrics["worst_tt"].window_start_time if metrics.get("worst_tt") else None

    return RunResult(
        method_id=method_id,
        method_name=method_name,
        seed=seed,
        diverged=metrics["diverged"] or (not np.isfinite(posterior).all()),
        n_valid_speed=metrics["n_valid_speed"],
        n_valid_tt=metrics["n_valid_tt"],
        ks_speed=ks_speed,
        ks_speed_passed=ks_speed_passed,
        ks_tt=ks_tt,
        ks_tt_passed=ks_tt_passed,
        worst_speed=worst_speed,
        worst_speed_start=worst_speed_start,
        worst_tt=worst_tt,
        worst_tt_start=worst_tt_start,
        posterior_capacityFactor=posterior[0],
        posterior_minGap=posterior[1],
        posterior_impatience=posterior[2],
        compute_time_sec=time.time() - start_time,
    )


def run_ies_baseline(
    seed: int,
    obs_df: pd.DataFrame,
    obs_variance: np.ndarray,
    param_bounds: List[Tuple[float, float]],
    output_dir: Path,
    scenario: str,
) -> RunResult:
    start_time = time.time()
    dapper_seed = seed if seed != 0 else 1

    ies = IESLoop(
        project_root=str(PROJECT_ROOT),
        label=f"a1_ies_seed{seed}",
        ensemble_size=L2_ENSEMBLE_SIZE,
        max_iters=L2_MAX_ITERS,
        seed=dapper_seed,
        use_baseline=False,
        update_damping=L2_DAMPING,
        t_min=L2_TIME_MIN,
        t_max=L2_TIME_MAX,
        tt_mode=L2_TT_MODE,
    )

    ies.obs_df = obs_df
    ies.Y_obs = obs_df["mean_speed_kmh"].values
    ies.obs_csv_path = str(L2_OBS_FILE)
    ies.obs_variance = obs_variance
    ies._compute_group_weights()

    final_params = ies.run()
    posterior = np.array(
        [
            final_params.get("capacityFactor"),
            final_params.get("minGap_background"),
            final_params.get("impatience"),
        ],
        dtype=float,
    )
    posterior = clip_params(posterior, param_bounds)
    if not np.isfinite(posterior).all():
        return RunResult(
            method_id="IES",
            method_name="IES (Ours)",
            seed=seed,
            diverged=True,
            n_valid_speed=0,
            n_valid_tt=0,
            ks_speed=None,
            ks_speed_passed=None,
            ks_tt=None,
            ks_tt_passed=None,
            worst_speed=None,
            worst_speed_start=None,
            worst_tt=None,
            worst_tt_start=None,
            posterior_capacityFactor=None,
            posterior_minGap=None,
            posterior_impatience=None,
            compute_time_sec=time.time() - start_time,
        )

    simulator = L2SumoSimulator(
        obs_file=L2_OBS_FILE,
        ensemble_size=1,
        seed=dapper_seed,
        output_root=output_dir / "sumo_runs" / "IES" / f"seed{seed}",
        run_prefix=f"IES_seed{seed}",
        use_baseline=False,
    )
    sim_speed, matched_mask = simulator.simulate_single(posterior)
    metrics = evaluate_metrics(obs_df, sim_speed, matched_mask, scenario)

    ks_speed = metrics["ks_speed"].ks_stat if metrics.get("ks_speed") else None
    ks_speed_passed = metrics["ks_speed"].passed if metrics.get("ks_speed") else None
    ks_tt = metrics["ks_tt"].ks_stat if metrics.get("ks_tt") else None
    ks_tt_passed = metrics["ks_tt"].passed if metrics.get("ks_tt") else None
    worst_speed = metrics["worst_speed"].worst_ks if metrics.get("worst_speed") else None
    worst_speed_start = metrics["worst_speed"].window_start_time if metrics.get("worst_speed") else None
    worst_tt = metrics["worst_tt"].worst_ks if metrics.get("worst_tt") else None
    worst_tt_start = metrics["worst_tt"].window_start_time if metrics.get("worst_tt") else None

    return RunResult(
        method_id="IES",
        method_name="IES (Ours)",
        seed=seed,
        diverged=metrics["diverged"] or (not np.isfinite(posterior).all()),
        n_valid_speed=metrics["n_valid_speed"],
        n_valid_tt=metrics["n_valid_tt"],
        ks_speed=ks_speed,
        ks_speed_passed=ks_speed_passed,
        ks_tt=ks_tt,
        ks_tt_passed=ks_tt_passed,
        worst_speed=worst_speed,
        worst_speed_start=worst_speed_start,
        worst_tt=worst_tt,
        worst_tt_start=worst_tt_start,
        posterior_capacityFactor=posterior[0],
        posterior_minGap=posterior[1],
        posterior_impatience=posterior[2],
        compute_time_sec=time.time() - start_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="A1: DAPPER L2 baselines (SUMO + PM Peak)")
    parser.add_argument("--scenario", type=str, default="pm_peak", choices=["pm_peak"])
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    obs_df = pd.read_csv(L2_OBS_FILE).reset_index(drop=True)
    _, param_mu, param_sigma, param_bounds = load_l2_priors()

    priors_path = PROJECT_ROOT / "config" / "calibration" / "l2_priors.json"
    with open(priors_path, "r", encoding="utf-8") as f:
        priors = json.load(f)
    obs_variance = compute_obs_variance(
        obs_df,
        priors.get("ensemble_config", {}).get("observation_noise_std", 2.0),
    )

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    methods = [
        (
            "EnKF",
            "EnKF (ETKF)",
            lambda: da.EnKF("Sqrt", N=L2_ENSEMBLE_SIZE),
        ),
        (
            "EnKS",
            "EnKS",
            lambda: da.EnKS("Sqrt", N=L2_ENSEMBLE_SIZE, Lag=1),
        ),
        (
            "iEnKS",
            "iEnKS",
            lambda: da.iEnKS("Sqrt", N=L2_ENSEMBLE_SIZE, Lag=1, nIter=L2_MAX_ITERS, wtol=0),
        ),
    ]

    results: List[RunResult] = []

    for seed in seeds:
        for method_id, method_name, xp_factory in methods:
            result = run_dapper_method(
                method_id=method_id,
                method_name=method_name,
                xp_factory=xp_factory,
                seed=seed,
                obs_df=obs_df,
                obs_variance=obs_variance,
                param_mu=param_mu,
                param_sigma=param_sigma,
                param_bounds=param_bounds,
                output_dir=output_dir,
                scenario=args.scenario,
            )
            results.append(result)

        ies_result = run_ies_baseline(
            seed=seed,
            obs_df=obs_df,
            obs_variance=obs_variance,
            param_bounds=param_bounds,
            output_dir=output_dir,
            scenario=args.scenario,
        )
        results.append(ies_result)

    df_results = pd.DataFrame([r.to_dict() for r in results])
    results_path = output_dir / "results.csv"
    df_results.to_csv(results_path, index=False)

    summary = df_results.groupby(["method_id", "method_name"]).agg(
        ks_speed_mean=("ks_speed", "mean"),
        ks_speed_std=("ks_speed", "std"),
        ks_tt_mean=("ks_tt", "mean"),
        ks_tt_std=("ks_tt", "std"),
        worst_speed_mean=("worst_speed", "mean"),
        worst_speed_std=("worst_speed", "std"),
        diverged_count=("diverged", "sum"),
        n_runs=("seed", "nunique"),
        capacityFactor_mean=("posterior_capacityFactor", "mean"),
        capacityFactor_std=("posterior_capacityFactor", "std"),
        minGap_mean=("posterior_minGap", "mean"),
        impatience_mean=("posterior_impatience", "mean"),
    ).reset_index()

    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    md_lines = [
        "# A1 DAPPER L2 Baselines (PM Peak)",
        "",
        f"- Observation: `{L2_OBS_FILE}`",
        f"- State: {', '.join(L2_STATE_NAMES)}",
        f"- Ne={L2_ENSEMBLE_SIZE}, K={L2_MAX_ITERS}, beta={L2_DAMPING}",
        "",
        "| Method | KS(speed) | KS(TT) | Worst-15min KS | capacityFactor | minGap | impatience | Diverged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        ks_speed = f"{row['ks_speed_mean']:.4f}" if pd.notna(row["ks_speed_mean"]) else "NA"
        ks_tt = f"{row['ks_tt_mean']:.4f}" if pd.notna(row["ks_tt_mean"]) else "NA"
        worst = f"{row['worst_speed_mean']:.4f}" if pd.notna(row["worst_speed_mean"]) else "NA"
        cap = f"{row['capacityFactor_mean']:.3f}" if pd.notna(row["capacityFactor_mean"]) else "NA"
        mingap = f"{row['minGap_mean']:.3f}" if pd.notna(row["minGap_mean"]) else "NA"
        impat = f"{row['impatience_mean']:.3f}" if pd.notna(row["impatience_mean"]) else "NA"
        md_lines.append(
            f"| {row['method_name']} | {ks_speed} | {ks_tt} | {worst} | {cap} | {mingap} | {impat} | {int(row['diverged_count'])}/{int(row['n_runs'])} |"
        )

    md_path = output_dir / "tables" / "dapper_l2_baselines.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("输出文件:")
    print(f"  - {results_path}")
    print(f"  - {summary_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
