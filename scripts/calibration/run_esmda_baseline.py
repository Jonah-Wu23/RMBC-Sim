#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_esmda_baseline.py
=====================
ES-MDA baseline using pyESMDA for L2 parameter estimation.

This script reuses the IES simulation pipeline to generate predictions,
but the update step is handled by pyESMDA.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from pyesmda import ESMDA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from calibration.run_ies_loop import IESLoop

def build_prior_ensemble(mu: np.ndarray, sigma: np.ndarray, bounds: list[Tuple[float, float]],
                         n_ensemble: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(loc=mu, scale=sigma, size=(n_ensemble, len(mu)))
    for i, (low, high) in enumerate(bounds):
        m[:, i] = np.clip(m[:, i], low, high)
    return m


def fill_missing_simulations(
    y_sim: np.ndarray,
    matched_masks: np.ndarray,
    y_obs: np.ndarray
) -> np.ndarray:
    filled = y_sim.copy()
    n_obs = y_obs.shape[0]
    for j in range(n_obs):
        valid_mask = matched_masks[:, j] & ~np.isnan(y_sim[:, j])
        if valid_mask.sum() > 0:
            fill_val = np.mean(y_sim[valid_mask, j])
            filled[~valid_mask, j] = fill_val
        else:
            filled[:, j] = y_obs[j]
    # Guard against any residual NaN
    filled = np.where(np.isnan(filled), y_obs, filled)
    return filled


def build_forward_model(ies: IESLoop):
    call_counter = {"idx": 0}

    def forward_model(m_ensemble: np.ndarray) -> np.ndarray:
        call_counter["idx"] += 1
        iter_id = call_counter["idx"]

        X = np.array(m_ensemble, dtype=float)
        for i, (low, high) in enumerate(ies.bounds):
            X[:, i] = np.clip(X[:, i], low, high)

        configs = ies.generate_sumo_configs(iter_id, X)
        edgedata_paths, _ = ies.run_parallel_simulations(configs)
        y_sim, matched_masks = ies.collect_simulation_results(edgedata_paths)
        return fill_missing_simulations(y_sim, matched_masks, ies.Y_obs)

    return forward_model


def compute_metrics(y_obs: np.ndarray, y_sim: np.ndarray) -> dict:
    y_bar = np.mean(y_sim, axis=0)
    rmse = float(np.sqrt(np.mean((y_bar - y_obs) ** 2)))
    ks_stat, p_value = ks_2samp(y_obs, y_bar)
    return {
        "rmse": rmse,
        "ks_clean": float(ks_stat),
        "p_value": float(p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ES-MDA baseline (pyESMDA) for L2 calibration")
    parser.add_argument("--label", type=str, default="ESMDA", help="Run label for output paths")
    parser.add_argument("--ensemble-size", type=int, default=20, help="Ensemble size (Ne)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of assimilations (K)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--obs-var-floor", type=float, default=1.0, help="Observation variance floor (km/h)^2")
    parser.add_argument("--tmin", type=float, default=0.0, help="Simulation time window start (s)")
    parser.add_argument("--tmax", type=float, default=3600.0, help="Simulation time window end (s)")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "calibration" / "ies_comparison"),
                        help="Output directory for results")
    args = parser.parse_args()

    # Reuse IES pipeline for observation definitions and SUMO forward model.
    ies = IESLoop(
        project_root=str(PROJECT_ROOT),
        label=f"{args.label}",
        ensemble_size=args.ensemble_size,
        max_iters=args.iterations,
        seed=args.seed,
        use_baseline=True,
        t_min=args.tmin,
        t_max=args.tmax
    )
    ies.output_dir = PROJECT_ROOT / "sumo" / "output" / "esmda_runs" / args.label
    ies.output_dir.mkdir(parents=True, exist_ok=True)

    mu = np.array(ies.mu, dtype=float)
    sigma = np.array(ies.sigma, dtype=float)
    bounds = ies.bounds
    m_init = build_prior_ensemble(mu, sigma, bounds, args.ensemble_size, args.seed)

    obs_var = np.maximum(ies.obs_variance, args.obs_var_floor)
    cov_obs = np.diag(obs_var)

    n_assim = args.iterations
    cov_obs_inflation_factors = [float(n_assim)] * n_assim

    forward_model = build_forward_model(ies)

    solver = ESMDA(
        obs=ies.Y_obs,
        m_init=m_init,
        cov_obs=cov_obs,
        forward_model=forward_model,
        n_assimilations=n_assim,
        cov_obs_inflation_factors=cov_obs_inflation_factors,
        m_bounds=np.array(bounds, dtype=float),
        save_ensembles_history=True,
        seed=args.seed
    )

    solver.solve()

    posterior_ensemble = solver.m_history[-1]
    y_sim = forward_model(posterior_ensemble)
    metrics = compute_metrics(ies.Y_obs, y_sim)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "config": "ES-MDA (pyESMDA)",
        "ks_clean": metrics["ks_clean"],
        "rmse": metrics["rmse"],
        "p_value": metrics["p_value"],
        "passed": metrics["ks_clean"] < 0.35,
        "ensemble_size": args.ensemble_size,
        "iterations": args.iterations,
        "seed": args.seed
    }

    result_path = output_dir / "es_mda_results.csv"
    pd.DataFrame([result]).to_csv(result_path, index=False)

    config_path = output_dir / "es_mda_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[ES-MDA] Results saved: {result_path}")
    print(f"[ES-MDA] Config saved:  {config_path}")


if __name__ == "__main__":
    main()
