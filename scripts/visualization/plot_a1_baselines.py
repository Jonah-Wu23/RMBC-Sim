#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_a1_baselines.py
====================
Figure 1: A1 baseline comparison (single-column width, 2 subplots).

(a) MAE mean ± std across seeds
(b) RMSE mean ± std across seeds
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS = PROJECT_ROOT / "data" / "experiments_v4" / "a1_dapper_baselines" / "results.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "experiments_v4" / "a1_dapper_baselines" / "figures"

METHOD_ORDER = ["IES", "EnKF", "EnKS", "iEnKS"]
METHOD_LABELS = {
    "IES": "IES (Ours)",
    "EnKF": "EnKF",
    "EnKS": "EnKS",
    "iEnKS": "iEnKS",
}

COLOR_MAP = {
    "IES": "#ff7f0e",
    "EnKF": "#1f77b4",
    "EnKS": "#6baed6",
    "iEnKS": "#ffbb78",
}


def summarize_metric(df: pd.DataFrame, value_col: str) -> Dict[str, Tuple[float, float]]:
    summary = {}
    for method_id in METHOD_ORDER:
        subset = df[df["method_id"] == method_id][value_col].dropna()
        if subset.empty:
            summary[method_id] = (np.nan, np.nan)
        else:
            summary[method_id] = (float(subset.mean()), float(subset.std()))
    return summary


def _set_zoom_limits(ax, means: List[float], stds: List[float], pad_ratio: float = 0.2) -> None:
    lows = []
    highs = []
    for mean, std in zip(means, stds):
        if not np.isfinite(mean):
            continue
        err = std if np.isfinite(std) else 0.0
        lows.append(mean - err)
        highs.append(mean + err)
    if not lows:
        return
    vmin = min(lows)
    vmax = max(highs)
    span = max(vmax - vmin, 1e-6)
    pad = span * pad_ratio
    ax.set_ylim(max(vmin - pad, 0.0), vmax + pad)


def plot_a1_baselines(mae_summary: Dict[str, Tuple[float, float]], rmse_summary: Dict[str, Tuple[float, float]], output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True)

    x = np.arange(len(METHOD_ORDER))
    colors = [COLOR_MAP[m] for m in METHOD_ORDER]

    mae_means = [mae_summary[m][0] for m in METHOD_ORDER]
    mae_stds = [mae_summary[m][1] for m in METHOD_ORDER]
    rmse_means = [rmse_summary[m][0] for m in METHOD_ORDER]
    rmse_stds = [rmse_summary[m][1] for m in METHOD_ORDER]

    ax1 = axes[0]
    ax1.errorbar(
        x,
        mae_means,
        yerr=mae_stds,
        fmt="-o",
        color="#1f77b4",
        ecolor="black",
        elinewidth=0.8,
        capsize=3,
        markersize=4,
    )
    ax1.scatter(x, mae_means, color=colors, edgecolor="black", linewidth=0.4, zorder=3)
    ax1.set_title("(a) MAE (km/h)")
    ax1.set_ylabel("MAE (km/h)")
    _set_zoom_limits(ax1, mae_means, mae_stds)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = axes[1]
    ax2.errorbar(
        x,
        rmse_means,
        yerr=rmse_stds,
        fmt="-o",
        color="#ff7f0e",
        ecolor="black",
        elinewidth=0.8,
        capsize=3,
        markersize=4,
    )
    ax2.scatter(x, rmse_means, color=colors, edgecolor="black", linewidth=0.4, zorder=3)
    ax2.set_title("(b) RMSE (km/h)")
    ax2.set_ylabel("RMSE (km/h)")
    _set_zoom_limits(ax2, rmse_means, rmse_stds)
    ax2.set_xticks(x)
    ax2.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3, which="both")

    fig.tight_layout(pad=0.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig1_a1_baselines.png"
    pdf_path = output_dir / "fig1_a1_baselines.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot A1 baseline comparison (MAE + RMSE)")
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.csv: {results_path}")

    results_df = pd.read_csv(results_path)
    if "method_id" not in results_df.columns:
        raise ValueError("results.csv missing method_id column")

    mae_summary = summarize_metric(results_df, "mae_speed")
    rmse_summary = summarize_metric(results_df, "rmse_speed")

    plot_a1_baselines(mae_summary, rmse_summary, output_dir)


if __name__ == "__main__":
    main()
