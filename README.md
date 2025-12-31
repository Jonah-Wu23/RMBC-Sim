# RCMDT: Observation-Operator-Aware Calibration of Mobility Digital Twins

Codebase for an IEEE SMC 2026 paper draft on robust calibration of bus corridor digital twins. The current paper storyline lives in `docs/paper_outline.md`.

## What This Project Claims (In One Page)

This repo implements and evaluates **RCMDT**, a hierarchical calibration loop (BO outer loop + IES inner loop) for bus corridor mobility digital twins, with a paper contribution that is deliberately **not** “we calibrated better”, but:

- **Observation Operator Audit:** “Real” door-to-door (D2D) measurements can be contaminated by non-transport operational semantics (schedule adherence, holding/layover). Robust conclusions require an auditable observation operator.
- **Regime Separation (Rule C):** We isolate non-transport “ghost jams” with a reproducible rule: flag samples if `(T > T*) ∧ (v_eff < v*)` using `T* = 325s`, `v* = 5 km/h`.
- **Robustness Evidence:** Robustness is validated at the **distribution level** (K-S + worst-window stress test). K-S is used as **validation evidence**, not as the calibration objective.
- **Diagnostics (Not Validation):** Trajectory decomposition and dwell/holding-proxy plots answer *where mechanisms differ*, not whether the calibrated model is “correct”.

## Paper Alignment

- Paper outline: `docs/paper_outline.md`
- Experiment registry + run notes: `docs/experiments.md`
- Data/compliance notes: `DATA.md`
- Main P14 figures (already generated): `plots/P14_ghost_audit.png`, `plots/P14_robustness_cdf.png`

## Repository Layout

- `scripts/`: analysis, calibration, visualization utilities
- `sumo/`: SUMO configs, networks, routes, outputs
- `data/`: peak-hour datasets (calibration side)
- `data2/`: off-peak datasets (robustness / transfer side)
- `plots/`: paper figures and diagnostics

Note: some folders (e.g., `data/`, `data2/`) may be git-ignored depending on your workflow; see `.gitignore`.

## Requirements

- Windows 10/11 (tested), Python 3.11+, SUMO 1.20.0 (`sumo --version`)
- Python packages:
  ```bash
  pip install -r requirements.txt
  ```
  Optional (used by other scripts): `pip install -r requirements-optional.txt`

## Quickstart: Reproduce The P14 Evidence Chain (Recommended)

The P14 story is: **operator audit → distributional robustness (K-S) → regime separation diagnostics**.

1) Run off-peak simulation:
```bash
sumo -c sumo/config/experiment_robustness.sumocfg
```

2) Compute robustness metrics (raw vs Rule C clean):
```bash
python scripts/evaluate_robustness.py --real data2/processed/link_stats_offpeak.csv --sim sumo/output/offpeak_stopinfo.xml --dist data/processed/kmb_route_stop_dist.csv --t_critical 325 --speed_kmh 5 --max_dist_m 1500
```

3) Generate the two main P14 figures (audit + CDF):
```bash
python scripts/visualization/plot_p14_robustness.py --raw data2/processed/link_stats_offpeak.csv --sim sumo/output/offpeak_stopinfo.xml --dist data/processed/kmb_route_stop_dist.csv --t_critical 325 --speed_kmh 5 --max_dist_m 1500 --worst_window_ks 0.3337
```
Outputs:
- `plots/P14_ghost_audit.png`
- `plots/P14_robustness_cdf.png`

## Smoke Test (No Real Data / No SUMO)

This repo includes synthetic fixtures so reviewers can run a minimal end-to-end check without any private data:

```bash
python scripts/smoke/p14_smoke.py
```

Or in PowerShell:
```powershell
.\reproduce.ps1
```

## Optional Diagnostics (Use For Mechanism Explanations Only)

Trajectory decomposition (stepped/full-time vs traffic-only) is diagnostic and should not be used as the primary validation evidence:

```bash
python scripts/visualization/plot_trajectory_stepped.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/trajectory_stepped_68X.png --route 68X --t_critical 325 --speed_kmh 5
python scripts/visualization/plot_trajectory_stepped.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/trajectory_stepped_960.png --route 960 --t_critical 325 --speed_kmh 5
```

Holding proxy vs simulated dwell (diagnostic):
```bash
python scripts/visualization/plot_dwell_distribution.py
```

## Calibration Runs (Advanced / Expensive)

- L1 Bayesian optimization (outer loop): `scripts/calibration/`
- L2 IES assimilation (inner loop): `scripts/calibration/`
- For the authoritative run settings and what each label means (B1–P14), follow `docs/experiments.md`.

## Troubleshooting

- SUMO outputs empty/small: check `sumo/output/*.xml` sizes and `sumo/output/*.log` logs.
- P14 “raw” K-S is high (~0.5+): this is expected before observation operator audit / regime separation (see `plots/P14_ghost_audit.png` and `docs/paper_outline.md`).
