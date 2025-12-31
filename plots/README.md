# Figures: Provenance & Generation

This folder contains paper figures and diagnostics. For the paper evidence chain, the two main P14 figures are:

- `plots/P14_ghost_audit.png` (operator audit / regime separation visualization)
- `plots/P14_robustness_cdf.png` (distribution-level robustness evidence via K-S)

## Regenerate (P14)

Using your real inputs (see `README.md` for defaults):

```bash
python scripts/visualization/plot_p14_robustness.py ^
  --raw data2/processed/link_stats_offpeak.csv ^
  --sim sumo/output/offpeak_stopinfo.xml ^
  --dist data/processed/kmb_route_stop_dist.csv ^
  --t_critical 325 --speed_kmh 5 --max_dist_m 1500 ^
  --worst_window_ks 0.3337 ^
  --out-audit plots/P14_ghost_audit.png ^
  --out-cdf plots/P14_robustness_cdf.png
```

## Notes

- `t_critical=325` and `speed_kmh=5` define Rule C in the paper narrative.
- Trajectory/dwell plots are diagnostics and should not be treated as primary validation evidence.
