param(
  [switch]$UseFixtures = $true
)

$ErrorActionPreference = "Stop"

if ($UseFixtures) {
  python scripts/smoke/p14_smoke.py
  exit $LASTEXITCODE
}

Write-Host "Running P14 pipeline with default real-data paths..." -ForegroundColor Cyan
python scripts/evaluate_robustness.py --real data2/processed/link_stats_offpeak.csv --sim sumo/output/offpeak_stopinfo.xml --dist data/processed/kmb_route_stop_dist.csv --t_critical 325 --speed_kmh 5 --max_dist_m 1500
python scripts/visualization/plot_p14_robustness.py --raw data2/processed/link_stats_offpeak.csv --sim sumo/output/offpeak_stopinfo.xml --dist data/processed/kmb_route_stop_dist.csv --t_critical 325 --speed_kmh 5 --max_dist_m 1500 --out-audit plots/P14_ghost_audit.png --out-cdf plots/P14_robustness_cdf.png
