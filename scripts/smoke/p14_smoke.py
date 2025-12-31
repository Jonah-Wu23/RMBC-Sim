#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    fixtures = repo_root / "tests" / "fixtures" / "p14"
    out_dir = repo_root / "plots" / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    real_csv = fixtures / "link_stats_offpeak.csv"
    dist_csv = fixtures / "kmb_route_stop_dist.csv"
    sim_xml = fixtures / "offpeak_stopinfo.xml"

    t_critical = 325
    speed_kmh = 5.0
    max_dist_m = 1500.0

    from evaluate_robustness import evaluate_robustness

    metrics = evaluate_robustness(
        str(real_csv),
        str(sim_xml),
        str(dist_csv),
        t_critical,
        speed_kmh,
        max_dist_m,
        peak_file=None,
        grade=False,
    )
    if not metrics or metrics["ks_raw"] is None or metrics["ks_clean"] is None:
        raise SystemExit("Smoke test failed: missing KS metrics")
    if not (0 <= metrics["ks_clean"] <= 1 and 0 <= metrics["ks_raw"] <= 1):
        raise SystemExit("Smoke test failed: KS outside [0,1]")
    if metrics["ks_clean"] >= metrics["ks_raw"]:
        raise SystemExit("Smoke test failed: cleaning did not improve KS on fixtures")

    audit_png = out_dir / "P14_ghost_audit.png"
    cdf_png = out_dir / "P14_robustness_cdf.png"

    # Import plotting script only after matplotlib is available in env.
    sys.path.insert(0, str(repo_root / "scripts" / "visualization"))
    from plot_p14_robustness import plot_ghost_audit, plot_robustness_cdf

    plot_ghost_audit(str(real_csv), str(audit_png), t_critical, speed_kmh, max_dist_m, fixture=True)
    plot_robustness_cdf(str(real_csv), str(sim_xml), str(dist_csv), str(cdf_png), t_critical, speed_kmh, max_dist_m, fixture=True)

    if not audit_png.exists() or not cdf_png.exists():
        raise SystemExit("Smoke test failed: expected output PNGs not found")

    print(f"OK: wrote {audit_png}")
    print(f"OK: wrote {cdf_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
