"""
Gaode traffic/status/road cleaner.

Inputs:
- data/raw/*-traffic.json (from gaode_downloader with --traffic-road)

Outputs:
- data/processed/traffic_road.csv : per-road-segment snapshot with speed/status/polyline and capture_ts
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def parse_ts_from_name(path: Path) -> str:
    m = re.search(r"(20\d{6}-\d{6})", path.name)
    return m.group(1) if m else ""


def iter_traffic_payloads(pattern: str) -> Iterable[Dict]:
    for path_str in glob.glob(pattern):
        path = Path(path_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            yield path, payload
        except Exception:
            continue


def build_rows(line: str) -> List[Dict]:
    rows: List[Dict] = []
    pattern = str(RAW_DIR / f"{line}-road-*-traffic.json")
    for path, payload in iter_traffic_payloads(pattern):
        ts = parse_ts_from_name(path)
        info = payload.get("trafficinfo") or {}
        description = info.get("description", "")
        evaluation = info.get("evaluation") or {}
        roads = info.get("roads") or []
        for rd in roads:
            rows.append(
                {
                    "capture_ts": ts,
                    "line": line,
                    "road_name": rd.get("name", ""),
                    "direction": rd.get("direction", ""),
                    "status": rd.get("status", ""),
                    "speed_kmh": rd.get("speed", ""),
                    "angle": rd.get("angle", ""),
                    "lcodes": rd.get("lcodes", ""),
                    "polyline": rd.get("polyline", ""),
                    "desc": description,
                    "expedite_pct": evaluation.get("expedite", ""),
                    "congested_pct": evaluation.get("congested", ""),
                    "blocked_pct": evaluation.get("blocked", ""),
                    "unknown_pct": evaluation.get("unknown", ""),
                    "eval_status": evaluation.get("status", ""),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean Gaode traffic/status/road snapshots.")
    ap.add_argument("--line", default="229", help="line label used in filenames (default: 229)")
    args = ap.parse_args()

    rows = build_rows(args.line)
    out_path = PROCESSED_DIR / "traffic_road.csv"
    write_csv(
        out_path,
        rows,
        [
            "capture_ts",
            "line",
            "road_name",
            "direction",
            "status",
            "speed_kmh",
            "angle",
            "lcodes",
            "polyline",
            "desc",
            "expedite_pct",
            "congested_pct",
            "blocked_pct",
            "unknown_pct",
            "eval_status",
        ],
    )
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
