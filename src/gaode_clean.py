"""
Gaode bus data cleaner (static stops/line meta) for captured lineid responses.

Inputs:
- data/raw/<line>-*-lineid.json : snapshots from gaode_downloader

Outputs (written to data/processed/):
- <line>_stops.csv      : deduped stop list with lon/lat and sequence
- <line>_line_meta.csv  : per-direction line meta (start/end stop, times, polyline length)

Notes:
- Focuses on static structure; ETA/velocity not available without higher privileges.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def parse_ts_from_name(path: Path) -> str:
    """Extract YYYYMMDD-HHMMSS from filename."""
    m = re.search(r"(20\d{6}-\d{6})", path.name)
    if m:
        return m.group(1)
    # fallback to mtime
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d-%H%M%S")


def parse_location(loc: str) -> Tuple[Optional[float], Optional[float]]:
    if not loc or "," not in loc:
        return None, None
    try:
        lon_s, lat_s = loc.split(",", 1)
        return float(lon_s), float(lat_s)
    except Exception:
        return None, None


def iter_lineid_payloads(line: str) -> Iterable[Tuple[Path, Dict]]:
    for path_str in glob.glob(str(RAW_DIR / f"{line}-*-lineid.json")):
        path = Path(path_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        yield path, payload


def dedup_latest(rows: List[Dict], key_fields: Tuple[str, ...]) -> List[Dict]:
    latest: Dict[Tuple, Dict] = {}
    for row in rows:
        key = tuple(row.get(k) for k in key_fields)
        prev = latest.get(key)
        if not prev or row.get("capture_ts", "") > prev.get("capture_ts", ""):
            latest[key] = row
    return list(latest.values())


def build_stop_rows(line: str) -> List[Dict]:
    rows: List[Dict] = []
    for path, payload in iter_lineid_payloads(line):
        ts = parse_ts_from_name(path)
        for busline in payload.get("buslines", []) or []:
            direction = busline.get("name") or "unknown"
            line_id = busline.get("id") or "unknown"
            for idx, stop in enumerate(busline.get("busstops", []) or [], start=1):
                seq = stop.get("sequence") or idx
                lon, lat = parse_location(stop.get("location"))
                rows.append(
                    {
                        "line": line,
                        "line_id": line_id,
                        "direction": direction,
                        "stop_seq": int(seq),
                        "stop_id": stop.get("id") or "",
                        "stop_name": stop.get("name") or "",
                        "lon": lon,
                        "lat": lat,
                        "capture_ts": ts,
                    }
                )
    return dedup_latest(rows, ("line_id", "direction", "stop_id"))


def build_line_meta_rows(line: str) -> List[Dict]:
    rows: List[Dict] = []
    for path, payload in iter_lineid_payloads(line):
        ts = parse_ts_from_name(path)
        for busline in payload.get("buslines", []) or []:
            rows.append(
                {
                    "line": line,
                    "line_id": busline.get("id") or "",
                    "direction": busline.get("name") or "unknown",
                    "start_stop": busline.get("start_stop") or "",
                    "end_stop": busline.get("end_stop") or "",
                    "start_time": busline.get("start_time") or "",
                    "end_time": busline.get("end_time") or "",
                    "direc": busline.get("direc") or "",
                    "polyline": busline.get("polyline") or "",
                    "capture_ts": ts,
                }
            )
    return dedup_latest(rows, ("line_id", "direction"))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean Gaode lineid snapshots into structured CSVs.")
    ap.add_argument("--line", default="229", help="line label used in filenames (default: 229)")
    args = ap.parse_args()

    stop_rows = build_stop_rows(args.line)
    meta_rows = build_line_meta_rows(args.line)

    write_csv(
        PROCESSED_DIR / f"{args.line}_stops.csv",
        stop_rows,
        ["line", "line_id", "direction", "stop_seq", "stop_id", "stop_name", "lon", "lat", "capture_ts"],
    )
    write_csv(
        PROCESSED_DIR / f"{args.line}_line_meta.csv",
        meta_rows,
        ["line", "line_id", "direction", "start_stop", "end_stop", "start_time", "end_time", "direc", "polyline", "capture_ts"],
    )

    print(f"wrote {len(stop_rows)} stop rows -> {PROCESSED_DIR / (args.line + '_stops.csv')}")
    print(f"wrote {len(meta_rows)} line meta rows -> {PROCESSED_DIR / (args.line + '_line_meta.csv')}")


if __name__ == "__main__":
    main()
