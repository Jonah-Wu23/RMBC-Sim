"""
Organize collected HK data files under data/raw into category folders.

Categories (matched by filename prefix):
- KMB: route, route-stop, stop, route-eta, stop-eta
- TSM notification: tsm-notification-*.csv
- IRN download: irnAvgSpeed-all-*.xml
- Detector locations: traffic_speed_volume_occ_info-*.csv
- STN: trafficnews-*.xml
- HKO: hko-rhrread-*.json
- JTI: Journeytimev2-*.xml

Usage:
  python scripts/organize_raw_files.py          # move files
  python scripts/organize_raw_files.py --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple

RAW_DIR = Path("data/raw")

# Ordered patterns so more specific matches (e.g. stop-eta) happen before stop.
PATTERN_TO_DIR: Tuple[Tuple[re.Pattern[str], Path], ...] = (
    (re.compile(r"^kmb-route-eta-"), RAW_DIR / "kmb" / "route-eta"),
    (re.compile(r"^kmb-route-stop-"), RAW_DIR / "kmb" / "route-stop"),
    (re.compile(r"^kmb-route-"), RAW_DIR / "kmb" / "route"),
    (re.compile(r"^kmb-stop-eta-"), RAW_DIR / "kmb" / "stop-eta"),
    (re.compile(r"^kmb-stop-"), RAW_DIR / "kmb" / "stop"),
    (re.compile(r"^tsm-notification-"), RAW_DIR / "tsm_notification"),
    (re.compile(r"^irnAvgSpeed-all-"), RAW_DIR / "irn_download"),
    (re.compile(r"^traffic_speed_volume_occ_info-"), RAW_DIR / "detector_locations"),
    (re.compile(r"^trafficnews-"), RAW_DIR / "stn"),
    (re.compile(r"^hko-rhrread-"), RAW_DIR / "hko"),
    (re.compile(r"^Journeytimev2-"), RAW_DIR / "jti"),
)


def find_target_dir(file_name: str) -> Path | None:
    for pattern, target_dir in PATTERN_TO_DIR:
        if pattern.match(file_name):
            return target_dir
    return None


def organize(dry_run: bool = False) -> Tuple[int, int]:
    moved = 0
    skipped = 0

    if not RAW_DIR.exists():
        raise SystemExit(f"raw directory not found: {RAW_DIR}")

    for path in RAW_DIR.iterdir():
        if not path.is_file():
            continue

        target_dir = find_target_dir(path.name)
        if target_dir is None:
            skipped += 1
            print(f"[skip] {path.name}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / path.name
        print(f"{'[move]' if not dry_run else '[dry]'} {path.name} -> {dest.relative_to(RAW_DIR)}")
        if not dry_run:
            # replace() overwrites existing files if any.
            path.replace(dest)
        moved += 1

    return moved, skipped


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Organize data/raw files into category folders.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without changing files.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    moved, skipped = organize(dry_run=args.dry_run)
    print(f"Done. moved={moved}, skipped={skipped}")


if __name__ == "__main__":
    main()
