"""
Simple downloader for Gaode bus line info and route-time snapshots.

Targets: Xi'an bus 229 (both directions). Captures:
- linename response (keyword search)
- lineid detail per direction
- optional etd/driving snapshot between first/last stop as route-speed proxy

Notes:
- Requires env GAODE_KEY or --key argument.
- Writes raw JSON into data/raw/ using naming: <line>-<direction>-<YYYYMMDD>-<HHMMSS>-<type>.json
- Logs to data/raw/<line>-fetch.log
"""

import argparse
import datetime as dt
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests

RAW_DIR = Path("data/raw")
DEFAULT_CITY = "610100"  # Xi'an adcode
DEFAULT_LINE_KEYWORD = "229"
LINE_NAME_URL = "https://restapi.amap.com/v3/bus/linename"
LINE_ID_URL = "https://restapi.amap.com/v3/bus/lineid"
ETD_URL = "https://restapi.amap.com/v4/etd/driving"
TRAFFIC_ROAD_URL = "https://restapi.amap.com/v3/traffic/status/road"


def setup_logging(line: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log_file = RAW_DIR / f"{line}-fetch.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def sanitize(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return safe.strip("_") or "unknown"


def utc_ts(offset_seconds: int = 0) -> int:
    return int(time.time()) + offset_seconds


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def request_json(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    retries: int = 3,
    backoff: float = 1.5,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                raise RuntimeError("rate limited (429)")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # pragma: no cover - simple CLI script
            last_err = exc
            wait = backoff ** attempt
            logging.warning("attempt %s failed: %s; sleep %.1fs", attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"request failed after {retries} attempts: {last_err}")


def fetch_linename(session: requests.Session, key: str, city: str, keyword: str) -> Dict[str, Any]:
    params = {
        "key": key,
        "city": city,
        "keywords": keyword,
        "extensions": "all",
        "output": "JSON",
        "page": 1,
        "offset": 20,
    }
    logging.info("fetch linename city=%s keyword=%s", city, keyword)
    return request_json(session, LINE_NAME_URL, params)


def fetch_lineid(session: requests.Session, key: str, line_id: str) -> Dict[str, Any]:
    params = {"key": key, "id": line_id, "extensions": "all", "output": "JSON"}
    logging.info("fetch lineid id=%s", line_id)
    return request_json(session, LINE_ID_URL, params)


def fetch_traffic_road(
    session: requests.Session,
    key: str,
    adcode: str,
    name: str,
    level: int = 5,
    extensions: str = "all",
) -> Dict[str, Any]:
    params = {
        "key": key,
        "adcode": adcode,
        "name": name,
        "level": level,
        "extensions": extensions,
        "output": "JSON",
    }
    logging.info("fetch traffic road name=%s adcode=%s level=%s", name, adcode, level)
    return request_json(session, TRAFFIC_ROAD_URL, params)


def fetch_etd(
    session: requests.Session,
    key: str,
    origin: str,
    destination: str,
    firsttime: int,
    interval: int,
    count: int,
    strategy: int,
) -> Dict[str, Any]:
    params = {
        "key": key,
        "origin": origin,
        "destination": destination,
        "firsttime": firsttime,
        "interval": interval,
        "count": count,
        "strategy": strategy,
    }
    logging.info(
        "fetch etd origin=%s destination=%s firsttime=%s interval=%s count=%s strategy=%s",
        origin,
        destination,
        firsttime,
        interval,
        count,
        strategy,
    )
    return request_json(session, ETD_URL, params)


def iter_buslines(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    buslines = payload.get("buslines") or []
    if isinstance(buslines, list):
        for item in buslines:
            if isinstance(item, dict):
                yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaode bus downloader (229 route).")
    parser.add_argument("--key", help="Gaode Web service key (or set GAODE_KEY)")
    parser.add_argument("--city", default=DEFAULT_CITY, help="adcode/citycode (default Xi'an)")
    parser.add_argument("--keyword", default=DEFAULT_LINE_KEYWORD, help="bus line keyword")
    parser.add_argument("--line", default="229", help="line name for filenames")
    parser.add_argument("--interval", type=int, default=60, help="polling interval seconds")
    parser.add_argument("--duration", type=int, default=3600, help="total duration seconds")
    parser.add_argument("--strategy", type=int, default=1, help="etd strategy code")
    parser.add_argument("--etd", action="store_true", help="enable etd/driving snapshot per direction")
    parser.add_argument("--count", type=int, default=30, help="etd time point count (<=48)")
    parser.add_argument("--firsttime-offset", type=int, default=300, help="offset seconds from now for etd firsttime")
    parser.add_argument("--traffic-road", action="store_true", help="enable traffic/status/road polling by road name")
    parser.add_argument("--road-name", default="长安南路", help="road name for traffic/status/road")
    parser.add_argument("--road-level", type=int, default=5, help="road level for traffic/status/road (1-6)")
    args = parser.parse_args()

    api_key = args.key or os.getenv("GAODE_KEY")
    if not api_key:
        raise SystemExit("GAODE_KEY not provided via --key or env GAODE_KEY")

    setup_logging(args.line)
    logging.info("use key prefix=%s***", api_key[:4])

    with requests.Session() as session:
        end_ts = time.time() + args.duration
        round_idx = 0
        while True:
            now = dt.datetime.now()
            ts_str = now.strftime("%Y%m%d-%H%M%S")
            linename_payload = fetch_linename(session, api_key, args.city, args.keyword)
            save_json(
                linename_payload,
                RAW_DIR / f"{args.line}-all-{ts_str}-linename.json",
            )

            if args.traffic_road:
                traffic_payload = fetch_traffic_road(
                    session,
                    api_key,
                    adcode=args.city,
                    name=args.road_name,
                    level=args.road_level,
                    extensions="all",
                )
                save_json(
                    traffic_payload,
                    RAW_DIR / f"{args.line}-road-{sanitize(args.road_name)}-{ts_str}-traffic.json",
                )

            for busline in iter_buslines(linename_payload):
                direction_name = sanitize(busline.get("name", "unknown"))
                line_id = busline.get("id") or "unknown"

                lineid_payload = fetch_lineid(session, api_key, line_id)
                save_json(
                    lineid_payload,
                    RAW_DIR / f"{args.line}-{direction_name}-{ts_str}-lineid.json",
                )

                busstops = busline.get("busstops") or []
                if args.etd and busstops:
                    origin_loc = busstops[0].get("location")
                    dest_loc = busstops[-1].get("location")
                    if origin_loc and dest_loc:
                        firsttime = utc_ts(args.firsttime_offset)
                        etd_payload = fetch_etd(
                            session=session,
                            key=api_key,
                            origin=origin_loc,
                            destination=dest_loc,
                            firsttime=firsttime,
                            interval=max(10, args.interval),
                            count=min(args.count, 48),
                            strategy=args.strategy,
                        )
                        save_json(
                            etd_payload,
                            RAW_DIR / f"{args.line}-{direction_name}-{ts_str}-etd.json",
                        )
                    else:
                        logging.warning("skip etd: missing origin/destination for %s", direction_name)

            round_idx += 1
            if time.time() >= end_ts:
                break
            sleep_s = max(1, args.interval)
            logging.info("round %s done, sleep %ss (until %.0f)", round_idx, sleep_s, end_ts)
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
