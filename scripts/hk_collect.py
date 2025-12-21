"""
One-click HK data collector (static + ETA + road speeds + events + weather).

Outputs to data/raw/ with timestamp in filename.
Endpoints:
- KMB: route, route-stop (inbound/outbound), stop, route-eta
- Road speeds: IRN processed (irnAvgSpeed-all.xml via res.data.gov.hk), raw detector data (rawSpeedVol-all.xml via res.data.gov.hk, optional)
- Detector info: traffic_speed_volume_occ_info.csv
- Road segments: speed_segments_info.csv
- Special Traffic News: trafficnews.xml
- Journey Time Indicators: Journeytimev2.xml
- HKO rhrread (JSON)

Usage (examples):
  python scripts/hk_collect.py --routes 1,2,68X --eta-interval 60
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Iterable, Tuple

import requests

RAW_DIR = Path("data/raw")
BASE_KMB = "https://data.etabus.gov.hk/v1/transport/kmb"

URL_IRN_SPEED_DL = (
    "https://res.data.gov.hk/api/get-download-file?"
    "name=https%3A%2F%2Fresource.data.one.gov.hk%2Ftd%2Ftraffic-detectors%2FirnAvgSpeed-all.xml"
)
URL_TSM_RAW = "https://resource.data.one.gov.hk/td/traffic-detectors/traffic-speed-volume-occupancy.xml"
URL_TSM_RAW_DL = (
    "https://res.data.gov.hk/api/get-download-file?"
    "name=https%3A%2F%2Fresource.data.one.gov.hk%2Ftd%2Ftraffic-detectors%2FrawSpeedVol-all.xml"
)
URL_TSM_LOC = "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/traffic_speed_volume_occ_info.csv"
URL_TSM_SEG = "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/speed_segments_info.csv"
URL_STN = "https://www.td.gov.hk/en/special_news/trafficnews.xml"
URL_JTI = "https://resource.data.one.gov.hk/td/jss/Journeytimev2.xml"
URL_HKO = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
URL_JSON_BUS = "https://static.data.gov.hk/td/routes-fares-geojson/JSON_BUS.json"
URL_TSM_NOTIFICATION = "https://static.data.gov.hk/td/traffic-speed-map/notification.csv"
URL_TDAS_ROUTE = "https://tdas-api.hkemobility.gov.hk/tdas/api/route"


def ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_text(name: str, content: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    path.write_text(content, encoding="utf-8")
    return path


def save_bytes(name: str, content: bytes) -> Path:
    path = RAW_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def fetch(url: str, params=None, binary: bool = False) -> Tuple[int, str]:
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.status_code, resp.content if binary else resp.text


def _fetch_route_stop(route: str, bound: str, svc: str, timestamp: str, stop_ids: set):
    """抓 route-stop，同时收集 stop_id 到集合里。"""
    status, txt = fetch(f"{BASE_KMB}/route-stop/{route}/{bound}/{svc}")
    save_text(f"kmb-route-stop-{route}-{bound}-{timestamp}.json", txt)
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            for item in data.get("data", []):
                sid = item.get("stop")
                if sid:
                    stop_ids.add(sid)
    except Exception as e:
        print(f"[warn] parse route-stop {route}-{bound} failed: {e}")


def poll_once(routes: Iterable[str], svc: str) -> None:
    """按 hk_api_smoke 轮询一次，去掉已知 fail 的接口；全量路段站点都会抓 stop/stop-eta。"""
    timestamp = ts()
    stop_ids = set()
    print(f"[{timestamp}] poll start, routes={','.join(routes)}, svc={svc}")

    def safe_call(name: str, fn):
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:
            print(f"  warn {name} failed: {e}")

    safe_call(
        "kmb-route",
        lambda: save_text(f"kmb-route-{timestamp}.json", fetch(f"{BASE_KMB}/route")[1]),
    )
    for route in routes:
        for bound in ("inbound", "outbound"):
            safe_call(
                f"kmb-route-stop-{route}-{bound}",
                lambda r=route, b=bound: _fetch_route_stop(r, b, svc, timestamp, stop_ids),
            )
        safe_call(
            f"kmb-route-eta-{route}",
            lambda r=route: save_text(f"kmb-route-eta-{r}-{timestamp}.json", fetch(f"{BASE_KMB}/route-eta/{r}/{svc}")[1]),
        )

    print(f"  info collected route-stop, unique stops={len(stop_ids)}")

    # 全量站点 stop/stop-eta
    for sid in sorted(stop_ids):
        safe_call(
            f"kmb-stop-{sid}",
            lambda s=sid: save_text(f"kmb-stop-{s}-{timestamp}.json", fetch(f"{BASE_KMB}/stop/{s}")[1]),
        )
        safe_call(
            f"kmb-stop-eta-{sid}",
            lambda s=sid: save_text(f"kmb-stop-eta-{s}-{timestamp}.json", fetch(f"{BASE_KMB}/stop-eta/{s}")[1]),
        )

    safe_call(
        "tsm-notification",
        lambda: save_text(f"tsm-notification-{timestamp}.csv", fetch(URL_TSM_NOTIFICATION)[1]),
    )

    safe_call(
        "irnAvgSpeed-all",
        lambda: save_bytes(f"irnAvgSpeed-all-{timestamp}.xml", fetch(URL_IRN_SPEED_DL, binary=True)[1]),
    )

    safe_call(
        "rawSpeedVol-all",
        lambda: save_bytes(f"detector_locations/rawSpeedVol-all-{timestamp}.xml", fetch(URL_TSM_RAW_DL, binary=True)[1]),
    )

    safe_call(
        "tsm-detector-loc",
        lambda: save_text(f"traffic_speed_volume_occ_info-{timestamp}.csv", fetch(URL_TSM_LOC)[1]),
    )

    safe_call("stn", lambda: save_text(f"trafficnews-{timestamp}.xml", fetch(URL_STN)[1]))
    safe_call("jti", lambda: save_text(f"Journeytimev2-{timestamp}.xml", fetch(URL_JTI)[1]))
    safe_call(
        "hko-rhrread",
        lambda: save_text(
            f"hko-rhrread-{timestamp}.json", fetch(URL_HKO, params={"dataType": "rhrread", "lang": "en"})[1]
        ),
    )

    print(f"[{timestamp}] poll done, stops={len(stop_ids)}")

def main():
    ap = argparse.ArgumentParser(description="HK data collector (KMB ETA + speeds/events).")
    ap.add_argument("--routes", default="1", help="Comma separated KMB routes, default 1")
    ap.add_argument("--service-type", default="1", help="service_type for KMB (default 1)")
    ap.add_argument("--interval", type=int, default=60, help="轮询间隔（秒），默认 60")
    ap.add_argument("--duration", type=int, default=0, help="轮询总时长（秒），0 表示只抓取一次")
    args = ap.parse_args()

    target_time = datetime.datetime(2025, 12, 21, 17, 0, 0)
    print(f"[{datetime.datetime.now()}] Timer started. Waiting until {target_time} (UTC+8)...")
    while True:
        now = datetime.datetime.now()
        if now >= target_time:
            print(f"[{now}] Target time reached! Starting collection.")
            break
        time.sleep(1)

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]

    # 其他接口按 interval/duration 轮询
    if args.duration <= 0:
        poll_once(routes, args.service_type)
    else:
        start_time = time.time()
        end_time = start_time + args.duration
        next_tick = start_time
        while True:
            now = time.time()
            if now >= end_time:
                break
            # 对齐到预期起点；若超时则立即执行
            if now < next_tick:
                sleep_head = min(next_tick - now, end_time - now)
                if sleep_head > 0:
                    time.sleep(sleep_head)
            loop_start = time.time()
            poll_once(routes, args.service_type)
            elapsed = time.time() - loop_start
            # 保持轮询起点间隔约为 interval 秒
            next_tick += args.interval
            remaining = end_time - time.time()
            sleep_sec = min(max(next_tick - time.time(), 0), remaining)
            if sleep_sec <= 0:
                print(f"[info] poll elapsed {elapsed:.1f}s, interval {args.interval}s reached, continue immediately")
                continue
            print(f"[info] poll elapsed {elapsed:.1f}s, sleep {sleep_sec:.1f}s to align interval")
            time.sleep(sleep_sec)

    print("Done.")


if __name__ == "__main__":
    main()
