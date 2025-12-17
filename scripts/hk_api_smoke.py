"""
Minimal smoke test for HK data sources (KMB ETA + data.gov.hk direct links).

Checks:
- KMB route / route-stop / stop / route-eta / stop-eta
- Public transport GeoJSON (JSON_BUS direct)
- Traffic Speed Map (notification.csv only)
- IRN average speed (irnAvgSpeed-all.xml)
- Special Traffic News (XML direct)
- HKO real-time weather (rhrread)
- Journey Time Indicators (Journeytimev2.xml)
- TDAS route API (optional; may be 403)
"""

from __future__ import annotations

import json
import requests

BASE_KMB = "https://data.etabus.gov.hk/v1/transport/kmb"
URL_JSON_BUS = "https://static.data.gov.hk/td/routes-fares-geojson/JSON_BUS.json"
URL_TSM_NOTIFICATION = "https://static.data.gov.hk/td/traffic-speed-map/notification.csv"
URL_TSM_RAW = "https://resource.data.one.gov.hk/td/traffic-detectors/traffic-speed-volume-occupancy.xml"
URL_TSM_RAW_DL = (
    "https://res.data.gov.hk/api/get-download-file?"
    "name=https%3A%2F%2Fresource.data.one.gov.hk%2Ftd%2Ftraffic-detectors%2FrawSpeedVol-all.xml"
)
URL_TSM_LOC = "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/traffic_speed_volume_occ_info.csv"
URL_TSM_SEG = "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/speed_segments_info.csv"
URL_IRN_SPEED = "https://resource.data.one.gov.hk/td/traffic-detectors/irnAvgSpeed-all.xml"
URL_IRN_SPEED_DL = (
    "https://res.data.gov.hk/api/get-download-file?"
    "name=https%3A%2F%2Fresource.data.one.gov.hk%2Ftd%2Ftraffic-detectors%2FirnAvgSpeed-all.xml"
)
URL_STN = "https://www.td.gov.hk/en/special_news/trafficnews.xml"
URL_JTI = "https://resource.data.one.gov.hk/td/jss/Journeytimev2.xml"
URL_TDAS_ROUTE = "https://tdas-api.hkemobility.gov.hk/tdas/api/route"


def hit(name: str, url: str, params=None):
    try:
        resp = requests.get(url, params=params, timeout=10)
        entry = {"status": resp.status_code}
        if resp.ok:
            ct = resp.headers.get("Content-Type", "")
            if "json" in ct:
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        entry["keys"] = list(data.keys())[:5]
                        if "data" in data and isinstance(data["data"], list):
                            entry["count"] = len(data["data"])
                    elif isinstance(data, list):
                        entry["len"] = len(data)
                except Exception:
                    entry["text"] = resp.text[:200]
            else:
                entry["text"] = resp.text[:200]
        else:
            entry["text"] = resp.text[:200]
    except Exception as e:  # pragma: no cover - diagnostic
        entry = {"error": str(e)}
    print(f"[{name}] {entry}")
    return entry


def main():
    summary = {}
    # KMB endpoints
    summary["route"] = hit("route", f"{BASE_KMB}/route")
    summary["route-stop"] = hit("route-stop", f"{BASE_KMB}/route-stop/1/outbound/1")
    summary["stop"] = hit("stop", f"{BASE_KMB}/stop/C88E34E485B43EFB")
    summary["route-eta"] = hit("route-eta", f"{BASE_KMB}/route-eta/1/1")
    summary["stop-eta"] = hit("stop-eta", f"{BASE_KMB}/stop-eta/C88E34E485B43EFB")

    # Public transport GeoJSON (JSON_BUS direct)
    summary["json_bus"] = hit("json_bus", URL_JSON_BUS)

    # Traffic Speed Map notification (5min) - 2nd Gen reminder
    summary["tsm_notification"] = hit("tsm_notification", URL_TSM_NOTIFICATION)

    # IRN average speed (processed, 2min)
    summary["irn_avg_speed"] = hit("irn_avg_speed", URL_IRN_SPEED)
    summary["irn_avg_speed_dl"] = hit("irn_avg_speed_dl", URL_IRN_SPEED_DL)
    # Detector locations (CSV)
    summary["detector_locations"] = hit("detector_locations", URL_TSM_LOC)
    # Road network segments (CSV)
    summary["road_network_segments"] = hit("road_network_segments", URL_TSM_SEG)
    # Raw speed/volume/occupancy (optional)
    summary["tsm_raw"] = hit("tsm_raw", URL_TSM_RAW)
    summary["tsm_raw_dl"] = hit("tsm_raw_dl", URL_TSM_RAW_DL)

    # Special Traffic News (XML direct)
    summary["stn"] = hit("stn", URL_STN)

    # HKO real-time weather
    summary["hko_rhrread"] = hit(
        "hko_rhrread",
        "https://data.weather.gov.hk/weatherAPI/opendata/weather.php",
        params={"dataType": "rhrread", "lang": "en"},
    )

    # Journey Time Indicators (2nd gen)
    summary["jti"] = hit("jti", URL_JTI)

    # TDAS driving route API (connectivity probe; may be 403)
    summary["tdas-route"] = hit("tdas-route", URL_TDAS_ROUTE)

    # Simple pass/fail judgment: HTTP 2xx -> pass, else fail.
    verdict = {
        k: ("pass" if v.get("status", 0) >= 200 and v.get("status", 0) < 300 else "fail")
        for k, v in summary.items()
    }
    print("\nSummary:\n", json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nVerdict (pass/fail):\n", json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
