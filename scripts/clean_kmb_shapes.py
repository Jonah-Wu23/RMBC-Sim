"""
Build KMB route-stop table with coordinates and cumulative distance using IRN road network.

Sources:
- data/raw/kmb-route-stop-<route>-<bound>-<ts>.json  (seq + stop)
- data/raw/kmb-stop-<stop>-<ts>.json                 (lat/long + names)
- data/RdNet_IRNP.gdb                                (centerline network，EPSG:2326)

Output:
- data/processed/kmb_route_stop_dist.csv with columns:
  route,bound,service_type,seq,stop_id,stop_name_en,stop_name_tc,lat,long,cum_dist_m,link_dist_m

Usage:
  python scripts/clean_kmb_shapes.py --routes 68X,960 --service-type 1
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from time import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

RAW_DIR = Path("data/raw")
IRN_GDB = Path("data/RdNet_IRNP.gdb")
PROCESSED_DIR = Path("data/processed")


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate great-circle distance between two points in meters."""
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def normalize_xy(x: float, y: float, ndigits: int = 3) -> Tuple[float, float]:
    """Round coordinates to merge floating duplicates when building graph."""
    return (round(x, ndigits), round(y, ndigits))


def latest_file(pattern: str) -> Path:
    matches = sorted(RAW_DIR.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_route_stop(route: str, bound: str, service_type: str) -> List[Tuple[int, str]]:
    fname = f"kmb-route-stop-{route}-{bound}-*.json"
    path = latest_file(fname)
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("data", [])
    result = []
    for item in items:
        seq = item.get("seq")
        stop_id = item.get("stop")
        if seq is None or stop_id is None:
            continue
        result.append((int(seq), str(stop_id)))
    result.sort(key=lambda x: x[0])
    return result


def load_stop(stop_id: str) -> Dict[str, str]:
    fname = f"kmb-stop-{stop_id}-*.json"
    path = latest_file(fname)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("data", {})


@dataclass
class StopProjection:
    node_id: str
    point: Point
    connect_edges: List[Tuple[Tuple[float, float], float]]  # (neighbor_node, weight_m)


def build_centerline_graph() -> Tuple[nx.Graph, gpd.GeoDataFrame]:
    """Build graph from IRN CENTERLINE segments (undirected)."""
    print("[info] loading CENTERLINE layer ...")
    t0 = time()
    gdf = gpd.read_file(IRN_GDB, layer="CENTERLINE")
    gdf = gdf.explode(index_parts=False)
    print(f"[info] centerline rows={len(gdf)}, time={time()-t0:.1f}s")
    segments = []
    G = nx.Graph()
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                p1 = normalize_xy(x1, y1)
                p2 = normalize_xy(x2, y2)
                seg = LineString([p1, p2])
                length = seg.length
                segments.append({"u": p1, "v": p2, "geometry": seg, "length": length})
                G.add_edge(p1, p2, length=length)
    seg_gdf = gpd.GeoDataFrame(segments, geometry="geometry", crs=gdf.crs)
    print(f"[info] graph built: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    return G, seg_gdf


def project_stop_to_graph(
    G: nx.Graph, seg_gdf: gpd.GeoDataFrame, stop_id: str, lon: float, lat: float, transformer
) -> StopProjection:
    """Project stop to nearest centerline segment, connect to graph endpoints."""
    pt_wgs = Point(lon, lat)
    x, y = transformer.transform(lon, lat)  # always_xy=True so pass lon,lat
    pt = Point(normalize_xy(x, y))
    # nearest segment (fallback: brute-force distance)
    try:
        idx_iter = seg_gdf.sindex.nearest(pt, return_all=False)
        idx = next(idx_iter)[0]
    except Exception:
        idx = seg_gdf.geometry.distance(pt).idxmin()
    seg = seg_gdf.iloc[idx]
    line: LineString = seg.geometry
    # project point to segment
    dist_on_line = line.project(pt)
    p_near_on_line = line.interpolate(dist_on_line)
    p_near = Point(normalize_xy(p_near_on_line.x, p_near_on_line.y))
    # offsets
    u = normalize_xy(line.coords[0][0], line.coords[0][1])
    v = normalize_xy(line.coords[-1][0], line.coords[-1][1])
    line_full = LineString([u, v])
    offset = line_full.project(p_near)
    total = line_full.length
    connect = [
        (u, offset),
        (v, max(total - offset, 0.0)),
    ]
    node_id = f"stop::{stop_id}"
    return StopProjection(node_id=node_id, point=p_near, connect_edges=connect)


def build_table(routes: Iterable[str], service_type: str) -> List[Dict[str, object]]:
    G, seg_gdf = build_centerline_graph()
    rows: List[Dict[str, object]] = []

    # transformer: WGS84 -> EPSG:2326
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)

    for route in routes:
        for bound in ("inbound", "outbound"):
            print(f"[info] processing route={route} bound={bound}")
            seq_stop = load_route_stop(route, bound, service_type)
            # add projected stops to graph
            stop_nodes = {}
            for seq, stop_id in seq_stop:
                stop_info = load_stop(stop_id)
                lat = float(stop_info.get("lat"))
                lon = float(stop_info.get("long"))
                proj = project_stop_to_graph(G, seg_gdf, stop_id, lon, lat, transformer)
                if proj.node_id not in G:
                    for neigh, w in proj.connect_edges:
                        G.add_edge(proj.node_id, neigh, length=w)
                stop_nodes[seq] = (proj, stop_info)

            # compute cumulative distances along shortest paths
            cum = 0.0
            prev_node = None
            for seq in sorted(stop_nodes):
                proj, stop_info = stop_nodes[seq]
                if prev_node is None:
                    path_len = 0.0
                else:
                    try:
                        path_len = nx.shortest_path_length(G, prev_node, proj.node_id, weight="length")
                    except nx.NetworkXNoPath:
                        path_len = float("nan")
                if not math.isnan(path_len):
                    cum += path_len
                rows.append(
                    {
                        "route": route,
                        "bound": bound,
                        "service_type": service_type,
                        "seq": seq,
                        "stop_id": proj.node_id.split("::", 1)[1],
                        "stop_name_en": stop_info.get("name_en", ""),
                        "stop_name_tc": stop_info.get("name_tc", ""),
                        "lat": float(stop_info.get("lat")),
                        "long": float(stop_info.get("long")),
                        "cum_dist_m": "" if math.isnan(cum) else round(cum, 3),
                        "link_dist_m": "" if math.isnan(path_len) else round(path_len, 3),
                    }
                )
                prev_node = proj.node_id
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "route",
        "bound",
        "service_type",
        "seq",
        "stop_id",
        "stop_name_en",
        "stop_name_tc",
        "lat",
        "long",
        "cum_dist_m",
        "link_dist_m",
    ]
    lines = [",".join(headers)]
    for r in rows:
        vals = [r.get(h, "") for h in headers]
        # simple CSV escaping for commas
        def esc(v: object) -> str:
            s = "" if v is None else str(v)
            return f"\"{s}\"" if "," in s else s

        lines.append(",".join(esc(v) for v in vals))
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KMB route-stop distance table from raw data and GeoJSON.")
    parser.add_argument("--routes", default="68X,960", help="Comma separated route list, default 68X,960")
    parser.add_argument("--service-type", default="1", help="service_type (default 1)")
    args = parser.parse_args()

    routes = [r.strip() for r in re.split(r"[;,]", args.routes) if r.strip()]
    rows = build_table(routes, args.service_type)
    write_csv(rows, PROCESSED_DIR / "kmb_route_stop_dist.csv")
    print("Done.")


if __name__ == "__main__":
    main()
