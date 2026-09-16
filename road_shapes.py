"""
road_shapes.py
==============
도로 폴리라인 / OSRM 헬퍼.

출동 경로의 본체는 osrm_router.py (OSRM).
여기 모듈은 costmap 엣지 곡선 캐시·OSRM HTTP 호출을 담당한다.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import networkx as nx

_HERE = Path(__file__).resolve().parent
CACHE_PATH = _HERE / "data" / "open" / "edge_shapes.json"

# 이보다 짧은 직선은 OSM과 거의 같아 보간 불필요
MIN_CHORD_M = 90.0
# 이보다 긴 직선(고속도로 등)은 맵을 가로질러 보이므로 costmap에서 제외
# (시내 간선은 노드 간격이 길 수 있어 280m는 과하게 잘림 → 완화)
MAX_COSTMAP_CHORD_M = float(os.environ.get("COSTMAP_MAX_CHORD_M", "480"))

# 기본은 로컬 osrm-backend. 공개 project-osrm 은 안심구역에서 사용 금지.
OSRM_URL = os.environ.get(
    "OSRM_URL",
    "http://127.0.0.1:5000/route/v1/driving",
)


def haversine_m(lat1, lng1, lat2, lng2) -> float:
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lng2 - lng1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def edge_chord_m(G: nx.DiGraph, u, v) -> float:
    a, b = G.nodes[u], G.nodes[v]
    return haversine_m(a["lat"], a["lng"], b["lat"], b["lng"])


def _load_cache() -> dict:
    if CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _key(u, v) -> str:
    return f"{int(u)}-{int(v)}"


def fetch_osrm_polyline(lat1, lng1, lat2, lng2, timeout: float = 8.0) -> list[list[float]] | None:
    """OSRM GeoJSON → [[lat,lng], ...] Leaflet 순서."""
    route = fetch_osrm_route([(lat1, lng1), (lat2, lng2)], timeout=timeout)
    if not route:
        return None
    return route.get("coords")


def fetch_osrm_route(
    waypoints: list[tuple[float, float]],
    timeout: float = 12.0,
) -> dict | None:
    """다중 경유 OSRM.

    waypoints: [(lat,lng), ...]
    returns {
      coords: [[lat,lng],...],
      legs: [{coords, duration_min, distance_m}, ...],  # waypoint 사이
      duration_min, distance_m
    }
    """
    if len(waypoints) < 2:
        return None
    parts = [f"{lng},{lat}" for lat, lng in waypoints]
    coords = ";".join(parts)
    url = (
        f"{OSRM_URL}/{coords}?"
        + urllib.parse.urlencode({
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
        })
    )
    req = urllib.request.Request(url, headers={"User-Agent": "daegu-ems-viz/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if data.get("code") != "Ok":
        return None
    routes = data.get("routes") or []
    if not routes:
        return None
    route = routes[0]
    geo = (route.get("geometry") or {}).get("coordinates") or []
    full = [[float(lat), float(lng)] for lng, lat in geo]
    if len(full) < 2:
        return None

    # legs: OSRM returns duration(sec), distance(m) per waypoint pair; geometry is whole route.
    # Split full polyline by cumulative distance ratio of legs.
    raw_legs = route.get("legs") or []
    leg_meta = []
    for lg in raw_legs:
        leg_meta.append({
            "duration_min": float(lg.get("duration") or 0.0) / 60.0,
            "distance_m": float(lg.get("distance") or 0.0),
        })
    if not leg_meta:
        # single blob
        dist = float(route.get("distance") or 0.0)
        dur = float(route.get("duration") or 0.0) / 60.0
        leg_meta = [{"duration_min": max(dur, 0.2), "distance_m": dist}]

    leg_coords = _split_polyline_by_leg_distances(full, [x["distance_m"] for x in leg_meta])
    legs = []
    for meta, coords_leg in zip(leg_meta, leg_coords):
        if len(coords_leg) < 2:
            continue
        legs.append({
            "coords": coords_leg,
            "duration_min": max(float(meta["duration_min"]), 0.15),
            "distance_m": float(meta["distance_m"]),
        })
    if not legs:
        return None
    return {
        "coords": full,
        "legs": legs,
        "duration_min": sum(x["duration_min"] for x in legs),
        "distance_m": sum(x["distance_m"] for x in legs),
        "source": "osrm",
    }


def _split_polyline_by_leg_distances(
    coords: list[list[float]], distances_m: list[float],
) -> list[list[list[float]]]:
    """전체 polyline을 leg 거리 비율로 나눈다."""
    if len(coords) < 2:
        return [coords]
    if len(distances_m) <= 1:
        return [coords]

    # cumulative length along coords
    seg_lens = []
    for a, b in zip(coords[:-1], coords[1:]):
        seg_lens.append(haversine_m(a[0], a[1], b[0], b[1]))
    total = sum(seg_lens) or 1.0
    target_total = sum(max(0.0, d) for d in distances_m) or total
    # scale OSRM distances to polyline length
    scale = total / target_total if target_total > 0 else 1.0

    out: list[list[list[float]]] = []
    i = 0
    acc = 0.0
    start_pt = coords[0]
    for di, dist in enumerate(distances_m):
        need = max(0.0, dist) * scale
        chunk = [start_pt]
        if di == len(distances_m) - 1:
            chunk.extend(coords[i + 1 :])
            if len(chunk) < 2:
                chunk = [coords[-2], coords[-1]] if len(coords) >= 2 else coords
            out.append(chunk)
            break
        while i < len(seg_lens) and acc + seg_lens[i] < need - 1e-6:
            acc += seg_lens[i]
            i += 1
            chunk.append(coords[i])
        # interpolate remaining on current segment
        if i < len(seg_lens):
            remain = need - acc
            t = remain / seg_lens[i] if seg_lens[i] > 0 else 1.0
            t = max(0.0, min(1.0, t))
            a, b = coords[i], coords[i + 1]
            mid = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
            chunk.append(mid)
            start_pt = mid
            # consume partial
            seg_lens[i] = seg_lens[i] * (1 - t)
            coords[i] = mid  # local copy issue - coords is shared
            acc = 0.0
        if len(chunk) < 2:
            chunk.append(chunk[0])
        out.append(chunk)
    return out


def straight_coords(G: nx.DiGraph, u, v) -> list[list[float]]:
    a, b = G.nodes[u], G.nodes[v]
    return [[float(a["lat"]), float(a["lng"])], [float(b["lat"]), float(b["lng"])]]


def get_edge_coords(G: nx.DiGraph, u, v, cache: dict | None = None, fetch: bool = True) -> list[list[float]]:
    """시각화용 좌표열. 캐시/OSRM/직선."""
    cache = _load_cache() if cache is None else cache
    k = _key(u, v)
    if k in cache and isinstance(cache[k], list) and len(cache[k]) >= 2:
        return cache[k]

    chord = edge_chord_m(G, u, v)
    coords = straight_coords(G, u, v)
    if fetch and chord >= MIN_CHORD_M:
        a, b = G.nodes[u], G.nodes[v]
        shaped = fetch_osrm_polyline(a["lat"], a["lng"], b["lat"], b["lng"])
        if shaped and len(shaped) >= 2:
            coords = shaped
            cache[k] = coords
    return coords


def ensure_shapes_for_edges(
    G: nx.DiGraph,
    edges: Iterable[tuple],
    *,
    max_fetch: int = 350,
    max_chord_m: float = MAX_COSTMAP_CHORD_M,
    verbose: bool = True,
) -> dict:
    """costmap에 쓸 엣지들의 도로 곡선 캐시를 채운다."""
    cache = _load_cache()
    fetched = 0
    skipped_long = 0
    for u, v in edges:
        k = _key(u, v)
        if k in cache:
            continue
        if u not in G.nodes or v not in G.nodes:
            continue
        chord = edge_chord_m(G, u, v)
        if chord > max_chord_m:
            skipped_long += 1
            continue
        if chord < MIN_CHORD_M:
            cache[k] = straight_coords(G, u, v)
            continue
        if fetched >= max_fetch:
            break
        shaped = fetch_osrm_polyline(
            G.nodes[u]["lat"], G.nodes[u]["lng"],
            G.nodes[v]["lat"], G.nodes[v]["lng"],
        )
        cache[k] = shaped if shaped else straight_coords(G, u, v)
        fetched += 1
        if fetched % 40 == 0:
            _save_cache(cache)
            time.sleep(0.15)
    _save_cache(cache)
    if verbose:
        print(f"[road_shapes] cache={len(cache)} newly_fetched={fetched} skipped_long={skipped_long}")
    return cache


def path_geometry(
    G: nx.DiGraph,
    path: list,
    cache: dict | None = None,
    *,
    fetch: bool = False,
) -> tuple[list[list[float]], list[list[list[float]]]]:
    """노드 경로 → (전체 polyline, 그래프 엣지별 polyline).

    각 구간 양끝은 반드시 그래프 노드 좌표로 고정한다.
    (OSRM 끝이 노드와 어긋나면 사고 마커에 못 닿은 채 '도착' 처리되는 버그 방지)
    """
    cache = _load_cache() if cache is None else cache
    if not path or len(path) < 2:
        return [], []
    segments: list[list[list[float]]] = []
    full: list[list[float]] = []
    for u, v in zip(path[:-1], path[1:]):
        a = straight_coords(G, u, v)[0]
        b = straight_coords(G, u, v)[1]
        seg = get_edge_coords(G, u, v, cache=cache, fetch=fetch)
        if len(seg) < 2:
            seg = [a, b]
        else:
            seg = [[float(x), float(y)] for x, y in seg]
            seg[0] = a
            seg[-1] = b
        segments.append(seg)
        if not full:
            full.extend(seg)
        else:
            full.extend(seg[1:])
    return full, segments


def path_latlng_shaped(
    G: nx.DiGraph,
    path: list,
    cache: dict | None = None,
    *,
    fetch: bool = False,
) -> list[list[float]]:
    """노드 경로 → 도로를 따라가는 latlng (캐시된 shape 연결)."""
    full, _ = path_geometry(G, path, cache=cache, fetch=fetch)
    return full


def should_draw_on_costmap(G: nx.DiGraph, u, v, max_chord_m: float = MAX_COSTMAP_CHORD_M) -> bool:
    """맵을 가로지르는 긴 직선(고속 등)은 costmap에서 숨김."""
    if u not in G or v not in G:
        return False
    road = str(G.edges[u, v].get("road_name") or "")
    if "고속도로" in road:
        return False
    return edge_chord_m(G, u, v) <= max_chord_m
