"""
osrm_router.py
==============
선택적 OSM driving 라우터 (로컬 osrm-backend 또는 공개 데모).

기본 엔진은 표준노드링크 DiGraph (`ROUTING_ENGINE=graph`).
안심구역(인터넷 차단)에서는 graph만 쓰고, OSRM은 로컬에 띄운 뒤에만 켠다.
"""

from __future__ import annotations

import os
from functools import lru_cache

from road_shapes import fetch_osrm_route, haversine_m

# 환경변수 ROUTING_ENGINE=graph|osrm  (기본 graph = 오프라인/안심구역)
ROUTING_ENGINE = os.environ.get("ROUTING_ENGINE", "graph").strip().lower()


def use_osrm() -> bool:
    return ROUTING_ENGINE in ("osrm", "osm", "online")


def route_latlng(
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    via: list[tuple[float, float]] | None = None,
    timeout: float = 12.0,
) -> dict | None:
    """(lat,lng) → OSRM 경로. 실패 시 None."""
    wps = [origin]
    if via:
        wps.extend(via)
    wps.append(destination)
    # 너무 가까우면 직선
    if haversine_m(origin[0], origin[1], destination[0], destination[1]) < 25 and not via:
        return {
            "coords": [[origin[0], origin[1]], [destination[0], destination[1]]],
            "legs": [{
                "coords": [[origin[0], origin[1]], [destination[0], destination[1]]],
                "duration_min": 0.2,
                "distance_m": haversine_m(origin[0], origin[1], destination[0], destination[1]),
            }],
            "duration_min": 0.2,
            "distance_m": haversine_m(origin[0], origin[1], destination[0], destination[1]),
            "source": "straight",
        }
    return fetch_osrm_route(wps, timeout=timeout)


@lru_cache(maxsize=512)
def _cached_pair(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> tuple | None:
    """캐시용: 좌표를 소수 5자리로 반올림."""
    r = route_latlng((a_lat, a_lng), (b_lat, b_lng))
    if not r:
        return None
    # tuple-ify for cache return (dicts not hashable as values ok)
    legs = []
    for lg in r["legs"]:
        legs.append((
            tuple((float(p[0]), float(p[1])) for p in lg["coords"]),
            float(lg["duration_min"]),
            float(lg["distance_m"]),
        ))
    full = tuple((float(p[0]), float(p[1])) for p in r["coords"])
    return (full, tuple(legs), float(r["duration_min"]), float(r["distance_m"]), r.get("source", "osrm"))


def route_pair(origin: tuple[float, float], destination: tuple[float, float]) -> dict | None:
    key = (
        round(origin[0], 5), round(origin[1], 5),
        round(destination[0], 5), round(destination[1], 5),
    )
    packed = _cached_pair(*key)
    if packed is None:
        # unrounded retry once
        return route_latlng(origin, destination)
    full, legs_t, dur, dist, src = packed
    legs = [
        {"coords": [[p[0], p[1]] for p in coords], "duration_min": dmin, "distance_m": dm}
        for coords, dmin, dm in legs_t
    ]
    return {
        "coords": [[p[0], p[1]] for p in full],
        "legs": legs,
        "duration_min": dur,
        "distance_m": dist,
        "source": src,
    }


def apply_congestion_factor(duration_min: float, congestion: float) -> float:
    """공개 혼잡을 ETA에 동일 공식으로 반영 (배차 비용 = 표시 ETA)."""
    c = max(0.0, min(1.0, float(congestion)))
    return float(duration_min) * (1.0 + 0.55 * c)
