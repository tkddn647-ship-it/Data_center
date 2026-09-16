"""
dispatch_center.py
====================
사용자가 "중앙관제센터" 역할을 맡아 사고 좌표를 직접 지정하고, 그 사고가
화재인지/인명피해(구급)인지/둘 다인지에 따라 알맞은 차량을 실제 대구 도로망
위에서 출동시키는 시뮬레이터. 학습된 RL 모델이 없어도(혼잡가중 다익스트라로)
바로 돌려볼 수 있다 — "학습 직전" 단계에서 전체 그림을 미리 확인하기 위한 도구.

핵심
----
1) 사고는 **여러 곳** 동시에 지정 가능 (각각 유형·규모가 달라도 됨).
2) 전역 최적화: 어느 소방서에서 나갈지 / 어느 응급실로 이송할지 /
   경로가 서로 안 겹치게 할지를 사고 전체를 보고 함께 결정.
3) 인명피해(구급)는 소방서→현장→응급실까지, 화재는 소방서→화재현장까지.

사용법
------
    # 사고 1곳 (기존과 동일)
    python dispatch_center.py --lat 35.87 --lng 128.60 --incident-type mixed_major --save dispatch.gif --no-show

    # 사고 여러 곳: TYPE:lat,lng 를 --incident 로 반복
    python dispatch_center.py ^
        --incident fire_1:35.87,128.60 ^
        --incident medical_1:35.85,128.55 ^
        --incident mixed_2:35.90,128.62 ^
        --save multi_dispatch.gif --no-show
"""

from __future__ import annotations
import argparse
import heapq
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from network_graph import build_combined_graph
from daegu_ems_geo import DAEGU_FIRE_STATIONS, DAEGU_ER_HOSPITALS
from standard_node_link import DAEGU_LAT_MIN, DAEGU_LAT_MAX, DAEGU_LNG_MIN, DAEGU_LNG_MAX, nearest_node


# ---------------------------------------------------------------------- #
# 1) 사고 유형·규모 프리셋
# ---------------------------------------------------------------------- #
@dataclass
class IncidentSpec:
    needs_fire: bool
    needs_medical: bool
    n_fire_trucks: int = 1
    n_ambulances: int = 1
    label: str = ""
    # 숫자가 클수록 먼저 배차 (대형사고 우선)
    severity: int = 1


INCIDENT_PRESETS: dict[str, IncidentSpec] = {
    "fire_1":        IncidentSpec(True,  False, 1, 0, "1단계 화재 (단순)", severity=1),
    "fire_major":    IncidentSpec(True,  False, 2, 0, "3단계 화재 (대형)", severity=3),
    "medical_1":     IncidentSpec(False, True,  0, 1, "1단계 구급 (단순 사고/인명피해 1명)", severity=1),
    "medical_major": IncidentSpec(False, True,  0, 2, "3단계 구급 (다수 인명피해)", severity=3),
    "mixed_2":       IncidentSpec(True,  True,  1, 1, "2단계 복합 (화재+인명피해, 예: 차량화재)", severity=2),
    "mixed_major":   IncidentSpec(True,  True,  2, 2, "3단계 대형복합 (다수 사상자 화재사고)", severity=3),
}


@dataclass
class Incident:
    """한 건의 사고 — 위치 + 유형."""
    incident_id: int
    node: int
    type_key: str
    spec: IncidentSpec
    lat: float = 0.0
    lng: float = 0.0


@dataclass
class UnitRequest:
    """배차가 필요한 차량 1대 분량의 요청 (전역 할당 전에 모음)."""
    incident_id: int
    incident_node: int
    vehicle_type: str          # "fire_truck" | "ambulance"
    unit_index: int            # 같은 사고·같은 차종 내 순번 (1-based)
    severity: int
    label: str


# ---------------------------------------------------------------------- #
# 2) 사고 좌표·목록 지정
# ---------------------------------------------------------------------- #
def pick_incidents_interactively(G: nx.DiGraph, default_type: str = "medical_1",
                                  max_clicks: int = 8) -> list[tuple[str, int]]:
    """지도에서 여러 번 클릭. Enter/우클릭으로 종료. 모두 같은 default_type 적용."""
    pos = {n: (d["lng"], d["lat"]) for n, d in G.nodes(data=True)}
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.4, arrows=False, ax=ax, edge_color="gray")
    for s in DAEGU_FIRE_STATIONS:
        ax.plot(s["lng"], s["lat"], "s", color="navy", markersize=8)
    for h in DAEGU_ER_HOSPITALS:
        ax.plot(h["lng"], h["lat"], "^", color="crimson", markersize=8)
    ax.set_title(f"사고 위치를 클릭 (최대 {max_clicks}곳, Enter로 종료) — type={default_type}")
    ax.set_xlim(DAEGU_LNG_MIN, DAEGU_LNG_MAX)
    ax.set_ylim(DAEGU_LAT_MIN, DAEGU_LAT_MAX)

    pts = plt.ginput(max_clicks, timeout=0)
    plt.close(fig)
    if not pts:
        raise RuntimeError("클릭이 감지되지 않았습니다.")
    out = []
    for lng, lat in pts:
        node = nearest_node(G, lat, lng)
        print(f"[dispatch_center] 클릭 (lat={lat:.5f}, lng={lng:.5f}) -> node {node}")
        out.append((default_type, node))
    return out


def parse_incident_arg(raw: str) -> tuple[str, Optional[int], Optional[float], Optional[float]]:
    """'fire_1:35.87,128.60' 또는 'medical_1:node:1500004600' 파싱."""
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"--incident 형식: TYPE:lat,lng 또는 TYPE:node:ID  (받은 값: {raw})"
        )
    type_key, rest = raw.split(":", 1)
    type_key = type_key.strip()
    if type_key not in INCIDENT_PRESETS:
        raise argparse.ArgumentTypeError(
            f"알 수 없는 사고 유형 '{type_key}'. 선택지: {list(INCIDENT_PRESETS)}"
        )
    rest = rest.strip()
    if rest.startswith("node:"):
        return type_key, int(rest.split(":", 1)[1]), None, None
    if "," not in rest:
        raise argparse.ArgumentTypeError(f"좌표는 lat,lng 형식이어야 합니다: {raw}")
    lat_s, lng_s = rest.split(",", 1)
    return type_key, None, float(lat_s), float(lng_s)


def resolve_incidents(G: nx.DiGraph, args) -> list[Incident]:
    """CLI 인자를 Incident 목록으로 변환. --incident 여러 개 우선, 없으면 단일 사고 모드."""
    items: list[tuple[str, int]] = []

    if args.incident:
        for type_key, node_id, lat, lng in args.incident:
            if node_id is not None:
                node = int(node_id)
            else:
                node = nearest_node(G, lat, lng)
            items.append((type_key, node))
    elif args.interactive:
        items = pick_incidents_interactively(G, default_type=args.incident_type)
    else:
        # 단일 사고 하위호환
        if args.node is not None:
            node = int(args.node)
        elif args.lat is not None and args.lng is not None:
            node = nearest_node(G, args.lat, args.lng)
        else:
            rng = np.random.default_rng(args.seed)
            candidates = [
                n for n in G.nodes
                if 35.82 <= G.nodes[n]["lat"] <= 35.92 and 128.52 <= G.nodes[n]["lng"] <= 128.68
            ]
            node = int(rng.choice(candidates))
        items.append((args.incident_type, node))

    incidents = []
    for i, (type_key, node) in enumerate(items, start=1):
        spec = INCIDENT_PRESETS[type_key]
        incidents.append(Incident(
            incident_id=i, node=node, type_key=type_key, spec=spec,
            lat=float(G.nodes[node]["lat"]), lng=float(G.nodes[node]["lng"]),
        ))
    return incidents


# ---------------------------------------------------------------------- #
# 3) 전역 배정 최적화 — 소방서 / 응급실 / 경로 겹침
# ---------------------------------------------------------------------- #
EMPTY_STATION_PENALTY = 1e6    # 잔여 0인 서는 사실상 제외 (전 서 고갈 시에만 허용)
HOSPITAL_LOAD_PENALTY = 18.0   # 같은 응급실로 여러 대 몰리면 비용↑
OVERLAP_PENALTY = 18.0         # 같은 도로 동시 사용 강하게 회피 (응급차 충돌 방지)
ALLEY_FACTOR = 1.55            # ~길 / 번길 은 간선보다 비싸게 (빙빙 골목 억제)
TURN_PENALTY = 1.15            # 급커브(분 단위 가산) — 지그재그 억제
U_TURN_PENALTY = 3.0
EMPTY_STATION_PENALTY = 1e6    # 잔여 0인 서는 사실상 제외 (전 서 고갈 시에만 허용)


@dataclass
class StationFleet:
    """소방서별 보유·잔여 차량. 119 구급은 소방서에서만 출발(병원 출발 아님)."""
    capacity_amb: dict[int, int] = field(default_factory=dict)
    capacity_fire: dict[int, int] = field(default_factory=dict)
    avail_amb: dict[int, int] = field(default_factory=dict)
    avail_fire: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_stations(cls, stations: list[dict] | None = None) -> "StationFleet":
        stations = stations if stations is not None else DAEGU_FIRE_STATIONS
        fle = cls()
        for s in stations:
            sid = int(s["station_id"])
            amb = int(s.get("ambulances", 4) or 4)
            fire = int(s.get("fire_trucks", 4) or 4)
            fle.capacity_amb[sid] = amb
            fle.capacity_fire[sid] = fire
            fle.avail_amb[sid] = amb
            fle.avail_fire[sid] = fire
        return fle

    def available(self, station_id: int, vehicle_type: str) -> int:
        if vehicle_type == "ambulance":
            return int(self.avail_amb.get(station_id, 0))
        return int(self.avail_fire.get(station_id, 0))

    def take(self, station_id: int, vehicle_type: str) -> bool:
        """출동 1대 차감. 잔여 없으면 False."""
        bag = self.avail_amb if vehicle_type == "ambulance" else self.avail_fire
        if bag.get(station_id, 0) <= 0:
            return False
        bag[station_id] -= 1
        return True

    def release(self, station_id: int, vehicle_type: str) -> None:
        """복귀 시 1대 환원 (보유 상한 초과 금지)."""
        if vehicle_type == "ambulance":
            cap, bag = self.capacity_amb, self.avail_amb
        else:
            cap, bag = self.capacity_fire, self.avail_fire
        bag[station_id] = min(cap.get(station_id, 0), bag.get(station_id, 0) + 1)

    def snapshot(self, station_names: dict[int, str] | None = None) -> list[dict]:
        names = station_names or {}
        rows = []
        for sid in sorted(set(self.capacity_amb) | set(self.capacity_fire)):
            rows.append({
                "station_id": sid,
                "station_name": names.get(sid, f"station#{sid}"),
                "ambulances": {
                    "capacity": self.capacity_amb.get(sid, 0),
                    "available": self.avail_amb.get(sid, 0),
                    "out": self.capacity_amb.get(sid, 0) - self.avail_amb.get(sid, 0),
                },
                "fire_trucks": {
                    "capacity": self.capacity_fire.get(sid, 0),
                    "available": self.avail_fire.get(sid, 0),
                    "out": self.capacity_fire.get(sid, 0) - self.avail_fire.get(sid, 0),
                },
            })
        return rows

    def totals(self) -> dict:
        return {
            "ambulances": {
                "capacity": sum(self.capacity_amb.values()),
                "available": sum(self.avail_amb.values()),
            },
            "fire_trucks": {
                "capacity": sum(self.capacity_fire.values()),
                "available": sum(self.avail_fire.values()),
            },
        }


def _is_alley(road: str) -> bool:
    r = (road or "").strip()
    if not r:
        return False
    return r.endswith("길") or "번길" in r


def _edge_cost(u, v, d) -> float:
    tt = float(d.get("travel_time", 1.0))
    length_m = max(float(d.get("length_m", 400.0)), 20.0)
    # API/통계 오매칭으로 travel_time만 비정상이면 골목·블록순회 우회가 생김
    sensible_min = (length_m / 1000.0) / 70.0 * 60.0
    sensible_max = (length_m / 1000.0) / 8.0 * 60.0
    tt = min(max(tt, sensible_min * 0.4), sensible_max)
    base = tt * (1.0 + float(d.get("congestion", 0.0)))
    if _is_alley(str(d.get("road_name") or "")):
        base *= ALLEY_FACTOR
    return base


def _bearing(G: nx.DiGraph, a, b) -> float:
    return math.atan2(
        G.nodes[b]["lng"] - G.nodes[a]["lng"],
        G.nodes[b]["lat"] - G.nodes[a]["lat"],
    )


def _turn_extra(G: nx.DiGraph, prev, u, v) -> float:
    """50° 이상 꺾이면 페널티 — 성긴 노드를 지그재그로 잇는 경로 억제."""
    d = abs(_bearing(G, prev, u) - _bearing(G, u, v))
    if d > math.pi:
        d = 2 * math.pi - d
    if d < math.radians(50):
        return 0.0
    return TURN_PENALTY * (d / (math.pi / 2))


def _path_cost(G: nx.DiGraph, path: list) -> float:
    if len(path) < 2:
        return 0.0
    total = _edge_cost(path[0], path[1], G.edges[path[0], path[1]])
    for i in range(1, len(path) - 1):
        u, v = path[i], path[i + 1]
        total += _edge_cost(u, v, G.edges[u, v])
        total += _turn_extra(G, path[i - 1], u, v)
        if path[i - 1] == v:
            total += U_TURN_PENALTY
    return total


def _route_cost(G: nx.DiGraph, src: int, dst: int, edge_usage: Counter | None = None) -> tuple[list, float]:
    """혼잡·골목·급커브·겹침을 반영한 경로. (노드,직전노드) 상태로 턴 비용 반영."""
    usage = edge_usage or Counter()
    if src not in G or dst not in G:
        return [src], float("inf")
    if src == dst:
        return [src], 0.0

    # state = (node, prev) ; start prev=None
    pq: list[tuple[float, int, Optional[int]]] = [(0.0, src, None)]
    best: dict[tuple[int, Optional[int]], float] = {(src, None): 0.0}
    came: dict[tuple[int, Optional[int]], tuple[int, Optional[int]]] = {}
    found: tuple[int, Optional[int]] | None = None

    while pq:
        cost, u, prev = heapq.heappop(pq)
        if cost != best.get((u, prev), float("inf")):
            continue
        if u == dst:
            found = (u, prev)
            break
        for _, v, d in G.out_edges(u, data=True):
            step = _edge_cost(u, v, d) + usage.get((u, v), 0) * OVERLAP_PENALTY
            if prev is not None:
                step += _turn_extra(G, prev, u, v)
                if v == prev:
                    step += U_TURN_PENALTY
            nd = cost + step
            st = (v, u)
            if nd < best.get(st, float("inf")):
                best[st] = nd
                came[st] = (u, prev)
                heapq.heappush(pq, (nd, v, u))

    if found is None:
        cands = [(c, st) for st, c in best.items() if st[0] == dst]
        if not cands:
            return [src], float("inf")
        _, found = min(cands)

    path: list[int] = []
    st: tuple[int, Optional[int]] | None = found
    while st is not None:
        path.append(st[0])
        st = came.get(st)
    path.reverse()
    return path, best[found]


def _euclid2(G: nx.DiGraph, a: int, b: int) -> float:
    return (G.nodes[a]["lat"] - G.nodes[b]["lat"]) ** 2 + (G.nodes[a]["lng"] - G.nodes[b]["lng"]) ** 2


def _collect_unit_requests(incidents: list[Incident]) -> list[UnitRequest]:
    reqs: list[UnitRequest] = []
    for inc in incidents:
        sp = inc.spec
        if sp.needs_fire:
            for i in range(1, sp.n_fire_trucks + 1):
                reqs.append(UnitRequest(
                    incident_id=inc.incident_id, incident_node=inc.node,
                    vehicle_type="fire_truck", unit_index=i, severity=sp.severity,
                    label=f"I{inc.incident_id}-Fire-{i}",
                ))
        if sp.needs_medical:
            for i in range(1, sp.n_ambulances + 1):
                reqs.append(UnitRequest(
                    incident_id=inc.incident_id, incident_node=inc.node,
                    vehicle_type="ambulance", unit_index=i, severity=sp.severity,
                    label=f"I{inc.incident_id}-Amb-{i}",
                ))
    # 대형사고·화재 우선 배차 (자원 선점)
    reqs.sort(key=lambda r: (-r.severity, 0 if r.vehicle_type == "fire_truck" else 1, r.incident_id))
    return reqs


def _station_xy(sid: int) -> tuple[float, float]:
    for s in DAEGU_FIRE_STATIONS:
        if int(s["station_id"]) == int(sid):
            return float(s["lat"]), float(s["lng"])
    raise KeyError(sid)


def _dispatch_origin_xy(sid: int, scene_xy: tuple[float, float]) -> tuple[float, float]:
    """소방서 재고(sid)는 유지하되, 현장에 더 가까운 119안전센터가 있으면 거기서 출발."""
    base = _station_xy(sid)
    try:
        from ems_extra_geo import nearest_safety_center
        from road_shapes import haversine_m
        c = nearest_safety_center(scene_xy[0], scene_xy[1])
        if not c:
            return base
        d_center = float(c.get("dist_m") or haversine_m(scene_xy[0], scene_xy[1], c["lat"], c["lng"]))
        d_station = haversine_m(scene_xy[0], scene_xy[1], base[0], base[1])
        # 센터가 더 가깝고, 소속 소방서와도 너무 멀지 않을 때
        if d_center + 80 < d_station and d_center < 12000:
            return float(c["lat"]), float(c["lng"])
    except Exception:
        pass
    return base


def _hospital_xy(hid: int) -> tuple[float, float]:
    for h in DAEGU_ER_HOSPITALS:
        if int(h["hospital_id"]) == int(hid):
            return float(h["lat"]), float(h["lng"])
    raise KeyError(hid)


def _scene_cong(G: nx.DiGraph, lat: float, lng: float) -> float:
    """사고 지점 근처 그래프 혼잡 (보정용)."""
    try:
        n = nearest_node(G, lat, lng)
    except Exception:
        return 0.35
    vals = []
    for u, v, d in list(G.in_edges(n, data=True))[:8]:
        vals.append(float(d.get("congestion", 0.35)))
    for u, v, d in list(G.out_edges(n, data=True))[:8]:
        vals.append(float(d.get("congestion", 0.35)))
    if not vals:
        return 0.35
    return float(sum(vals) / len(vals))


def _assign_station_osrm(
    G: nx.DiGraph,
    station_map: dict,
    scene_xy: tuple[float, float],
    fleet: StationFleet,
    vehicle_type: str,
    congestion: float,
) -> tuple[int, dict, float]:
    """잔여 차량 있는 서 중 OSRM ETA(혼잡보정) 최소."""
    from osrm_router import apply_congestion_factor, route_pair
    from road_shapes import haversine_m

    best = None
    for sid in station_map:
        try:
            sxy = _dispatch_origin_xy(sid, scene_xy)
        except KeyError:
            continue
        r = route_pair(sxy, scene_xy)
        if not r:
            cost = EMPTY_STATION_PENALTY
            path_info = {
                "coords": [[sxy[0], sxy[1]], [scene_xy[0], scene_xy[1]]],
                "legs": [{
                    "coords": [[sxy[0], sxy[1]], [scene_xy[0], scene_xy[1]]],
                    "duration_min": 30.0,
                    "distance_m": haversine_m(sxy[0], sxy[1], scene_xy[0], scene_xy[1]),
                }],
                "duration_min": 30.0,
                "source": "fallback",
            }
        else:
            path_info = r
            cost = apply_congestion_factor(r["duration_min"], congestion)
        avail = fleet.available(sid, vehicle_type)
        if avail <= 0:
            cost += EMPTY_STATION_PENALTY
        else:
            cost -= 0.15 * avail
        if best is None or cost < best[0]:
            best = (cost, sid, path_info)
    assert best is not None
    return best[1], best[2], best[0]


def _assign_hospital_osrm(
    G: nx.DiGraph,
    hospital_map: dict,
    scene_xy: tuple[float, float],
    hospital_load: Counter,
    congestion: float,
) -> tuple[int, dict, float]:
    from osrm_router import apply_congestion_factor, route_pair

    best = None
    for hid in hospital_map:
        try:
            hxy = _hospital_xy(hid)
        except KeyError:
            continue
        r = route_pair(scene_xy, hxy)
        if not r:
            continue
        cost = apply_congestion_factor(r["duration_min"], congestion)
        cost += hospital_load[hid] * HOSPITAL_LOAD_PENALTY
        if best is None or cost < best[0]:
            best = (cost, hid, r)
    if best is None:
        # fallback first hospital
        hid = next(iter(hospital_map))
        hxy = _hospital_xy(hid)
        r = route_pair(scene_xy, hxy) or {
            "coords": [[scene_xy[0], scene_xy[1]], [hxy[0], hxy[1]]],
            "legs": [{
                "coords": [[scene_xy[0], scene_xy[1]], [hxy[0], hxy[1]]],
                "duration_min": 20.0,
                "distance_m": 1000.0,
            }],
            "duration_min": 20.0,
            "source": "fallback",
        }
        return hid, r, 20.0
    return best[1], best[2], best[0]


def _merge_osrm_legs(leg_a: dict, leg_b: dict | None) -> dict:
    """출동 leg + 이송 leg → 하나의 mission route."""
    legs = list(leg_a.get("legs") or [])
    coords = list(leg_a.get("coords") or [])
    if leg_b:
        for lg in leg_b.get("legs") or []:
            legs.append(lg)
        bcoords = leg_b.get("coords") or []
        if coords and bcoords:
            coords = coords + bcoords[1:]
        elif bcoords:
            coords = bcoords
    return {
        "coords": coords,
        "legs": legs,
        "duration_min": sum(float(x["duration_min"]) for x in legs),
        "distance_m": sum(float(x.get("distance_m") or 0) for x in legs),
        "source": leg_a.get("source") or "osrm",
    }


def plan_global_dispatch(
    G: nx.DiGraph,
    station_map: dict,
    hospital_map: dict,
    incidents: list[Incident],
    verbose: bool = True,
    fleet: StationFleet | None = None,
) -> list[dict]:
    """배차·경로·ETA.

    기본(ROUTING_ENGINE=graph): 표준노드링크 혼잡가중 최단경로 (오프라인).
    ROUTING_ENGINE=osrm 이고 로컬 OSRM 가능하면 OSM driving 본체.
    """
    from osrm_router import apply_congestion_factor, use_osrm
    from road_shapes import haversine_m

    if not use_osrm():
        return _plan_global_dispatch_graph(
            G, station_map, hospital_map, incidents, verbose=verbose, fleet=fleet,
        )

    missions: list[dict] = []
    hospital_load: Counter = Counter()
    fle = fleet if fleet is not None else StationFleet.from_stations()
    skipped = 0

    # incident_id → Incident
    by_id = {inc.incident_id: inc for inc in incidents}

    for req in _collect_unit_requests(incidents):
        inc = by_id[req.incident_id]
        scene_xy = (float(inc.lat), float(inc.lng))
        cong = _scene_cong(G, *scene_xy)

        sid, dispatch_route, cost = _assign_station_osrm(
            G, station_map, scene_xy, fle, req.vehicle_type, cong,
        )
        if not fle.take(sid, req.vehicle_type):
            skipped += 1

        if req.vehicle_type == "fire_truck":
            eta = apply_congestion_factor(dispatch_route["duration_min"], cong)
            missions.append({
                "label": req.label,
                "vehicle_type": "fire_truck",
                "incident_id": req.incident_id,
                "station_id": sid,
                "path": [station_map[sid], req.incident_node],  # 호환용 노드 힌트
                "hospital_id": None,
                "dispatch_len": 1,  # OSRM leg 1개 = 현장
                "fleet_exhausted": cost >= EMPTY_STATION_PENALTY / 2,
                "osrm": dispatch_route,
                "congestion_factor": cong,
                "eta_min_osrm": eta,
                "scene_xy": scene_xy,
                "station_xy": _dispatch_origin_xy(sid, scene_xy),
                "hospital_xy": None,
            })
            continue

        hid, transport_route, _ = _assign_hospital_osrm(
            G, hospital_map, scene_xy, hospital_load, cong,
        )
        hospital_load[hid] += 1
        merged = _merge_osrm_legs(dispatch_route, transport_route)
        eta = apply_congestion_factor(merged["duration_min"], cong)
        missions.append({
            "label": req.label,
            "vehicle_type": "ambulance",
            "incident_id": req.incident_id,
            "station_id": sid,
            "path": [station_map[sid], req.incident_node, hospital_map[hid]],
            "hospital_id": hid,
            "dispatch_len": 1,  # 첫 leg 끝나면 현장
            "fleet_exhausted": cost >= EMPTY_STATION_PENALTY / 2,
            "osrm": merged,
            "congestion_factor": cong,
            "eta_min_osrm": eta,
            "scene_xy": scene_xy,
            "station_xy": _dispatch_origin_xy(sid, scene_xy),
            "hospital_xy": _hospital_xy(hid),
        })

    if verbose:
        print(f"[dispatch_center] OSRM 배차 사고 {len(incidents)}건 / {len(missions)}대 "
              f"| 고갈강제 {skipped}")
        for m in missions:
            src = (m.get("osrm") or {}).get("source", "?")
            print(f"   {m['label']:14s} station#{m['station_id']} "
                  f"ETA {m.get('eta_min_osrm', 0):.1f}min ({src})")

    return missions


def _plan_global_dispatch_graph(
    G: nx.DiGraph,
    station_map: dict,
    hospital_map: dict,
    incidents: list[Incident],
    verbose: bool = True,
    fleet: StationFleet | None = None,
) -> list[dict]:
    """레거시: 표준노드 최단경로 (ROUTING_ENGINE=graph)."""
    missions: list[dict] = []
    edge_usage: Counter = Counter()
    hospital_load: Counter = Counter()
    fle = fleet if fleet is not None else StationFleet.from_stations()

    def _commit_path(path: list):
        for u, v in zip(path[:-1], path[1:]):
            edge_usage[(u, v)] += 1

    skipped = 0
    for req in _collect_unit_requests(incidents):
        sid, dispatch_path, cost = _assign_station(
            G, station_map, req.incident_node, fle, req.vehicle_type, edge_usage,
        )
        if not fle.take(sid, req.vehicle_type):
            skipped += 1
        _commit_path(dispatch_path)

        # 클릭 좌표 보존 (그래프 경로여도 현장 표시용)
        _inc_xy = None
        for _inc in incidents:
            if _inc.incident_id == req.incident_id:
                if _inc.lat and _inc.lng:
                    _inc_xy = (float(_inc.lat), float(_inc.lng))
                break

        if req.vehicle_type == "fire_truck":
            missions.append({
                "label": req.label,
                "vehicle_type": "fire_truck",
                "incident_id": req.incident_id,
                "station_id": sid,
                "path": dispatch_path,
                "hospital_id": None,
                "dispatch_len": len(dispatch_path) - 1,
                "fleet_exhausted": cost >= EMPTY_STATION_PENALTY / 2,
                "scene_xy": _inc_xy,
                "station_xy": _dispatch_origin_xy(sid, _inc_xy) if _inc_xy else _station_xy(sid),
                "hospital_xy": None,
            })
            continue

        hid, transport_path, _ = _assign_hospital(
            G, hospital_map, req.incident_node, hospital_load, edge_usage,
        )
        hospital_load[hid] += 1
        _commit_path(transport_path)
        full = dispatch_path + transport_path[1:]
        missions.append({
            "label": req.label,
            "vehicle_type": "ambulance",
            "incident_id": req.incident_id,
            "station_id": sid,
            "path": full,
            "hospital_id": hid,
            "dispatch_len": len(dispatch_path) - 1,
            "fleet_exhausted": cost >= EMPTY_STATION_PENALTY / 2,
            "scene_xy": _inc_xy,
            "station_xy": _dispatch_origin_xy(sid, _inc_xy) if _inc_xy else _station_xy(sid),
            "hospital_xy": _hospital_xy(hid),
        })

    if verbose:
        print(f"[dispatch_center] GRAPH 배차 {len(missions)}대 skipped={skipped}")
    return missions


# keep old names used above
def _assign_station(
    G: nx.DiGraph,
    station_map: dict,
    incident_node: int,
    fleet: StationFleet,
    vehicle_type: str,
    edge_usage: Counter,
) -> tuple[int, list, float]:
    """잔여 차량이 있는 소방서 중 경로비용 최소. 고갈되면 페널티로만 허용."""
    best = None
    for sid, snode in station_map.items():
        path, cost = _route_cost(G, snode, incident_node, edge_usage)
        avail = fleet.available(sid, vehicle_type)
        if avail <= 0:
            cost += EMPTY_STATION_PENALTY
        else:
            cost -= 0.15 * avail
        cost += 0.5 * _euclid2(G, snode, incident_node) * 1e4
        if best is None or cost < best[0]:
            best = (cost, sid, path)
    assert best is not None
    return best[1], best[2], best[0]


def _assign_hospital(G: nx.DiGraph, hospital_map: dict, incident_node: int,
                     hospital_load: Counter, edge_usage: Counter) -> tuple[int, list, float]:
    """혼잡·이미 배정된 환자 수·경로 겹침을 보고 응급실 선택. (출발지는 아님)"""
    best = None
    for hid, hnode in hospital_map.items():
        path, cost = _route_cost(G, incident_node, hnode, edge_usage)
        cost += hospital_load[hid] * HOSPITAL_LOAD_PENALTY
        if best is None or cost < best[0]:
            best = (cost, hid, path)
    assert best is not None
    return best[1], best[2], best[0]


# 하위호환 래퍼 (단일 사고)
def plan_multi_vehicle_dispatch(G, station_map, hospital_map, incident_node, spec, verbose=True,
                                fleet: StationFleet | None = None):
    inc = Incident(1, incident_node, "custom", spec,
                   lat=G.nodes[incident_node]["lat"], lng=G.nodes[incident_node]["lng"])
    return plan_global_dispatch(
        G, station_map, hospital_map, [inc], verbose=verbose, fleet=fleet,
    )


# ---------------------------------------------------------------------- #
# 4) 다중 사고·다중 차량 애니메이션
# ---------------------------------------------------------------------- #
VEHICLE_COLORS = [
    "darkorange", "purple", "teal", "darkgoldenrod", "deeppink",
    "royalblue", "olivedrab", "crimson", "sienna", "dodgerblue",
]
INCIDENT_MARKERS = ["*", "P", "X", "D", "v", "^"]


def animate_multi(G: nx.DiGraph, missions: list, incidents: list[Incident],
                   title: str, out_gif: Optional[str] = None, interval_ms: int = 90):
    pos = {n: (d["lng"], d["lat"]) for n, d in G.nodes(data=True)}
    fig, ax = plt.subplots(figsize=(10.5, 11))
    ax.set_aspect("equal")

    edges = list(G.edges())
    if len(edges) > 2500:
        rng0 = np.random.default_rng(0)
        edges = [edges[i] for i in rng0.choice(len(edges), size=2500, replace=False)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.12, width=0.4,
                            edge_color="#6b8e6b", arrows=False, ax=ax)

    for s in DAEGU_FIRE_STATIONS:
        ax.plot(s["lng"], s["lat"], "s", color="navy", markersize=8, zorder=5)
    for h in DAEGU_ER_HOSPITALS:
        ax.plot(h["lng"], h["lat"], "^", color="crimson", markersize=9, zorder=5)

    for i, inc in enumerate(incidents):
        mk = INCIDENT_MARKERS[i % len(INCIDENT_MARKERS)]
        ax.plot(inc.lng, inc.lat, mk, color="black", markersize=16, zorder=6,
                label=f"I{inc.incident_id}:{inc.type_key}")
        ax.annotate(f"I{inc.incident_id}", (inc.lng, inc.lat),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color="black")

    lines, points = [], []
    xy_by_mission = []
    for i, m in enumerate(missions):
        color = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]
        xy = [pos[n] for n in m["path"] if n in pos]
        xy_by_mission.append(xy)
        line, = ax.plot([], [], color=color, lw=2.4, zorder=7,
                        label=f"{m['label']} ({m['vehicle_type']})")
        pt, = ax.plot([], [], "o", color=color, markersize=9, zorder=8,
                      markeredgecolor="black", markeredgewidth=0.7)
        lines.append(line)
        points.append(pt)

    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.92))

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(DAEGU_LNG_MIN, DAEGU_LNG_MAX)
    ax.set_ylim(DAEGU_LAT_MIN, DAEGU_LAT_MAX)
    ax.legend(loc="lower right", fontsize=7)

    max_len = max((len(xy) for xy in xy_by_mission), default=1)

    def init():
        for line, pt in zip(lines, points):
            line.set_data([], [])
            pt.set_data([], [])
        return lines + points + [status]

    def update(frame):
        active = 0
        for xy, line, pt in zip(xy_by_mission, lines, points):
            if not xy:
                continue
            i = min(frame, len(xy) - 1)
            xs = [p[0] for p in xy[: i + 1]]
            ys = [p[1] for p in xy[: i + 1]]
            line.set_data(xs, ys)
            pt.set_data([xy[i][0]], [xy[i][1]])
            if frame < len(xy):
                active += 1
        status.set_text(
            f"step {frame}/{max_len-1}  |  incidents={len(incidents)}  "
            f"active={active}/{len(missions)}"
        )
        return lines + points + [status]

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init,
                         interval=interval_ms, blit=False, repeat=False)
    if out_gif:
        anim.save(out_gif, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
        print(f"[dispatch_center] saved {out_gif}")
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass
    return anim


# ---------------------------------------------------------------------- #
# 5) CLI
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="중앙관제 시뮬레이터 — 다중 사고 + 소방서/응급실/경로 전역 배정",
    )
    parser.add_argument(
        "--incident", action="append", default=None, type=parse_incident_arg,
        metavar="TYPE:lat,lng",
        help="사고 1건. 여러 번 지정 가능. 예: fire_1:35.87,128.60 / medical_1:node:1500004600",
    )
    parser.add_argument("--interactive", action="store_true",
                        help="지도 클릭으로 사고 여러 곳 지정 (로컬 GUI)")
    # 단일 사고 하위호환
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lng", type=float, default=None)
    parser.add_argument("--node", type=int, default=None)
    parser.add_argument("--incident-type", type=str, default="medical_1",
                        choices=list(INCIDENT_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--n-intersections", type=int, default=40)
    parser.add_argument("--save", type=str, default="dispatch.gif")
    parser.add_argument("--interval-ms", type=int, default=80)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    print("[dispatch_center] 실제 대구 도로망 로딩...")
    G, hospital_map = build_combined_graph(
        verbose=True, data_dir=args.data_dir, n_intersections=args.n_intersections,
    )
    station_map = G.graph["station_node_map"]

    incidents = resolve_incidents(G, args)

    print(f"\n[dispatch_center] === 사고 {len(incidents)}건 ===")
    for inc in incidents:
        print(f"  I{inc.incident_id} [{inc.type_key}] {inc.spec.label}")
        print(f"      node={inc.node} ({inc.lat:.5f}, {inc.lng:.5f})  "
              f"소방{inc.spec.n_fire_trucks}/구급{inc.spec.n_ambulances}")
    print()

    missions = plan_global_dispatch(G, station_map, hospital_map, incidents)

    title = f"Dispatch Center: {len(incidents)} incidents / {len(missions)} units (global assign)"
    animate_multi(G, missions, incidents, title,
                  out_gif=args.save, interval_ms=args.interval_ms)


if __name__ == "__main__":
    main()
