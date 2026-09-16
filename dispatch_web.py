"""
dispatch_web.py
================
중앙관제 실시간 웹 UI.

- 지도 클릭으로 다중 사고 지정
- 시간축 혼잡 costmap (단계 색상)
- 시뮬 시계에 맞춰 차량이 travel_time 기준으로 실제 이동
- 혼잡이 바뀌면 남은 구간을 재탐색(실시간 재경로)

실행:
    python dispatch_web.py
    # http://127.0.0.1:5050
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from network_graph import build_combined_graph
from daegu_ems_geo import DAEGU_FIRE_STATIONS, DAEGU_ER_HOSPITALS
from standard_node_link import (
    DAEGU_LAT_MIN, DAEGU_LAT_MAX, DAEGU_LNG_MIN, DAEGU_LNG_MAX, nearest_node,
)
from dispatch_center import (
    INCIDENT_PRESETS, Incident, StationFleet, plan_global_dispatch, _route_cost,
)
from datetime import date, datetime

from congestion_time import (
    TimeCongestionField, path_edge_times, path_edge_speeds, format_clock, traffic_color,
)

HERE = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(HERE / "templates"),
            static_folder=str(HERE / "static"))

G = None
STATION_MAP: dict = {}
HOSPITAL_MAP: dict = {}
FIELD: TimeCongestionField | None = None
FLEET: StationFleet | None = None
# 현재 출동 중(잔여에서 차감된) 유닛 — 재배차 시 환원 후 다시 배정
ACTIVE_OUT: list[dict] = []

STATION_NAME = {s["station_id"]: s["station_name"] for s in DAEGU_FIRE_STATIONS}
HOSPITAL_NAME = {h["hospital_id"]: h["hospital_name"] for h in DAEGU_ER_HOSPITALS}


def _path_geometry(path_nodes: list) -> tuple[list[list[float]], list[list[list[float]]]]:
    """지도 polyline + 홉마다 한 구간. 양끝=노드 좌표 고정 (도착 판정과 일치)."""
    from road_shapes import path_geometry
    import os
    # 기본 off: OSRM이 노드를 비끼면 '사고 미도달인데 완료'처럼 보임
    fetch = os.environ.get("PATH_SHAPES_FETCH", "0").strip().lower() in ("1", "true", "yes")
    try:
        full, segs = path_geometry(G, path_nodes, fetch=fetch)
        if full and len(full) >= 2 and segs and len(segs) == len(path_nodes) - 1:
            return full, segs
    except Exception:
        pass
    nodes = [[float(G.nodes[n]["lat"]), float(G.nodes[n]["lng"])] for n in path_nodes]
    segs = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)]
    return nodes, segs


def _mission_payload(m: dict, t_min: float) -> dict:
    """미션 API. OSRM legs가 있으면 배차·표시·이동이 동일 경로."""
    path = m["path"]
    osrm = m.get("osrm")
    cong = float(m.get("congestion_factor") or 0.35)

    if osrm and osrm.get("legs"):
        from osrm_router import apply_congestion_factor
        legs = osrm["legs"]
        edge_segments = [lg["coords"] for lg in legs if lg.get("coords") and len(lg["coords"]) >= 2]
        edge_times = [
            round(apply_congestion_factor(float(lg["duration_min"]), cong), 3)
            for lg in legs
        ]
        n = min(len(edge_segments), len(edge_times))
        edge_segments, edge_times = edge_segments[:n], edge_times[:n]
        # 현장·병원 좌표를 구간 끝점에 강제 (도로 스냅이 병원 앞으로 밀려도 클릭=현장)
        scene_xy = list(m["scene_xy"]) if m.get("scene_xy") else None
        hospital_xy = list(m["hospital_xy"]) if m.get("hospital_xy") else None
        station_xy = list(m["station_xy"]) if m.get("station_xy") else None
        if n >= 1 and scene_xy and len(edge_segments[0]) >= 2:
            edge_segments[0] = list(edge_segments[0])
            edge_segments[0][-1] = [float(scene_xy[0]), float(scene_xy[1])]
            if station_xy:
                edge_segments[0][0] = [float(station_xy[0]), float(station_xy[1])]
        if n >= 2 and scene_xy and hospital_xy:
            edge_segments[1] = list(edge_segments[1])
            edge_segments[1][0] = [float(scene_xy[0]), float(scene_xy[1])]
            edge_segments[1][-1] = [float(hospital_xy[0]), float(hospital_xy[1])]
        path_line = []
        for i, seg in enumerate(edge_segments):
            path_line.extend(seg if i == 0 else seg[1:])
        if len(path_line) < 2:
            path_line = list(osrm.get("coords") or [])
        edge_speeds = []
        for lg, et in zip(legs[:n], edge_times):
            dist_km = float(lg.get("distance_m") or 0) / 1000.0
            edge_speeds.append(round(dist_km / max(et / 60.0, 1e-3), 1) if et else 30.0)
        node_latlng = []
        if station_xy:
            node_latlng.append([station_xy[0], station_xy[1]])
        if scene_xy:
            node_latlng.append([scene_xy[0], scene_xy[1]])
        if hospital_xy:
            node_latlng.append([hospital_xy[0], hospital_xy[1]])
        dispatch_len = 1 if m["vehicle_type"] == "ambulance" else max(n, 1)
        hid = m.get("hospital_id")
        waypoints = {
            "scene_node": int(path[min(1, len(path) - 1)]) if path else None,
            "hospital_node": int(HOSPITAL_MAP[hid]) if hid is not None and hid in HOSPITAL_MAP else None,
        }
        return {
            "id": m["label"],
            "label": m["label"],
            "vehicle_type": m["vehicle_type"],
            "incident_id": m["incident_id"],
            "station_id": m["station_id"],
            "station_name": STATION_NAME.get(m["station_id"], ""),
            "hospital_id": hid,
            "hospital_name": HOSPITAL_NAME.get(hid, "") if hid is not None else None,
            "hops": n,
            "dispatch_len": dispatch_len,
            "path_nodes": [int(x) for x in path],
            "path_node_latlng": node_latlng,
            "path": path_line,
            "edge_segments": edge_segments,
            "edge_times": edge_times,
            "edge_speeds": edge_speeds,
            "eta_min": round(sum(edge_times), 2),
            "phase": "to_scene",
            "waypoints": waypoints,
            "incident_node": int(path[1]) if len(path) > 1 else (int(path[0]) if path else None),
            "start_node": int(path[0]) if path else None,
            "goal_node": int(path[-1]) if path else None,
            "route_source": osrm.get("source", "osrm"),
            "scene_xy": scene_xy,
            "station_xy": station_xy,
            "hospital_xy": hospital_xy,
        }

    times = path_edge_times(FIELD, path, t_min)
    speeds = path_edge_speeds(FIELD, path, t_min)
    n_hops = max(0, len(path) - 1)
    if len(times) != n_hops:
        times = (times + [0.5] * n_hops)[:n_hops]
    if len(speeds) != n_hops:
        speeds = (speeds + [30.0] * n_hops)[:n_hops]

    goal_node = path[-1] if path else None
    phase = "to_scene"
    incident_node = None
    if m["vehicle_type"] == "ambulance":
        cut = int(m.get("dispatch_len") or 0)
        cut = max(0, min(cut, len(path) - 1))
        scene_node = path[cut]
        incident_node = int(scene_node)
        hospital_node = HOSPITAL_MAP.get(m["hospital_id"])
        goal_node = hospital_node
        waypoints = {
            "scene_node": int(scene_node),
            "hospital_node": int(hospital_node) if hospital_node is not None else None,
        }
    else:
        incident_node = int(path[-1]) if path else None
        waypoints = {"scene_node": incident_node, "hospital_node": None}

    path_line, edge_segments = _path_geometry(path)
    if len(edge_segments) != n_hops:
        nodes = [[float(G.nodes[n]["lat"]), float(G.nodes[n]["lng"])] for n in path]
        path_line = nodes
        edge_segments = [[nodes[i], nodes[i + 1]] for i in range(n_hops)]

    node_latlng = [[float(G.nodes[n]["lat"]), float(G.nodes[n]["lng"])] for n in path]
    # 그래프 폴백이어도 현장은 클릭 좌표 (스냅 노드 ≠ 현장 표시)
    scene_xy = list(m["scene_xy"]) if m.get("scene_xy") else None
    if scene_xy is None and incident_node is not None and incident_node in G.nodes:
        # 최후: 클릭이 안 넘어온 경우만 — 그래도 클라이언트는 click_lat 우선
        scene_xy = [float(G.nodes[incident_node]["lat"]), float(G.nodes[incident_node]["lng"])]
    station_xy = list(m["station_xy"]) if m.get("station_xy") else None
    hospital_xy = list(m["hospital_xy"]) if m.get("hospital_xy") else None
    if hospital_xy is None and m.get("hospital_id") is not None:
        try:
            from dispatch_center import _hospital_xy
            hospital_xy = list(_hospital_xy(m["hospital_id"]))
        except Exception:
            hospital_xy = None

    return {
        "id": m["label"],
        "label": m["label"],
        "vehicle_type": m["vehicle_type"],
        "incident_id": m["incident_id"],
        "station_id": m["station_id"],
        "station_name": STATION_NAME.get(m["station_id"], ""),
        "hospital_id": m["hospital_id"],
        "hospital_name": HOSPITAL_NAME.get(m["hospital_id"], "") if m["hospital_id"] is not None else None,
        "hops": n_hops,
        "dispatch_len": int(m.get("dispatch_len") or 0),
        "path_nodes": [int(n) for n in path],
        "path_node_latlng": node_latlng,
        "path": path_line,
        "edge_segments": edge_segments,
        "edge_times": [round(float(t), 3) for t in times],
        "edge_speeds": speeds,
        "eta_min": round(sum(times), 2),
        "phase": phase,
        "waypoints": waypoints,
        "incident_node": incident_node,
        "start_node": int(path[0]) if path else None,
        "goal_node": int(goal_node) if goal_node is not None else None,
        "route_source": "graph",
        "scene_xy": scene_xy,
        "station_xy": station_xy,
        "hospital_xy": hospital_xy,
    }


def _apply_time(t_min: float, horizon_min: float = 45.0) -> None:
    FIELD.apply_to_graph(G, t_min, horizon_min=horizon_min)


def _parse_day(raw) -> date:
    if not raw:
        return date.today()
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    return date.fromisoformat(str(raw)[:10])


def _set_context_from_request(body_or_args) -> dict:
    """쿼리/바디에서 date 읽어 캘린더 맥락 설정."""
    if hasattr(body_or_args, "get"):
        raw = body_or_args.get("date")
    else:
        raw = None
    day = _parse_day(raw)
    ctx = FIELD.set_date(day)
    return ctx.to_dict()


def _fleet_payload() -> dict:
    if FLEET is None:
        return {"stations": [], "totals": {}}
    return {
        "stations": FLEET.snapshot(STATION_NAME),
        "totals": FLEET.totals(),
        "note": "119 구급·소방은 소방서 잔여 대수에서 출동. 병원은 이송 목적지(출발지 아님).",
    }


def _release_active_out() -> None:
    global ACTIVE_OUT
    if FLEET is None:
        ACTIVE_OUT = []
        return
    for u in ACTIVE_OUT:
        FLEET.release(int(u["station_id"]), u["vehicle_type"])
    ACTIVE_OUT = []


def load_graph(data_dir: str | None = None, verbose: bool = True):
    global G, STATION_MAP, HOSPITAL_MAP, FIELD, STATION_NAME, HOSPITAL_NAME, FLEET, ACTIVE_OUT
    from daegu_open_data import print_open_data_status
    from daegu_ems_geo import DAEGU_FIRE_STATIONS as _FS, DAEGU_ER_HOSPITALS as _ER, EMS_GEO_SOURCE

    if verbose:
        print_open_data_status()
        print(f"[dispatch_web] EMS 좌표 소스: {EMS_GEO_SOURCE}")

    G, HOSPITAL_MAP = build_combined_graph(verbose=verbose, data_dir=data_dir)
    STATION_MAP = G.graph["station_node_map"]
    STATION_NAME = {s["station_id"]: s["station_name"] for s in _FS}
    HOSPITAL_NAME = {h["hospital_id"]: h["hospital_name"] for h in _ER}
    FLEET = StationFleet.from_stations(_FS)
    ACTIVE_OUT = []
    FIELD = TimeCongestionField.from_graph(G, sample_edges=3800)
    # 실시간 매칭이 있으면 소스에 표시
    if any(d.get("realtime") for _, _, d in G.edges(data=True)):
        FIELD.source = f"{FIELD.source}+linkspeed_realtime"
    if data_dir:
        FIELD.source = f"{FIELD.source}+its_overlay"

    # 맵에 길을 뚫지 않도록: costmap용 짧은 구간의 도로곡선 캐시
    try:
        from road_shapes import ensure_shapes_for_edges, should_draw_on_costmap
        FIELD.edge_list = [e for e in FIELD.edge_list if should_draw_on_costmap(G, *e)]
        ensure_shapes_for_edges(G, FIELD.edge_list[:180], max_fetch=0, verbose=verbose)
        # max_fetch=0: 기동 지연 없이 긴 직선만 제외. 곡선 캐시는 백그라운드/수동.
        # ROAD_SHAPES_FETCH=1 이면 OSRM으로 시내 구간 곡선 채움
        import os
        if os.environ.get("ROAD_SHAPES_FETCH", "").strip() in ("1", "true", "yes"):
            ensure_shapes_for_edges(G, FIELD.edge_list[:200], max_fetch=80, verbose=verbose)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[dispatch_web] road_shapes 생략: {e}")

    now = datetime.now()
    FIELD.set_date(now.date())
    _apply_time(now.hour * 60 + now.minute)
    if verbose:
        print(f"[dispatch_web] time-congestion field ready ({FIELD.source}), "
              f"costmap edges={len(FIELD.edge_list)}, now={now.strftime('%Y-%m-%d %H:%M')}")


@app.get("/")
def index():
    resp = send_from_directory(HERE / "templates", "dispatch_live.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/api/meta")
def api_meta():
    now = datetime.now()
    t_now = now.hour * 60 + now.minute + now.second / 60.0
    if FIELD:
        FIELD.set_date(now.date())
    presets = {
        k: {
            "label": s.label,
            "n_fire_trucks": s.n_fire_trucks,
            "n_ambulances": s.n_ambulances,
            "severity": s.severity,
        }
        for k, s in INCIDENT_PRESETS.items()
    }
    return jsonify({
        "bounds": {
            "lat_min": DAEGU_LAT_MIN, "lat_max": DAEGU_LAT_MAX,
            "lng_min": DAEGU_LNG_MIN, "lng_max": DAEGU_LNG_MAX,
        },
        "center": {"lat": 35.87, "lng": 128.60},
        "presets": presets,
        "stations": [
            {
                "station_id": s["station_id"], "name": s["station_name"],
                "lat": s["lat"], "lng": s["lng"],
                "snap_lat": float(G.nodes[STATION_MAP[s["station_id"]]]["lat"]) if s["station_id"] in STATION_MAP else s["lat"],
                "snap_lng": float(G.nodes[STATION_MAP[s["station_id"]]]["lng"]) if s["station_id"] in STATION_MAP else s["lng"],
                "ambulances": int(s.get("ambulances", 4) or 4),
                "fire_trucks": int(s.get("fire_trucks", 4) or 4),
            }
            for s in DAEGU_FIRE_STATIONS
        ],
        "hospitals": [
            {
                "hospital_id": h["hospital_id"], "name": h["hospital_name"],
                "lat": h["lat"], "lng": h["lng"],
                "snap_lat": float(G.nodes[HOSPITAL_MAP[h["hospital_id"]]]["lat"]) if h["hospital_id"] in HOSPITAL_MAP else h["lat"],
                "snap_lng": float(G.nodes[HOSPITAL_MAP[h["hospital_id"]]]["lng"]) if h["hospital_id"] in HOSPITAL_MAP else h["lng"],
            }
            for h in DAEGU_ER_HOSPITALS
        ],
        "fleet": _fleet_payload(),
        "note_geo": "시설 아이콘=실제 좌표, 경로 출발/도착=표준노드 스냅. 사고 I마커=스냅 위치(클릭과 점선으로 연결).",
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "congestion_levels": FIELD.costmap(G, t_now)["levels"] if FIELD else [],
        "congestion_source": FIELD.source if FIELD else None,
        "congestion_legend": FIELD.costmap(G, t_now).get("legend") if FIELD else None,
        "default_t_min": t_now,
        "default_date": now.date().isoformat(),
        "server_now": now.isoformat(timespec="seconds"),
        "calendar": FIELD.context.to_dict() if FIELD and FIELD.context else None,
    })


@app.get("/api/costmap")
def api_costmap():
    """시간 t (+lookahead) 교통량 지도 — 파랑(한산)→빨강(혼잡)."""
    _set_context_from_request(request.args)
    t_min = float(request.args.get("t_min", datetime.now().hour * 60 + datetime.now().minute))
    lookahead = float(request.args.get("lookahead_min", 0))
    _apply_time(t_min)
    return jsonify(FIELD.costmap(G, t_min, lookahead_min=lookahead))


@app.get("/api/forecast")
def api_forecast():
    """현재 날짜·시각 기준 향후 N시간 혼잡 예측."""
    cal = _set_context_from_request(request.args)
    t_min = float(request.args.get("t_min", datetime.now().hour * 60 + datetime.now().minute))
    hours = float(request.args.get("hours", 6))
    fc = FIELD.forecast(t_min, horizon_hours=hours, step_min=30)
    fc["calendar"] = cal
    return jsonify(fc)


@app.post("/api/dispatch")
def api_dispatch():
    """사고 목록 + 시뮬 시각 → 전역 배차 (edge_times 포함)."""
    body = request.get_json(force=True, silent=True) or {}
    raw = body.get("incidents") or []
    t_min = float(body.get("t_min", datetime.now().hour * 60 + datetime.now().minute))
    cal = _set_context_from_request(body)
    if not isinstance(raw, list):
        return jsonify({"error": "incidents must be a list"}), 400
    if len(raw) > 20:
        return jsonify({"error": "too many incidents (max 20)"}), 400

    # 앞으로 45분 예측 혼잡을 반영해 경로 비용 설정
    _apply_time(t_min, horizon_min=float(body.get("horizon_min", 45)))

    incidents: list[Incident] = []
    snapped = []
    for i, item in enumerate(raw, start=1):
        type_key = str(item.get("type", "medical_1"))
        if type_key not in INCIDENT_PRESETS:
            return jsonify({"error": f"unknown type: {type_key}"}), 400
        try:
            lat = float(item["lat"])
            lng = float(item["lng"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": f"incident {i}: lat/lng required"}), 400
        node = nearest_node(G, lat, lng)
        spec = INCIDENT_PRESETS[type_key]
        snap_lat = float(G.nodes[node]["lat"])
        snap_lng = float(G.nodes[node]["lng"])
        from road_shapes import haversine_m
        snap_m = round(haversine_m(lat, lng, snap_lat, snap_lng), 1)
        # OSRM 본체: 사고 좌표는 클릭 그대로 (스냅은 참고·costmap용)
        incidents.append(Incident(
            incident_id=i, node=node, type_key=type_key, spec=spec,
            lat=lat, lng=lng,
        ))
        snapped.append({
            "incident_id": i, "type": type_key, "label": spec.label,
            "click_lat": lat, "click_lng": lng,
            "lat": lat, "lng": lng,
            "snap_lat": snap_lat, "snap_lng": snap_lng,
            "node": int(node),
            "snap_m": snap_m,
            "n_fire": spec.n_fire_trucks, "n_amb": spec.n_ambulances,
        })

    if not incidents:
        return jsonify({
            "t_min": t_min, "time_label": format_clock(t_min),
            "incidents": [], "missions": [],
            "fleet": _fleet_payload(),
            "summary": {"n_incidents": 0, "n_units": 0, "overlap": 0},
        })

    # 재배차: 이전 출동 차감 환원 후, 현재 사고 목록 기준으로 다시 배정·차감
    _release_active_out()
    missions_raw = plan_global_dispatch(
        G, STATION_MAP, HOSPITAL_MAP, incidents, verbose=False, fleet=FLEET,
    )
    ACTIVE_OUT.clear()
    for m in missions_raw:
        ACTIVE_OUT.append({
            "id": m["label"],
            "station_id": int(m["station_id"]),
            "vehicle_type": m["vehicle_type"],
        })
    # OSRM 경로면 그래프 엣지 겹침 카운트는 의미 약함
    overlap = 0
    if any(not m.get("osrm") for m in missions_raw):
        usage: Counter = Counter()
        for m in missions_raw:
            for u, v in zip(m["path"][:-1], m["path"][1:]):
                usage[(u, v)] += 1
        overlap = sum(c - 1 for c in usage.values() if c > 1)

    missions = [_mission_payload(m, t_min) for m in missions_raw]
    route_src = missions[0].get("route_source", "graph") if missions else "none"
    fc = FIELD.forecast(t_min, horizon_hours=6, step_min=30)
    return jsonify({
        "t_min": t_min,
        "time_label": format_clock(t_min),
        "calendar": cal,
        "forecast": {
            "now": fc["now"], "trend": fc["trend"], "peak": fc["peak"],
            "series": fc["series"][:8],
        },
        "incidents": snapped,
        "missions": missions,
        "fleet": _fleet_payload(),
        "route_source": route_src,
        "summary": {
            "n_incidents": len(incidents),
            "n_units": len(missions),
            "overlap": overlap,
            "total_eta_min": round(sum(m["eta_min"] for m in missions), 1),
            "amb_available": FLEET.totals()["ambulances"]["available"] if FLEET else 0,
            "fire_available": FLEET.totals()["fire_trucks"]["available"] if FLEET else 0,
            "route_source": route_src,
        },
    })


@app.get("/api/fleet")
def api_fleet_get():
    return jsonify(_fleet_payload())


@app.post("/api/fleet/release")
def api_fleet_release():
    """차량 도착·임무 종료 → 해당 서 잔여 +1."""
    global ACTIVE_OUT
    body = request.get_json(force=True, silent=True) or {}
    released = []
    for item in body.get("units") or []:
        try:
            sid = int(item["station_id"])
            vtype = str(item.get("vehicle_type", "ambulance"))
            uid = item.get("id")
        except (KeyError, TypeError, ValueError):
            continue
        if FLEET is not None:
            FLEET.release(sid, vtype)
        ACTIVE_OUT = [u for u in ACTIVE_OUT if u.get("id") != uid]
        released.append({"id": uid, "station_id": sid, "vehicle_type": vtype})
    return jsonify({"released": released, "fleet": _fleet_payload()})


@app.post("/api/fleet/reset")
def api_fleet_reset():
    global FLEET, ACTIVE_OUT
    FLEET = StationFleet.from_stations(DAEGU_FIRE_STATIONS)
    ACTIVE_OUT = []
    return jsonify(_fleet_payload())


@app.post("/api/reroute")
def api_reroute():
    """이동 중 재탐색. OSRM 본체면 현재 좌표→목표 OSRM 재조회."""
    from osrm_router import apply_congestion_factor, route_pair, use_osrm

    body = request.get_json(force=True, silent=True) or {}
    t_min = float(body.get("t_min", datetime.now().hour * 60 + datetime.now().minute))
    _set_context_from_request(body)
    vehicles = body.get("vehicles") or []
    _apply_time(t_min, horizon_min=float(body.get("horizon_min", 45)))

    out = []
    if use_osrm():
        for v in vehicles:
            vid = v.get("id")
            try:
                cur_lat = float(v["current_lat"])
                cur_lng = float(v["current_lng"])
            except (KeyError, TypeError, ValueError):
                continue
            vtype = v.get("vehicle_type", "fire_truck")
            phase = v.get("phase", "to_scene")
            scene_xy = v.get("scene_xy")
            hospital_xy = v.get("hospital_xy")
            cong = 0.35
            if scene_xy and len(scene_xy) >= 2:
                from dispatch_center import _scene_cong
                cong = _scene_cong(G, float(scene_xy[0]), float(scene_xy[1]))

            legs_routes = []
            if vtype == "ambulance" and phase == "to_scene" and scene_xy and hospital_xy:
                r1 = route_pair((cur_lat, cur_lng), (float(scene_xy[0]), float(scene_xy[1])))
                r2 = route_pair((float(scene_xy[0]), float(scene_xy[1])),
                                (float(hospital_xy[0]), float(hospital_xy[1])))
                if not r1 or not r2:
                    continue
                from dispatch_center import _merge_osrm_legs
                merged = _merge_osrm_legs(r1, r2)
                dispatch_len = 1
            else:
                goal = hospital_xy if phase == "to_hospital" else scene_xy
                if not goal or len(goal) < 2:
                    continue
                merged = route_pair((cur_lat, cur_lng), (float(goal[0]), float(goal[1])))
                if not merged:
                    continue
                dispatch_len = 0 if phase == "to_hospital" else 1

            segs = [lg["coords"] for lg in merged["legs"]]
            times = [round(apply_congestion_factor(float(lg["duration_min"]), cong), 3)
                     for lg in merged["legs"]]
            out.append({
                "id": vid,
                "path_nodes": v.get("path_nodes") or [],
                "path": merged["coords"],
                "edge_segments": segs,
                "edge_times": times,
                "edge_speeds": [],
                "eta_min": round(sum(times), 2),
                "dispatch_len": dispatch_len,
                "rerouted": True,
                "route_source": merged.get("source", "osrm"),
                "t_min": t_min,
                "time_label": format_clock(t_min),
            })
        return jsonify({
            "t_min": t_min,
            "time_label": format_clock(t_min),
            "vehicles": out,
            "n_rerouted": len(out),
        })

    edge_usage: Counter = Counter()
    for v in vehicles:
        vid = v.get("id")
        try:
            cur = int(v["current_node"])
        except (KeyError, TypeError, ValueError):
            continue
        if cur not in G:
            continue

        vtype = v.get("vehicle_type", "fire_truck")
        phase = v.get("phase", "to_scene")
        scene = v.get("scene_node")
        hospital = v.get("hospital_node")

        path: list = []
        if vtype == "ambulance" and phase == "to_scene" and scene is not None and hospital is not None:
            scene, hospital = int(scene), int(hospital)
            p1, _ = _route_cost(G, cur, scene, edge_usage)
            for a, b in zip(p1[:-1], p1[1:]):
                edge_usage[(a, b)] += 1
            p2, _ = _route_cost(G, scene, hospital, edge_usage)
            for a, b in zip(p2[:-1], p2[1:]):
                edge_usage[(a, b)] += 1
            path = p1 + p2[1:]
        else:
            goal = v.get("goal_node")
            if goal is None:
                goal = hospital if phase == "to_hospital" else scene
            if goal is None:
                continue
            goal = int(goal)
            if goal not in G:
                continue
            path, _ = _route_cost(G, cur, goal, edge_usage)
            for a, b in zip(path[:-1], path[1:]):
                edge_usage[(a, b)] += 1

        if not path:
            continue
        times = path_edge_times(FIELD, path, t_min)
        speeds = path_edge_speeds(FIELD, path, t_min)
        remaining = v.get("remaining_nodes")
        changed = True
        if remaining is not None:
            changed = [int(n) for n in path] != [int(n) for n in remaining]

        path_line, edge_segments = _path_geometry(path)
        out.append({
            "id": vid,
            "path_nodes": [int(n) for n in path],
            "path": path_line,
            "edge_segments": edge_segments,
            "edge_times": [round(t, 3) for t in times],
            "edge_speeds": speeds,
            "eta_min": round(sum(times), 2),
            "rerouted": bool(changed),
            "t_min": t_min,
            "time_label": format_clock(t_min),
        })

    n_changed = sum(1 for x in out if x["rerouted"])
    return jsonify({
        "t_min": t_min,
        "time_label": format_clock(t_min),
        "vehicles": out,
        "n_rerouted": n_changed,
    })


@app.post("/api/snap")
def api_snap():
    """클릭 좌표 → 도로 노드 스냅만."""
    body = request.get_json(force=True, silent=True) or {}
    lat, lng = float(body["lat"]), float(body["lng"])
    node = nearest_node(G, lat, lng)
    return jsonify({
        "node": int(node),
        "lat": float(G.nodes[node]["lat"]),
        "lng": float(G.nodes[node]["lng"]),
    })


def main():
    parser = argparse.ArgumentParser(description="중앙관제 실시간 웹 (시간축 costmap)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("[dispatch_web] 도로망 + 시간축 혼잡 로딩 중…")
    load_graph(data_dir=args.data_dir, verbose=True)
    print(f"[dispatch_web] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
