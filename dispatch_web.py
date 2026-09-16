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
    INCIDENT_PRESETS, Incident, plan_global_dispatch, _route_cost,
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

STATION_NAME = {s["station_id"]: s["station_name"] for s in DAEGU_FIRE_STATIONS}
HOSPITAL_NAME = {h["hospital_id"]: h["hospital_name"] for h in DAEGU_ER_HOSPITALS}


def _path_latlng(path_nodes: list) -> list[list[float]]:
    return [[float(G.nodes[n]["lat"]), float(G.nodes[n]["lng"])] for n in path_nodes]


def _mission_payload(m: dict, t_min: float) -> dict:
    path = m["path"]
    times = path_edge_times(FIELD, path, t_min)
    speeds = path_edge_speeds(FIELD, path, t_min)
    goal_node = path[-1] if path else None
    # 구급: dispatch_len 이후는 병원 이송
    phase = "to_scene"
    if m["vehicle_type"] == "ambulance" and m.get("hospital_id") is not None:
        phase = "to_hospital" if m.get("dispatch_len") is not None else "to_hospital"
    if m["vehicle_type"] == "fire_truck":
        phase = "to_scene"

    # goal: 화재=현장, 구급 출동중이면 현장, 이송중이면 병원 — 초기 배차는 전체 path
    if m["vehicle_type"] == "ambulance":
        # path = station→scene→hospital ; scene index = dispatch_len
        cut = m.get("dispatch_len", 0)
        scene_node = path[cut] if cut < len(path) else path[-1]
        hospital_node = HOSPITAL_MAP.get(m["hospital_id"])
        goal_node = hospital_node
        waypoints = {
            "scene_node": int(scene_node),
            "hospital_node": int(hospital_node) if hospital_node is not None else None,
        }
    else:
        waypoints = {"scene_node": int(goal_node) if goal_node is not None else None,
                     "hospital_node": None}

    return {
        "id": m["label"],
        "label": m["label"],
        "vehicle_type": m["vehicle_type"],
        "incident_id": m["incident_id"],
        "station_id": m["station_id"],
        "station_name": STATION_NAME.get(m["station_id"], ""),
        "hospital_id": m["hospital_id"],
        "hospital_name": HOSPITAL_NAME.get(m["hospital_id"], "") if m["hospital_id"] is not None else None,
        "hops": max(0, len(path) - 1),
        "dispatch_len": m.get("dispatch_len"),
        "path_nodes": [int(n) for n in path],
        "path": _path_latlng(path),
        "edge_times": [round(t, 3) for t in times],  # 분
        "edge_speeds": speeds,  # km/h (응급차 GPS 속도)
        "eta_min": round(sum(times), 2),
        "phase": phase,
        "waypoints": waypoints,
        "start_node": int(path[0]) if path else None,
        "goal_node": int(goal_node) if goal_node is not None else None,
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


def load_graph(data_dir: str | None = None, verbose: bool = True):
    global G, STATION_MAP, HOSPITAL_MAP, FIELD
    G, HOSPITAL_MAP = build_combined_graph(verbose=verbose, data_dir=data_dir)
    STATION_MAP = G.graph["station_node_map"]
    FIELD = TimeCongestionField.from_graph(G, sample_edges=2200)
    if data_dir:
        FIELD.source = "its_overlay+diurnal"
    _apply_time(8 * 60)  # 기본 출근 피크
    if verbose:
        print(f"[dispatch_web] time-congestion field ready ({FIELD.source}), "
              f"costmap edges={len(FIELD.edge_list)}")


@app.get("/")
def index():
    resp = send_from_directory(HERE / "templates", "dispatch_live.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/api/meta")
def api_meta():
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
            {"station_id": s["station_id"], "name": s["station_name"],
             "lat": s["lat"], "lng": s["lng"]}
            for s in DAEGU_FIRE_STATIONS
        ],
        "hospitals": [
            {"hospital_id": h["hospital_id"], "name": h["hospital_name"],
             "lat": h["lat"], "lng": h["lng"]}
            for h in DAEGU_ER_HOSPITALS
        ],
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "congestion_levels": FIELD.costmap(G, 0)["levels"] if FIELD else [],
        "congestion_source": FIELD.source if FIELD else None,
        "congestion_legend": FIELD.costmap(G, 0).get("legend") if FIELD else None,
        "default_t_min": 8 * 60,
        "default_date": date.today().isoformat(),
        "calendar": FIELD.context.to_dict() if FIELD and FIELD.context else None,
    })


@app.get("/api/costmap")
def api_costmap():
    """시간 t (+lookahead) 교통량 지도 — 파랑(한산)→빨강(혼잡)."""
    _set_context_from_request(request.args)
    t_min = float(request.args.get("t_min", 8 * 60))
    lookahead = float(request.args.get("lookahead_min", 0))
    _apply_time(t_min)
    return jsonify(FIELD.costmap(G, t_min, lookahead_min=lookahead))


@app.get("/api/forecast")
def api_forecast():
    """현재 날짜·시각 기준 향후 N시간 혼잡 예측."""
    cal = _set_context_from_request(request.args)
    t_min = float(request.args.get("t_min", 8 * 60))
    hours = float(request.args.get("hours", 6))
    fc = FIELD.forecast(t_min, horizon_hours=hours, step_min=30)
    fc["calendar"] = cal
    return jsonify(fc)


@app.post("/api/dispatch")
def api_dispatch():
    """사고 목록 + 시뮬 시각 → 전역 배차 (edge_times 포함)."""
    body = request.get_json(force=True, silent=True) or {}
    raw = body.get("incidents") or []
    t_min = float(body.get("t_min", 8 * 60))
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
        incidents.append(Incident(
            incident_id=i, node=node, type_key=type_key, spec=spec,
            lat=snap_lat, lng=snap_lng,
        ))
        snapped.append({
            "incident_id": i, "type": type_key, "label": spec.label,
            "lat": snap_lat, "lng": snap_lng, "node": int(node),
            "n_fire": spec.n_fire_trucks, "n_amb": spec.n_ambulances,
        })

    if not incidents:
        return jsonify({
            "t_min": t_min, "time_label": format_clock(t_min),
            "incidents": [], "missions": [],
            "summary": {"n_incidents": 0, "n_units": 0, "overlap": 0},
        })

    missions_raw = plan_global_dispatch(
        G, STATION_MAP, HOSPITAL_MAP, incidents, verbose=False,
    )
    usage: Counter = Counter()
    for m in missions_raw:
        for u, v in zip(m["path"][:-1], m["path"][1:]):
            usage[(u, v)] += 1
    overlap = sum(c - 1 for c in usage.values() if c > 1)

    missions = [_mission_payload(m, t_min) for m in missions_raw]
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
        "summary": {
            "n_incidents": len(incidents),
            "n_units": len(missions),
            "overlap": overlap,
            "total_eta_min": round(sum(m["eta_min"] for m in missions), 1),
        },
    })


@app.post("/api/reroute")
def api_reroute():
    """이동 중 재탐색. 현재 시각 혼잡을 반영.

    소방차: current → 화재현장
    구급차(현장 전): current → 현장 → 응급실  (현장 경유 강제)
    구급차(이송 중): current → 응급실
    """
    body = request.get_json(force=True, silent=True) or {}
    t_min = float(body.get("t_min", 8 * 60))
    _set_context_from_request(body)
    vehicles = body.get("vehicles") or []
    _apply_time(t_min, horizon_min=float(body.get("horizon_min", 45)))

    edge_usage: Counter = Counter()
    out = []
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
        old = [int(n) for n in (v.get("path_nodes") or [])]
        # 진행 중이면 old는 '남은 경로'가 아니라 전체일 수 있음 → 노드열 비교는
        # 현재 위치부터의 suffix 와 새 path 비교
        changed = path != old and path != ([cur] + old[1:] if old else path)
        # 더 단순: 새 path 와 클라이언트가 보낸 remaining 비교
        remaining = v.get("remaining_nodes")
        if remaining is not None:
            changed = [int(n) for n in path] != [int(n) for n in remaining]

        out.append({
            "id": vid,
            "path_nodes": [int(n) for n in path],
            "path": _path_latlng(path),
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
