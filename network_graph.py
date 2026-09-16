"""
network_graph.py
=================
대구시 응급 골든타임 — **오픈데이터 표준노드링크** 위에
오픈 교통 통계(및 선택적 실시간 API / 안심구역 ITS)를 얹고,
오픈 소방서·응급실 좌표를 스냅한다.

기본 (오픈데이터)
-----------------
- data/standard_node_link/  표준노드·링크 (data.go.kr)
- data/open/link_hourly_stats.csv  링크 시간별 속도
- data/open/fire_stations.csv , er_hospitals.csv

선택
----
- DAEGU_TRAFFIC_API_KEY  실시간 소통 (data.go.kr/15126266) + 돌발 (15126267 /dgincident)
- --data-dir 경북대 ITS   안심구역 보강
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import networkx as nx

from standard_node_link import build_backbone_graph, nearest_node
from daegu_ems_geo import DAEGU_FIRE_STATIONS, DAEGU_ER_HOSPITALS

URBAN_NODE_OFFSET = 0

_HERE = Path(__file__).resolve().parent
_DEFAULT_BACKBONE = _HERE / "data" / "standard_node_link"


def _overlay_open_congestion(G: nx.DiGraph, verbose: bool = True) -> str:
    """오픈 속도 프로파일 → 엣지 혼잡. 실시간·돌발 API 있으면 추가 보정."""
    from daegu_open_data import (
        apply_incidents_to_graph,
        apply_open_profile_to_graph,
        apply_realtime_speeds,
        fetch_incident_events,
        fetch_realtime_traffic,
        load_speed_profile,
    )

    source = "none"
    profile = load_speed_profile()
    if profile:
        hits = apply_open_profile_to_graph(G, profile, hour=8, day_type="weekday")
        source = profile.get("source", "open_profile")
        if verbose:
            print(
                f"[network_graph] 오픈 속도프로파일 적용 roads_hit~{hits}, "
                f"n_roads={profile.get('n_roads')}"
            )
    try:
        items = fetch_realtime_traffic()
        rt = apply_realtime_speeds(G, items)
        if rt:
            source = f"{source}+realtime_api" if source != "none" else "realtime_api"
            if verbose:
                print(f"[network_graph] 실시간 소통 API 매칭 {rt}개")
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[network_graph] 실시간 API 생략: {e}")
    try:
        events = fetch_incident_events()
        n = apply_incidents_to_graph(G, events)
        if events:
            source = f"{source}+dgincident" if source != "none" else "dgincident"
            if verbose:
                print(f"[network_graph] 돌발(공사·사고) {len(events)}건 → 엣지 {n}개 페널티")
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[network_graph] 돌발 API 생략: {e}")
    return source


def _overlay_its_congestion(G: nx.DiGraph, data_dir: str, verbose: bool = True) -> None:
    """경북대 ITS 교통량을 최근접 표준노드 진입 엣지에 혼잡으로 반영(선택)."""
    from data_schema import (
        load_daegu_camera_data,
        load_daegu_traffic_volume_data,
        load_daegu_intersection_data,
    )
    from gis_matching import check_daegu_bounds

    inter = load_daegu_intersection_data(data_dir=data_dir)
    cam = load_daegu_camera_data(data_dir=data_dir)
    vol = load_daegu_traffic_volume_data(data_dir=data_dir)

    ratio = check_daegu_bounds(inter["lat"], inter["lng"])
    if verbose:
        print(f"[network_graph] ITS 교차로 대구권 비율: {ratio:.1%}")

    vehicle_cols = [c for c in vol.columns if c not in ("t", "c", "i", "re", "p")]
    vol = vol.copy()
    vol["total"] = vol[vehicle_cols].sum(axis=1)
    cam_vol = vol.groupby("c")["total"].mean()
    max_vol = float(cam_vol.max()) if len(cam_vol) else 1.0

    hits = 0
    for _, row in cam.iterrows():
        try:
            node = nearest_node(G, float(row["lat"]), float(row["lng"]))
        except Exception:
            continue
        cong = float(cam_vol.get(row["cameraNo"], 0) / (max_vol + 1e-6))
        for pred in list(G.predecessors(node)):
            ed = G.edges[pred, node]
            ed["congestion"] = max(float(ed.get("congestion", 0.0)), cong)
            length_km = float(ed.get("length_m", 400.0)) / 1000.0
            speed = max(15.0, 45.0 * (1 - 0.55 * ed["congestion"]))
            ed["travel_time"] = (length_km / speed) * 60.0
            hits += 1
    if verbose:
        print(f"[network_graph] ITS 혼잡 오버레이 엣지 {hits}개 (cameras={len(cam)})")


def snap_ems_facilities(G: nx.DiGraph, max_snap_m: float = 800.0) -> tuple[dict, dict]:
    """시설 좌표 → 최근접 표준노드.

    예전 unique 강제 스냅은 가까운 병원·서가 서로 밀어내 수백 m~km 밀림.
    같은 노드를 공유해도 되며, 너무 먼 스냅은 경고만 남긴다.
    """
    from road_shapes import haversine_m

    def _snap(lat: float, lng: float, label: str) -> int:
        nid = nearest_node(G, lat, lng)
        d = haversine_m(lat, lng, G.nodes[nid]["lat"], G.nodes[nid]["lng"])
        if d > max_snap_m:
            print(f"[network_graph] WARN snap {label}: {d:.0f}m (>{max_snap_m:.0f}m)")
        return int(nid)

    station_map = {}
    for s in DAEGU_FIRE_STATIONS:
        nid = _snap(s["lat"], s["lng"], s["station_name"])
        station_map[int(s["station_id"])] = nid
        G.nodes[nid]["is_station"] = True
        G.nodes[nid]["station_name"] = s["station_name"]
        G.nodes[nid]["facility_lat"] = s["lat"]
        G.nodes[nid]["facility_lng"] = s["lng"]

    hospital_map = {}
    for h in DAEGU_ER_HOSPITALS:
        nid = _snap(h["lat"], h["lng"], h["hospital_name"])
        hospital_map[int(h["hospital_id"])] = nid
        G.nodes[nid]["is_hospital"] = True
        G.nodes[nid]["hospital_name"] = h["hospital_name"]
        G.nodes[nid]["facility_lat"] = h["lat"]
        G.nodes[nid]["facility_lng"] = h["lng"]

    return station_map, hospital_map


def build_urban_graph(n_intersections: int = 40, verbose: bool = True,
                       data_dir: str | None = None,
                       backbone_dir: str | None = None) -> nx.DiGraph:
    """실제 대구 표준노드링크 뼈대 + 오픈 혼잡 (+ 선택 ITS)."""
    bdir = backbone_dir or str(_DEFAULT_BACKBONE)
    if not os.path.isdir(bdir):
        bdir = None
    G = build_backbone_graph(data_dir=bdir, verbose=verbose, clip_to_daegu=True)

    open_src = _overlay_open_congestion(G, verbose=verbose)
    G.graph["open_congestion_source"] = open_src

    if open_src == "none":
        rng = np.random.default_rng(0)
        for u, v, ed in G.edges(data=True):
            if "congestion" not in ed or ed.get("congestion") is None:
                ed["congestion"] = float(rng.uniform(0.15, 0.55))
            length_km = float(ed.get("length_m", 400.0)) / 1000.0
            speed = max(15.0, 40.0 * (1 - 0.5 * ed["congestion"]))
            ed["travel_time"] = (length_km / speed) * 60.0
            ed.setdefault("risk", 0.25)
    else:
        for _, _, ed in G.edges(data=True):
            ed.setdefault("risk", 0.25)

    if data_dir:
        _overlay_its_congestion(G, data_dir, verbose=verbose)

    if verbose:
        print(
            f"[network_graph] 대구 실도로망: "
            f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}"
        )
    return G


def build_combined_graph(n_hw_links: int = 200, n_intersections: int = 40, verbose: bool = True,
                          data_dir: str | None = None, backbone_dir: str | None = None, **_kw):
    if verbose:
        print("[network_graph] 스코프: 대구시 응급 골든타임 (오픈데이터 기본)")
        print("[network_graph] 지도: 표준노드링크 + 오픈 소방/응급의료")
        if data_dir:
            print(f"[network_graph] ITS 보강: {data_dir}")

    G = build_urban_graph(
        n_intersections=n_intersections, verbose=verbose,
        data_dir=data_dir, backbone_dir=backbone_dir,
    )
    station_map, hospital_map = snap_ems_facilities(G)
    G.graph["station_node_map"] = station_map
    G.graph["hospital_node_map"] = hospital_map

    if verbose:
        print("[network_graph] 소방서 스냅:")
        for sid, nid in station_map.items():
            name = next(
                s["station_name"] for s in DAEGU_FIRE_STATIONS if s["station_id"] == sid
            )
            print(f"  #{sid} {name} -> node {nid}")
        print("[network_graph] 응급실 스냅:")
        for hid, nid in hospital_map.items():
            name = next(
                h["hospital_name"] for h in DAEGU_ER_HOSPITALS if h["hospital_id"] == hid
            )
            print(f"  #{hid} {name} -> node {nid}")

    return G, hospital_map


def build_highway_graph(*args, **kwargs):
    raise RuntimeError("대구 시내 스코프만 사용. build_combined_graph()를 쓰세요.")


if __name__ == "__main__":
    G, hosp = build_combined_graph()
    print("hospitals", hosp)
