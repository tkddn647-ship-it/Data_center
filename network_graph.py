"""
network_graph.py
=================
대구시 응급 골든타임 — **실제 대구 도로망(표준노드링크)** 위에
경북대 ITS 혼잡(있으면)을 얹고, 실제 소방서·응급실을 스냅한다.

지도 뼈대 (항상 실제 대구)
----------------------------
data/standard_node_link/daegu_nodes.csv + daegu_links.csv
(공공데이터포털 대구시 표준노드/링크)

ITS (안심구역 data_dir 있을 때)
--------------------------------
camerainfo / traffic → 최근접 표준노드에 혼잡 오버레이
intersectioninfo → 참고(좌표 검증)

ITS 없을 때(로컬 데모)
----------------------
실제 도로망 + 합성 혼잡으로도 **진짜 대구 모양** 경로 데모 가능.
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


def _overlay_its_congestion(G: nx.DiGraph, data_dir: str, verbose: bool = True) -> None:
    """경북대 ITS 교통량을 최근접 표준노드 진입 엣지에 혼잡으로 반영."""
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


def snap_ems_facilities(G: nx.DiGraph) -> tuple[dict, dict]:
    used: set[int] = set()

    def _snap(lat: float, lng: float) -> int:
        ranked = sorted(
            G.nodes,
            key=lambda n: (G.nodes[n]["lat"] - lat) ** 2 + (G.nodes[n]["lng"] - lng) ** 2,
        )
        for n in ranked:
            if n not in used:
                used.add(n)
                return int(n)
        return int(ranked[0])

    station_map = {}
    for s in DAEGU_FIRE_STATIONS:
        nid = _snap(s["lat"], s["lng"])
        station_map[int(s["station_id"])] = nid
        G.nodes[nid]["is_station"] = True
        G.nodes[nid]["station_name"] = s["station_name"]
        G.nodes[nid]["facility_lat"] = s["lat"]
        G.nodes[nid]["facility_lng"] = s["lng"]

    hospital_map = {}
    for h in DAEGU_ER_HOSPITALS:
        nid = _snap(h["lat"], h["lng"])
        hospital_map[int(h["hospital_id"])] = nid
        G.nodes[nid]["is_hospital"] = True
        G.nodes[nid]["hospital_name"] = h["hospital_name"]
        G.nodes[nid]["facility_lat"] = h["lat"]
        G.nodes[nid]["facility_lng"] = h["lng"]

    return station_map, hospital_map


def build_urban_graph(n_intersections: int = 40, verbose: bool = True,
                       data_dir: str | None = None,
                       backbone_dir: str | None = None) -> nx.DiGraph:
    """실제 대구 표준노드링크 뼈대 (+ 선택적 ITS 혼잡)."""
    bdir = backbone_dir or str(_DEFAULT_BACKBONE)
    if not os.path.isdir(bdir):
        bdir = None
    G = build_backbone_graph(data_dir=bdir, verbose=verbose, clip_to_daegu=True)

    # 데모용 기본 혼잡 (ITS 없을 때)
    rng = np.random.default_rng(0)
    for u, v, ed in G.edges(data=True):
        if "congestion" not in ed or ed.get("congestion") is None:
            ed["congestion"] = float(rng.uniform(0.15, 0.55))
        length_km = float(ed.get("length_m", 400.0)) / 1000.0
        speed = max(15.0, 40.0 * (1 - 0.5 * ed["congestion"]))
        ed["travel_time"] = (length_km / speed) * 60.0
        ed.setdefault("risk", 0.25)

    if data_dir:
        _overlay_its_congestion(G, data_dir, verbose=verbose)

    if verbose:
        print(f"[network_graph] 대구 실도로망: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    return G


def build_combined_graph(n_hw_links: int = 200, n_intersections: int = 40, verbose: bool = True,
                          data_dir: str | None = None, backbone_dir: str | None = None, **_kw):
    if verbose:
        print("[network_graph] 스코프: 대구시 응급 골든타임")
        print("[network_graph] 지도: 대구 표준노드링크(실제 도로) + 실제 병원/소방서 좌표")
        if data_dir:
            print(f"[network_graph] ITS 혼잡 오버레이: {data_dir}")

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
            name = next(s["station_name"] for s in DAEGU_FIRE_STATIONS if s["station_id"] == sid)
            print(f"   {name} -> node {nid} @ ({G.nodes[nid]['lat']:.5f},{G.nodes[nid]['lng']:.5f})")
        print("[network_graph] 응급실 스냅:")
        for hid, nid in hospital_map.items():
            name = next(h["hospital_name"] for h in DAEGU_ER_HOSPITALS if h["hospital_id"] == hid)
            print(f"   {name} -> node {nid} @ ({G.nodes[nid]['lat']:.5f},{G.nodes[nid]['lng']:.5f})")
        s0 = next(iter(station_map.values()))
        ok = sum(1 for h in hospital_map.values() if nx.has_path(G, s0, h))
        print(f"[network_graph] 샘플 소방서→병원 도달 {ok}/{len(hospital_map)}")

    return G, hospital_map


def build_highway_graph(*args, **kwargs):
    raise RuntimeError("대구 시내 스코프만 사용. build_combined_graph()를 쓰세요.")


if __name__ == "__main__":
    G, hosp = build_combined_graph()
    print("hospitals", hosp)
