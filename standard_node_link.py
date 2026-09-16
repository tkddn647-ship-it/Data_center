"""
standard_node_link.py
=====================
국가표준노드링크(국토교통부 ITS)를 '도로망 뼈대(backbone)'로 읽어 networkx 그래프로
만든다.

설계 의도
---------
이전 버전은 안심구역 관측 데이터(카메라 directionNo, 개별차량이동경로)에서
연결성을 "추론"했다. 그 방식은 관측되지 않은 구간이 그래프에서 통째로
빠지는 근본 결함이 있다.

올바른 구조:
  1) 표준노드링크 = 전국(또는 대구) 도로망의 진짜 연결성 + 좌표  (뼈대)
  2) 사고/혼잡/카메라 = 뼈대 위 노드·엣지의 "속성"으로 얹음     (오버레이)

데이터 출처 (안심구역과 무관, 지금 바로 공개)
--------------------------------------------
- 대구광역시_도로 표준노드 현황 (data.go.kr/15049953)
  컬럼: 표준노드ID, 노드명, 노드유형코드, 회전제한여부, 경도, 위도, 적용시작일자
- 대구광역시_도로 표준링크 현황 (data.go.kr/15049952)
  컬럼: 표준링크ID, 출발표준노드, 도착표준노드, 지도상거리, 도로등급코드, ...
- 전국 SHP: ITS 표준노드링크 관리시스템 (nodelink.its.go.kr)
  MOCT_NODE.shp / MOCT_LINK.shp  (EPSG:5186, 선택 — geopandas 필요)

사용법
------
    # 공공데이터 CSV를 data/standard_node_link/ 아래에 두고:
    G = build_backbone_graph(data_dir="data/standard_node_link")

    # 파일이 없으면 대구 bbox 안의 "합성 도로망"을 만들어 구조만 검증
    G = build_backbone_graph()   # synthetic=True 자동
"""

from __future__ import annotations

import glob
import math
import os
import warnings
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

# 대구 대략 bbox (위도/경도) — gis_matching.check_daegu_bounds 와 동일 기준
# 대구 대략 bbox — 칠곡경북대병원(북구 호국로) 포함
DAEGU_LAT_MIN, DAEGU_LAT_MAX = 35.70, 36.00
DAEGU_LNG_MIN, DAEGU_LNG_MAX = 128.35, 128.80

# 공개 CSV / SHP 에서 자주 쓰이는 컬럼명 별칭
NODE_ID_ALIASES = ("표준노드ID", "NODE_ID", "node_id", "NODE_ID")
LAT_ALIASES = ("위도", "lat", "LAT", "Y", "좌표Y")
LNG_ALIASES = ("경도", "lng", "LON", "lon", "X", "좌표X")
LINK_ID_ALIASES = ("표준링크ID", "LINK_ID", "link_id")
F_NODE_ALIASES = ("출발표준노드", "F_NODE", "f_node", "FROM_NODE", "시작노드")
T_NODE_ALIASES = ("도착표준노드", "T_NODE", "t_node", "TO_NODE", "종료노드")
LENGTH_ALIASES = ("지도상거리", "LENGTH", "length", "SHAPE_STLE", "거리")
ROAD_NAME_ALIASES = ("도로명", "ROAD_NAME", "road_name")
ROAD_RANK_ALIASES = ("도로등급코드", "ROAD_RANK", "road_rank")


def _pick_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    cols = {c.strip(): c for c in df.columns}
    lower = {c.strip().lower(): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return cols[a]
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _to_int_id(series: pd.Series) -> pd.Series:
    """표준노드/링크 ID는 보통 10자리 숫자 문자열. 정수로 변환(앞자리 0 제거 없음)."""
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce").astype("Int64")


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _find_file(data_dir: str, patterns: list[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hits:
            return hits[0]
        hits = sorted(glob.glob(os.path.join(data_dir, "**", pat), recursive=True))
        if hits:
            return hits[0]
    return None


def load_daegu_node_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "utf-8", "euc-kr"):
        try:
            raw = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"인코딩을 알 수 없음: {path}")

    id_col = _pick_col(raw, NODE_ID_ALIASES)
    lat_col = _pick_col(raw, LAT_ALIASES)
    lng_col = _pick_col(raw, LNG_ALIASES)
    if not id_col or not lat_col or not lng_col:
        raise ValueError(
            f"표준노드 CSV 필수 컬럼 없음 (ID/위도/경도). 실제 컬럼={list(raw.columns)}"
        )

    out = pd.DataFrame({
        "node_id": _to_int_id(raw[id_col]),
        "lat": pd.to_numeric(raw[lat_col], errors="coerce"),
        "lng": pd.to_numeric(raw[lng_col], errors="coerce"),
    })
    name_col = _pick_col(raw, ("노드명", "NODE_NAME", "node_name"))
    if name_col:
        out["node_name"] = raw[name_col].astype(str)
    type_col = _pick_col(raw, ("노드유형코드", "NODE_TYPE", "node_type"))
    if type_col:
        out["node_type"] = raw[type_col].astype(str)
    out = out.dropna(subset=["node_id", "lat", "lng"]).drop_duplicates("node_id")
    return out.reset_index(drop=True)


def load_daegu_link_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "utf-8", "euc-kr"):
        try:
            raw = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"인코딩을 알 수 없음: {path}")

    link_col = _pick_col(raw, LINK_ID_ALIASES)
    f_col = _pick_col(raw, F_NODE_ALIASES)
    t_col = _pick_col(raw, T_NODE_ALIASES)
    if not link_col or not f_col or not t_col:
        raise ValueError(
            f"표준링크 CSV 필수 컬럼 없음 (링크ID/출발/도착). 실제 컬럼={list(raw.columns)}"
        )

    out = pd.DataFrame({
        "link_id": _to_int_id(raw[link_col]),
        "f_node": _to_int_id(raw[f_col]),
        "t_node": _to_int_id(raw[t_col]),
    })
    len_col = _pick_col(raw, LENGTH_ALIASES)
    out["length_m"] = (
        pd.to_numeric(raw[len_col], errors="coerce") if len_col else np.nan
    )
    name_col = _pick_col(raw, ROAD_NAME_ALIASES)
    if name_col:
        out["road_name"] = raw[name_col].astype(str)
    rank_col = _pick_col(raw, ROAD_RANK_ALIASES)
    if rank_col:
        out["road_rank"] = raw[rank_col].astype(str)
    out = out.dropna(subset=["link_id", "f_node", "t_node"]).drop_duplicates("link_id")
    return out.reset_index(drop=True)


def load_moct_shapefile(node_shp: str, link_shp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전국 MOCT_NODE / MOCT_LINK shapefile (EPSG:5186) → WGS84 lat/lng."""
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "SHP 로딩에는 geopandas가 필요합니다: pip install geopandas"
        ) from e

    nodes = gpd.read_file(node_shp, encoding="cp949")
    links = gpd.read_file(link_shp, encoding="cp949")
    if nodes.crs is None:
        nodes = nodes.set_crs(5186)
    if links.crs is None:
        links = links.set_crs(5186)
    nodes = nodes.to_crs(4326)
    links = links.to_crs(4326)

    id_col = _pick_col(nodes, NODE_ID_ALIASES) or "NODE_ID"
    node_df = pd.DataFrame({
        "node_id": _to_int_id(nodes[id_col]),
        "lat": nodes.geometry.y,
        "lng": nodes.geometry.x,
    }).dropna()

    link_id = _pick_col(links, LINK_ID_ALIASES) or "LINK_ID"
    f_col = _pick_col(links, F_NODE_ALIASES) or "F_NODE"
    t_col = _pick_col(links, T_NODE_ALIASES) or "T_NODE"
    len_col = _pick_col(links, LENGTH_ALIASES)
    link_df = pd.DataFrame({
        "link_id": _to_int_id(links[link_id]),
        "f_node": _to_int_id(links[f_col]),
        "t_node": _to_int_id(links[t_col]),
        "length_m": pd.to_numeric(links[len_col], errors="coerce") if len_col else np.nan,
    }).dropna(subset=["link_id", "f_node", "t_node"])
    return node_df, link_df


def synthesize_daegu_backbone(
    n_arterials: int = 6,
    grid_step: float = 0.012,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """실제 파일이 없을 때 쓰는 '대구 모양' 합성 도로망.

    무작위 점구름이 아니라, 대구 bbox 안에 격자 + 방사 간선으로 연결성을
    만든다. 그래서 지도에 그리면 '도로망처럼' 보인다 (진짜 대구 외곽선은
    아니지만, 뼈대 그래프 방식이 동작한다는 걸 증명하기에 충분).
    """
    rng = np.random.default_rng(seed)
    lats = np.arange(DAEGU_LAT_MIN + 0.02, DAEGU_LAT_MAX - 0.02, grid_step)
    lngs = np.arange(DAEGU_LNG_MIN + 0.02, DAEGU_LNG_MAX - 0.02, grid_step)

    # 노드: 격자점 + 약간의 지터(실제 도로가 격자에 딱 안 맞는 느낌)
    nodes = []
    node_id = 2240000000  # 대구권역 코드(224)를 흉내 낸 10자리 ID
    coord_to_id = {}
    for i, lat in enumerate(lats):
        for j, lng in enumerate(lngs):
            # 모서리를 깎아 타원형 도시 외곽처럼 보이게
            cy, cx = 35.87, 128.57
            if ((lat - cy) / 0.16) ** 2 + ((lng - cx) / 0.20) ** 2 > 1.0:
                continue
            jlat = float(lat + rng.normal(0, 0.0015))
            jlng = float(lng + rng.normal(0, 0.0015))
            nid = node_id
            node_id += 1
            nodes.append({"node_id": nid, "lat": jlat, "lng": jlng, "node_name": f"SYN_{i}_{j}"})
            coord_to_id[(i, j)] = nid

    node_df = pd.DataFrame(nodes)

    # 엣지: 격자 이웃 + 몇 개 대각선 간선
    links = []
    link_id = 2241000000
    id_to_coord = {v: k for k, v in coord_to_id.items()}
    pos = node_df.set_index("node_id")[["lat", "lng"]]

    def _add(a, b):
        nonlocal link_id
        if a not in pos.index or b not in pos.index:
            return
        la, loa = pos.loc[a, ["lat", "lng"]]
        lb, lob = pos.loc[b, ["lat", "lng"]]
        dist_m = _haversine_km(la, loa, lb, lob) * 1000.0
        links.append({
            "link_id": link_id, "f_node": a, "t_node": b, "length_m": dist_m, "road_name": "SYN",
        })
        link_id += 1
        links.append({
            "link_id": link_id, "f_node": b, "t_node": a, "length_m": dist_m, "road_name": "SYN",
        })
        link_id += 1

    for (i, j), nid in coord_to_id.items():
        for di, dj in ((0, 1), (1, 0)):
            nb = coord_to_id.get((i + di, j + dj))
            if nb is not None:
                _add(nid, nb)

    # 방사형 간선: 중심에서 외곽 방향으로 n_arterials 개
    center_keys = sorted(coord_to_id.keys(), key=lambda ij: (ij[0] - len(lats) / 2) ** 2 + (ij[1] - len(lngs) / 2) ** 2)
    center_id = coord_to_id[center_keys[0]]
    for k in range(n_arterials):
        angle = 2 * math.pi * k / n_arterials
        target_i = int(len(lats) / 2 + math.cos(angle) * (len(lats) / 2 - 1))
        target_j = int(len(lngs) / 2 + math.sin(angle) * (len(lngs) / 2 - 1))
        # 중심 → 외곽으로 가까운 격자점을 따라 연결
        path_keys = []
        ci, cj = center_keys[0]
        ti, tj = target_i, target_j
        while (ci, cj) != (ti, tj) and len(path_keys) < 40:
            path_keys.append((ci, cj))
            if abs(ti - ci) >= abs(tj - cj) and ci != ti:
                ci += 1 if ti > ci else -1
            elif cj != tj:
                cj += 1 if tj > cj else -1
            else:
                break
            if (ci, cj) not in coord_to_id:
                break
        path_keys.append((ci, cj))
        for a, b in zip(path_keys[:-1], path_keys[1:]):
            if a in coord_to_id and b in coord_to_id:
                _add(coord_to_id[a], coord_to_id[b])

    link_df = pd.DataFrame(links).drop_duplicates(subset=["f_node", "t_node"])
    return node_df, link_df


def resolve_backbone_tables(
    data_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """표준노드/링크 테이블을 찾고, 없으면 합성 데이터로 대체.

    Returns
    -------
    node_df, link_df, source : source ∈ {'csv','shp','synthetic'}
    """
    search_dirs = []
    if data_dir:
        search_dirs.append(data_dir)
    # 프로젝트 기본 위치
    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs.append(os.path.join(here, "data", "standard_node_link"))
    search_dirs.append(os.path.join(here, "data"))

    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        node_csv = _find_file(d, [
            "*표준노드*.csv", "*node*.csv", "*NODE*.csv", "MOCT_NODE.csv",
        ])
        link_csv = _find_file(d, [
            "*표준링크*.csv", "*link*.csv", "*LINK*.csv", "MOCT_LINK.csv",
        ])
        # node/link 둘 다 있어야 하고, 서로 다른 파일이어야 함
        if node_csv and link_csv and os.path.abspath(node_csv) != os.path.abspath(link_csv):
            # 파일명에 node/link가 명확히 구분되는지 한 번 더 확인
            nl = os.path.basename(node_csv).lower()
            ll = os.path.basename(link_csv).lower()
            if ("node" in nl or "노드" in os.path.basename(node_csv)) and (
                "link" in ll or "링크" in os.path.basename(link_csv)
            ):
                return load_daegu_node_csv(node_csv), load_daegu_link_csv(link_csv), "csv"

        node_shp = _find_file(d, ["MOCT_NODE.shp", "*NODE*.shp"])
        link_shp = _find_file(d, ["MOCT_LINK.shp", "*LINK*.shp"])
        if node_shp and link_shp:
            return (*load_moct_shapefile(node_shp, link_shp), "shp")

    node_df, link_df = synthesize_daegu_backbone()
    return node_df, link_df, "synthetic"


def _filter_daegu_bbox(node_df: pd.DataFrame) -> pd.DataFrame:
    """공개 CSV에 간혹 대구 밖/이상치 좌표가 섞여 있어 권역으로 자른다."""
    m = (
        (node_df["lat"] >= DAEGU_LAT_MIN) & (node_df["lat"] <= DAEGU_LAT_MAX)
        & (node_df["lng"] >= DAEGU_LNG_MIN) & (node_df["lng"] <= DAEGU_LNG_MAX)
    )
    return node_df.loc[m].copy()


def build_backbone_graph(
    data_dir: Optional[str] = None,
    default_speed_kmh: float = 40.0,
    clip_to_daegu: bool = True,
    verbose: bool = True,
) -> nx.DiGraph:
    """표준노드링크 → DiGraph.

    노드 속성: lat, lng, domain='urban', source, (optional) node_name
    엣지 속성: link_id, length_m, travel_time(분), congestion=0.3, risk=0.3
               (혼잡/위험은 이후 오버레이에서 덮어씀)
    """
    node_df, link_df, source = resolve_backbone_tables(data_dir=data_dir)

    if source == "synthetic":
        warnings.warn(
            "[standard_node_link] 실제 표준노드링크 파일이 없어 합성 대구 도로망을 "
            "사용합니다. data.go.kr에서 '대구광역시_도로 표준노드/링크 현황' CSV를 "
            "받아 data/standard_node_link/ 에 두면 진짜 대구 도로망으로 교체됩니다.",
            stacklevel=2,
        )

    if clip_to_daegu and source != "synthetic":
        before = len(node_df)
        node_df = _filter_daegu_bbox(node_df)
        if verbose:
            print(f"[standard_node_link] Daegu bbox filter: {before} -> {len(node_df)} nodes")

    # 링크 양끝이 노드 테이블에 있는 것만 쓴다 (공개 CSV는 노드⊃링크 끝점)
    node_ids = set(node_df["node_id"].astype(int))
    link_df = link_df[
        link_df["f_node"].astype(int).isin(node_ids)
        & link_df["t_node"].astype(int).isin(node_ids)
    ].copy()

    G = nx.DiGraph()
    for _, row in node_df.iterrows():
        attrs = {
            "lat": float(row["lat"]),
            "lng": float(row["lng"]),
            "domain": "urban",
            "backbone_source": source,
            "risk": 0.3,
        }
        if "node_name" in row and pd.notna(row["node_name"]):
            attrs["node_name"] = str(row["node_name"])
        G.add_node(int(row["node_id"]), **attrs)

    for _, row in link_df.iterrows():
        a, b = int(row["f_node"]), int(row["t_node"])
        length_m = row.get("length_m", np.nan)
        if pd.isna(length_m) or float(length_m) <= 0:
            length_m = _haversine_km(
                G.nodes[a]["lat"], G.nodes[a]["lng"],
                G.nodes[b]["lat"], G.nodes[b]["lng"],
            ) * 1000.0
        length_km = float(length_m) / 1000.0
        travel_time = (length_km / max(default_speed_kmh, 5.0)) * 60.0
        edge_attrs = {
            "link_id": int(row["link_id"]) if pd.notna(row.get("link_id")) else None,
            "length_m": float(length_m),
            "travel_time": travel_time,
            "congestion": 0.3,
            "risk": 0.3,
            "backbone": True,
        }
        if "road_name" in row and pd.notna(row["road_name"]):
            edge_attrs["road_name"] = str(row["road_name"])
        G.add_edge(a, b, **edge_attrs)

    # 링크에 안 걸린 고립 노드 제거 (경로 탐색에 방해)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if verbose:
        print(f"[standard_node_link] source={source}, "
              f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
              f"dropped_isolates={len(isolates)}")
        if G.number_of_nodes():
            lats = [d["lat"] for _, d in G.nodes(data=True)]
            lngs = [d["lng"] for _, d in G.nodes(data=True)]
            print(f"[standard_node_link] bbox lat=[{min(lats):.4f},{max(lats):.4f}] "
                  f"lng=[{min(lngs):.4f},{max(lngs):.4f}]")

    return G


def nearest_node(G: nx.DiGraph, lat: float, lng: float) -> int:
    """위경도에 가장 가까운 백본 노드 ID (단순 선형 탐색 — 대구 규모면 충분)."""
    best, best_d = None, float("inf")
    for n, d in G.nodes(data=True):
        if "lat" not in d:
            continue
        dist = (d["lat"] - lat) ** 2 + (d["lng"] - lng) ** 2
        if dist < best_d:
            best, best_d = n, dist
    if best is None:
        raise ValueError("그래프에 좌표 노드가 없습니다.")
    return best


def snap_points_to_nodes(
    G: nx.DiGraph,
    points: pd.DataFrame,
    lat_col: str = "lat",
    lng_col: str = "lng",
    id_col: Optional[str] = None,
) -> dict:
    """관측점(교차로/병원/IC) → 최근접 백본 노드 매핑 dict.

    Returns
    -------
    mapping : {원본 id 또는 row index -> backbone node_id}
    """
    mapping = {}
    for idx, row in points.iterrows():
        key = int(row[id_col]) if id_col and id_col in row else idx
        mapping[key] = nearest_node(G, float(row[lat_col]), float(row[lng_col]))
    return mapping


def export_backbone_map(G: nx.DiGraph, out_path: str = "backbone_map.png") -> str:
    """뼈대 도로망만 그려서 저장 — 진짜 대구처럼 보이는지 눈으로 확인용."""
    import matplotlib.pyplot as plt

    pos = {n: (d["lng"], d["lat"]) for n, d in G.nodes(data=True) if "lat" in d}
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_aspect("equal")
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=0.6, arrows=False, edge_color="seagreen", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=8, node_color="darkgreen", alpha=0.7, ax=ax)
    src = next((d.get("backbone_source", "?") for _, d in G.nodes(data=True)), "?")
    ax.set_title(f"Standard node-link backbone ({src}): "
                 f"{G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    note = ("Real map if source=csv/shp. Synthetic grid-arterial if source=synthetic "
            "(still road-shaped — not a random cloud).")
    ax.text(0.01, -0.07, note, transform=ax.transAxes, fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[standard_node_link] backbone map saved: {out_path}")
    return out_path


if __name__ == "__main__":
    G = build_backbone_graph()
    export_backbone_map(G)
    # 연결성 스모크 테스트: 임의 두 노드 사이 경로 존재 여부
    nodes = list(G.nodes)
    if len(nodes) >= 2:
        a, b = nodes[0], nodes[len(nodes) // 2]
        print(f"path {a} -> {b}: {nx.has_path(G, a, b)}")
