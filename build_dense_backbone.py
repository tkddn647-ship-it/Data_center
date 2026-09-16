"""
build_dense_backbone.py
=======================
안심구역·오프라인용으로 **최대한 촘촘한** 표준노드링크 뼈대를 만든다.

추천 (오프라인 학습에 가장 맞음)
--------------------------------
1) ITS 전국 표준노드링크 SHP 다운로드 (무료, 한 번만)
   https://www.its.go.kr/nodelink/nodelinkRef
   → 압축 해제 후 MOCT_NODE.shp / MOCT_LINK.shp 를
     data/standard_node_link/ 에 둔다.

2) 이 스크립트 실행:
   python build_dense_backbone.py

   대구 bbox만 잘라 data/standard_node_link/daegu_nodes.csv
   · daegu_links.csv 로 저장 (기존 CSV는 .bak).

왜 이게 "오픈+오프라인 최대 촘촘"인가
------------------------------------
- 공공·ITS 표준이라 linkspeed/혼잡 ID와 맞추기 쉬움
- 포털 대구 CSV 샘플보다 전국 SHP 원본이 링크가 더 많음
- 한 번 받으면 인터넷 없이 학습·관제 가능

골목 내비 수준이 필요하면 (선택)
--------------------------------
OSM PBF는 더 촘촘하지만 표준링크ID가 없어 혼잡 매칭이 약해짐.
오프라인 OSRM용으로만 별도 권장 (ROUTING_ENGINE=osrm + 로컬 backend).

실행:
    python build_dense_backbone.py
    python build_dense_backbone.py --shp-dir path/to/NODELINKDATA
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from standard_node_link import (
    DAEGU_LAT_MAX,
    DAEGU_LAT_MIN,
    DAEGU_LNG_MAX,
    DAEGU_LNG_MIN,
    load_moct_shapefile,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "standard_node_link"


def _find_moct(shp_dir: Path) -> tuple[Path, Path] | None:
    node = None
    link = None
    for p in shp_dir.rglob("*.shp"):
        name = p.name.upper()
        if "NODE" in name and "LINK" not in name:
            node = p
        elif "LINK" in name:
            link = p
    if node and link:
        return node, link
    return None


def export_daegu_csv(node_shp: Path, link_shp: Path, out_dir: Path) -> tuple[int, int]:
    node_df, link_df = load_moct_shapefile(str(node_shp), str(link_shp))
    # 대구권 ID (15xxxxxx…) + bbox
    id_mask = node_df["node_id"].astype("Int64").astype(str).str.startswith(
        ("150", "151", "152", "153", "154", "155", "156", "157", "158", "159")
    )
    bbox = (
        (node_df["lat"] >= DAEGU_LAT_MIN) & (node_df["lat"] <= DAEGU_LAT_MAX)
        & (node_df["lng"] >= DAEGU_LNG_MIN) & (node_df["lng"] <= DAEGU_LNG_MAX)
    )
    nodes = node_df[id_mask | bbox].copy()
    keep = set(nodes["node_id"].dropna().astype(int).tolist())
    links = link_df[
        link_df["f_node"].isin(keep) & link_df["t_node"].isin(keep)
    ].copy()
    # 고아 제거
    used = set(links["f_node"].astype(int)) | set(links["t_node"].astype(int))
    nodes = nodes[nodes["node_id"].isin(used)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    n_path = out_dir / "daegu_nodes.csv"
    l_path = out_dir / "daegu_links.csv"
    for p in (n_path, l_path):
        if p.is_file():
            bak = p.with_suffix(p.suffix + ".bak")
            shutil.copy2(p, bak)
            print(f"[backup] {p.name} → {bak.name}")

    # 기존 로더가 기대하는 한글 컬럼명으로 저장
    nodes_out = pd.DataFrame({
        "표준노드ID": nodes["node_id"].astype("Int64"),
        "위도": nodes["lat"].astype(float),
        "경도": nodes["lng"].astype(float),
    })
    links_out = pd.DataFrame({
        "표준링크ID": links["link_id"].astype("Int64"),
        "출발표준노드": links["f_node"].astype("Int64"),
        "도착표준노드": links["t_node"].astype("Int64"),
        "지도상거리": links["length_m"].astype(float),
    })
    nodes_out.to_csv(n_path, index=False, encoding="utf-8-sig")
    links_out.to_csv(l_path, index=False, encoding="utf-8-sig")
    print(f"[dense] nodes={len(nodes_out)} links={len(links_out)}")
    print(f"[dense] wrote {n_path}")
    print(f"[dense] wrote {l_path}")
    return len(nodes_out), len(links_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shp-dir",
        type=str,
        default=str(OUT_DIR),
        help="MOCT_NODE.shp / MOCT_LINK.shp 가 있는 폴더 (기본 data/standard_node_link)",
    )
    args = ap.parse_args()
    shp_dir = Path(args.shp_dir)
    found = _find_moct(shp_dir)
    if not found:
        print(
            "MOCT_NODE.shp / MOCT_LINK.shp 를 찾을 수 없습니다.\n"
            "1) https://www.its.go.kr/nodelink/nodelinkRef 에서 전국표준노드링크 다운로드\n"
            f"2) 압축 해제 후 SHP를 {OUT_DIR} 에 복사\n"
            "3) python build_dense_backbone.py 다시 실행\n"
            "\n"
            "이 데이터가 오프라인·안심구역에서 쓸 수 있는 오픈 도로망 중\n"
            "표준링크ID까지 맞는 '가장 촘촘한' 선택입니다.\n"
            "(골목 100%는 OSM이지만 혼잡 ID 매칭이 깨짐 → 학습 본체는 ITS 권장)"
        )
        raise SystemExit(1)
    node_shp, link_shp = found
    print(f"[dense] node={node_shp}")
    print(f"[dense] link={link_shp}")
    export_daegu_csv(node_shp, link_shp, OUT_DIR)


if __name__ == "__main__":
    main()
