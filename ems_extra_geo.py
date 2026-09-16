"""
ems_extra_geo.py
================
추가 EMS 공간데이터 로더.

1) 119안전센터 (data.go.kr/15117114) → data/open/safety_centers_119.csv
2) 긴급차 진출입로 아파트 (data.go.kr/15156867) → data/open/emergency_entrances_daegu.csv

용도
----
- 안전센터: 출동 **출발 좌표**를 소방서보다 가까운 센터로 (차량 재고는 소속 소방서)
- 진출입로: 클릭이 단지 근처면 **현장 좌표를 입구로 스냅** (실제로 들어가는 길)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
OPEN = _HERE / "data" / "open"
CENTERS_CSV = OPEN / "safety_centers_119.csv"
ENTRANCES_CSV = OPEN / "emergency_entrances_daegu.csv"
SHP_DIR = _HERE / "data" / "대구광역시119공간센터"


def _haversine_m(lat1, lng1, lat2, lng2) -> float:
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lng2 - lng1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def ensure_safety_centers_csv(force: bool = False) -> Path | None:
    """SHP(cp949) → CSV. 이미 CSV 있으면 유지."""
    if CENTERS_CSV.is_file() and not force and CENTERS_CSV.stat().st_size > 100:
        return CENTERS_CSV
    shp = SHP_DIR / "gi_fire_p.shp"
    if not shp.is_file():
        return CENTERS_CSV if CENTERS_CSV.is_file() else None
    try:
        import geopandas as gpd
    except ImportError:
        return CENTERS_CSV if CENTERS_CSV.is_file() else None
    g = gpd.read_file(shp, encoding="cp949")
    if g.crs and str(g.crs) != "EPSG:4326":
        g = g.to_crs(4326)
    rows = []
    for i, r in g.iterrows():
        geom = r.geometry
        if geom is None:
            continue
        rows.append({
            "center_id": int(r.get("gid") or i),
            "ward_id": str(r.get("ward_id") or ""),
            "name": str(r.get("ward_nm") or ""),
            "type_cd": str(r.get("type_cd") or ""),
            "lat": round(float(geom.y), 7),
            "lng": round(float(geom.x), 7),
        })
    OPEN.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CENTERS_CSV, index=False, encoding="utf-8-sig")
    return CENTERS_CSV


def load_safety_centers() -> list[dict]:
    ensure_safety_centers_csv()
    if not CENTERS_CSV.is_file():
        return []
    df = pd.read_csv(CENTERS_CSV, encoding="utf-8-sig")
    out = []
    for _, r in df.iterrows():
        try:
            out.append({
                "center_id": int(r["center_id"]),
                "ward_id": str(r.get("ward_id", "")),
                "name": str(r.get("name", "")),
                "type_cd": str(r.get("type_cd", "")),
                "lat": float(r["lat"]),
                "lng": float(r["lng"]),
            })
        except (TypeError, ValueError, KeyError):
            continue
    return out


def load_emergency_entrances(path: Path | None = None) -> list[dict]:
    """
    국토부 긴급자동차 진출입로 CSV.
    포털에서 받아 data/open/emergency_entrances_daegu.csv 로 저장.
    https://www.data.go.kr/data/15156867/fileData.do
    """
    p = Path(path) if path else ENTRANCES_CSV
    if not p.is_file():
        # 흔한 다운로드 파일명도 탐색
        for cand in OPEN.glob("*진출입*.csv"):
            p = cand
            break
        else:
            return []
    raw = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            raw = pd.read_csv(p, encoding=enc)
            break
        except Exception:
            continue
    if raw is None or raw.empty:
        return []

    def pick(*names):
        cols = {str(c).strip(): c for c in raw.columns}
        low = {str(c).strip().lower().replace(" ", ""): c for c in raw.columns}
        for n in names:
            if n in cols:
                return cols[n]
            key = n.lower().replace(" ", "")
            if key in low:
                return low[key]
        for c in raw.columns:
            cl = str(c).lower()
            for n in names:
                if n.lower() in cl:
                    return c
        return None

    c_code = pick("단지코드", "k-apt", "단지 코드")
    c_name = pick("단지명", "단지이름")
    c_addr = pick("도로명주소", "단지주소", "주소")
    c_gate = pick("입구번호", "출입로명", "출입구")
    c_memo = pick("출입로 참고메모", "메모", "참고")
    c_lat = pick("좌표 Y(WGS84경위도)", "좌표Y(WGS84경위도)", "위도", "lat", "y(wgs84)")
    c_lng = pick("좌표 X(WGS84경위도)", "좌표X(WGS84경위도)", "경도", "lng", "lon", "x(wgs84)")
    # 투영좌표만 있으면 스킵(WGS 필요). 일부 파일은 좌표x/y 가 WGS일 수도.
    if c_lat is None or c_lng is None:
        c_lat = pick("좌표y", "좌표Y", "Y")
        c_lng = pick("좌표x", "좌표X", "X")

    out = []
    for i, r in raw.iterrows():
        try:
            lat = float(str(r[c_lat]).replace(",", ""))
            lng = float(str(r[c_lng]).replace(",", ""))
        except Exception:
            continue
        # 대구 대략 범위 (투영좌표 오인 방지)
        if not (35.5 < lat < 36.4 and 128.2 < lng < 129.0):
            continue
        out.append({
            "apt_code": str(r[c_code]) if c_code is not None else "",
            "name": str(r[c_name]) if c_name is not None else "",
            "address": str(r[c_addr]) if c_addr is not None else "",
            "gate": str(r[c_gate]) if c_gate is not None else "",
            "memo": str(r[c_memo]) if c_memo is not None else "",
            "lat": lat,
            "lng": lng,
        })
    return out


def nearest_safety_center(lat: float, lng: float, centers: list[dict] | None = None) -> dict | None:
    centers = centers if centers is not None else load_safety_centers()
    if not centers:
        return None
    best, best_d = None, 1e18
    for c in centers:
        d = _haversine_m(lat, lng, c["lat"], c["lng"])
        if d < best_d:
            best, best_d = c, d
    if best is None:
        return None
    return {**best, "dist_m": round(best_d, 1)}


def nearest_entrance(lat: float, lng: float, entrances: list[dict] | None = None,
                     max_m: float = 350.0) -> dict | None:
    """max_m 이내면 아파트 진출입로로 스냅."""
    entrances = entrances if entrances is not None else load_emergency_entrances()
    if not entrances:
        return None
    best, best_d = None, 1e18
    for e in entrances:
        d = _haversine_m(lat, lng, e["lat"], e["lng"])
        if d < best_d:
            best, best_d = e, d
    if best is None or best_d > max_m:
        return None
    return {**best, "dist_m": round(best_d, 1)}


def attach_parent_station(centers: list[dict], fire_stations: list[dict]) -> list[dict]:
    """각 안전센터 → 최근접 소방서(차량 재고 소속)."""
    out = []
    for c in centers:
        parent = None
        best = 1e18
        for s in fire_stations:
            d = _haversine_m(c["lat"], c["lng"], float(s["lat"]), float(s["lng"]))
            if d < best:
                best = d
                parent = s
        row = dict(c)
        if parent:
            row["parent_station_id"] = int(parent["station_id"])
            row["parent_station_name"] = parent.get("station_name", "")
            row["parent_dist_m"] = round(best, 1)
        out.append(row)
    return out
