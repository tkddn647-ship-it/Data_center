"""
daegu_open_data.py
==================
공공데이터포털(오픈데이터) 기반 대구 로더.

필수(권장) 자료 — data/open/ 에 두고 사용:
  1) 표준노드/링크     → 이미 data/standard_node_link/ (data.go.kr/15049953, 15049952)
  2) 링크별 시간별 통계 → link_hourly_stats.csv     (data.go.kr/15117329)
  3) 응급의료기관 현황  → er_hospitals.csv           (data.go.kr/15132528 + 공개좌표)
  4) 소방서 좌표        → fire_stations.csv          (소방청 전국소방서 좌표 등)
  5) (선택) 실시간 소통 → DAEGU_TRAFFIC_API_KEY      (data.go.kr/15126266)
  6) (선택) 돌발(공사·사고) → DAEGU_INCIDENT_API_URL (data.go.kr/15126267 /dgincident)

안심구역 경북대 ITS는 선택 보강이다. 데모·예측의 기본 축은 오픈데이터다.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
OPEN_DIR = _HERE / "data" / "open"
PROFILE_PATH = OPEN_DIR / "hourly_speed_profile.json"

# data.go.kr 안내
OPEN_CATALOG = [
    {
        "id": "standard_node",
        "name": "대구 도로 표준노드 현황",
        "url": "https://www.data.go.kr/data/15049953/fileData.do",
        "local": "data/standard_node_link/daegu_nodes.csv",
    },
    {
        "id": "standard_link",
        "name": "대구 도로 표준링크 현황",
        "url": "https://www.data.go.kr/data/15049952/fileData.do",
        "local": "data/standard_node_link/daegu_links.csv",
    },
    {
        "id": "link_hourly",
        "name": "대구 교통 링크별 시간별 통계",
        "url": "https://www.data.go.kr/data/15117329/fileData.do",
        "local": "data/open/link_hourly_stats.csv",
    },
    {
        "id": "traffic_api",
        "name": "대구 교통소통정보(신) API",
        "url": "https://www.data.go.kr/data/15126266/openapi.do",
        "local": "env:DAEGU_TRAFFIC_API_KEY",
    },
    {
        "id": "incident_api",
        "name": "대구 돌발 교통정보 조회 서비스(신) API",
        "url": "https://www.data.go.kr/data/15126267/openapi.do",
        "local": "env:DAEGU_TRAFFIC_API_KEY",
    },
    {
        "id": "er_hospitals",
        "name": "대구 응급의료기관 현황",
        "url": "https://www.data.go.kr/data/15132528/fileData.do",
        "local": "data/open/er_hospitals.csv",
    },
    {
        "id": "fire_stations",
        "name": "소방서 좌표(소방청/대구 공개)",
        "url": "https://www.data.go.kr/",
        "local": "data/open/fire_stations.csv",
    },
    {
        "id": "station_fleet",
        "name": "소방서별 구급·소방차 대수",
        "url": "https://www.data.go.kr/data/15066554/fileData.do",
        "local": "data/open/station_fleet.csv",
    },
]


def _read_csv(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"CSV 로드 실패: {path} ({last_err})")


def _pick_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    cols = {str(c).strip(): c for c in df.columns}
    lower = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return cols[a]
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _norm_road(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\s+", "", s)
    return s


def find_open_file(*patterns: str) -> Path | None:
    if not OPEN_DIR.is_dir():
        return None
    for pat in patterns:
        hits = sorted(OPEN_DIR.glob(pat))
        if hits:
            return hits[0]
        hits = sorted(OPEN_DIR.glob("**/" + pat))
        if hits:
            return hits[0]
    return None


def speed_to_congestion(speed_kmh: float, free_flow: float = 50.0) -> float:
    """통행속도 → 혼잡도[0,1]. 자유속도 대비 느릴수록 혼잡."""
    sp = float(np.clip(speed_kmh, 1.0, 120.0))
    ff = max(free_flow, 20.0)
    ratio = sp / ff
    # 1.0 이상 → 한산, 0.3 이하 → 정체
    cong = 1.0 - np.clip((ratio - 0.25) / 0.75, 0.0, 1.0)
    return float(np.clip(cong, 0.02, 0.98))


def load_link_hourly_stats(path: str | Path | None = None) -> pd.DataFrame:
    """
    컬럼(공개 정의): 년월일, 시, 가로명, 링크명, 속도
    """
    p = Path(path) if path else find_open_file(
        "link_hourly_stats.csv",
        "*링크별*시간*.csv",
        "*시간별*통계*.csv",
        "*hourly*.csv",
    )
    if p is None or not p.is_file():
        raise FileNotFoundError(
            "오픈데이터 '링크별 시간별 통계' CSV가 없습니다. "
            f"{OPEN_DIR} 에 link_hourly_stats.csv 를 두세요. "
            "https://www.data.go.kr/data/15117329/fileData.do"
        )
    raw = _read_csv(p)
    date_c = _pick_col(raw, ("년월일", "일자", "date", "DATE"))
    hour_c = _pick_col(raw, ("시", "시간", "hour", "HOUR"))
    road_c = _pick_col(raw, ("가로명", "도로명", "road_name", "ROAD_NAME"))
    link_c = _pick_col(raw, ("링크명", "구간명", "link_name", "LINK_NAME"))
    speed_c = _pick_col(raw, ("속도", "평균속도", "speed", "SPEED"))
    if not hour_c or not speed_c:
        raise ValueError(f"시간/속도 컬럼 없음: {list(raw.columns)}")
    if not road_c and not link_c:
        raise ValueError(f"가로명/링크명 컬럼 없음: {list(raw.columns)}")

    hour = raw[hour_c].astype(str).str.extract(r"(\d{1,2})", expand=False)
    out = pd.DataFrame({
        "hour": pd.to_numeric(hour, errors="coerce"),
        "speed": pd.to_numeric(raw[speed_c], errors="coerce"),
        "road_name": raw[road_c].astype(str).map(_norm_road) if road_c else "",
        "link_name": raw[link_c].astype(str) if link_c else "",
    })
    if date_c:
        out["date"] = pd.to_datetime(raw[date_c], errors="coerce")
        out["weekday"] = out["date"].dt.weekday
    out = out.dropna(subset=["hour", "speed"])
    out["hour"] = out["hour"].astype(int) % 24
    return out.reset_index(drop=True)


def build_hourly_speed_profile(df: pd.DataFrame) -> dict[str, Any]:
    """도로명×시간대 평균속도 + 도시 전체 시간대 프로파일."""
    city = (
        df.groupby("hour")["speed"].mean()
        .reindex(range(24))
        .interpolate(limit_direction="both")
        .fillna(35.0)
    )
    city_list = [round(float(x), 2) for x in city.tolist()]

    by_road: dict[str, list[float]] = {}
    if "road_name" in df.columns and df["road_name"].astype(str).str.len().gt(0).any():
        g = df.groupby(["road_name", "hour"])["speed"].mean().unstack("hour")
        g = g.reindex(columns=range(24))
        for road, row in g.iterrows():
            series = row.astype(float).interpolate(limit_direction="both").fillna(city)
            key = _norm_road(road)
            if not key or key == "nan":
                continue
            by_road[key] = [round(float(x), 2) for x in series.tolist()]

    weekday_city = None
    weekend_city = None
    if "weekday" in df.columns and df["weekday"].notna().any():
        wd = df[df["weekday"] < 5]
        we = df[df["weekday"] >= 5]
        if len(wd):
            weekday_city = [
                round(float(x), 2)
                for x in wd.groupby("hour")["speed"].mean()
                .reindex(range(24)).interpolate(limit_direction="both").fillna(35.0)
            ]
        if len(we):
            weekend_city = [
                round(float(x), 2)
                for x in we.groupby("hour")["speed"].mean()
                .reindex(range(24)).interpolate(limit_direction="both").fillna(35.0)
            ]

    return {
        "source": "data.go.kr/15117329",
        "unit": "km/h",
        "city_hourly": city_list,
        "weekday_hourly": weekday_city or city_list,
        "weekend_hourly": weekend_city or city_list,
        "by_road": by_road,
        "n_rows": int(len(df)),
        "n_roads": int(len(by_road)),
    }


def save_speed_profile(profile: dict, path: Path | None = None) -> Path:
    path = path or PROFILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_speed_profile(path: Path | None = None) -> dict | None:
    path = path or PROFILE_PATH
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    # CSV가 있으면 즉시 프로파일 생성
    try:
        df = load_link_hourly_stats()
    except FileNotFoundError:
        return None
    profile = build_hourly_speed_profile(df)
    save_speed_profile(profile, path)
    return profile


def profile_speed(profile: dict, road_name: str | None, hour: int, day_type: str = "weekday") -> float:
    h = int(hour) % 24
    key = _norm_road(road_name) if road_name else ""
    by_road = profile.get("by_road") or {}
    if key and key in by_road:
        return float(by_road[key][h])
    if day_type in ("weekend", "holiday", "holiday_travel"):
        series = profile.get("weekend_hourly") or profile.get("city_hourly")
    else:
        series = profile.get("weekday_hourly") or profile.get("city_hourly")
    if not series:
        return 35.0
    return float(series[h])


def apply_open_profile_to_graph(G, profile: dict, hour: int = 8, day_type: str = "weekday") -> int:
    """오픈 속도 프로파일로 엣지 congestion/travel_time 갱신. 매칭 건수 반환."""
    hits = 0
    for u, v, ed in G.edges(data=True):
        road = ed.get("road_name")
        sp = profile_speed(profile, road, hour, day_type=day_type)
        if road and _norm_road(road) in (profile.get("by_road") or {}):
            hits += 1
        cong = speed_to_congestion(sp)
        ed["congestion"] = cong
        ed["speed_kmh_open"] = sp
        length_km = float(ed.get("length_m", 400.0)) / 1000.0
        ed["travel_time"] = (length_km / max(sp, 5.0)) * 60.0
    return hits


def load_fire_stations_open(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else find_open_file(
        "fire_stations.csv", "*소방*.csv", "*fire*.csv",
    )
    if p is None or not p.is_file():
        raise FileNotFoundError(f"소방서 오픈 CSV 없음: {OPEN_DIR}/fire_stations.csv")
    raw = _read_csv(p)
    name_c = _pick_col(raw, ("소방서명", "관서명", "station_name", "이름", "시설명"))
    lat_c = _pick_col(raw, ("위도", "Y좌표", "y", "lat", "LAT"))
    lng_c = _pick_col(raw, ("경도", "X좌표", "x", "lng", "LON", "lon"))
    if not name_c or not lat_c or not lng_c:
        raise ValueError(f"소방서 CSV 필수컬럼 없음: {list(raw.columns)}")
    rows = []
    for i, r in raw.iterrows():
        name = str(r[name_c]).strip()
        # 대구만 (전국 파일이면 필터)
        if "대구" not in name and "달성" not in name:
            # 주소 컬럼이 있으면 대구 필터
            addr_c = _pick_col(raw, ("주소", "소재지", "address"))
            if addr_c and "대구" not in str(r.get(addr_c, "")):
                continue
            if addr_c is None and "대구" not in name:
                # 시드 파일이면 이름에 대구가 있음
                if not name:
                    continue
        lat, lng = float(r[lat_c]), float(r[lng_c])
        # 일부 파일은 X=경도 Y=위도, 일부는 반대 — 대구 bbox로 판별
        if not (35.6 <= lat <= 36.2 and 128.2 <= lng <= 129.0):
            if 35.6 <= lng <= 36.2 and 128.2 <= lat <= 129.0:
                lat, lng = lng, lat
            else:
                continue
        rows.append({
            "station_id": len(rows),
            "station_name": name,
            "lat": lat,
            "lng": lng,
            "source": "open_data",
        })
    if not rows:
        raise ValueError("대구 소방서 행을 찾지 못함")
    fleet = load_station_fleet()
    for row in rows:
        f = fleet.get(_norm_road(row["station_name"])) or fleet.get(row["station_name"])
        if f:
            row["ambulances"] = int(f["ambulances"])
            row["fire_trucks"] = int(f["fire_trucks"])
        else:
            row.setdefault("ambulances", 4)
            row.setdefault("fire_trucks", 4)
    return rows


def load_station_fleet(path: str | Path | None = None) -> dict[str, dict]:
    """소방서명 → {ambulances, fire_trucks}. 키는 원문 + 정규화명."""
    p = Path(path) if path else find_open_file(
        "station_fleet.csv", "*fleet*.csv", "*구급*현황*.csv",
    )
    out: dict[str, dict] = {}
    if p is None or not p.is_file():
        return out
    raw = _read_csv(p)
    name_c = _pick_col(raw, ("소방서명", "관서명", "station_name", "센터명", "시설명"))
    amb_c = _pick_col(raw, ("ambulances", "구급차", "구급차수", "구급", "수량"))
    fire_c = _pick_col(raw, ("fire_trucks", "소방차", "펌프차", "소방차수"))
    if not name_c:
        return out
    for _, r in raw.iterrows():
        name = str(r[name_c]).strip()
        if not name or name.lower() == "nan":
            continue
        try:
            amb = int(float(r[amb_c])) if amb_c else 4
        except (TypeError, ValueError):
            amb = 4
        try:
            fire = int(float(r[fire_c])) if fire_c else 4
        except (TypeError, ValueError):
            fire = 4
        rec = {"ambulances": max(0, amb), "fire_trucks": max(0, fire)}
        out[name] = rec
        out[_norm_road(name)] = rec
    return out


def load_er_hospitals_open(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else find_open_file(
        "er_hospitals.csv", "*응급의료*.csv", "*hospital*.csv",
    )
    if p is None or not p.is_file():
        raise FileNotFoundError(f"응급의료 오픈 CSV 없음: {OPEN_DIR}/er_hospitals.csv")
    raw = _read_csv(p)
    name_c = _pick_col(raw, ("응급의료기관명", "기관명", "병원명", "hospital_name", "시설명"))
    lat_c = _pick_col(raw, ("위도", "Y좌표", "lat", "LAT"))
    lng_c = _pick_col(raw, ("경도", "X좌표", "lng", "LON", "lon"))
    type_c = _pick_col(raw, ("구분", "유형", "specialty", "센터구분"))
    addr_c = _pick_col(raw, ("소재지", "주소", "address"))
    if not name_c:
        raise ValueError(f"병원명 컬럼 없음: {list(raw.columns)}")
    if not lat_c or not lng_c:
        raise ValueError(
            "응급의료 현황 CSV에 좌표가 없습니다. "
            "공개 주소 기준 위도/경도 컬럼(위도,경도)을 추가한 er_hospitals.csv 를 사용하세요."
        )
    rows = []
    for _, r in raw.iterrows():
        name = str(r[name_c]).strip()
        if not name or name == "nan":
            continue
        lat, lng = float(r[lat_c]), float(r[lng_c])
        if not (35.6 <= lat <= 36.2 and 128.2 <= lng <= 129.0):
            if 35.6 <= lng <= 36.2 and 128.2 <= lat <= 129.0:
                lat, lng = lng, lat
            else:
                continue
        spec = str(r[type_c]).strip() if type_c else ""
        rows.append({
            "hospital_id": len(rows),
            "hospital_name": name,
            "lat": lat,
            "lng": lng,
            "specialty": spec,
            "address": str(r[addr_c]).strip() if addr_c else "",
            "source": "open_data",
        })
    if not rows:
        raise ValueError("응급의료기관 행을 찾지 못함")
    return rows


def _load_dotenv() -> None:
    """프로젝트 루트 .env 를 환경변수로 로드 (이미 있는 키는 덮어쓰지 않음)."""
    env_path = _HERE / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        pass


_load_dotenv()


# 포털 상세기능: API 목록 → linkspeed
# https://www.data.go.kr/data/15126266/openapi.do#/API%20목록/linkspeed
DEFAULT_TRAFFIC_API_URL = (
    "https://apis.data.go.kr/6270000/service/rest1/linkspeed"
)

_TRAFFIC_URL_CANDIDATES = (
    "https://apis.data.go.kr/6270000/service/rest1/linkspeed",
)


def fetch_realtime_traffic(api_key: str | None = None, timeout: float = 20.0) -> list[dict]:
    """
    대구 교통소통정보(신) API (JSON).

    필수: DAEGU_TRAFFIC_API_KEY
    권장: DAEGU_TRAFFIC_API_URL = 포털 상세기능의 **요청주소 전체**
          (예: https://apis.data.go.kr/6270000/service/rest1/<연산명>)

    End Point 만 `.../rest1` 로 오면 연산명 후보를 순회한다.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    _load_dotenv()
    key = api_key or os.environ.get("DAEGU_TRAFFIC_API_KEY") or os.environ.get("DATA_GO_KR_API_KEY")
    if not key:
        raise RuntimeError("DAEGU_TRAFFIC_API_KEY 가 필요합니다 (data.go.kr/15126266 활용신청)")

    configured = os.environ.get("DAEGU_TRAFFIC_API_URL", DEFAULT_TRAFFIC_API_URL).rstrip("/")
    urls: list[str] = []
    if configured.endswith("/rest1") or configured.endswith("/rest"):
        urls.append(f"{configured}/linkspeed")
    else:
        urls.append(configured)
    for u in _TRAFFIC_URL_CANDIDATES:
        if u not in urls:
            urls.append(u)

    params = {
        "serviceKey": key,
        "pageNo": "1",
        "numOfRows": "5000",
        "type": "json",
        "resultType": "json",
        "getType": "json",
        "dataType": "JSON",
    }
    qs = urllib.parse.urlencode(params, safe="%")

    last_err: Exception | None = None
    raw = ""
    used_url = urls[0]
    for base in urls:
        url = f"{base}?{qs}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "daegu-ems-open-data/1.0", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            used_url = base
            # NO_OPENAPI 가 JSON 본문으로 200 오는 경우도 있음
            if "NO_OPENAPI_SERVICE_ERROR" in raw:
                last_err = RuntimeError(f"NO_OPENAPI @ {base}")
                continue
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if "NO_OPENAPI_SERVICE_ERROR" in body:
                last_err = RuntimeError(f"NO_OPENAPI @ {base}")
                continue
            last_err = RuntimeError(f"교통소통 API HTTP {e.code}: {body[:500]}")
            continue
    else:
        raise RuntimeError(
            "교통소통 API 연산명을 찾지 못했습니다. "
            "공공데이터포털 해당 API의 '상세기능' 탭에서 **요청주소** 전체를 복사해 "
            ".env 의 DAEGU_TRAFFIC_API_URL 에 넣어 주세요. "
            f"(마지막 오류: {last_err})"
        )

    raw_stripped = raw.lstrip()
    if raw_stripped.startswith("<"):
        raise RuntimeError(
            "교통소통 API가 XML을 반환했습니다. getType=json 파라미터/URL을 확인하세요. "
            f"url={used_url} head={raw_stripped[:200]!r}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"교통소통 API JSON 파싱 실패 ({used_url}): {raw[:300]!r}") from e

    err = _find_api_error(payload)
    if err:
        raise RuntimeError(f"교통소통 API 오류 ({used_url}): {err}")

    items = _extract_items(payload)
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        low = {str(k).lower().replace("_", ""): v for k, v in it.items()}
        # 대구 linkspeed 응답: LINK_SPEED, STD_LINK_ID, ROAD_NM, DIST, ...
        link = (
            low.get("stdlinkid") or low.get("linkid") or low.get("링크id")
            or low.get("표준링크id") or low.get("linkno") or low.get("링크번호")
            or low.get("dsrclinksn")
        )
        speed = (
            low.get("linkspeed") or low.get("speed") or low.get("속도")
            or low.get("avgspeed") or low.get("평균속도") or low.get("spd")
        )
        snode = low.get("startnodeid") or low.get("시작노드id") or low.get("fnode")
        enode = low.get("endnodeid") or low.get("종료노드id") or low.get("tnode")
        road = (
            low.get("roadnm") or low.get("roadname") or low.get("가로명") or low.get("도로명")
            or low.get("sectionnm") or low.get("sectionname") or low.get("구간명")
            or low.get("linkname") or low.get("링크명") or ""
        )
        dist = low.get("dist") or low.get("distance") or low.get("거리") or low.get("length")
        travel = low.get("linktime") or low.get("traveltime") or low.get("통행시간")
        start_nm = low.get("startfacnm") or low.get("시작지점명") or ""
        end_nm = low.get("endfacnm") or low.get("도착지점명") or ""
        if speed is None or speed == "":
            continue
        try:
            sp = float(str(speed).replace(",", ""))
        except ValueError:
            continue
        out.append({
            "link_id": _safe_int(link),
            "speed": sp,
            "start_node_id": _safe_int(snode),
            "end_node_id": _safe_int(enode),
            "road_name": _norm_road(str(road)) if road else "",
            "section_name": str(low.get("sectionnm") or ""),
            "start_name": str(start_nm),
            "end_name": str(end_nm),
            "distance_m": float(dist) if dist not in (None, "") else None,
            "travel_time_sec": float(travel) if travel not in (None, "") else None,
            "raw": it,
            "api_url": used_url,
        })
    if not out and items:
        sample = items[0] if isinstance(items[0], dict) else {"_": items[0]}
        raise RuntimeError(
            f"교통소통 API 응답에서 속도를 못 찾음 ({used_url}). "
            f"샘플 키={list(sample.keys())[:20]}"
        )
    if not out:
        raise RuntimeError(
            f"교통소통 API 응답이 비어 있습니다 ({used_url}). "
            "상세기능 요청주소/파라미터를 확인하세요."
        )
    return out


DEFAULT_INCIDENT_API_URL = (
    "https://apis.data.go.kr/6270000/service/rest/dgincident"
)

# 돌발 유형 코드 → 혼잡 가산 (대략: 사고/통제 > 공사 > 기타)
_INCIDENT_CODE_PENALTY = {
    "1": 0.55,  # 사고 계열(추정)
    "2": 0.45,  # 공사
    "3": 0.35,  # 행사 등
    "4": 0.40,
    "5": 0.50,  # 통제/제한
}


def fetch_incident_events(api_key: str | None = None, timeout: float = 20.0) -> list[dict]:
    """
    대구 돌발 교통정보(신) — 공사·사고·통제 등.

    https://www.data.go.kr/data/15126267/openapi.do
    End Point: .../service/rest  +  /dgincident
    환경변수: DAEGU_INCIDENT_API_URL (기본 위 URL)
              DAEGU_TRAFFIC_API_KEY 또는 DATA_GO_KR_API_KEY
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    _load_dotenv()
    key = api_key or os.environ.get("DAEGU_TRAFFIC_API_KEY") or os.environ.get("DATA_GO_KR_API_KEY")
    if not key:
        raise RuntimeError("DAEGU_TRAFFIC_API_KEY 가 필요합니다 (돌발 API도 동일 인증키)")

    base = os.environ.get("DAEGU_INCIDENT_API_URL", DEFAULT_INCIDENT_API_URL).rstrip("/")
    params = {
        "serviceKey": key,
        "pageNo": "1",
        "numOfRows": "5000",
        "type": "json",
        "resultType": "json",
    }
    url = f"{base}?{urllib.parse.urlencode(params, safe='%')}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "daegu-ems-open-data/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"돌발 API HTTP {e.code}: {body[:400]}") from e

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"돌발 API JSON 파싱 실패: {raw[:300]!r}") from e

    err = _find_api_error(payload)
    if err:
        raise RuntimeError(f"돌발 API 오류: {err}")

    items = _extract_items(payload)
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        low = {str(k).lower().replace("_", ""): v for k, v in it.items()}
        # 응답: COORDX=경도, COORDY=위도, LINKID, INCIDENTCODE, INCIDENTTITLE, ...
        lng = low.get("coordx") or low.get("lng") or low.get("lon") or low.get("x")
        lat = low.get("coordy") or low.get("lat") or low.get("y")
        link = low.get("linkid") or low.get("stdlinkid") or low.get("링크id")
        code = str(low.get("incidentcode") or low.get("type") or "")
        grade = str(low.get("troublegrade") or low.get("trafficgrade") or "")
        title = str(low.get("incidenttitle") or low.get("title") or low.get("내용") or "")
        loc = str(low.get("location") or low.get("주소") or "")
        iid = str(low.get("incidentid") or low.get("id") or "")
        start = str(low.get("startdate") or "")
        end = str(low.get("enddate") or "")
        if lat in (None, "") or lng in (None, ""):
            continue
        try:
            lat_f, lng_f = float(lat), float(lng)
        except (TypeError, ValueError):
            continue
        penalty = float(_INCIDENT_CODE_PENALTY.get(code, 0.40))
        try:
            g = int(str(grade).lstrip("0") or "0")
            if g >= 3:
                penalty = min(0.85, penalty + 0.15)
            elif g >= 2:
                penalty = min(0.75, penalty + 0.08)
        except ValueError:
            pass
        out.append({
            "incident_id": iid,
            "lat": lat_f,
            "lng": lng_f,
            "link_id": _safe_int(link),
            "code": code,
            "grade": grade,
            "title": title,
            "location": loc,
            "start": start,
            "end": end,
            "penalty": penalty,
            "raw": it,
        })
    return out


def apply_incidents_to_graph(G, events: list[dict], radius_m: float = 180.0) -> int:
    """
    돌발 지점 근처 엣지(및 LINKID 일치)에 혼잡 페널티를 올려 우회하게 함.
    반환: 페널티가 적용된 엣지 수.
    """
    from standard_node_link import nearest_node
    from road_shapes import haversine_m

    if not events:
        return 0

    by_link: dict[int, float] = {}
    for ev in events:
        lid = ev.get("link_id")
        if lid is not None:
            by_link[int(lid)] = max(by_link.get(int(lid), 0.0), float(ev["penalty"]))

    # 좌표 → 최근접 노드 주변
    node_pen: dict[int, float] = {}
    for ev in events:
        try:
            n = nearest_node(G, float(ev["lat"]), float(ev["lng"]))
        except Exception:
            continue
        node_pen[n] = max(node_pen.get(n, 0.0), float(ev["penalty"]))
        # 이웃 1홉
        for nb in list(G.successors(n)) + list(G.predecessors(n)):
            node_pen[nb] = max(node_pen.get(nb, 0.0), float(ev["penalty"]) * 0.7)

    hits = 0
    for u, v, ed in G.edges(data=True):
        pen = 0.0
        lid = ed.get("link_id")
        if lid is not None and int(lid) in by_link:
            pen = max(pen, by_link[int(lid)])
        if u in node_pen:
            pen = max(pen, node_pen[u])
        if v in node_pen:
            pen = max(pen, node_pen[v])
        # 거리 기반 보강: 돌발 좌표와 엣지 중점
        if pen <= 0 and events:
            try:
                mlat = (float(G.nodes[u]["lat"]) + float(G.nodes[v]["lat"])) / 2
                mlng = (float(G.nodes[u]["lng"]) + float(G.nodes[v]["lng"])) / 2
            except Exception:
                continue
            for ev in events:
                d = haversine_m(mlat, mlng, float(ev["lat"]), float(ev["lng"]))
                if d <= radius_m:
                    pen = max(pen, float(ev["penalty"]) * (1.0 - d / radius_m))
        if pen <= 0:
            continue
        old = float(ed.get("congestion", 0.35))
        ed["congestion"] = float(min(0.95, max(old, pen)))
        ed["incident"] = True
        ed["incident_penalty"] = pen
        # travel_time 재계산
        length_km = float(ed.get("length_m", 400.0)) / 1000.0
        sp = float(ed.get("speed_kmh_open") or max(8.0, 45.0 * (1.0 - 0.6 * ed["congestion"])))
        ed["travel_time"] = (length_km / max(sp, 5.0)) * 60.0
        hits += 1
    return hits


def _safe_int(v) -> int | None:
    if v in (None, ""):
        return None
    try:
        return int(float(str(v)))
    except ValueError:
        return None


def _find_api_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    # OpenAPI GW
    for path in (
        ("response", "header"),
        ("Response", "header"),
        ("header",),
        ("result",),
    ):
        cur: Any = payload
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if not ok or not isinstance(cur, dict):
            continue
        code = str(cur.get("resultCode") or cur.get("resultcode") or cur.get("RETURN_CODE") or "")
        msg = str(cur.get("resultMsg") or cur.get("resultmsg") or cur.get("RETURN_MSG") or "")
        if code and code not in ("00", "0", "000", "NORMAL_SERVICE", "SUCCESS"):
            return f"{code} {msg}".strip()
        if msg.upper() not in ("", "SUCCESS", "NORMAL_SERVICE", "NORMAL SERVICE") and code in ("",):
            if "ERROR" in msg.upper() or "거부" in msg or "제한" in msg:
                return msg
    return None


def _extract_items(payload: Any) -> list:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    # 재귀적으로 item/items/data 리스트 탐색
    def walk(obj: Any, depth: int = 0) -> list | None:
        if depth > 6:
            return None
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                return obj
            return None
        if not isinstance(obj, dict):
            return None
        for key in ("item", "items", "data", "list", "row", "rows", "trafficInfo", "trafficinfo"):
            if key in obj:
                found = walk(obj[key], depth + 1)
                if found:
                    return found
                v = obj[key]
                if isinstance(v, dict):
                    # single item wrapped
                    if any(k.lower() in ("speed", "속도", "linkid") or "속도" in str(k) for k in v.keys()):
                        return [v]
        for v in obj.values():
            found = walk(v, depth + 1)
            if found:
                return found
        return None

    found = walk(payload)
    return found or []


def apply_realtime_speeds(G, items: list[dict]) -> int:
    """실시간 속도를 link_id / (start,end) / 도로명 으로 매칭."""
    by_link = {it["link_id"]: it for it in items if it.get("link_id") is not None}
    by_pair = {
        (it["start_node_id"], it["end_node_id"]): it
        for it in items
        if it.get("start_node_id") is not None and it.get("end_node_id") is not None
    }
    by_road: dict[str, list[float]] = {}
    for it in items:
        rn = _norm_road(str(it.get("road_name") or ""))
        if rn:
            by_road.setdefault(rn, []).append(float(it["speed"]))
    road_avg = {k: float(np.mean(v)) for k, v in by_road.items()}

    hits = 0
    for u, v, ed in G.edges(data=True):
        it = None
        lid = ed.get("link_id")
        if lid in by_link:
            it = by_link[lid]
        elif (u, v) in by_pair:
            it = by_pair[(u, v)]
        if it:
            sp = float(it["speed"])
        else:
            rn = _norm_road(str(ed.get("road_name") or ""))
            if rn and rn in road_avg:
                sp = road_avg[rn]
            else:
                continue
        ed["congestion"] = speed_to_congestion(sp)
        ed["speed_kmh_open"] = sp
        length_km = float(ed.get("length_m", 400.0)) / 1000.0
        ed["travel_time"] = (length_km / max(sp, 5.0)) * 60.0
        ed["realtime"] = True
        hits += 1
    return hits


def open_data_status() -> dict:
    """로컬에 어떤 오픈데이터가 준비됐는지."""
    st = {}
    for row in OPEN_CATALOG:
        local = row["local"]
        if local.startswith("env:"):
            ok = bool(os.environ.get(local.split(":", 1)[1]) or os.environ.get("DATA_GO_KR_API_KEY"))
            st[row["id"]] = {"ready": ok, "path": local, "url": row["url"], "name": row["name"]}
            continue
        p = _HERE / local
        st[row["id"]] = {
            "ready": p.is_file() and p.stat().st_size > 0,
            "path": local,
            "url": row["url"],
            "name": row["name"],
            "bytes": p.stat().st_size if p.is_file() else 0,
        }
    st["speed_profile"] = {
        "ready": PROFILE_PATH.is_file(),
        "path": str(PROFILE_PATH.relative_to(_HERE)),
    }
    return st


def print_open_data_status() -> None:
    st = open_data_status()
    print("[open_data] 대구 오픈데이터 준비 상태")
    for k, v in st.items():
        if k == "speed_profile":
            mark = "OK" if v["ready"] else "--"
            print(f"  [{mark}] speed_profile → {v['path']}")
            continue
        mark = "OK" if v["ready"] else "NEED"
        print(f"  [{mark}] {v['name']}")
        print(f"         local: {v['path']}")
        if not v["ready"]:
            print(f"         받기: {v['url']}")
