"""
daegu_ems_geo.py
================
대구 소방서·응급의료기관 좌표.

우선순위:
  1) data/open/fire_stations.csv , er_hospitals.csv  (오픈데이터)
  2) 아래 내장 목록 (동일 공개좌표 — 파일 없을 때만)
"""

from __future__ import annotations

_FALLBACK_FIRE = [
    {"station_id": 0, "station_name": "대구중부소방서", "lat": 35.862455, "lng": 128.575284, "ambulances": 5, "fire_trucks": 6},
    {"station_id": 1, "station_name": "대구동부소방서", "lat": 35.881700, "lng": 128.628800, "ambulances": 5, "fire_trucks": 5},
    {"station_id": 2, "station_name": "대구서부소방서", "lat": 35.869800, "lng": 128.555500, "ambulances": 5, "fire_trucks": 5},
    {"station_id": 3, "station_name": "대구북부소방서", "lat": 35.878005, "lng": 128.592298, "ambulances": 6, "fire_trucks": 6},
    {"station_id": 4, "station_name": "대구수성소방서", "lat": 35.825275, "lng": 128.644549, "ambulances": 5, "fire_trucks": 5},
    {"station_id": 5, "station_name": "대구달서소방서", "lat": 35.831053, "lng": 128.534706, "ambulances": 7, "fire_trucks": 6},
    {"station_id": 6, "station_name": "대구달성소방서", "lat": 35.698750, "lng": 128.442024, "ambulances": 4, "fire_trucks": 5},
    {"station_id": 7, "station_name": "대구남부소방서", "lat": 35.834193, "lng": 128.604390, "ambulances": 5, "fire_trucks": 5},
]

_FALLBACK_ER = [
    {"hospital_id": 0, "hospital_name": "경북대학교병원", "lat": 35.866469, "lng": 128.604834, "specialty": "권역응급"},
    {"hospital_id": 1, "hospital_name": "영남대학교병원", "lat": 35.847322, "lng": 128.585045, "specialty": "권역응급"},
    {"hospital_id": 2, "hospital_name": "계명대학교동산병원", "lat": 35.854052, "lng": 128.480213, "specialty": "권역응급"},
    {"hospital_id": 3, "hospital_name": "대구가톨릭대학교병원", "lat": 35.844746, "lng": 128.567923, "specialty": "지역응급"},
    {"hospital_id": 4, "hospital_name": "칠곡경북대학교병원", "lat": 35.956565, "lng": 128.563887, "specialty": "권역외상"},
    {"hospital_id": 5, "hospital_name": "대구파티마병원", "lat": 35.884310, "lng": 128.624395, "specialty": "지역응급"},
    {"hospital_id": 6, "hospital_name": "대구의료원", "lat": 35.859597, "lng": 128.540303, "specialty": "지역응급"},
]


def _load_open():
    try:
        from daegu_open_data import load_er_hospitals_open, load_fire_stations_open
        fire = load_fire_stations_open()
        er = load_er_hospitals_open()
        for s in fire:
            s.setdefault("ambulances", 4)
            s.setdefault("fire_trucks", 4)
        return fire, er, "open_data"
    except Exception as e:  # noqa: BLE001
        print(f"[daegu_ems_geo] 오픈 CSV 로드 실패 → 내장 공개좌표 사용 ({e})")
        return _FALLBACK_FIRE, _FALLBACK_ER, "builtin_public_coords"


DAEGU_FIRE_STATIONS, DAEGU_ER_HOSPITALS, EMS_GEO_SOURCE = _load_open()
