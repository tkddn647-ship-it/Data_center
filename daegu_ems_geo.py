"""
daegu_ems_geo.py
================
대구시 실제 응급의료기관·소방서 좌표 (WGS84, 공개 지도 기준).
"""

from __future__ import annotations

DAEGU_FIRE_STATIONS = [
    {"station_id": 0, "station_name": "대구중부소방서", "lat": 35.86980, "lng": 128.60600},
    {"station_id": 1, "station_name": "대구동부소방서", "lat": 35.87950, "lng": 128.65180},
    {"station_id": 2, "station_name": "대구서부소방서", "lat": 35.87190, "lng": 128.55020},
    {"station_id": 3, "station_name": "대구북부소방서", "lat": 35.91020, "lng": 128.59550},
    {"station_id": 4, "station_name": "대구수성소방서", "lat": 35.84480, "lng": 128.63150},
    {"station_id": 5, "station_name": "대구달서소방서", "lat": 35.82960, "lng": 128.53280},
    {"station_id": 6, "station_name": "대구달성소방서", "lat": 35.77480, "lng": 128.43150},
    {"station_id": 7, "station_name": "대구남부소방서", "lat": 35.84620, "lng": 128.59010},
]

# 주소 기반 공개 좌표
DAEGU_ER_HOSPITALS = [
    # 중구 동덕로 130
    {"hospital_id": 0, "hospital_name": "경북대학교병원", "lat": 35.86670, "lng": 128.60440, "specialty": "권역응급"},
    # 남구 현충로 170
    {"hospital_id": 1, "hospital_name": "영남대학교병원", "lat": 35.84860, "lng": 128.58860, "specialty": "권역응급"},
    # 달서구 달구벌대로 1035 (성서 캠퍼스)
    {"hospital_id": 2, "hospital_name": "계명대학교동산병원", "lat": 35.85360, "lng": 128.48030, "specialty": "권역응급"},
    # 남구 두류
    {"hospital_id": 3, "hospital_name": "대구가톨릭대학교병원", "lat": 35.84470, "lng": 128.62790, "specialty": "지역응급"},
    # 북구 호국로 807
    {"hospital_id": 4, "hospital_name": "칠곡경북대학교병원", "lat": 35.94280, "lng": 128.56440, "specialty": "권역외상"},
    # 동구
    {"hospital_id": 5, "hospital_name": "대구파티마병원", "lat": 35.88490, "lng": 128.62340, "specialty": "지역응급"},
    # 서구
    {"hospital_id": 6, "hospital_name": "대구의료원", "lat": 35.85720, "lng": 128.56580, "specialty": "지역응급"},
    # 수성구
    {"hospital_id": 7, "hospital_name": "대구가톨릭대병원(수성)", "lat": 35.85800, "lng": 128.63000, "specialty": "지역응급"},
]
