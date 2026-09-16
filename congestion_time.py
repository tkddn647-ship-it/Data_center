"""
congestion_time.py
==================
표현/데모용 시간·캘린더 혼잡 모델.

목표(학습 전 단계)
------------------
현재 날짜·시각을 기준으로
  1) 앞으로 몇 시간 동안 교통이 얼마나 빡셀지 **예측**하고
  2) 그 예측으로 **최적 경로**를 짜며
  3) 지도에 교통량을 **파랑(한산)→빨강(혼잡)** 으로 한눈에 보이게 한다.

요일/주말/추석·설 등 연휴를 반영한다.
나중에 안심구역 ITS + RL/시계열 학습 모델이 이 인터페이스를 대체하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import networkx as nx


# 단계 범례(보조) + 연속 파랑→빨강 스케일(메인 시각화)
CONGESTION_LEVELS = [
    {"level": 0, "max": 0.25, "name": "원활", "color": "#2166ac"},
    {"level": 1, "max": 0.50, "name": "서행", "color": "#67a9cf"},
    {"level": 2, "max": 0.75, "name": "혼잡", "color": "#ef8a62"},
    {"level": 3, "max": 1.01, "name": "정체", "color": "#b2182b"},
]

EMS_SPEED_FACTOR = 1.20

# 한국 공휴일/연휴 (데모용 고정표 — 필요 시 확장)
# 추석·설은 연도별 음력이라 주요 연도만 명시
_KR_HOLIDAYS: dict[date, str] = {}


def _add_holiday(y: int, m: int, d: int, name: str) -> None:
    _KR_HOLIDAYS[date(y, m, d)] = name


for _y in range(2024, 2031):
    _add_holiday(_y, 1, 1, "신정")
    _add_holiday(_y, 3, 1, "삼일절")
    _add_holiday(_y, 5, 5, "어린이날")
    _add_holiday(_y, 6, 6, "현충일")
    _add_holiday(_y, 8, 15, "광복절")
    _add_holiday(_y, 10, 3, "개천절")
    _add_holiday(_y, 10, 9, "한글날")
    _add_holiday(_y, 12, 25, "성탄절")

# 설·추석 연휴 (양력 근사 표 — 데모용)
for d0, name in [
    (date(2025, 1, 28), "설연휴"), (date(2025, 1, 29), "설날"), (date(2025, 1, 30), "설연휴"),
    (date(2025, 10, 5), "추석연휴"), (date(2025, 10, 6), "추석"), (date(2025, 10, 7), "추석연휴"), (date(2025, 10, 8), "추석연휴"),
    (date(2026, 2, 16), "설연휴"), (date(2026, 2, 17), "설날"), (date(2026, 2, 18), "설연휴"),
    (date(2026, 9, 24), "추석연휴"), (date(2026, 9, 25), "추석"), (date(2026, 9, 26), "추석연휴"), (date(2026, 9, 27), "추석연휴"),
    (date(2027, 2, 6), "설연휴"), (date(2027, 2, 7), "설날"), (date(2027, 2, 8), "설연휴"),
    (date(2027, 10, 14), "추석연휴"), (date(2027, 10, 15), "추석"), (date(2027, 10, 16), "추석연휴"),
]:
    _KR_HOLIDAYS[d0] = name


def congestion_level(c: float) -> dict:
    c = float(np.clip(c, 0.0, 1.0))
    for row in CONGESTION_LEVELS:
        if c < row["max"]:
            return row
    return CONGESTION_LEVELS[-1]


def traffic_color(c: float) -> str:
    """파랑(한산) → 하늘 → 주황 → 빨강(정체) 연속 색."""
    c = float(np.clip(c, 0.0, 1.0))
    # stops: 0 #2166ac, 0.35 #67a9cf, 0.55 #fddbc7, 0.75 #ef8a62, 1 #b2182b
    stops = [
        (0.0, (33, 102, 172)),
        (0.35, (103, 169, 207)),
        (0.55, (253, 219, 199)),
        (0.75, (239, 138, 98)),
        (1.0, (178, 24, 43)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if c <= t1 or i == len(stops) - 2:
            u = 0.0 if t1 == t0 else (c - t0) / (t1 - t0)
            u = float(np.clip(u, 0.0, 1.0))
            rgb = tuple(int(c0[j] + u * (c1[j] - c0[j])) for j in range(3))
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return "#b2182b"


def _diurnal_factor(t_min: float, day_type: str = "weekday") -> float:
    """하루 중 혼잡 배율. day_type에 따라 피크 모양이 달라짐."""
    h = (t_min % (24 * 60)) / 60.0
    if day_type == "holiday_travel":
        # 추석/설 귀성·귀경: 오전~오후 길게 막힘, 심야도 어느 정도
        hump = np.exp(-0.5 * ((h - 11.0) / 3.2) ** 2) + 0.85 * np.exp(-0.5 * ((h - 17.5) / 2.8) ** 2)
        return float(np.clip(0.45 + 1.55 * hump, 0.35, 2.4))
    if day_type == "weekend":
        # 주말: 출근 피크 약함, 오후 나들이 피크
        afternoon = np.exp(-0.5 * ((h - 14.5) / 2.2) ** 2)
        evening = 0.55 * np.exp(-0.5 * ((h - 19.0) / 1.4) ** 2)
        return float(np.clip(0.28 + 1.1 * afternoon + evening, 0.2, 1.7))
    # weekday
    morning = np.exp(-0.5 * ((h - 8.0) / 0.75) ** 2)
    evening = np.exp(-0.5 * ((h - 18.0) / 0.9) ** 2)
    lunch = 0.45 * np.exp(-0.5 * ((h - 12.5) / 0.6) ** 2)
    base = 0.22 + 0.08 * np.sin((h / 24.0) * 2 * np.pi + 1.0)
    return float(np.clip(base + 1.35 * morning + 1.45 * evening + lunch, 0.15, 2.2))


@dataclass
class CalendarContext:
    """현재 날짜의 교통 패턴 맥락."""
    day: date
    weekday: int            # 0=월 … 6=일
    day_type: str           # weekday | weekend | holiday_travel | holiday
    label: str              # UI 표시
    volume_factor: float    # 전체 교통량 배율

    @classmethod
    def from_date(cls, day: date) -> "CalendarContext":
        wd = day.weekday()
        holiday = _KR_HOLIDAYS.get(day)
        # 연휴 전날/다음날도 귀성·귀경 패턴에 가깝게
        near = None
        for delta in (-1, 1, -2, 2):
            name = _KR_HOLIDAYS.get(day + timedelta(days=delta))
            if name and ("추석" in name or "설" in name):
                near = name
                break

        if holiday and ("추석" in holiday or "설" in holiday):
            return cls(day, wd, "holiday_travel", f"{holiday}", 1.55)
        if near:
            return cls(day, wd, "holiday_travel", f"연휴이동·{near}권", 1.40)
        if holiday:
            return cls(day, wd, "holiday", holiday, 0.75)
        if wd >= 5:
            return cls(day, wd, "weekend", "주말", 0.90)
        return cls(day, wd, "weekday", "평일", 1.00)

    def to_dict(self) -> dict:
        return {
            "date": self.day.isoformat(),
            "weekday": self.weekday,
            "weekday_name": ["월", "화", "수", "목", "금", "토", "일"][self.weekday],
            "day_type": self.day_type,
            "label": self.label,
            "volume_factor": self.volume_factor,
        }


@dataclass
class TimeCongestionField:
    """엣지별 base 혼잡 + 시간·캘린더 변조 + 단기 예측."""

    base: dict
    phase: dict
    length_m: dict
    edge_list: list
    latlng: dict = field(default_factory=dict)
    source: str = "synthetic_calendar_forecast"
    context: CalendarContext | None = None

    @classmethod
    def from_graph(cls, G: nx.DiGraph, sample_edges: int = 2200, seed: int = 0) -> "TimeCongestionField":
        rng = np.random.default_rng(seed)
        base, phase, length_m, latlng = {}, {}, {}, {}
        for u, v, d in G.edges(data=True):
            b = float(d.get("congestion", 0.3))
            lat = 0.5 * (G.nodes[u]["lat"] + G.nodes[v]["lat"])
            lng = 0.5 * (G.nodes[u]["lng"] + G.nodes[v]["lng"])
            center_boost = 0.18 * np.exp(-((lat - 35.87) ** 2 + (lng - 128.60) ** 2) / (2 * 0.028 ** 2))
            length = float(d.get("length_m", 400.0))
            short_boost = 0.08 if length < 250 else 0.0
            base[(u, v)] = float(np.clip(b + center_boost + short_boost, 0.08, 0.92))
            phase[(u, v)] = float(rng.uniform(0, 2 * np.pi))
            length_m[(u, v)] = length
            latlng[(u, v)] = (lat, lng)

        edges = list(base.keys())
        if len(edges) > sample_edges:
            pick = rng.choice(len(edges), size=sample_edges, replace=False)
            edge_list = [edges[i] for i in pick]
        else:
            edge_list = edges
        ctx = CalendarContext.from_date(date.today())
        return cls(base=base, phase=phase, length_m=length_m, edge_list=edge_list,
                   latlng=latlng, context=ctx)

    def set_date(self, day: date) -> CalendarContext:
        self.context = CalendarContext.from_date(day)
        return self.context

    def _day_type(self) -> str:
        return self.context.day_type if self.context else "weekday"

    def _volume_factor(self) -> float:
        return self.context.volume_factor if self.context else 1.0

    def congestion(self, u, v, t_min: float, lookahead_min: float = 0.0) -> float:
        """시각 t (+lookahead) 의 예측 혼잡도."""
        key = (u, v)
        if key not in self.base:
            return 0.3
        t_eff = t_min + lookahead_min
        day_type = self._day_type()
        h = (t_eff % (24 * 60)) / 60.0
        side = np.sin(self.phase[key])
        morning_extra = 0.45 * max(0.0, -side) * np.exp(-0.5 * ((h - 8.0) / 0.65) ** 2)
        evening_extra = 0.50 * max(0.0, side) * np.exp(-0.5 * ((h - 18.2) / 0.75) ** 2)
        wave = 0.12 * np.sin(2 * np.pi * h / 24.0 + self.phase[key])
        c = self.base[key] * _diurnal_factor(t_eff, day_type) + wave + morning_extra + evening_extra
        c *= self._volume_factor()
        return float(np.clip(c, 0.02, 0.98))

    def predicted_congestion(self, u, v, t_min: float, eta_along_path_min: float = 0.0) -> float:
        """경로 진행 중 해당 엣지 도착 예정 시각의 예측 혼잡 (최적경로용)."""
        return self.congestion(u, v, t_min, lookahead_min=eta_along_path_min)

    def speed_kmh(self, u, v, t_min: float, ems: bool = True, lookahead_min: float = 0.0) -> float:
        cong = self.congestion(u, v, t_min, lookahead_min=lookahead_min)
        speed = max(10.0, 48.0 * (1.0 - 0.72 * cong))
        if ems:
            speed *= EMS_SPEED_FACTOR
        return float(speed)

    def travel_time_min(self, u, v, t_min: float, ems: bool = True, lookahead_min: float = 0.0) -> float:
        length_km = self.length_m.get((u, v), 400.0) / 1000.0
        speed = self.speed_kmh(u, v, t_min, ems=ems, lookahead_min=lookahead_min)
        return (length_km / max(speed, 5.0)) * 60.0

    def apply_to_graph(self, G: nx.DiGraph, t_min: float, horizon_min: float = 45.0) -> None:
        """경로탐색용: '지금~곧' 예측 혼잡을 엣지 weight에 반영.

        단순 현재시각만 쓰지 않고, 앞으로 horizon_min 동안의 평균 예측을 써서
        '앞으로 빡세질 길'을 미리 피하게 한다.
        """
        samples = [0.0, horizon_min * 0.35, horizon_min * 0.7, horizon_min]
        for u, v in G.edges():
            preds = [self.congestion(u, v, t_min, lookahead_min=s) for s in samples]
            # 평균과 최악 사이 — 앞으로 막힐 길을 보수적으로 회피
            cong = 0.6 * float(np.mean(preds)) + 0.4 * float(np.max(preds))
            ed = G.edges[u, v]
            ed["congestion"] = float(np.clip(cong, 0.02, 0.98))
            length_km = self.length_m.get((u, v), float(ed.get("length_m", 400.0))) / 1000.0
            speed = max(10.0, 48.0 * (1.0 - 0.72 * cong)) * EMS_SPEED_FACTOR
            ed["travel_time"] = (length_km / max(speed, 5.0)) * 60.0
            ed["speed_kmh"] = speed

    def city_congestion(self, t_min: float, lookahead_min: float = 0.0, sample: int = 400) -> float:
        """도시 평균 혼잡 (예측 곡선용)."""
        edges = self.edge_list[:sample] if self.edge_list else list(self.base.keys())[:sample]
        if not edges:
            return 0.3
        vals = [self.congestion(u, v, t_min, lookahead_min=lookahead_min) for u, v in edges]
        return float(np.mean(vals))

    def forecast(self, t_min: float, horizon_hours: float = 6.0, step_min: int = 30) -> dict:
        """현재 시각부터 horizon 동안의 도시 혼잡 예측 시계열."""
        steps = []
        horizon = int(horizon_hours * 60)
        peak = 0.0
        peak_t = int(t_min)
        for dt in range(0, horizon + 1, step_min):
            c = self.city_congestion(t_min, lookahead_min=float(dt))
            abs_t = int(t_min + dt) % (24 * 60)
            if c > peak:
                peak, peak_t = c, abs_t
            steps.append({
                "offset_min": dt,
                "t_min": abs_t,
                "time_label": f"{abs_t // 60:02d}:{abs_t % 60:02d}",
                "congestion": round(c, 3),
                "level": congestion_level(c)["level"],
                "color": traffic_color(c),
                "severity": congestion_level(c)["name"],
            })
        now = steps[0]["congestion"] if steps else 0.3
        later = steps[min(2, len(steps) - 1)]["congestion"] if steps else now
        trend = "증가" if later > now + 0.05 else ("감소" if later < now - 0.05 else "유지")
        return {
            "horizon_hours": horizon_hours,
            "step_min": step_min,
            "now": round(now, 3),
            "trend": trend,
            "peak": {"congestion": round(peak, 3), "t_min": peak_t,
                     "time_label": f"{peak_t // 60:02d}:{peak_t % 60:02d}",
                     "color": traffic_color(peak)},
            "series": steps,
            "calendar": self.context.to_dict() if self.context else None,
            "note": "데모용 캘린더·시간대 예측(학습 모델 대체 예정)",
        }

    def costmap(self, G: nx.DiGraph, t_min: float, lookahead_min: float = 0.0) -> dict:
        """파랑→빨강 교통량 지도."""
        segments = []
        hist = [0, 0, 0, 0]
        for u, v in self.edge_list:
            if u not in G.nodes or v not in G.nodes:
                continue
            cong = self.congestion(u, v, t_min, lookahead_min=lookahead_min)
            lvl = congestion_level(cong)
            hist[lvl["level"]] += 1
            du, dv = G.nodes[u], G.nodes[v]
            segments.append({
                "coords": [
                    [float(du["lat"]), float(du["lng"])],
                    [float(dv["lat"]), float(dv["lng"])],
                ],
                "congestion": round(cong, 3),
                "level": lvl["level"],
                "color": traffic_color(cong),
            })
        t = int(t_min + lookahead_min) % (24 * 60)
        return {
            "t_min": t,
            "time_label": f"{t // 60:02d}:{t % 60:02d}",
            "lookahead_min": lookahead_min,
            "source": self.source,
            "levels": CONGESTION_LEVELS,
            "histogram": hist,
            "segments": segments,
            "legend": {
                "low": {"label": "한산", "color": traffic_color(0.05)},
                "mid": {"label": "보통", "color": traffic_color(0.5)},
                "high": {"label": "혼잡", "color": traffic_color(0.95)},
                "scale": "blue_to_red",
            },
            "calendar": self.context.to_dict() if self.context else None,
            "city_avg": round(self.city_congestion(t_min, lookahead_min), 3),
        }


def path_edge_times(field: TimeCongestionField, path: list, t_min: float) -> list[float]:
    """경로를 따라가며 도착 예정 시각의 예측 혼잡으로 구간 시간 계산."""
    times = []
    acc = 0.0
    for u, v in zip(path[:-1], path[1:]):
        dt = field.travel_time_min(u, v, t_min, ems=True, lookahead_min=acc)
        times.append(dt)
        acc += dt
    return times


def path_edge_speeds(field: TimeCongestionField, path: list, t_min: float) -> list[float]:
    speeds = []
    acc = 0.0
    for u, v in zip(path[:-1], path[1:]):
        speeds.append(round(field.speed_kmh(u, v, t_min, ems=True, lookahead_min=acc), 1))
        acc += field.travel_time_min(u, v, t_min, ems=True, lookahead_min=acc)
    return speeds


def format_clock(t_min: float) -> str:
    t = int(t_min) % (24 * 60)
    return f"{t // 60:02d}:{t % 60:02d}"
