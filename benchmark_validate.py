"""
benchmark_validate.py
=====================
제출·발표용 끝까지 검증 스크립트.

1) 예측 정확도 (오픈 vs 안심구역 ITS) — MAE/RMSE
   - 지금: 오픈 link_hourly 시간대 holdout 백테스트 (항상 가능)
   - --data-dir 있으면 ITS 실측과 같은 구간·시간으로 나란히 비교
2) RL(PPO) vs 혼잡가중 Dijkstra — 시나리오 N회
   - --model-path 있을 때만 PPO. 없으면 Dijkstra baseline만 + 안내
3) 혼잡회피 효과 통계 — 단순최단 vs 혼잡가중, N 시나리오 평균 개선율
4) 다중차량 겹침 — Prioritized Planning on/off, mixed_major N회

실행 예:
    python benchmark_validate.py --n 100
    python benchmark_validate.py --n 100 --model-path emergency_ppo.zip
    python benchmark_validate.py --n 50 --data-dir /path/to/경북대ITS
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------- #
# 공통
# ---------------------------------------------------------------------- #
def _path_travel_cost(G: nx.DiGraph, path: list, *, use_cong: bool) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G.edges[u, v]
        tt = float(d.get("travel_time", 1.0))
        if use_cong:
            tt *= 1.0 + float(d.get("congestion", 0.0))
        total += tt
    return total


def _plain_path(G: nx.DiGraph, src: int, dst: int) -> list:
    try:
        return nx.shortest_path(G, src, dst, weight="travel_time")
    except nx.NetworkXNoPath:
        return [src]


def _cong_path(G: nx.DiGraph, src: int, dst: int) -> list:
    def w(u, v, d):
        return float(d.get("travel_time", 1.0)) * (1.0 + float(d.get("congestion", 0.0)))

    try:
        return nx.shortest_path(G, src, dst, weight=w)
    except nx.NetworkXNoPath:
        return [src]


def _edge_pairs(path: list) -> set[tuple]:
    return {(u, v) for u, v in zip(path[:-1], path[1:])}


def _overlap_count(paths: list[list]) -> int:
    usage: Counter = Counter()
    for p in paths:
        usage.update(_edge_pairs(p))
    return int(sum(c - 1 for c in usage.values() if c > 1))


# ---------------------------------------------------------------------- #
# 1) 예측 정확도
# ---------------------------------------------------------------------- #
def bench_prediction_open() -> dict:
    """오픈 시간대 프로파일 → holdout 시간 MAE/RMSE (km/h)."""
    from daegu_open_data import load_link_hourly_stats, build_hourly_speed_profile

    df = load_link_hourly_stats()
    if df.empty or "hour" not in df.columns or "speed" not in df.columns:
        return {"status": "skip", "reason": "link_hourly_stats 없음"}

    # 시 0~17 로 프로파일, 18~23 으로 평가 (또는 데이터 절반)
    hours = sorted(df["hour"].dropna().unique().tolist())
    if len(hours) < 8:
        return {"status": "skip", "reason": "시간대 샘플 부족"}

    cut = hours[len(hours) * 2 // 3]
    train = df[df["hour"] <= cut]
    test = df[df["hour"] > cut]
    if train.empty or test.empty:
        train = df.sample(frac=0.7, random_state=0)
        test = df.drop(train.index)

    profile = build_hourly_speed_profile(train)
    city = profile.get("city_hourly") or [35.0] * 24
    by_road = profile.get("by_road") or {}

    preds, acts = [], []
    for _, row in test.iterrows():
        h = int(row["hour"]) % 24
        act = float(row["speed"])
        key = str(row.get("road_name", "") or "")
        pred = float((by_road.get(key) or city)[h])
        preds.append(pred)
        acts.append(act)

    pred_a = np.asarray(preds, dtype=float)
    act_a = np.asarray(acts, dtype=float)
    err = pred_a - act_a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "status": "ok",
        "source": "open_link_hourly_holdout",
        "n_test": int(len(acts)),
        "mae_kmh": round(mae, 3),
        "rmse_kmh": round(rmse, 3),
        "note": "오픈 통계만으로 시간대 속도 예측 holdout. ITS 전역 실측과 비교하려면 --data-dir",
    }


def bench_prediction_its(data_dir: str | None) -> dict:
    if not data_dir:
        return {
            "status": "pending",
            "reason": "안심구역 전역 데이터 경로 없음 (--data-dir)",
            "how": "같은 링크·시간으로 오픈 예측 vs ITS 실측 MAE를 나란히 보고, 오차 감소 %p 를 발표 숫자로 씀",
        }
    try:
        from data_schema import load_daegu_traffic_volume_data
    except ImportError:
        return {"status": "skip", "reason": "data_schema.py 없음 (안심구역 PC에서 연결)"}

    try:
        vol = load_daegu_traffic_volume_data(data_dir=data_dir)
    except Exception as e:
        return {"status": "skip", "reason": str(e)}

    if vol is None or len(vol) == 0:
        return {"status": "skip", "reason": "ITS 교통량 비어 있음"}

    # 카메라별 평균 교통량 분포 — 오픈 프로파일과 직접 km/h 비교는 단위가 다름.
    # 여기서는 'ITS가 로드됨 + 커버리지' 를 보고, 실제 MAE는 안심구역에서 속도 환산 후 재실행.
    return {
        "status": "loaded",
        "n_rows": int(len(vol)),
        "note": "ITS 로드 OK. 속도(km/h)로 환산 매핑 후 open MAE와 동일 지표로 재비교 필요",
    }


# ---------------------------------------------------------------------- #
# 2) RL vs Dijkstra
# ---------------------------------------------------------------------- #
def bench_rl_vs_dijkstra(G, hospital_map, n: int, model_path: str | None, seed: int) -> dict:
    from env import EmergencyRouteEnv, _HAS_GYM

    model = None
    if model_path and Path(model_path).exists():
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        except Exception as e:
            return {"status": "skip", "reason": f"모델 로드 실패: {e}"}
    else:
        return {
            "status": "pending",
            "reason": "emergency_ppo.zip 없음 — python train.py 후 --model-path 로 재실행",
            "baseline_hint": "Dijkstra(혼잡가중 greedy) 단독 수치는 아래 congestion_avoid 참고",
        }

    env = EmergencyRouteEnv(G, hospital_map, seed=seed)

    def run_one(use_model, s: int) -> dict:
        reset_out = env.reset(seed=s, vehicle_type="ambulance", station_id=s % max(1, len(G.graph.get("station_node_map", {0: 0}))))
        obs = reset_out[0] if _HAS_GYM else reset_out
        hops = 0
        reached = False
        tt = 0.0
        for _ in range(env.MAX_STEPS):
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                # 혼잡가중 greedy (evaluate._policy_action 와 동일 계열)
                neighbors = list(env.graph.successors(env.current_node))[: env.MAX_NEIGHBORS]
                if not neighbors:
                    action = 0
                else:
                    scores = []
                    for i, nb in enumerate(neighbors):
                        try:
                            hops_left = nx.shortest_path_length(env.graph, nb, env.goal_node)
                        except nx.NetworkXNoPath:
                            hops_left = 999
                        cong = env.graph.edges[env.current_node, nb].get("congestion", 0.5)
                        scores.append((hops_left + 0.5 * cong, i))
                    action = int(min(scores)[1])
            step_out = env.step(action)
            if _HAS_GYM:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out
            hops += 1
            # travel proxy
            if len(env.path_history if hasattr(env, "path_history") else []) >= 2:
                pass
            if done:
                reached = env.current_node == env.goal_node
                break
        return {"hops": hops, "reached": bool(reached)}

    dij, ppo = [], []
    for i in range(n):
        s = seed + i * 17
        dij.append(run_one(False, s))
        # reset same seed for fair-ish compare
        ppo.append(run_one(True, s))

    def agg(rows):
        return {
            "mean_hops": round(float(np.mean([r["hops"] for r in rows])), 2),
            "reach_rate": round(float(np.mean([1.0 if r["reached"] else 0.0 for r in rows])), 3),
            "n": len(rows),
        }

    a, b = agg(dij), agg(ppo)
    improve = None
    if a["mean_hops"] > 0:
        improve = round(100.0 * (a["mean_hops"] - b["mean_hops"]) / a["mean_hops"], 2)
    return {
        "status": "ok",
        "dijkstra_greedy": a,
        "ppo": b,
        "hops_improve_pct": improve,
        "note": "양수면 PPO가 평균 홉 수 감소",
    }


# ---------------------------------------------------------------------- #
# 3) 혼잡회피 통계
# ---------------------------------------------------------------------- #
def bench_congestion_avoid(G, station_map: dict, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nodes = [nd for nd, d in G.nodes(data=True) if "lat" in d]
    if len(nodes) < 10 or not station_map:
        return {"status": "skip", "reason": "그래프/소방서 부족"}

    station_nodes = list(station_map.values())
    improves = []
    plain_costs = []
    aware_costs = []
    for _ in range(n):
        scene = int(rng.choice(nodes))
        # 최근접 서
        ilat, ilng = G.nodes[scene]["lat"], G.nodes[scene]["lng"]
        best_sid, best_d = None, 1e18
        for sid, nid in station_map.items():
            d = (G.nodes[nid]["lat"] - ilat) ** 2 + (G.nodes[nid]["lng"] - ilng) ** 2
            if d < best_d:
                best_sid, best_d = sid, d
        src = station_map[best_sid]
        plain = _plain_path(G, src, scene)
        aware = _cong_path(G, src, scene)
        # 평가 비용은 항상 혼잡 반영 travel (실제 체감)
        pc = _path_travel_cost(G, plain, use_cong=True)
        ac = _path_travel_cost(G, aware, use_cong=True)
        if pc <= 1e-6:
            continue
        plain_costs.append(pc)
        aware_costs.append(ac)
        improves.append(100.0 * (pc - ac) / pc)

    if not improves:
        return {"status": "skip", "reason": "유효 시나리오 0"}

    return {
        "status": "ok",
        "n": len(improves),
        "mean_plain_cost": round(float(np.mean(plain_costs)), 2),
        "mean_aware_cost": round(float(np.mean(aware_costs)), 2),
        "mean_improve_pct": round(float(np.mean(improves)), 2),
        "median_improve_pct": round(float(np.median(improves)), 2),
        "pct_scenarios_aware_better": round(100.0 * float(np.mean([x > 0 for x in improves])), 1),
        "note": "양수=혼잡가중 경로가 혼잡반영 비용에서 더 좋음",
    }


# ---------------------------------------------------------------------- #
# 4) 다중차량 겹침
# ---------------------------------------------------------------------- #
def _route_naive(G, src, dst) -> list:
    return _cong_path(G, src, dst)


def _route_with_usage(G, src, dst, edge_usage: Counter, penalty: float = 8.0) -> list:
    def w(u, v, d):
        base = float(d.get("travel_time", 1.0)) * (1.0 + float(d.get("congestion", 0.0)))
        return base + penalty * edge_usage[(u, v)]

    try:
        return nx.shortest_path(G, src, dst, weight=w)
    except nx.NetworkXNoPath:
        return [src]


def bench_multi_overlap(G, station_map, hospital_map, n: int, seed: int) -> dict:
    from standard_node_link import nearest_node as sn

    rng = np.random.default_rng(seed + 99)
    nodes = [nd for nd, d in G.nodes(data=True) if "lat" in d]
    if len(nodes) < 20:
        return {"status": "skip", "reason": "노드 부족"}

    naive_ov, aware_ov = [], []
    for i in range(n):
        lat = float(rng.uniform(35.82, 35.92))
        lng = float(rng.uniform(128.52, 128.65))
        try:
            node = sn(G, lat, lng)
        except Exception:
            node = int(rng.choice(nodes))
        # 소방서 상위 2
        scored = []
        ilat, ilng = G.nodes[node]["lat"], G.nodes[node]["lng"]
        for sid, nid in station_map.items():
            d = (G.nodes[nid]["lat"] - ilat) ** 2 + (G.nodes[nid]["lng"] - ilng) ** 2
            scored.append((d, sid, nid))
        scored.sort()
        picks = scored[:2]
        # 병원 1
        h_scored = []
        for hid, hnid in hospital_map.items():
            d = (G.nodes[hnid]["lat"] - ilat) ** 2 + (G.nodes[hnid]["lng"] - ilng) ** 2
            h_scored.append((d, hid, hnid))
        h_scored.sort()
        hnid = h_scored[0][2]

        paths_n = []
        for _, sid, snid in picks:
            paths_n.append(_route_naive(G, snid, node))
            paths_n.append(_route_naive(G, snid, node) + _route_naive(G, node, hnid)[1:])
        usage: Counter = Counter()
        paths_a = []
        for _, sid, snid in picks:
            p1 = _route_with_usage(G, snid, node, usage)
            for e in _edge_pairs(p1):
                usage[e] += 1
            paths_a.append(p1)
            p2a = _route_with_usage(G, snid, node, usage)
            for e in _edge_pairs(p2a):
                usage[e] += 1
            p2b = _route_with_usage(G, node, hnid, usage)
            for e in _edge_pairs(p2b):
                usage[e] += 1
            paths_a.append(p2a + p2b[1:])

        naive_ov.append(_overlap_count(paths_n))
        aware_ov.append(_overlap_count(paths_a))

    mean_n = float(np.mean(naive_ov))
    mean_a = float(np.mean(aware_ov))
    reduce_pct = 100.0 * (mean_n - mean_a) / mean_n if mean_n > 0 else 0.0
    return {
        "status": "ok",
        "n": n,
        "mean_overlap_naive": round(mean_n, 2),
        "mean_overlap_prioritized": round(mean_a, 2),
        "reduce_pct": round(reduce_pct, 2),
        "note": "겹침=여러 대가 같은 방향 엣지를 공유한 횟수 합",
    }


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="시나리오 수 (2·3·4순위)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None, help="안심구역 ITS 폴더")
    ap.add_argument("--out", type=str, default="benchmark_results.json")
    ap.add_argument("--quick", action="store_true", help="그래프 로드만 가볍게 (n 자동 30)")
    args = ap.parse_args()
    if args.quick:
        args.n = min(args.n, 30)

    print("[benchmark] 그래프 로드…")
    from network_graph import build_combined_graph

    G, hospital_map = build_combined_graph(verbose=False, data_dir=args.data_dir)
    station_map = G.graph.get("station_node_map", {})

    results = {
        "n_scenarios": args.n,
        "seed": args.seed,
        "1_prediction_open": bench_prediction_open(),
        "1_prediction_its": bench_prediction_its(args.data_dir),
        "2_rl_vs_dijkstra": bench_rl_vs_dijkstra(
            G, hospital_map, min(args.n, 80), args.model_path, args.seed
        ),
        "3_congestion_avoid": bench_congestion_avoid(G, station_map, args.n, args.seed),
        "4_multi_overlap": bench_multi_overlap(G, station_map, hospital_map, args.n, args.seed),
    }

    out = HERE / args.out
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(results, ensure_ascii=True, indent=2))
    print(f"[benchmark] saved {out}")


if __name__ == "__main__":
    main()
