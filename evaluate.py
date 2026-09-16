"""
evaluate.py
===========
대구시 응급 골든타임 경로 시각화.
출동(소방서→현장, 주황) + 이송(현장→응급실, 보라).
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from network_graph import build_combined_graph
from daegu_ems_geo import DAEGU_ER_HOSPITALS
from env import EmergencyRouteEnv, _HAS_GYM


def _policy_action(env: EmergencyRouteEnv, obs, model=None) -> int:
    if model is not None:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    # 학습 모델 없을 때: 혼잡 가중 최단경로에 가까운 greedy
    neighbors = list(env.graph.successors(env.current_node))[: env.MAX_NEIGHBORS]
    if not neighbors:
        return 0
    scores = []
    for i, nb in enumerate(neighbors):
        try:
            hops = nx.shortest_path_length(env.graph, nb, env.goal_node)
        except nx.NetworkXNoPath:
            hops = 999
        cong = env.graph.edges[env.current_node, nb].get("congestion", 0.5)
        scores.append((hops + 0.5 * cong, i))
    return int(min(scores)[1])


def rollout(env: EmergencyRouteEnv, model=None, seed: int = 0,
            vehicle_type: str = "ambulance", target_hospital_id: int = 0, station_id: int = 0,
            switch_step: int | None = None, switch_to: int | None = None):
    reset_kwargs = dict(seed=seed, vehicle_type=vehicle_type, station_id=station_id)
    if vehicle_type == "ambulance":
        reset_kwargs["target_hospital_id"] = target_hospital_id
    reset_out = env.reset(**reset_kwargs)
    obs = reset_out[0] if _HAS_GYM else reset_out
    path = [env.current_node]
    total_reward = 0.0
    hospital_switch_at = None

    max_total = env.MAX_STEPS + (env.REROUTE_STEP_BONUS if switch_step is not None else 0)
    for step_i in range(max_total):
        if (vehicle_type == "ambulance" and switch_step is not None
                and step_i == switch_step and switch_to is not None):
            obs = env.reroute(switch_to)
            hospital_switch_at = len(path) - 1
            print(f"[evaluate] step {step_i}: hospital -> #{switch_to} "
                  f"(pos={env.current_node}, phase={env.phase})")

        action = _policy_action(env, obs, model)
        step_out = env.step(action)
        if _HAS_GYM:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        total_reward += reward
        path.append(env.current_node)
        if done:
            break

    return path, env.phase_switch_index, hospital_switch_at, total_reward, env


def visualize(graph, path, phase_switch_index, hospital_switch_at, env, out_path="route.png"):
    from daegu_ems_geo import DAEGU_FIRE_STATIONS
    from standard_node_link import DAEGU_LAT_MIN, DAEGU_LAT_MAX, DAEGU_LNG_MIN, DAEGU_LNG_MAX

    pos = {n: (d["lng"], d["lat"]) for n, d in graph.nodes(data=True) if "lat" in d}
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_aspect("equal")

    # 실제 대구 도로망 (샘플링)
    edges = list(graph.edges())
    if len(edges) > 2500:
        rng = np.random.default_rng(0)
        edges = [edges[i] for i in rng.choice(len(edges), size=2500, replace=False)]
    nx.draw_networkx_edges(graph, pos, edgelist=edges, alpha=0.12, width=0.4,
                            edge_color="#6b8e6b", arrows=False, ax=ax)

    # 실제 시설 좌표(스냅 노드 아님)
    for s in DAEGU_FIRE_STATIONS:
        ax.plot(s["lng"], s["lat"], "s", color="navy", markersize=8, zorder=5)
    if env.vehicle_type == "ambulance":
        for h in DAEGU_ER_HOSPITALS:
            ax.plot(h["lng"], h["lat"], "^", color="crimson", markersize=9, zorder=5)
    if env.incident_node in pos:
        marker = "X" if env.vehicle_type == "fire_truck" else "*"
        color = "orangered" if env.vehicle_type == "fire_truck" else "black"
        ax.plot(pos[env.incident_node][0], pos[env.incident_node][1],
                marker, color=color, markersize=16, zorder=6)

    path_draw = [n for n in path if n in pos]
    if env.vehicle_type == "fire_truck" or phase_switch_index is None:
        # 소방차(1단계) 이거나 구급차가 아직 현장 미도착 — 전부 출동색
        edges = list(zip(path_draw[:-1], path_draw[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color="darkorange",
                                width=2.4, arrows=True, ax=ax)
    else:
        # path index: phase_switch_index is env.steps when incident reached
        # path has start + one node per step, so incident is at index phase_switch_index
        cut = min(phase_switch_index, len(path_draw) - 1)
        before = path_draw[: cut + 1]
        after = path_draw[cut:]
        nx.draw_networkx_edges(graph, pos, edgelist=list(zip(before[:-1], before[1:])),
                                edge_color="darkorange", width=2.4, arrows=True, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=list(zip(after[:-1], after[1:])),
                                edge_color="purple", width=2.4, arrows=True, ax=ax)

    if env.vehicle_type == "ambulance":
        hname = next(
            (h["hospital_name"] for h in DAEGU_ER_HOSPITALS
             if h["hospital_id"] == env.target_hospital_id),
            f"#{env.target_hospital_id}",
        )
        title = (f"Daegu EMS golden-time: station#{env.station_id} -> incident -> "
                 f"hospital#{env.target_hospital_id} ({len(path)-1} hops)")
        if hospital_switch_at is not None:
            title += f" | hospital reroute@{hospital_switch_at}"
        legend = ("Green=Daegu standard node-link roads | Navy=fire station | "
                  "Red triangle=ER | Black star=incident | Orange=dispatch | Purple=transport")
    else:
        hname = None
        title = (f"Daegu FIRE dispatch: station#{env.station_id} -> fire scene "
                 f"({len(path)-1} hops)")
        legend = ("Green=Daegu standard node-link roads | Navy=fire station | "
                  "Orange X=fire scene | Orange=dispatch route (single phase)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(DAEGU_LNG_MIN, DAEGU_LNG_MAX)
    ax.set_ylim(DAEGU_LAT_MIN, DAEGU_LAT_MAX)
    ax.text(0.01, -0.06, legend, transform=ax.transAxes, fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[evaluate] saved {out_path}")
    if env.vehicle_type == "ambulance":
        print(f"[evaluate] hospital={hname}, phase_end={env.phase}, "
              f"reached_ER={env.current_node == env.hospital_node_map[env.target_hospital_id]}")
    else:
        print(f"[evaluate] reached_fire_scene={env.current_node == env.incident_node}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vehicle-type", type=str, default="ambulance",
                         choices=["ambulance", "fire_truck"])
    parser.add_argument("--target-hospital", type=int, default=0)
    parser.add_argument("--station-id", type=int, default=0)
    parser.add_argument("--switch-step", type=int, default=None)
    parser.add_argument("--switch-to", type=int, default=None)
    parser.add_argument("--out", type=str, default="route.png")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--n-intersections", type=int, default=40)
    args = parser.parse_args()

    graph, hospital_node_map = build_combined_graph(
        verbose=False, data_dir=args.data_dir, n_intersections=args.n_intersections,
    )
    env = EmergencyRouteEnv(graph, hospital_node_map, seed=args.seed)

    model = None
    if args.model_path:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(args.model_path)
        except ImportError:
            print("[evaluate] SB3 missing — random policy")

    if (args.vehicle_type == "ambulance" and args.switch_step is not None
            and args.switch_to is None):
        args.switch_to = (args.target_hospital + 1) % len(hospital_node_map)

    path, phase_sw, hosp_sw, reward, env = rollout(
        env, model=model, seed=args.seed, vehicle_type=args.vehicle_type,
        target_hospital_id=args.target_hospital, station_id=args.station_id,
        switch_step=args.switch_step, switch_to=args.switch_to,
    )
    print(f"[evaluate] vehicle_type={args.vehicle_type}, hops={len(path)-1}, reward={reward:.3f}, "
          f"phase_switch_step={phase_sw}")
    visualize(graph, path, phase_sw, hosp_sw, env, out_path=args.out)


if __name__ == "__main__":
    main()
