"""
demo_fire_dispatch.py
======================
구급차용 demo_showcase.py의 소방차 버전.

무작위 화재현장 발생 → 최근접 소방서 자동 출동 → 혼잡·위험 회피 경로로
현장 도착까지의 전 과정을 실제 대구 도로망 위에 애니메이션으로 보여준다.
구급차와 달리 목표 병원 비교 단계가 없다(1단계 임무) — 대신 혼잡도가
낮은 경로를 택했다는 걸 보여주기 위해, 단순 최단경로(다익스트라, 혼잡 미반영)와
혼잡가중 경로를 나란히 비교해서 그린다.

실행:
    python demo_fire_dispatch.py --seed 5 --save fire_showcase.gif --no-show
"""

from __future__ import annotations
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from network_graph import build_combined_graph
from daegu_ems_geo import DAEGU_FIRE_STATIONS
from standard_node_link import DAEGU_LAT_MIN, DAEGU_LAT_MAX, DAEGU_LNG_MIN, DAEGU_LNG_MAX
from demo_live_incident import _setup_kr_font, _HAS_KR_FONT, congestion_shortest_path, pick_nearest_station, STATION_EN
from demo_showcase import sample_background_traffic


def plain_shortest_path(G: nx.DiGraph, src: int, dst: int) -> list:
    """혼잡·위험을 무시한 순수 거리(이동시간)만 최소화하는 경로 — 비교용 baseline."""
    try:
        return nx.shortest_path(G, src, dst, weight="travel_time")
    except nx.NetworkXNoPath:
        return [src]


def path_cost(G: nx.DiGraph, path: list) -> float:
    def w(u, v, d):
        return float(d.get("travel_time", 1.0)) * (1.0 + float(d.get("congestion", 0.0)))
    return sum(w(u, v, G.edges[u, v]) for u, v in zip(path[:-1], path[1:]))


def animate_fire(G, aware_path, plain_path, fire_scene, station_id, out_gif=None, interval_ms=90, seed=0):
    pos = {n: (d["lng"], d["lat"]) for n, d in G.nodes(data=True)}
    fig, ax = plt.subplots(figsize=(10.5, 11))
    ax.set_aspect("equal")

    edges = list(G.edges())
    if len(edges) > 2500:
        rng0 = np.random.default_rng(0)
        edges = [edges[i] for i in rng0.choice(len(edges), size=2500, replace=False)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.12, width=0.4,
                            edge_color="#6b8e6b", arrows=False, ax=ax)

    (veh_lng, veh_lat), (ped_lng, ped_lat) = sample_background_traffic(G, seed=seed)
    ax.scatter(veh_lng, veh_lat, s=7, c="dimgray", alpha=0.55, zorder=3, marker="o")
    ax.scatter(ped_lng, ped_lat, s=5, c="steelblue", alpha=0.45, zorder=3, marker=".")

    for s in DAEGU_FIRE_STATIONS:
        ax.plot(s["lng"], s["lat"], "s", color="navy", markersize=8, zorder=5)

    # 비교용: 혼잡 무시 최단경로(회색 점선)를 먼저 깔아둔다
    plain_xy = [pos[n] for n in plain_path if n in pos]
    if len(plain_xy) > 1:
        pxs, pys = zip(*plain_xy)
        ax.plot(pxs, pys, "--", color="gray", lw=1.6, alpha=0.7, zorder=4,
                label="Shortest-distance route (congestion-blind)")

    ax.plot(G.nodes[fire_scene]["lng"], G.nodes[fire_scene]["lat"],
            "X", color="orangered", markersize=18, zorder=7, label="fire scene")

    path_line, = ax.plot([], [], color="darkorange", lw=2.8, zorder=8,
                          label="Congestion-aware route (this system)")
    agent_pt, = ax.plot([], [], "o", color="lime", markersize=10, zorder=9,
                        markeredgecolor="black", markeredgewidth=0.8)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9.5,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.92))

    s_en = STATION_EN.get(station_id, f"Station{station_id}")
    aware_cost = path_cost(G, aware_path)
    plain_cost = path_cost(G, plain_path)
    saved_pct = 100.0 * (plain_cost - aware_cost) / plain_cost if plain_cost > 0 else 0.0

    ax.set_title("Daegu FIRE dispatch: random fire scene -> nearest station -> "
                 "congestion-aware route", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(DAEGU_LNG_MIN, DAEGU_LNG_MAX)
    ax.set_ylim(DAEGU_LAT_MIN, DAEGU_LAT_MAX)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.01, -0.045,
            f"Congestion-aware route cost={aware_cost:.1f} vs shortest-distance cost={plain_cost:.1f} "
            f"({saved_pct:+.1f}% {'better' if saved_pct > 0 else 'worse'})",
            transform=ax.transAxes, fontsize=8.5, color="gray")

    xs = [pos[n][0] for n in aware_path if n in pos]
    ys = [pos[n][1] for n in aware_path if n in pos]

    def init():
        path_line.set_data([], [])
        agent_pt.set_data([], [])
        return path_line, agent_pt, status

    def update(frame):
        i = min(frame, len(xs) - 1)
        path_line.set_data(xs[: i + 1], ys[: i + 1])
        agent_pt.set_data([xs[i]], [ys[i]])
        status.set_text(f"DISPATCH  {s_en} -> fire scene\nstep {i}/{len(xs)-1}")
        return path_line, agent_pt, status

    anim = FuncAnimation(fig, update, frames=len(xs), init_func=init,
                         interval=interval_ms, blit=False, repeat=False)
    if out_gif:
        anim.save(out_gif, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
        print(f"[fire_demo] saved {out_gif}")
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass
    return anim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--fire-scene", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save", type=str, default="fire_showcase.gif")
    parser.add_argument("--interval-ms", type=int, default=80)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    print("[fire_demo] Loading REAL Daegu road network (standard node-link)...")
    G, _hospital_map = build_combined_graph(verbose=True, data_dir=args.data_dir)
    station_map = G.graph["station_node_map"]

    rng = np.random.default_rng(args.seed)
    reserved = set(station_map.values())
    candidates = [n for n in G.nodes if n not in reserved]
    center = [
        n for n in candidates
        if 35.82 <= G.nodes[n]["lat"] <= 35.92 and 128.52 <= G.nodes[n]["lng"] <= 128.68
    ]
    pool = center if center else candidates
    fire_scene = args.fire_scene if args.fire_scene is not None else int(rng.choice(pool))

    sid = pick_nearest_station(G, fire_scene, station_map)
    station_node = station_map[sid]

    aware_path = congestion_shortest_path(G, station_node, fire_scene)
    plain_path = plain_shortest_path(G, station_node, fire_scene)

    print("=" * 60)
    print(f"[fire_demo] REAL Daegu roads: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    print(f"[fire_demo] FIRE SCENE node={fire_scene} "
          f"({G.nodes[fire_scene]['lat']:.5f}, {G.nodes[fire_scene]['lng']:.5f})")
    s = next(x for x in DAEGU_FIRE_STATIONS if x["station_id"] == sid)
    print(f"[fire_demo] Nearest fire station: {s['station_name']}")
    print(f"[fire_demo] aware_cost={path_cost(G, aware_path):.1f}, "
          f"plain_cost={path_cost(G, plain_path):.1f}, hops={len(aware_path)-1}")
    print("=" * 60)

    animate_fire(G, aware_path, plain_path, fire_scene, sid,
                 out_gif=args.save, interval_ms=args.interval_ms, seed=args.seed)


if __name__ == "__main__":
    main()
