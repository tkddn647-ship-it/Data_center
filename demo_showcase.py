"""
demo_showcase.py
=================
발표/보고서용 "보여주기" 데모.

demo_live_incident.py와 다른 점 두 가지:
1) 응급실을 --hospital로 고정 지정하지 않고, **혼잡도까지 반영한 이동비용이
   가장 낮은 응급실을 자동으로 고른다** ("안 막히는 쪽 응급실로 가는" 시나리오).
   8개 응급실 전부에 대해 혼잡가중 다익스트라 비용을 계산해 비교한다.
2) 배경에 **무작위 차량·시민**을 뿌려서 "살아있는 도시" 느낌을 준다.
   실제 교통 시뮬레이션이 아니라 순수 시각적 장치다 — 엣지 혼잡도가 높을수록
   그 도로 근처에 차량 점이 더 많이 찍히도록 확률을 가중해서, 왜 특정
   응급실이 선택됐는지 배경만 봐도 짐작할 수 있게 했다.

실행:
    python demo_showcase.py --seed 3 --save showcase.gif --no-show
"""

from __future__ import annotations
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from network_graph import build_combined_graph
from daegu_ems_geo import DAEGU_ER_HOSPITALS, DAEGU_FIRE_STATIONS
from standard_node_link import DAEGU_LAT_MIN, DAEGU_LAT_MAX, DAEGU_LNG_MIN, DAEGU_LNG_MAX
from demo_live_incident import (
    _setup_kr_font, _HAS_KR_FONT, congestion_shortest_path, pick_nearest_station,
    HOSPITAL_EN, STATION_EN,
)


def pick_best_hospital(G: nx.DiGraph, incident: int, hospital_map: dict):
    """혼잡가중 이동비용이 가장 낮은(=안 막히는) 응급실을 고른다.

    Returns
    -------
    best_hid, best_path, cost_table (모든 병원의 비용 — 발표에서 '왜 이 병원인지' 보여줄 근거)
    """
    def edge_cost(u, v, d):
        return float(d.get("travel_time", 1.0)) * (1.0 + float(d.get("congestion", 0.0)))

    cost_table = []
    best_hid, best_cost, best_path = None, float("inf"), None
    for hid, node in hospital_map.items():
        try:
            path = nx.shortest_path(G, incident, node, weight=edge_cost)
            cost = sum(edge_cost(u, v, G.edges[u, v]) for u, v in zip(path[:-1], path[1:]))
        except nx.NetworkXNoPath:
            path, cost = None, float("inf")
        cost_table.append((hid, cost))
        if cost < best_cost:
            best_hid, best_cost, best_path = hid, cost, path

    return best_hid, best_path, cost_table


def sample_background_traffic(G: nx.DiGraph, n_vehicles: int = 220, n_pedestrians: int = 60,
                               seed: int = 0):
    """혼잡도에 비례한 확률로 엣지를 골라 그 위에 무작위 차량 점을 뿌린다.
    (실제 교통 시뮬레이션이 아니라 발표용 시각적 장치)"""
    rng = np.random.default_rng(seed)
    edges = list(G.edges(data=True))
    weights = np.array([max(d.get("congestion", 0.2), 0.02) for _, _, d in edges])
    weights = weights / weights.sum()
    idx = rng.choice(len(edges), size=min(n_vehicles, len(edges)), replace=True, p=weights)

    veh_lng, veh_lat = [], []
    for i in idx:
        u, v, _ = edges[i]
        t = rng.uniform(0.15, 0.85)  # 엣지 위 임의 지점(양 끝 교차로에 겹치지 않게)
        lu, lv = G.nodes[u], G.nodes[v]
        veh_lng.append(lu["lng"] + t * (lv["lng"] - lu["lng"]))
        veh_lat.append(lu["lat"] + t * (lv["lat"] - lu["lat"]))

    # 시민(보행자)은 교차로 근처에 완전 무작위로
    nodes = list(G.nodes(data=True))
    pick = rng.choice(len(nodes), size=min(n_pedestrians, len(nodes)), replace=False)
    ped_lng = [nodes[i][1]["lng"] + rng.normal(0, 0.0008) for i in pick]
    ped_lat = [nodes[i][1]["lat"] + rng.normal(0, 0.0008) for i in pick]

    return (np.array(veh_lng), np.array(veh_lat)), (np.array(ped_lng), np.array(ped_lat))


def animate_showcase(G, path, cut, incident, station_id, hospital_id, cost_table,
                      out_gif=None, interval_ms=90, seed=0):
    pos = {n: (d["lng"], d["lat"]) for n, d in G.nodes(data=True)}
    fig, ax = plt.subplots(figsize=(10.5, 11))
    ax.set_aspect("equal")

    edges = list(G.edges())
    if len(edges) > 2500:
        rng0 = np.random.default_rng(0)
        edges = [edges[i] for i in rng0.choice(len(edges), size=2500, replace=False)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.12, width=0.4,
                            edge_color="#6b8e6b", arrows=False, ax=ax)

    # --- 배경: 무작위 차량/시민 (혼잡도 가중 샘플링) ---
    (veh_lng, veh_lat), (ped_lng, ped_lat) = sample_background_traffic(G, seed=seed)
    ax.scatter(veh_lng, veh_lat, s=7, c="dimgray", alpha=0.55, zorder=3, marker="o",
               label="vehicles (bg)")
    ax.scatter(ped_lng, ped_lat, s=5, c="steelblue", alpha=0.45, zorder=3, marker=".",
               label="pedestrians (bg)")

    # --- 소방서 전체 ---
    for s in DAEGU_FIRE_STATIONS:
        ax.plot(s["lng"], s["lat"], "s", color="navy", markersize=8, zorder=5)

    # --- 응급실 전체: 혼잡가중 비용에 따라 색상(초록=한산 ~ 빨강=혼잡) ---
    costs = dict(cost_table)
    finite = [c for c in costs.values() if np.isfinite(c)]
    cmin, cmax = (min(finite), max(finite)) if finite else (0, 1)

    def cost_color(c):
        if not np.isfinite(c):
            return "black"
        t = 0.0 if cmax <= cmin else (c - cmin) / (cmax - cmin)
        return plt.cm.RdYlGn_r(t)  # 초록(낮음/한산) -> 빨강(높음/혼잡)

    for h in DAEGU_ER_HOSPITALS:
        hid = h["hospital_id"]
        c = costs.get(hid, float("inf"))
        marker = "^" if hid != hospital_id else "*"
        size = 10 if hid != hospital_id else 20
        ax.plot(h["lng"], h["lat"], marker, color=cost_color(c), markersize=size,
                markeredgecolor="black", markeredgewidth=0.6, zorder=6)
        label = h["hospital_name"][:4] if _HAS_KR_FONT else HOSPITAL_EN[hid].split()[0]
        tag = f"{label}\n({c:.0f})" if np.isfinite(c) else label
        ax.annotate(tag, (h["lng"], h["lat"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=6.5, color="black")

    ax.plot(G.nodes[incident]["lng"], G.nodes[incident]["lat"],
            "*", color="black", markersize=18, zorder=7, label="incident")

    path_line, = ax.plot([], [], color="darkorange", lw=2.8, zorder=8)
    agent_pt, = ax.plot([], [], "o", color="lime", markersize=10, zorder=9,
                        markeredgecolor="black", markeredgewidth=0.8)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9.5,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.92))

    h_en = HOSPITAL_EN[hospital_id]
    s_en = STATION_EN[station_id]
    ax.set_title("Daegu live demo: random incident -> nearest station -> "
                 "LEAST-CONGESTED ER (auto-selected)", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(DAEGU_LNG_MIN, DAEGU_LNG_MAX)
    ax.set_ylim(DAEGU_LAT_MIN, DAEGU_LAT_MAX)
    ax.text(0.01, -0.045,
            "Gray/blue dots = simulated background vehicles/pedestrians (illustrative only). "
            "ER triangle color: green=least congested -> red=most congested. Star ER = chosen.",
            transform=ax.transAxes, fontsize=7.5, color="gray")

    xs = [pos[n][0] for n in path if n in pos]
    ys = [pos[n][1] for n in path if n in pos]

    def init():
        path_line.set_data([], [])
        agent_pt.set_data([], [])
        return path_line, agent_pt, status

    def update(frame):
        i = min(frame, len(xs) - 1)
        path_line.set_data(xs[: i + 1], ys[: i + 1])
        agent_pt.set_data([xs[i]], [ys[i]])
        if i <= cut:
            path_line.set_color("darkorange")
            phase = f"DISPATCH  {s_en} -> incident"
        else:
            path_line.set_color("purple")
            phase = f"TRANSPORT  incident -> {h_en} (auto-picked: least congested)"
        status.set_text(f"{phase}\nstep {i}/{len(xs)-1}")
        return path_line, agent_pt, status

    anim = FuncAnimation(fig, update, frames=len(xs), init_func=init,
                         interval=interval_ms, blit=False, repeat=False)
    if out_gif:
        anim.save(out_gif, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
        print(f"[showcase] saved {out_gif}")
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass
    return anim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--incident", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save", type=str, default="showcase.gif")
    parser.add_argument("--interval-ms", type=int, default=80)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    print("[showcase] Loading REAL Daegu road network (standard node-link)...")
    G, hospital_map = build_combined_graph(verbose=True, data_dir=args.data_dir)
    station_map = G.graph["station_node_map"]

    rng = np.random.default_rng(args.seed)
    reserved = set(station_map.values()) | set(hospital_map.values())
    candidates = [n for n in G.nodes if n not in reserved]
    center = [
        n for n in candidates
        if 35.82 <= G.nodes[n]["lat"] <= 35.92 and 128.52 <= G.nodes[n]["lng"] <= 128.68
    ]
    pool = center if center else candidates
    incident = args.incident if args.incident is not None else int(rng.choice(pool))

    sid = pick_nearest_station(G, incident, station_map)
    dispatch = congestion_shortest_path(G, station_map[sid], incident)

    print("[showcase] 8개 응급실 혼잡가중 비용 비교 중...")
    best_hid, transport, cost_table = pick_best_hospital(G, incident, hospital_map)
    if transport is None:
        raise SystemExit("사고지점에서 도달 가능한 응급실이 없습니다 (그래프 연결성 확인 필요)")

    path = dispatch + transport[1:]
    cut = len(dispatch) - 1

    print("=" * 60)
    print(f"[showcase] REAL Daegu roads: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    print(f"[showcase] INCIDENT node={incident} "
          f"({G.nodes[incident]['lat']:.5f}, {G.nodes[incident]['lng']:.5f})")
    s = next(x for x in DAEGU_FIRE_STATIONS if x["station_id"] == sid)
    print(f"[showcase] Nearest fire station: {s['station_name']}")
    print("[showcase] 응급실별 혼잡가중 비용 (낮을수록 안 막힘):")
    for hid, c in sorted(cost_table, key=lambda x: x[1]):
        h = next(x for x in DAEGU_ER_HOSPITALS if x["hospital_id"] == hid)
        marker = " <== 선택" if hid == best_hid else ""
        cost_str = f"{c:.1f}" if np.isfinite(c) else "도달불가"
        print(f"   [{hid}] {h['hospital_name']:20s} cost={cost_str}{marker}")
    print(f"[showcase] hops={len(path)-1}, dispatch_cut={cut}")
    print("=" * 60)

    animate_showcase(G, path, cut, incident, sid, best_hid, cost_table,
                      out_gif=args.save, interval_ms=args.interval_ms, seed=args.seed)


if __name__ == "__main__":
    main()
