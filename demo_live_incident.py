"""
demo_live_incident.py
=====================
**실제 대구 도로망(표준노드링크)** + **실제 병원·소방서 좌표** 위에서
임의 사고 지점 → 출동 → 응급실 경로를 실시간처럼 재생.

실행:
    python demo_live_incident.py --seed 7 --hospital 0 --save live_incident.gif --no-show
    python demo_live_incident.py --hospital 1 --seed 3
"""

from __future__ import annotations
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter

from network_graph import build_combined_graph
from daegu_ems_geo import DAEGU_ER_HOSPITALS, DAEGU_FIRE_STATIONS
from env import AmbulanceRouteEnv, _HAS_GYM
from standard_node_link import DAEGU_LAT_MIN, DAEGU_LAT_MAX, DAEGU_LNG_MIN, DAEGU_LNG_MAX

# Windows 한글 폰트 (없으면 영문 라벨로 폴백)
def _setup_kr_font():
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path and "DejaVu" not in path:
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                return True
        except Exception:
            continue
    return False


_HAS_KR_FONT = _setup_kr_font()

# GIF/콘솔용 짧은 영문 표기
HOSPITAL_EN = {
    0: "Kyungpook Nat'l Univ Hosp",
    1: "Yeungnam Univ Hosp",
    2: "Keimyung Dongsan Hosp",
    3: "Daegu Catholic Univ Hosp",
    4: "Chilgok Kyungpook Hosp",
    5: "Daegu Fatima Hosp",
    6: "Daegu Medical Center",
    7: "Daegu Catholic (Suseong)",
}
STATION_EN = {
    0: "Jungbu FS",
    1: "Dongbu FS",
    2: "Seobu FS",
    3: "Bukbu FS",
    4: "Suseong FS",
    5: "Dalseo FS",
    6: "Dalseong FS",
    7: "Nambu FS",
}

def congestion_shortest_path(G: nx.DiGraph, src: int, dst: int) -> list:
    def w(u, v, d):
        return float(d.get("travel_time", 1.0)) * (1.0 + float(d.get("congestion", 0.0)))
    try:
        return nx.shortest_path(G, src, dst, weight=w)
    except nx.NetworkXNoPath:
        return [src]


def pick_nearest_station(G: nx.DiGraph, incident: int, station_map: dict) -> int:
    best_sid, best_d = None, float("inf")
    ilat, ilng = G.nodes[incident]["lat"], G.nodes[incident]["lng"]
    for sid, nid in station_map.items():
        d = (G.nodes[nid]["lat"] - ilat) ** 2 + (G.nodes[nid]["lng"] - ilng) ** 2
        if d < best_d:
            best_sid, best_d = sid, d
    return int(best_sid)


def build_full_mission_path(G, hospital_map, station_map, incident, hospital_id, model=None, seed=0):
    sid = pick_nearest_station(G, incident, station_map)
    station_node = station_map[sid]
    hospital_node = hospital_map[hospital_id]
    if model is None:
        dispatch = congestion_shortest_path(G, station_node, incident)
        transport = congestion_shortest_path(G, incident, hospital_node)
        return dispatch + transport[1:], len(dispatch) - 1, sid

    env = AmbulanceRouteEnv(G, hospital_map, seed=seed)
    env.reset(station_id=sid, target_hospital_id=hospital_id, incident_node=incident)
    path = [env.current_node]
    for _ in range(200):
        neighbors = list(env.graph.successors(env.current_node))[: env.MAX_NEIGHBORS]
        if not neighbors:
            break
        scores = []
        for i, nb in enumerate(neighbors):
            try:
                hops = nx.shortest_path_length(env.graph, nb, env.goal_node)
            except nx.NetworkXNoPath:
                hops = 999
            cong = env.graph.edges[env.current_node, nb].get("congestion", 0.5)
            scores.append((hops + 0.5 * cong, i))
        action = min(scores)[1]
        step_out = env.step(action)
        done = step_out[2] if not _HAS_GYM else (step_out[2] or step_out[3])
        path.append(env.current_node)
        if done:
            break
    cut = env.phase_switch_index if env.phase_switch_index is not None else len(path) // 2
    return path, cut, sid


def animate(G, path, cut, incident, station_id, hospital_id, out_gif=None, interval_ms=120):
    pos = {n: (d["lng"], d["lat"]) for n, d in G.nodes(data=True)}
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_aspect("equal")

    # 실제 대구 도로망 (샘플링해 그리기 속도 확보)
    edges = list(G.edges())
    if len(edges) > 2500:
        rng = np.random.default_rng(0)
        edges = [edges[i] for i in rng.choice(len(edges), size=2500, replace=False)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.12, width=0.4,
                            edge_color="#6b8e6b", arrows=False, ax=ax)

    # 실제 시설 좌표로 마커 (스냅 노드가 아니라 진짜 병원/서 위치)
    for s in DAEGU_FIRE_STATIONS:
        ax.plot(s["lng"], s["lat"], "s", color="navy", markersize=8, zorder=5)
    for h in DAEGU_ER_HOSPITALS:
        ax.plot(h["lng"], h["lat"], "^", color="crimson", markersize=9, zorder=5)
        label = h["hospital_name"][:4] if _HAS_KR_FONT else HOSPITAL_EN[h["hospital_id"]].split()[0]
        ax.annotate(label, (h["lng"], h["lat"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="crimson")

    ax.plot(G.nodes[incident]["lng"], G.nodes[incident]["lat"],
            "*", color="black", markersize=18, zorder=6, label="incident")

    path_line, = ax.plot([], [], color="darkorange", lw=2.8, zorder=7)
    agent_pt, = ax.plot([], [], "o", color="lime", markersize=9, zorder=8)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    h_en = HOSPITAL_EN[hospital_id]
    s_en = STATION_EN[station_id]
    ax.set_title("Daegu REAL roads + real ER/fire stations", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(DAEGU_LNG_MIN, DAEGU_LNG_MAX)
    ax.set_ylim(DAEGU_LAT_MIN, DAEGU_LAT_MAX)
    ax.text(0.01, -0.05,
            "Green=Daegu standard node-link | Navy=fire station | "
            "Red triangle=ER | Black star=incident",
            transform=ax.transAxes, fontsize=8, color="gray")

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
            phase = f"TRANSPORT  incident -> {h_en}"
        status.set_text(f"{phase}\nstep {i}/{len(xs)-1}")
        return path_line, agent_pt, status

    anim = FuncAnimation(fig, update, frames=len(xs), init_func=init,
                         interval=interval_ms, blit=False, repeat=False)
    if out_gif:
        anim.save(out_gif, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
        print(f"[demo] saved {out_gif}")
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass
    return anim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hospital", type=int, default=0)
    parser.add_argument("--incident", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="경북대 ITS (혼잡 오버레이)")
    parser.add_argument("--save", type=str, default="live_incident.gif")
    parser.add_argument("--interval-ms", type=int, default=80)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    print("[demo] Loading REAL Daegu road network (standard node-link)...")
    G, hospital_map = build_combined_graph(verbose=True, data_dir=args.data_dir)
    station_map = G.graph["station_node_map"]

    rng = np.random.default_rng(args.seed)
    reserved = set(station_map.values()) | set(hospital_map.values())
    candidates = [n for n in G.nodes if n not in reserved]
    # 도심 쪽 사고 비중을 높이려고 중심부 bbox 우선
    center = [
        n for n in candidates
        if 35.82 <= G.nodes[n]["lat"] <= 35.92 and 128.52 <= G.nodes[n]["lng"] <= 128.68
    ]
    pool = center if center else candidates
    incident = args.incident if args.incident is not None else int(rng.choice(pool))

    path, cut, sid = build_full_mission_path(
        G, hospital_map, station_map, incident, args.hospital, seed=args.seed,
    )

    print("=" * 60)
    print(f"[demo] REAL Daegu roads: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    print(f"[demo] INCIDENT node={incident} "
          f"({G.nodes[incident]['lat']:.5f}, {G.nodes[incident]['lng']:.5f})")
    s = next(x for x in DAEGU_FIRE_STATIONS if x["station_id"] == sid)
    h = next(x for x in DAEGU_ER_HOSPITALS if x["hospital_id"] == args.hospital)
    print(f"[demo] Fire station: {s['station_name']}")
    print(f"[demo] ER hospital:  {h['hospital_name']}")
    print(f"[demo] hops={len(path)-1}, dispatch_cut={cut}")
    print("=" * 60)

    animate(G, path, cut, incident, sid, args.hospital,
            out_gif=args.save, interval_ms=args.interval_ms)


if __name__ == "__main__":
    main()
