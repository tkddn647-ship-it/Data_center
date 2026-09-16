"""선택: OSRM으로 시내 구간 도로곡선 캐시 생성.

  set ROAD_SHAPES_FETCH=1
  python build_road_shapes.py
"""
from network_graph import build_combined_graph
from congestion_time import TimeCongestionField
from road_shapes import ensure_shapes_for_edges, should_draw_on_costmap, _load_cache

G, _ = build_combined_graph(verbose=True)
F = TimeCongestionField.from_graph(G, sample_edges=3000)
edges = [e for e in F.edge_list if should_draw_on_costmap(G, *e)]
print("drawable", len(edges))
ensure_shapes_for_edges(G, edges, max_fetch=400, verbose=True)
c = _load_cache()
print("cache", len(c), "polylines>2", sum(1 for v in c.values() if isinstance(v, list) and len(v) > 2))
