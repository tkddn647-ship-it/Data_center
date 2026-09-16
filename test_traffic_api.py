"""실시간 교통소통 API 스모크 테스트 (키·URL은 .env)."""
from daegu_open_data import fetch_realtime_traffic, print_open_data_status

print_open_data_status()
try:
    items = fetch_realtime_traffic()
except Exception as e:
    print("FAIL:", e)
    print()
    print("포털에서 확인할 것:")
    print("  1) https://www.data.go.kr/data/15126266/openapi.do")
    print("  2) [상세기능] 탭 → '요청주소' 전체 복사")
    print("  3) .env 의 DAEGU_TRAFFIC_API_URL=그주소")
    raise SystemExit(1)

print(f"OK items={len(items)}")
print("api_url", items[0].get("api_url"))
print("sample_keys", sorted({k for it in items[:3] for k in (it.get("raw") or {}).keys()}))
print("sample", {k: items[0][k] for k in ("link_id", "speed", "road_name", "start_node_id", "end_node_id")})
