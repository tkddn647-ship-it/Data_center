"""돌발 교통정보 API 스모크 테스트 (dgincident)."""
from daegu_open_data import fetch_incident_events, print_open_data_status

if __name__ == "__main__":
    print_open_data_status()
    try:
        ev = fetch_incident_events()
    except Exception as e:
        print("FAIL:", e)
        print("  포털에서 15126267 활용신청 후")
        print("  DAEGU_INCIDENT_API_URL=https://apis.data.go.kr/6270000/service/rest/dgincident")
        raise SystemExit(1)
    print(f"OK incidents={len(ev)}")
    for e in ev[:5]:
        print(f"  [{e.get('code')}] {e['lat']:.5f},{e['lng']:.5f} {(e.get('title') or '')[:60]}")
