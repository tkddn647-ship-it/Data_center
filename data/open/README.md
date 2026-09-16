# 대구 오픈데이터 배치 폴더

공공데이터포털에서 받은 파일을 여기에 둡니다.

| 파일 | 출처 | 용도 |
|---|---|---|
| `../standard_node_link/daegu_nodes.csv` | [ITS SHP](https://www.its.go.kr/nodelink/nodelinkRef) → `build_dense_backbone.py` (또는 [포털 노드](https://www.data.go.kr/data/15049953/fileData.do)) | 도로 교차점 (~37k) |
| `../standard_node_link/daegu_links.csv` | 동일 (또는 [포털 링크](https://www.data.go.kr/data/15049952/fileData.do)) | 도로 연결 (~52k) |
| `link_hourly_stats.csv` | [링크 시간별 통계](https://www.data.go.kr/data/15117329/fileData.do) | 시간대 속도→혼잡 예측 |
| `fire_stations.csv` | 소방청 전국소방서 좌표 등 | 출동 출발지(본부) |
| `safety_centers_119.csv` | [119안전센터 SHP](https://www.data.go.kr/data/15117114/fileData.do) | 더 가까운 출발(센터) |
| `emergency_entrances_daegu.csv` | [긴급차 진출입로](https://www.data.go.kr/data/15156867/fileData.do) | 아파트 현장→입구 스냅 |
| `station_fleet.csv` | [안전센터별 구급차 현황](https://www.data.go.kr/data/15066554/fileData.do) 스키마 시드 | 서별 구급·소방 **보유 대수** (출동 시 차감) |
| `er_hospitals.csv` | [응급의료기관 현황](https://www.data.go.kr/data/15132528/fileData.do) + 공개좌표 | 이송 목적지 |
| `hourly_speed_profile.json` | `python build_open_speed_profile.py` 로 생성 | 런타임 캐시 |

### 필수 작업

1. 포털에서 **링크별 시간별 통계** CSV를 받아 `link_hourly_stats.csv` 로 **교체** (현재 파일은 파이프라인용 샘플)
2. (선택) [교통소통정보 API](https://www.data.go.kr/data/15126266/openapi.do) 활용신청 후  
   `.env`에 키 + **상세기능 요청주소** 설정:
   ```bash
   DAEGU_TRAFFIC_API_KEY=...
   DAEGU_TRAFFIC_API_URL=https://apis.data.go.kr/6270000/service/rest1/<연산명>
   ```
   End Point(`.../rest1`)만 있으면 안 되고, 포털 **상세기능**의 요청주소 전체가 필요합니다.
3. `python build_open_speed_profile.py`
4. 실시간 연결 확인: `python test_traffic_api.py`
5. (선택) 지도 곡선·끊김 완화: `python build_road_shapes.py` → `edge_shapes.json`  
   costmap 직선 길이 한도: 환경변수 `COSTMAP_MAX_CHORD_M` (기본 480)
6. 도로망을 ITS로 다시 뽑을 때: 루트 `README.md` **「지금 상태」** + `python build_dense_backbone.py`

응급의료 원본 CSV에는 좌표가 없으므로, 공개 주소 기준 위도·경도 컬럼을 붙인 `er_hospitals.csv` 형식을 사용합니다.

지도 **파랑→빨강**이 끊겨 보이는 이유·완화는 루트 `README.md` **§4-3** 을 보면 된다.
