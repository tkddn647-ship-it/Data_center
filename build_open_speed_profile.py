"""
build_open_speed_profile.py
===========================
공공데이터 '대구 교통 링크별 시간별 통계' CSV → hourly_speed_profile.json

사용:
  1) https://www.data.go.kr/data/15117329/fileData.do 에서 CSV 다운로드
  2) data/open/link_hourly_stats.csv 로 저장
  3) python build_open_speed_profile.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from daegu_open_data import (
    OPEN_DIR,
    PROFILE_PATH,
    build_hourly_speed_profile,
    load_link_hourly_stats,
    print_open_data_status,
    save_speed_profile,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="링크별 시간별 통계 CSV 경로")
    ap.add_argument("--out", type=str, default=str(PROFILE_PATH))
    args = ap.parse_args()

    print_open_data_status()
    df = load_link_hourly_stats(args.csv)
    profile = build_hourly_speed_profile(df)
    out = save_speed_profile(profile, Path(args.out))
    print(f"[build] rows={profile['n_rows']} roads={profile['n_roads']} → {out}")


if __name__ == "__main__":
    main()
