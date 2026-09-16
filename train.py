"""
train.py
========
EmergencyRouteEnv(고속도로+대구시내 통합 그래프) 위에서 PPO(이산행동) 에이전트를 학습한다.

2026-09 업데이트: 구급차(2단계: 출동→현장→이송)와 소방차(1단계: 출동→화재현장)를
**하나의 정책**이 함께 학습한다. 매 에피소드마다 vehicle_type과 목표(병원 또는
화재현장)가 무작위로 바뀐다(goal-conditioned 학습) — 정책이 "이 병원 하나로 가는 법"이
아니라 "차종과 목표가 무엇이든 혼잡·위험을 피해 최적경로를 찾는 법"을 배우게 하기 위함.

병원 선정 자체는 실시간 병상 데이터가 없어 구급대원이 수동으로 하고,
정책은 "어느 목표가 주어지든" 최적경로를 찾도록 학습된다.
학습된 정책은 이송 도중 목표가 바뀌어도(env.reroute(), 구급차 한정) 재학습 없이 바로 대응한다.

실행 전 준비 (안심구역 분석 PC 또는 로컬 머신에서):
    pip install gymnasium stable-baselines3 torch

실행:
    python train.py --timesteps 300000

안심구역 실제 데이터로 교체하려면:
    data_schema.py 의 load_* 함수 내부만 실제 CSV 경로로 바꾸면
    이 스크립트는 수정할 필요가 없다.
"""

from __future__ import annotations
import argparse

from network_graph import build_combined_graph
from env import EmergencyRouteEnv
from policy import make_policy_kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-links", type=int, default=200)
    parser.add_argument("--n-intersections", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="emergency_ppo.zip")
    parser.add_argument("--vehicle-type", type=str, default=None, choices=["ambulance", "fire_truck"],
                         help="지정하면 그 차종만 학습(단일 미션). 생략하면 두 차종을 무작위로 섞어 " +
                              "하나의 정책으로 함께 학습(권장, goal-conditioned 일반화).")
    parser.add_argument("--data-dir", type=str, default=None,
                         help="안심구역 실제 데이터 폴더 경로. 지정하지 않으면 합성 데이터로 실행.")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as e:
        raise SystemExit(
            "stable-baselines3 / gymnasium 이 설치되어 있지 않습니다.\n"
            "  pip install gymnasium stable-baselines3 torch\n"
            f"(원본 에러: {e})"
        )

    print("[train] 대구시 응급 골든타임 그래프(경북대 ITS) 구성 중...")
    if args.data_dir:
        print(f"[train] 실제 데이터 모드: {args.data_dir}")
    graph, hospital_node_map = build_combined_graph(
        n_intersections=args.n_intersections, verbose=True,
        data_dir=args.data_dir,
    )

    randomize = args.vehicle_type is None
    if randomize:
        print("[train] 차종 무작위 혼합 학습: 구급차(2단계) + 소방차(1단계)를 한 정책으로 학습합니다.")
    else:
        print(f"[train] 단일 차종 학습: {args.vehicle_type}")

    def _env_fn():
        env = EmergencyRouteEnv(graph, hospital_node_map, seed=args.seed,
                                 randomize_vehicle_type=randomize)
        if not randomize:
            # 고정 차종 학습을 위해 reset()에 vehicle_type을 강제하는 얇은 래퍼
            orig_reset = env.reset
            env.reset = lambda **kw: orig_reset(vehicle_type=args.vehicle_type, **kw)
        return env

    env = make_vec_env(_env_fn, n_envs=4, seed=args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=make_policy_kwargs(hidden_dim=128, features_dim=64),
        verbose=1,
        seed=args.seed,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )

    print(f"[train] 학습 시작: timesteps={args.timesteps}")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    print(f"[train] 모델 저장 완료: {args.save_path}")


if __name__ == "__main__":
    main()
