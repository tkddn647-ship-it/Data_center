"""
policy.py
=========
"입력(state)과 마지막 출력층만 바꿔서 로보레이서로 재사용한다"는 목표를 위한
공용 body(특징추출기) + 문제별 head 구조.

Stable-Baselines3(SB3)는 policy_kwargs={"features_extractor_class": ...} 로
커스텀 특징추출기를 꽂을 수 있고, SB3 정책은 이 특징추출기 뒤에
자동으로 actor/critic head를 붙여준다. 즉:

    [ 이 파일의 SharedBodyExtractor ]  <- 문제마다 입력 차원만 다르게 생성자에 넘기면 됨
                    │
        SB3가 자동으로 붙여주는 actor head (Discrete 또는 Continuous)
        SB3가 자동으로 붙여주는 critic head (가치함수)

안심구역 대회 -> 로보레이서로 넘어갈 때 바뀌는 것은 딱 두 가지뿐이다.
  1) SharedBodyExtractor(observation_space) 생성자에 넘기는 obs 차원 (22 -> LiDAR 빔 수 등)
  2) PPO(...) -> SAC(...) 로 알고리즘 교체 (이산 행동 -> 연속 행동이므로)
   MLP body 자체(hidden_dim, 레이어 구성)는 그대로 재사용 가능.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = None
    BaseFeaturesExtractor = object


if _HAS_TORCH:

    class SharedBodyExtractor(BaseFeaturesExtractor):
        """문제(안심구역 경로탐색 / 로보레이서 주행)에 상관없이 재사용하는 MLP body.

        Parameters
        ----------
        observation_space : gymnasium.spaces.Box
            입력 상태 벡터 공간. 안심구역 버전은 22차원(HighwayRouteEnv),
            로보레이서 버전은 LiDAR 빔 수 + 속도/조향 등으로 차원만 바뀐다.
        hidden_dim : int
            은닉층 크기. 두 프로젝트에서 동일하게 128을 권장(과적합 방지+전이 용이성).
        """

        def __init__(self, observation_space, hidden_dim: int = 128, features_dim: int = 64):
            super().__init__(observation_space, features_dim=features_dim)
            input_dim = int(observation_space.shape[0])
            self.body = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations):
            return self.body(observations)

    def make_policy_kwargs(hidden_dim: int = 128, features_dim: int = 64) -> dict:
        """PPO/SAC 생성 시 policy_kwargs=make_policy_kwargs() 로 그대로 전달."""
        return {
            "features_extractor_class": SharedBodyExtractor,
            "features_extractor_kwargs": {"hidden_dim": hidden_dim, "features_dim": features_dim},
        }

else:
    def make_policy_kwargs(*args, **kwargs):
        raise ImportError(
            "torch / stable-baselines3 가 설치되어 있지 않습니다. "
            "실제 학습 환경(안심구역 분석PC 또는 로컬 GPU 머신)에서 "
            "`pip install torch stable-baselines3` 후 사용하세요."
        )
