"""
env.py
======
'위험회피형 최적경로' 강화학습 환경 — HighwayRouteEnv

MDP 설계 (이 설계가 나중에 로보레이서로 그대로 옮겨감. README.md의 대응표 참고)
--------------------------------------------------------------------------
- State  : 현재 노드에서 갈 수 있는 최대 K개 이웃 구간의 [이동시간, 위험도, 혼잡도]
           + 이웃 유효성 마스크 + 목표까지 남은 거리(hop) 추정치 + 지금까지 누적한 위험도
- Action : K개 이웃 중 하나를 선택 (이산행동, Discrete(K))
- Reward : -이동시간비용 - λ*위험도비용 - 스텝페널티
           도착 시 + 도착보너스 / 막다른길·최대스텝 초과 시 -페널티

로보레이서 대응 (README 참고):
    상태(state)  : 그래프 이웃 특징 벡터   ->  LiDAR 거리 스캔
    행동(action) : 이웃 링크 선택(이산)    ->  조향각/가속(연속, SAC로 교체)
    보상(reward) : -시간 - λ*사고위험      ->  -랩타임 - λ*충돌/트랙이탈
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import networkx as nx

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:  # 안심구역 등 gymnasium이 없는 환경에서도 로직 테스트가 가능하도록 폴백
    _HAS_GYM = False
    gym = None
    spaces = None


class _NoGymBase:
    """gymnasium이 설치되지 않은 환경에서 reset()/step() 동작만 검증하기 위한 최소 폴백."""
    pass


_EnvBase = gym.Env if _HAS_GYM else _NoGymBase


class HighwayRouteEnv(_EnvBase):
    """networkx DiGraph 위에서 동작하는 위험회피 경로탐색 강화학습 환경."""

    MAX_NEIGHBORS = 5     # 행동 공간 크기 K
    MAX_STEPS = 30
    RISK_LAMBDA = 2.0      # 위험도에 대한 페널티 가중치 (커질수록 안전제일 경로 선호)
    INVALID_ACTION_PENALTY = -5.0
    GOAL_BONUS = 10.0
    STEP_PENALTY = -0.05

    def __init__(self, graph: nx.DiGraph, seed: Optional[int] = None):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.rng = np.random.default_rng(seed)

        obs_dim = self.MAX_NEIGHBORS * 3 + self.MAX_NEIGHBORS + 2
        if _HAS_GYM:
            self.observation_space = spaces.Box(low=-1.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(self.MAX_NEIGHBORS)
        self._obs_dim = obs_dim

        self.current_node = None
        self.goal_node = None
        self.steps = 0
        self.cum_risk = 0.0
        self._goal_hops_cache: dict = {}

    # ------------------------------------------------------------------ #
    # gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 출발-목적지가 실제로 경로 연결되어 있는 쌍만 선택
        for _ in range(50):
            start, goal = self.rng.choice(self.nodes, size=2, replace=False)
            if nx.has_path(self.graph, start, goal):
                self.current_node = int(start)
                self.goal_node = int(goal)
                break
        else:
            # 연결된 쌍을 못 찾으면 그래프에서 가장 큰 연결요소 안에서 강제로 선택
            largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
            start, goal = self.rng.choice(list(largest_cc), size=2, replace=False)
            self.current_node = int(start)
            self.goal_node = int(goal)

        self.steps = 0
        self.cum_risk = 0.0
        self._goal_hops_cache = {}
        obs = self._get_obs()
        info = {"start": self.current_node, "goal": self.goal_node}
        if _HAS_GYM:
            return obs, info
        return obs

    def step(self, action: int):
        self.steps += 1
        neighbors = list(self.graph.successors(self.current_node))[: self.MAX_NEIGHBORS]

        terminated = False
        truncated = False
        reward = self.STEP_PENALTY

        if action >= len(neighbors):
            # 존재하지 않는 이웃을 선택 = 로보레이서에서 벽에 부딪히는 것과 동일한 '위법 행동'
            reward += self.INVALID_ACTION_PENALTY
            terminated = True
        else:
            next_node = neighbors[action]
            edge = self.graph.edges[self.current_node, next_node]
            time_cost = edge["travel_time"] / 10.0   # 정규화
            risk_cost = edge["risk"]
            reward += -time_cost - self.RISK_LAMBDA * risk_cost
            self.cum_risk += risk_cost
            self.current_node = next_node

            if self.current_node == self.goal_node:
                reward += self.GOAL_BONUS
                terminated = True

        if self.steps >= self.MAX_STEPS and not terminated:
            truncated = True

        obs = self._get_obs()
        info = {
            "current_node": self.current_node,
            "goal_node": self.goal_node,
            "cum_risk": self.cum_risk,
        }
        if _HAS_GYM:
            return obs, reward, terminated, truncated, info
        return obs, reward, (terminated or truncated), info

    # ------------------------------------------------------------------ #
    # 내부 유틸
    # ------------------------------------------------------------------ #
    def _hops_to_goal(self, node: int) -> float:
        if node in self._goal_hops_cache:
            return self._goal_hops_cache[node]
        try:
            hops = nx.shortest_path_length(self.graph, node, self.goal_node)
        except nx.NetworkXNoPath:
            hops = self.MAX_STEPS
        self._goal_hops_cache[node] = hops
        return hops

    def _get_obs(self) -> np.ndarray:
        neighbors = list(self.graph.successors(self.current_node))[: self.MAX_NEIGHBORS]
        feat = np.zeros(self.MAX_NEIGHBORS * 3, dtype=np.float32)
        mask = np.zeros(self.MAX_NEIGHBORS, dtype=np.float32)
        for i, nb in enumerate(neighbors):
            e = self.graph.edges[self.current_node, nb]
            feat[i * 3 + 0] = e["travel_time"] / 10.0
            feat[i * 3 + 1] = e["risk"]
            feat[i * 3 + 2] = e["congestion"]
            mask[i] = 1.0
        hops_norm = min(self._hops_to_goal(self.current_node), self.MAX_STEPS) / self.MAX_STEPS
        obs = np.concatenate([feat, mask, [hops_norm], [min(self.cum_risk, 5.0) / 5.0]]).astype(np.float32)
        return obs

    def render(self):
        print(f"node={self.current_node} goal={self.goal_node} "
              f"steps={self.steps} cum_risk={self.cum_risk:.2f}")


class EmergencyRouteEnv(HighwayRouteEnv):
    """대구시 응급 골든타임 env — 구급차·소방차 공용.

    핵심 아이디어는 차종과 무관하게 동일하다: 혼잡도·위험도를 반영해
    "지금 이 목표까지 가장 빠르고 안전한 경로"를 실시간으로 찾는 것.
    차종에 따라 임무 구조만 달라진다.

    vehicle_type="ambulance" : 소방서(출동) → 사고현장 → 응급실(이송)  [2단계]
        - 병원 선정은 사람이 함(병상정보 부재). 이송 중 재탐색(reroute) 가능.
    vehicle_type="fire_truck" : 소방서(출동) → 화재현장                [1단계]
        - 현장 도착이 곧 임무 종료. 목표는 응급실이 아니라 화재/사고 현장 자체.

    두 차종은 같은 소방서(station_node_map)에서 출동한다 — 국내 119는
    소방과 구급을 같은 소방서가 함께 운영하므로 이는 현실을 그대로 반영한 것이다.
    같은 정책망이 vehicle_type을 상태의 일부(조건화 변수)로 받아 두 임무를
    동시에 학습한다(goal-conditioned 일반화의 연장).
    """

    MAX_STEPS = 80
    REROUTE_STEP_BONUS = 20
    PHASE_BONUS = 5.0   # 사고현장(구급차) 또는 화재현장 도착 보너스
    RISK_LAMBDA = 1.0   # 시내는 사고위험보다 혼잡·시간이 핵심 → 위험 가중 완화

    VEHICLE_TYPES = ("ambulance", "fire_truck")

    def __init__(self, graph: nx.DiGraph, hospital_node_map: Optional[dict] = None,
                 seed: Optional[int] = None, randomize_vehicle_type: bool = False):
        super().__init__(graph, seed=seed)
        self.hospital_node_map = hospital_node_map or {}
        self.station_node_map = dict(graph.graph.get("station_node_map", {}))
        if not self.station_node_map:
            # 안전망: 소방서 정보가 없으면 임의 노드를 거점으로
            self.station_node_map = {0: list(graph.nodes)[0]}
        self.randomize_vehicle_type = randomize_vehicle_type

        self.vehicle_type = "ambulance"
        self.target_hospital_id = None
        self.station_id = None
        self.incident_node = None  # 사고현장(구급차) / 화재현장(소방차) 공용 필드
        self.phase = "dispatch"
        self._max_steps_this_episode = self.MAX_STEPS
        self.phase_switch_index = None

        # 상태벡터에 vehicle_type 원핫(2차원)을 추가한다 — 부모 클래스가 만든
        # 22차원 관측(이웃15+마스크5+hops1+누적위험1)에 이어붙여 총 24차원.
        # bigbrain 아키텍처 설계(기획서 6.2.3절)의 목표조건화에서 vehicle_type이
        # 하나의 축으로 들어가는 것과 대응된다.
        obs_dim = self._obs_dim + len(self.VEHICLE_TYPES)
        if _HAS_GYM:
            self.observation_space = spaces.Box(low=-1.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self._obs_dim = obs_dim

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
              start_node: Optional[int] = None, vehicle_type: Optional[str] = None,
              target_hospital_id: Optional[int] = None, station_id: Optional[int] = None,
              incident_node: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 차종 결정: 명시 지정 > (randomize_vehicle_type이면 무작위) > 기본값 ambulance
        # 기본값을 ambulance로 둔 이유는 기존 데모(demo_live_incident.py 등)가
        # vehicle_type을 몰라도 그대로 동작해야 하기 때문 (하위호환).
        if vehicle_type is not None:
            self.vehicle_type = vehicle_type
        elif self.randomize_vehicle_type:
            self.vehicle_type = str(self.rng.choice(self.VEHICLE_TYPES))
        else:
            self.vehicle_type = "ambulance"
        if self.vehicle_type not in self.VEHICLE_TYPES:
            raise ValueError(f"알 수 없는 vehicle_type: {self.vehicle_type} (허용: {self.VEHICLE_TYPES})")

        # 출동 거점(소방서) — 구급차·소방차 공용
        sid = station_id if station_id is not None else int(self.rng.choice(list(self.station_node_map.keys())))
        self.station_id = sid
        station_node = int(self.station_node_map[sid])

        if self.vehicle_type == "ambulance":
            if not self.hospital_node_map:
                raise ValueError("vehicle_type='ambulance'에는 hospital_node_map이 필요합니다.")
            if target_hospital_id is None:
                target_hospital_id = int(self.rng.choice(list(self.hospital_node_map.keys())))
            self.target_hospital_id = target_hospital_id
            hospital_node = int(self.hospital_node_map[target_hospital_id])

            # 사고현장: ITS에 사고좌표가 없어 교차로 노드로 근사.
            # 소방서·병원 양쪽에서 도달 가능한 노드만 후보로 삼는다(2단계 임무이므로).
            if incident_node is not None:
                self.incident_node = int(incident_node)
            else:
                candidates = [
                    n for n in self.nodes
                    if n != station_node and n != hospital_node
                    and nx.has_path(self.graph, station_node, n)
                    and nx.has_path(self.graph, n, hospital_node)
                ]
                if not candidates:
                    candidates = [n for n in self.nodes if n != station_node]
                self.incident_node = int(self.rng.choice(candidates))
        else:
            # 소방차: 목표 병원 개념이 없다 — 화재현장 도착이 곧 임무 종료(1단계).
            self.target_hospital_id = None
            if incident_node is not None:
                self.incident_node = int(incident_node)
            else:
                candidates = [
                    n for n in self.nodes
                    if n != station_node and nx.has_path(self.graph, station_node, n)
                ]
                if not candidates:
                    candidates = [n for n in self.nodes if n != station_node]
                self.incident_node = int(self.rng.choice(candidates))

        self.current_node = int(start_node) if start_node is not None else station_node
        self.phase = "dispatch"
        self.goal_node = self.incident_node
        self.phase_switch_index = None
        self.steps = 0
        self.cum_risk = 0.0
        self._goal_hops_cache = {}
        self._max_steps_this_episode = self.MAX_STEPS

        obs = self._get_obs()
        info = {
            "start": self.current_node,
            "goal": self.goal_node,
            "phase": self.phase,
            "vehicle_type": self.vehicle_type,
            "incident_node": self.incident_node,
            "target_hospital_id": self.target_hospital_id,
            "station_id": self.station_id,
        }
        if _HAS_GYM:
            return obs, info
        return obs

    def reroute(self, new_hospital_id: int):
        """이송 단계 mid-flight에서 응급실만 변경 (구급차 전용).

        소방차는 임무가 1단계(현장 도착=종료)라 재탐색 대상 목표가 없다 —
        화재현장 좌표 자체가 바뀌는 경우는 새 incident_node로 reset()하는 것이
        맞으므로 reroute는 구급차 전용으로 제한한다.
        """
        if self.vehicle_type != "ambulance":
            raise ValueError(
                "reroute()는 vehicle_type='ambulance'의 이송(transport) 단계 전용입니다. "
                "소방차는 화재현장 좌표가 바뀌면 reset(vehicle_type='fire_truck', incident_node=...)을 사용하세요."
            )
        if new_hospital_id not in self.hospital_node_map:
            raise ValueError(f"알 수 없는 hospital_id: {new_hospital_id}")
        self.target_hospital_id = new_hospital_id
        new_goal = int(self.hospital_node_map[new_hospital_id])
        if self.phase == "transport":
            self.goal_node = new_goal
        # dispatch 중이면 현장 도착 후 새 병원이 적용되도록 id만 갱신
        self._goal_hops_cache = {}
        self._max_steps_this_episode += self.REROUTE_STEP_BONUS
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1
        neighbors = list(self.graph.successors(self.current_node))[: self.MAX_NEIGHBORS]

        terminated = False
        truncated = False
        reward = self.STEP_PENALTY

        if action >= len(neighbors):
            reward += self.INVALID_ACTION_PENALTY
            terminated = True
        else:
            next_node = neighbors[action]
            edge = self.graph.edges[self.current_node, next_node]
            time_cost = edge["travel_time"] / 10.0
            # 시내: 혼잡을 위험처럼 페널티 (골든타임 = 시간)
            congestion_cost = float(edge.get("congestion", 0.0))
            risk_cost = edge.get("risk", 0.3)
            reward += -time_cost - self.RISK_LAMBDA * risk_cost - 0.8 * congestion_cost
            self.cum_risk += risk_cost
            self.current_node = next_node

            if self.vehicle_type == "ambulance":
                if self.phase == "dispatch" and self.current_node == self.incident_node:
                    # 사고현장 도착 → 이송 단계로 전환
                    reward += self.PHASE_BONUS
                    self.phase = "transport"
                    self.goal_node = int(self.hospital_node_map[self.target_hospital_id])
                    self.phase_switch_index = self.steps
                    self._goal_hops_cache = {}
                elif self.phase == "transport" and self.current_node == self.goal_node:
                    reward += self.GOAL_BONUS
                    terminated = True
            else:
                # 소방차: 화재현장 도착 = 임무 종료 (1단계이므로 phase 전환 없음)
                if self.current_node == self.goal_node:
                    reward += self.GOAL_BONUS
                    terminated = True

        if self.steps >= self._max_steps_this_episode and not terminated:
            truncated = True

        obs = self._get_obs()
        info = {
            "current_node": self.current_node,
            "goal_node": self.goal_node,
            "phase": self.phase,
            "vehicle_type": self.vehicle_type,
            "cum_risk": self.cum_risk,
            "target_hospital_id": self.target_hospital_id,
            "phase_switch_index": self.phase_switch_index,
        }
        if _HAS_GYM:
            return obs, reward, terminated, truncated, info
        return obs, reward, (terminated or truncated), info

    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()  # 부모: 이웃15+마스크5+hops1+누적위험1 = 22차원
        vt = np.zeros(len(self.VEHICLE_TYPES), dtype=np.float32)
        vt[self.VEHICLE_TYPES.index(self.vehicle_type)] = 1.0
        return np.concatenate([base, vt]).astype(np.float32)


# 하위호환: 기존 코드(demo_live_incident.py 등)가 AmbulanceRouteEnv를 그대로
# import해도 동작하도록 별칭을 남긴다. 신규 코드는 EmergencyRouteEnv를 직접 사용.
AmbulanceRouteEnv = EmergencyRouteEnv


if __name__ == "__main__":
    from network_graph import build_combined_graph

    G, hosp_map = build_combined_graph(verbose=False)

    print("=== 구급차 미션 (2단계: 출동→현장→이송) ===")
    env = EmergencyRouteEnv(G, hosp_map, seed=0)
    reset_out = env.reset(vehicle_type="ambulance", target_hospital_id=0, station_id=0)
    obs = reset_out[0] if _HAS_GYM else reset_out
    print(f"phase={env.phase} start={env.current_node} incident={env.incident_node} "
          f"hospital={env.hospital_node_map[0]}")

    total_reward = 0.0
    for _ in range(env.MAX_STEPS):
        n_valid = len(list(env.graph.successors(env.current_node))[: env.MAX_NEIGHBORS])
        action = int(np.random.randint(0, max(n_valid, 1)))
        result = env.step(action)
        if _HAS_GYM:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        total_reward += reward
        if env.phase_switch_index == env.steps:
            print(f"[현장도착] step={env.steps}, 이제 병원으로 phase={env.phase}")
        if done:
            break
    print(f"done phase={env.phase} reward={total_reward:.3f} "
          f"at_hospital={env.current_node == env.hospital_node_map[env.target_hospital_id]}")

    print("\n=== 소방차 미션 (1단계: 출동→화재현장) ===")
    env2 = EmergencyRouteEnv(G, hosp_map, seed=1)
    reset_out2 = env2.reset(vehicle_type="fire_truck", station_id=2)
    obs2 = reset_out2[0] if _HAS_GYM else reset_out2
    print(f"obs_dim={obs2.shape}, phase={env2.phase} start={env2.current_node} "
          f"fire_scene={env2.incident_node}")

    total_reward2 = 0.0
    for _ in range(env2.MAX_STEPS):
        n_valid = len(list(env2.graph.successors(env2.current_node))[: env2.MAX_NEIGHBORS])
        action = int(np.random.randint(0, max(n_valid, 1)))
        result = env2.step(action)
        if _HAS_GYM:
            obs2, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs2, reward, done, info = result
        total_reward2 += reward
        if done:
            break
    print(f"done reward={total_reward2:.3f} at_fire_scene={env2.current_node == env2.incident_node}")
