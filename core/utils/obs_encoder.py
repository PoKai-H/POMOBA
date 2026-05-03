from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _safe_bool(value: Any) -> float:
    return float(bool(value))


def _encode_team(team: Any) -> List[float]:
    if team in {"ally", "blue"}:
        return [1.0, 0.0, 0.0]
    if team in {"enemy", "red"}:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


@dataclass
class ObsEncoderConfig:
    max_agents: int = 3 # at most 3 other agents (2v2)
    max_objects: int = 7 # at most 3 minions from each team + 1 tower
    include_timestep: bool = True
    # Optional experiment hook for future tracking/belief work.
    # PPO should usually stay with the default slot-based encoding.
    include_ids: bool = False
    # Only used when include_ids=True.
    normalize_ids: bool = True
    # Only used when include_ids=True and normalize_ids=True.
    id_scale: float = 10.0
    timestep_scale: float = 100.0
    position_scale: float = 10.0
    hp_scale: float = 1.0


class ObservationEncoder:
    """Encode nested observations into fixed-size float32 vectors for PPO.

    By default this encoder is slot-based: entities are represented by fixed
    slots and their features, not by raw ids. Id encoding is kept only as an
    optional future experiment hook.

    Output: length = 106 vector
    """

    def __init__(self, config: ObsEncoderConfig | None = None):
        self.config = config or ObsEncoderConfig()

        self.self_dim = 9 if self.config.include_ids else 8
        self.agent_dim = 10 if self.config.include_ids else 9
        self.object_dim = 11 if self.config.include_ids else 10
        self.obs_dim = (
            (1 if self.config.include_timestep else 0)
            + self.self_dim
            + self.config.max_agents * self.agent_dim
            + self.config.max_objects * self.object_dim
        )

    def encode(self, obs: Dict[str, Any], array_lib=np):
        features: List[float] = []

        if self.config.include_timestep:
            timestep = _safe_float(obs.get("timestep", 0))
            features.append(timestep / self.config.timestep_scale)

        features.extend(self._encode_self(obs.get("self", {})))
        features.extend(self._encode_agents(obs.get("agents", [])))
        features.extend(self._encode_objects(obs.get("objects", [])))

        return array_lib.asarray(features, dtype=array_lib.float32)

    def batch_encode(self, observations: Sequence[Dict[str, Any]], array_lib=np):
        encoded = [self.encode(obs, array_lib=np) for obs in observations]
        return array_lib.asarray(encoded, dtype=array_lib.float32)

    def _encode_self(self, self_obs: Dict[str, Any]) -> List[float]:
        """
        Output:
        [
            team_ally,
            team_enemy,
            team_unknown,
            hp,
            pos_x,
            pos_y,
            alive,
            mask,
            ]
        """
        position = self_obs.get("position") or [0.0, 0.0]
        status = self_obs.get("status", {})
        hp = self_obs.get("hp", status.get("hp"))

        features = [
            *_encode_team(self_obs.get("team")),
            _safe_float(hp) / self.config.hp_scale,
            _safe_float(position[0]) / self.config.position_scale,
            _safe_float(position[1]) / self.config.position_scale,
            _safe_bool(status.get("alive")),
            1.0,
        ]

        if self.config.include_ids:
            return [self._encode_id(self_obs.get("id")), *features]

        return features

    def _encode_agents(self, agents: Sequence[Dict[str, Any]]) -> List[float]:
        """
        Output:
        [
            team_ally,
            team_enemy,
            team_unknown,
            visible,
            hp,
            rel_x,
            rel_y,
            alive,
            mask,
        ]
        """
        encoded: List[float] = []

        for agent in list(agents)[: self.config.max_agents]:
            relative_position = agent.get("relative_position") or [0.0, 0.0]
            status = agent.get("status", {})
            visible = bool(agent.get("visible", False))

            encoded.extend(
                self._agent_features(agent, visible, relative_position, status)
            )

        missing = self.config.max_agents - min(len(agents), self.config.max_agents)
        if missing > 0:
            encoded.extend([0.0] * (missing * self.agent_dim))

        return encoded

    def _encode_objects(self, objects: Sequence[Dict[str, Any]]) -> List[float]:
        encoded: List[float] = []

        for obj in list(objects)[: self.config.max_objects]:
            relative_position = obj.get("relative_position") or [0.0, 0.0]
            status = obj.get("status", {})
            visible = bool(obj.get("visible", False))

            encoded.extend(
                self._object_features(obj, visible, relative_position, status)
            )

        missing = self.config.max_objects - min(len(objects), self.config.max_objects)
        if missing > 0:
            encoded.extend([0.0] * (missing * self.object_dim))

        return encoded

    def _encode_id(self, value: Any) -> float:
        if not self.config.normalize_ids:
            return _safe_float(value)
        return _safe_float(value) / self.config.id_scale

    def _agent_features(
        self,
        agent: Dict[str, Any],
        visible: bool,
        relative_position: Sequence[Any],
        status: Dict[str, Any],
    ) -> List[float]:
        features = [
            *_encode_team(agent.get("team")),
            _safe_bool(visible),
            (_safe_float(agent.get("hp", status.get("hp"))) / self.config.hp_scale) if visible else 0.0,
            (_safe_float(relative_position[0]) / self.config.position_scale) if visible else 0.0,
            (_safe_float(relative_position[1]) / self.config.position_scale) if visible else 0.0,
            _safe_bool(status.get("alive")),
            1.0,
        ]

        if self.config.include_ids:
            return [self._encode_id(agent.get("id")), *features]

        return features

    def _object_features(
        self,
        obj: Dict[str, Any],
        visible: bool,
        relative_position: Sequence[Any],
        status: Dict[str, Any],
    ) -> List[float]:
        hp = obj.get("hp", status.get("hp"))
        features = [
            *_encode_team(obj.get("team")),
            _safe_bool(visible),
            self._encode_object_type(obj.get("type")),
            (_safe_float(hp) / self.config.hp_scale) if visible else 0.0,
            (_safe_float(relative_position[0]) / self.config.position_scale) if visible else 0.0,
            (_safe_float(relative_position[1]) / self.config.position_scale) if visible else 0.0,
            _safe_bool(status.get("alive")),
            1.0,
        ]

        if self.config.include_ids:
            return [self._encode_id(obj.get("id")), *features]

        return features

    def _encode_object_type(self, value: Any) -> float:
        object_types = {
            "tower": 1.0,
            "minion": 2.0,
            "ward": 3.0,
            "monster": 4.0,
        }
        return object_types.get(value, 0.0)



