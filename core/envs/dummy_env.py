import numpy as np


class DummyEnv:
    """Minimal multi-agent env shaped like the intended Godot contract."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.max_steps = 50
        self.agents_per_team = 1
        self.num_envs = 2
        self.agent_states = []
        self.minion_states = []
        self.tower_states = []
        self.vision_radius = 3.0
        self.action_space = {"action": {"size": 12, "action_type": "discrete"}}

    def reset(self, config):
        core_config = (config or {}).get("core", {})
        self.t = 0
        self.max_steps = int(core_config.get("max_steps", 50))
        self.agents_per_team = int(core_config.get("agents_per_team", 1))
        self.num_envs = self.agents_per_team * 2
        self.agent_states = self._build_agent_states()
        self.minion_states = self._build_minion_states()
        self.tower_states = self._build_tower_states()

        obs_list = [self._wrap_obs(self._get_obs_for_agent(i)) for i in range(self.num_envs)]
        info_list = [{} for _ in range(self.num_envs)]
        return obs_list, info_list

    def step(self, action):
        self.t += 1
        action_indices = self._extract_actions(action)

        for idx, action_index in enumerate(action_indices):
            move = self._action_to_delta(action_index)
            self.agent_states[idx]["position"] += move
            self.agent_states[idx]["last_action"] = int(action_index)

        self._apply_attack_actions(action_indices)
        self._step_minions()

        obs_list = [self._wrap_obs(self._get_obs_for_agent(i)) for i in range(self.num_envs)]
        reward_list = [self._get_reward_for_agent(i) for i in range(self.num_envs)]
        done = self.t >= self.max_steps
        done_list = [done for _ in range(self.num_envs)]
        truncated_list = [done for _ in range(self.num_envs)]
        info_list = [self._get_info_for_agent(i) for i in range(self.num_envs)]
        return obs_list, reward_list, done_list, truncated_list, info_list

    def close(self):
        return None

    def _build_agent_states(self):
        agent_states = []
        team_names = ("blue", "red")

        for team_name in team_names:
            base_x = -5.0 if team_name == "blue" else 5.0
            for slot in range(self.agents_per_team):
                y = (slot - (self.agents_per_team - 1) / 2.0) * 3.0
                agent_states.append(
                    {
                        "id": len(agent_states) + 1,
                        "team": team_name,
                        "hp": 100.0,
                        "alive": True,
                        "position": np.array([base_x, y], dtype=np.float32),
                        "last_action": 8,
                    }
                )

        return agent_states

    def _build_minion_states(self):
        minion_states = []
        team_names = ("blue", "red")

        for team_name in team_names:
            base_x = -2.0 if team_name == "blue" else 2.0
            direction = 1.0 if team_name == "blue" else -1.0

            for slot in range(self.agents_per_team):
                y = (slot - (self.agents_per_team - 1) / 2.0) * 2.0
                minion_states.append(
                    {
                        "id": 100 + len(minion_states) + 1,
                        "team": team_name,
                        "hp": 25.0,
                        "alive": True,
                        "position": np.array([base_x, y], dtype=np.float32),
                        "direction": direction,
                    }
                )

        return minion_states

    def _build_tower_states(self):
        return [
            {
                "id": 2001,
                "team": "blue",
                "hp": 200.0,
                "alive": True,
                "position": np.array([-1.0, 0.0], dtype=np.float32),
            },
            {
                "id": 2002,
                "team": "red",
                "hp": 200.0,
                "alive": True,
                "position": np.array([1.0, 0.0], dtype=np.float32),
            },
        ]

    def _wrap_obs(self, raw_obs):
        return {"obs": raw_obs}

    def _get_obs_for_agent(self, agent_idx):
        agent = self.agent_states[agent_idx]
        agent_pos = agent["position"]

        other_agents = []
        observed_enemy_actions = {}
        for other_idx, other in enumerate(self.agent_states):
            if other_idx == agent_idx:
                continue

            relative_position = other["position"] - agent_pos
            visible = self._is_visible(agent_pos, other["position"])
            if visible and other["team"] != agent["team"]:
                observed_enemy_actions[str(other["id"])] = int(other["last_action"])
            other_agents.append(
                {
                    "id": other["id"],
                    "team": other["team"],
                    "visible": visible,
                    "relative_position": (
                        relative_position.astype(np.float32).tolist() if visible else None
                    ),
                    "status": {
                        "alive": bool(other["alive"]),
                        "hp": float(other["hp"]) if visible else None,
                    },
                }
            )

        objects = []
        for minion in self.minion_states:
            relative_position = minion["position"] - agent_pos
            visible = self._is_visible(agent_pos, minion["position"])
            objects.append(
                {
                    "id": minion["id"],
                    "type": "minion",
                    "team": minion["team"],
                    "visible": visible,
                    "relative_position": (
                        relative_position.astype(np.float32).tolist() if visible else None
                    ),
                    "status": {
                        "alive": bool(minion["alive"]),
                        "hp": float(minion["hp"]) if visible else None,
                    },
                }
            )

        for tower in self.tower_states:
            relative_position = tower["position"] - agent_pos
            visible = self._is_visible(agent_pos, tower["position"])
            objects.append(
                {
                    "id": tower["id"],
                    "type": "tower",
                    "team": tower["team"],
                    "visible": visible,
                    "relative_position": (
                        relative_position.astype(np.float32).tolist() if visible else None
                    ),
                    "status": {
                        "alive": bool(tower["alive"]),
                        "hp": float(tower["hp"]) if visible else None,
                    },
                }
            )

        return {
            "timestep": self.t,
            "self": {
                "id": agent["id"],
                "team": agent["team"],
                "position": agent_pos.astype(np.float32).tolist(),
                "status": {
                    "alive": bool(agent["alive"]),
                    "hp": float(agent["hp"]),
                },
            },
            "agents": other_agents,
            "objects": objects,
            "extensions": {
                "observed_enemy_actions": observed_enemy_actions,
            },
        }

    def _extract_actions(self, action):
        if not isinstance(action, (list, tuple)) or len(action) == 0:
            return [int(action) for _ in range(self.num_envs)]

        branch = action[0]
        if isinstance(branch, np.ndarray):
            flat = branch.reshape(-1)
            if flat.size == 1:
                return [int(flat[0]) for _ in range(self.num_envs)]
            if flat.size == self.num_envs:
                return [int(v) for v in flat.tolist()]

        if isinstance(branch, (list, tuple)):
            if len(branch) == 1:
                return [int(branch[0]) for _ in range(self.num_envs)]
            if len(branch) == self.num_envs:
                return [int(v) for v in branch]

        return [int(branch) for _ in range(self.num_envs)]

    def _action_to_delta(self, action_index):
        """
        1: move up      		  -> [0, -1]
        2: move up-right          -> [1, -1]
        3: move right             -> [1, 0]
        4: move down-right        -> [1, 1]
        5: move down		      -> [0, 1]
        6: move down-left         -> [-1, 1]
        7: move left              -> [-1, 0]
        8: move up-left           -> [-1, -1]
        0: hold (no movement)     -> [0, 0]
        """
        deltas = {
            1: np.array([0.0, -1.0], dtype=np.float32),
            7: np.array([-1.0, 0.0], dtype=np.float32),
            3: np.array([1.0, 0.0], dtype=np.float32),
            5: np.array([0.0, 1.0], dtype=np.float32),
            8: np.array([-1.0, -1.0], dtype=np.float32),
            5: np.array([1.0, -1.0], dtype=np.float32),
            6: np.array([-1.0, 1.0], dtype=np.float32),
            4: np.array([1.0, 1.0], dtype=np.float32),
            0: np.array([0.0, 0.0], dtype=np.float32),
        }
        return deltas.get(int(action_index), np.array([0.0, 0.0], dtype=np.float32))

    def _apply_attack_actions(self, action_indices):
        for agent_idx, action_index in enumerate(action_indices):
            attacker = self.agent_states[agent_idx]
            if not attacker["alive"]:
                continue

            if action_index == 9:
                target = self._nearest_visible_enemy_agent(agent_idx)
                if target is not None:
                    self._apply_damage(target, damage=10.0)
            elif action_index == 10:
                target = self._nearest_visible_minion(attacker)
                if target is not None:
                    self._apply_damage(target, damage=8.0)
            elif action_index == 11:
                target = self._nearest_visible_tower(attacker)
                if target is not None:
                    self._apply_damage(target, damage=12.0)

    def _nearest_visible_enemy_agent(self, agent_idx):
        attacker = self.agent_states[agent_idx]
        candidates = [
            other
            for other_idx, other in enumerate(self.agent_states)
            if other_idx != agent_idx
            and other["team"] != attacker["team"]
            and other["alive"]
            and self._is_visible(attacker["position"], other["position"])
        ]
        return self._nearest_entity(attacker["position"], candidates)

    def _nearest_visible_minion(self, attacker):
        candidates = [
            minion
            for minion in self.minion_states
            if minion["alive"] and self._is_visible(attacker["position"], minion["position"])
        ]
        return self._nearest_entity(attacker["position"], candidates)

    def _nearest_visible_tower(self, attacker):
        candidates = [
            tower
            for tower in self.tower_states
            if tower["alive"]
            and self._is_visible(attacker["position"], tower["position"])
        ]
        return self._nearest_entity(attacker["position"], candidates)

    def _nearest_entity(self, origin, entities):
        if not entities:
            return None
        return min(
            entities,
            key=lambda entity: float(np.linalg.norm(entity["position"] - origin)),
        )

    def _apply_damage(self, entity, damage):
        entity["hp"] = max(0.0, float(entity["hp"]) - float(damage))
        entity["alive"] = entity["hp"] > 0.0

    def _step_minions(self):
        for minion in self.minion_states:
            if not minion["alive"]:
                continue
            minion["position"][0] += 0.25 * minion["direction"]

    def _is_visible(self, observer_position, target_position):
        return float(np.linalg.norm(target_position - observer_position)) <= self.vision_radius

    def _get_reward_for_agent(self, agent_idx):
        agent = self.agent_states[agent_idx]
        position = agent["position"]

        nearest_enemy_distance = min(
            float(np.linalg.norm(other["position"] - position))
            for other in self.agent_states
            if other["team"] != agent["team"]
        )
        reward = 0.1 if nearest_enemy_distance < 3.0 else -0.01
        return float(reward)

    def _get_info_for_agent(self, agent_idx):
        agent = self.agent_states[agent_idx]
        return {
            "agent_id": agent["id"],
            "team": agent["team"],
            "last_action": int(agent["last_action"]),
        }
