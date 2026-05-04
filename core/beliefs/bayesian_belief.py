

from core.utils.obs_encoder import unwrap_obs


class BayesianBelief:
    SCORE_ORDER = ["aggressive", "neutral", "farming", "observation_craving"]
    LANE_ENEMY_ID = 2
    ACTION_MOVE_SET = tuple(range(0, 9))
    ACTION_ATTACK_HERO = 9
    ACTION_ATTACK_MINION = 10
    ACTION_ATTACK_TOWER = 11
    STRONG = 0.6
    MEDIUM = 0.3
    WEAK = 0.1
    BELIEF_UPDATE_ALPHA = 0.3

    def __init__(self, num_strategies=4):
        self.num_strategies = num_strategies
        self.belief = {
            "aggressive": 0.25,
            "neutral": 0.25,
            "farming": 0.25,
            "observation_craving": 0.25,
        }
        self.curr_obs = None
        self.prev_obs = None
        self.opponent_last_action = None
        self.opponent_last2t_action = None

    def update_observaition(self, obs):
        self.curr_obs = unwrap_obs(obs)

    def _current_lane_enemy(self):
        if self.curr_obs is None:
            return None
        return next(
            (
                agent
                for agent in self.curr_obs.get("agents", [])
                if agent.get("id") == self.LANE_ENEMY_ID
            ),
            None,
        )

    def _current_observed_enemy_action(self):
        if self.curr_obs is None:
            return None

        observed_actions = self.curr_obs.get("extensions", {}).get("observed_enemy_actions", {})
        return observed_actions.get(str(self.LANE_ENEMY_ID))

    def _ally_alive_minions(self, obs):
        if obs is None:
            return []

        self_team = obs.get("self", {}).get("team")
        return [
            obj
            for obj in obs.get("objects", [])
            if obj.get("type") == "minion"
            and obj.get("team") == self_team
            and obj.get("status", {}).get("alive")
        ]

    def _visible_enemy_minions(self, obs):
        if obs is None:
            return []

        self_team = obs.get("self", {}).get("team")
        return [
            obj
            for obj in obs.get("objects", [])
            if obj.get("type") == "minion"
            and obj.get("team") != self_team
            and obj.get("visible")
            and obj.get("status", {}).get("alive")
        ]

    def _ally_tower(self, obs):
        if obs is None:
            return None

        self_team = obs.get("self", {}).get("team")
        return next(
            (
                obj
                for obj in obs.get("objects", [])
                if obj.get("type") == "tower"
                and obj.get("team") == self_team
                and obj.get("status", {}).get("alive")
            ),
            None,
        )

    def _self_hp_drop(self):
        if self.curr_obs is None or self.prev_obs is None:
            return False

        curr_hp = self.curr_obs["self"]["status"]["hp"]
        prev_hp = self.prev_obs["self"]["status"]["hp"]
        return curr_hp < prev_hp

    def _ally_tower_hp_drop(self):
        curr_tower = self._ally_tower(self.curr_obs)
        prev_tower = self._ally_tower(self.prev_obs)

        if curr_tower is None or prev_tower is None:
            return False

        curr_hp = curr_tower.get("status", {}).get("hp")
        prev_hp = prev_tower.get("status", {}).get("hp")
        if curr_hp is None or prev_hp is None:
            return False

        return curr_hp < prev_hp

    def _ally_minion_pressure(self):
        if self.curr_obs is None:
            return False

        ally_minions = self._ally_alive_minions(self.curr_obs)
        visible_enemy_minions = self._visible_enemy_minions(self.curr_obs)
        return len(ally_minions) < len(visible_enemy_minions)

    def _soft_add(self, scores, indices, value):
        for idx in indices:
            scores[idx] += value

    def _action_is_move(self, action):
        return action in self.ACTION_MOVE_SET

    def _action_is_object_attack(self, action):
        return action in (self.ACTION_ATTACK_MINION, self.ACTION_ATTACK_TOWER)

    def _enemy_visible_now(self):
        enemy = self._current_lane_enemy()
        return enemy is not None and enemy.get("visible")

    def _last_t_in_vision(self):
        obs = self.prev_obs
        if obs is None:
            return None, []

        visible_agents = [
            agent
            for agent in obs.get("agents", [])
            if agent.get("visible") is True
        ]
        lane_enemy = next((agent for agent in visible_agents if agent.get("id") == 2), None)

        visible_ally_objects = [
            obj
            for obj in obs.get("objects", [])
            if obj.get("visible") is True and obj.get("team") == obs.get("self", {}).get("team")
        ]

        return lane_enemy, visible_ally_objects

    def _last_t_ally_in_enemy_vision(self):
        lane_enemy, visible_ally_objects = self._last_t_in_vision()
        if lane_enemy is None:
            return []

        relative_position = lane_enemy.get("relative_position")
        if relative_position is None:
            return []

        dx_enemy, dy_enemy = relative_position
        enemy_dist_sq = dx_enemy * dx_enemy + dy_enemy * dy_enemy

        ally_objects_in_enemy_vision = []
        for obj in visible_ally_objects:
            rel = obj.get("relative_position")
            if rel is None:
                continue

            dx_obj, dy_obj = rel
            obj_dist_sq = dx_obj * dx_obj + dy_obj * dy_obj

            if obj_dist_sq > enemy_dist_sq:
                continue

            cross = dx_enemy * dy_obj - dy_enemy * dx_obj
            if abs(cross) > 1e-6:
                continue

            dot = dx_enemy * dx_obj + dy_enemy * dy_obj
            if dot < 0:
                continue

            ally_objects_in_enemy_vision.append(obj)

        return ally_objects_in_enemy_vision

    def _build_context(self):
        ally_in_enemy_vision_last_t = self._last_t_ally_in_enemy_vision()
        enemy_action = self._current_observed_enemy_action()

        return {
            "enemy_visible_now": self._enemy_visible_now(),
            "enemy_action_observed": enemy_action is not None,
            "enemy_action": enemy_action,
            "prev_enemy_action": self.opponent_last_action,
            "prev2_enemy_action": self.opponent_last2t_action,
            "ally_in_enemy_vision_last_t": bool(ally_in_enemy_vision_last_t),
            "self_hp_drop": self._self_hp_drop(),
            "tower_hp_drop": self._ally_tower_hp_drop(),
            "ally_minion_pressure": self._ally_minion_pressure(),
        }

    def _rule_definitions(self):
        return [
            {
                "name": "hidden_no_action_self_hp_drop",
                "when": lambda c: (
                    not c["enemy_visible_now"]
                    and not c["enemy_action_observed"]
                    and c["self_hp_drop"]
                ),
                "add": {"aggressive": self.WEAK, "neutral": self.WEAK},
            },
            {
                "name": "hidden_no_action_lane_pressure",
                "when": lambda c: (
                    not c["enemy_visible_now"]
                    and not c["enemy_action_observed"]
                    and (c["ally_minion_pressure"] or c["tower_hp_drop"])
                ),
                "add": {"neutral": self.WEAK, "farming": self.WEAK},
            },
            {
                "name": "visible_no_action_self_hp_drop",
                "when": lambda c: (
                    c["enemy_visible_now"]
                    and not c["enemy_action_observed"]
                    and c["self_hp_drop"]
                ),
                "add": {"aggressive": self.MEDIUM, "neutral": self.MEDIUM},
            },
            {
                "name": "visible_no_action_lane_pressure",
                "when": lambda c: (
                    c["enemy_visible_now"]
                    and not c["enemy_action_observed"]
                    and (c["ally_minion_pressure"] or c["tower_hp_drop"])
                ),
                "add": {"neutral": self.MEDIUM, "farming": self.MEDIUM},
            },
            {
                "name": "observe_attack_hero_in_vision",
                "when": lambda c: (
                    c["enemy_action"] == self.ACTION_ATTACK_HERO
                    and c["ally_in_enemy_vision_last_t"]
                ),
                "add": {"aggressive": self.STRONG, "neutral": self.STRONG},
            },
            {
                "name": "observe_attack_hero_outside_vision_context",
                "when": lambda c: (
                    c["enemy_action"] == self.ACTION_ATTACK_HERO
                    and not c["ally_in_enemy_vision_last_t"]
                ),
                "add": {"aggressive": self.MEDIUM, "neutral": self.MEDIUM},
            },
            {
                "name": "observe_attack_minion",
                "when": lambda c: c["enemy_action"] == self.ACTION_ATTACK_MINION,
                "add": {"neutral": self.STRONG, "farming": self.STRONG},
            },
            {
                "name": "observe_attack_tower",
                "when": lambda c: c["enemy_action"] == self.ACTION_ATTACK_TOWER,
                "add": {"aggressive": self.WEAK, "neutral": self.WEAK, "farming": self.WEAK},
            },
            {
                "name": "observe_move_with_low_visibility_context",
                "when": lambda c: (
                    self._action_is_move(c["enemy_action"])
                    and not c["ally_in_enemy_vision_last_t"]
                ),
                "add": {"observation_craving": self.STRONG},
            },
            {
                "name": "observe_move_with_targets_available",
                "when": lambda c: (
                    self._action_is_move(c["enemy_action"])
                    and c["ally_in_enemy_vision_last_t"]
                ),
                "add": {"observation_craving": self.WEAK, "neutral": self.WEAK},
            },
            {
                "name": "hero_to_hero",
                "when": lambda c: (
                    c["prev_enemy_action"] == self.ACTION_ATTACK_HERO
                    and c["enemy_action"] == self.ACTION_ATTACK_HERO
                ),
                "add": {"aggressive": self.STRONG},
            },
            {
                "name": "hero_to_minion",
                "when": lambda c: (
                    c["prev_enemy_action"] == self.ACTION_ATTACK_HERO
                    and c["enemy_action"] == self.ACTION_ATTACK_MINION
                ),
                "add": {"neutral": self.MEDIUM},
            },
            {
                "name": "hero_to_tower",
                "when": lambda c: (
                    c["prev_enemy_action"] == self.ACTION_ATTACK_HERO
                    and c["enemy_action"] == self.ACTION_ATTACK_TOWER
                ),
                "add": {"aggressive": self.WEAK, "neutral": self.WEAK},
            },
            {
                "name": "object_to_hero",
                "when": lambda c: (
                    self._action_is_object_attack(c["prev_enemy_action"])
                    and c["enemy_action"] == self.ACTION_ATTACK_HERO
                ),
                "add": {"neutral": self.MEDIUM, "farming": self.MEDIUM},
            },
            {
                "name": "object_to_object",
                "when": lambda c: (
                    self._action_is_object_attack(c["prev_enemy_action"])
                    and self._action_is_object_attack(c["enemy_action"])
                ),
                "add": {"farming": self.MEDIUM},
            },
            {
                "name": "move_to_hero",
                "when": lambda c: (
                    self._action_is_move(c["prev_enemy_action"])
                    and c["enemy_action"] == self.ACTION_ATTACK_HERO
                ),
                "add": {
                    "aggressive": self.MEDIUM,
                    "neutral": self.MEDIUM,
                    "observation_craving": self.MEDIUM,
                },
            },
            {
                "name": "move_to_object",
                "when": lambda c: (
                    self._action_is_move(c["prev_enemy_action"])
                    and self._action_is_object_attack(c["enemy_action"])
                ),
                "add": {"farming": self.MEDIUM, "observation_craving": self.MEDIUM},
            },
            {
                "name": "move_to_move",
                "when": lambda c: (
                    self._action_is_move(c["prev_enemy_action"])
                    and self._action_is_move(c["enemy_action"])
                ),
                "add": {"observation_craving": self.WEAK},
            },
            {
                "name": "hero_action_matches_self_hp_drop",
                "when": lambda c: (
                    c["enemy_action"] == self.ACTION_ATTACK_HERO
                    and c["self_hp_drop"]
                ),
                "add": {"aggressive": self.WEAK},
            },
            {
                "name": "tower_action_matches_tower_hp_drop",
                "when": lambda c: (
                    c["enemy_action"] == self.ACTION_ATTACK_TOWER
                    and c["tower_hp_drop"]
                ),
                "add": {"neutral": self.WEAK, "farming": self.WEAK},
            },
            {
                "name": "minion_action_matches_lane_pressure",
                "when": lambda c: (
                    c["enemy_action"] == self.ACTION_ATTACK_MINION
                    and c["ally_minion_pressure"]
                ),
                "add": {"farming": self.WEAK},
            },
        ]

    def _apply_rule_scores(self, context):
        score_map = {name: 0.0 for name in self.SCORE_ORDER}

        for rule in self._rule_definitions():
            if rule["when"](context):
                for strategy_name, value in rule["add"].items():
                    score_map[strategy_name] += value

        return [score_map[name] for name in self.SCORE_ORDER]

    def _extract_signal(self):
        self.curr_obs = unwrap_obs(self.curr_obs)
        self.prev_obs = unwrap_obs(self.prev_obs)
        context = self._build_context()
        return self._apply_rule_scores(context)

    def bayesian_update(self, obs):
        """
        Heuristic posterior update using extracted evidence scores.
        """
        next_obs = unwrap_obs(obs)
        self.curr_obs = next_obs
        signal = self._extract_signal()

        if any(signal):
            total = sum(signal)
            normalized_signal = [value / total for value in signal]
            for key, value in zip(self.SCORE_ORDER, normalized_signal):
                self.belief[key] = (
                    (1.0 - self.BELIEF_UPDATE_ALPHA) * self.belief[key]
                    + self.BELIEF_UPDATE_ALPHA * value
                )

        self.opponent_last2t_action = self.opponent_last_action
        self.opponent_last_action = self._current_observed_enemy_action()
        self.prev_obs = self.curr_obs

        return self.belief
