import random


class BaseStrategy:
    name = "base"

    ATTACK_HERO = 9
    ATTACK_MINION = 10
    ATTACK_TOWER = 11
    MOVE_UP = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_DOWN = 3
    LANE_CENTER_Y = 325.0
    HOLD = 8
    RETREAT = 12
    RANDOM_MOVE_ACTIONS = [MOVE_UP, MOVE_LEFT, MOVE_RIGHT, MOVE_DOWN, 4, 5, 6, 7, HOLD]

    def select_action(self, obs):
        tower_avoidance_action = self._avoid_enemy_tower_without_ally_minions(obs)
        if tower_avoidance_action is not None:
            return tower_avoidance_action

        if not self._visible_enemy_objects(obs):
            return self._select_observation_craving_action(obs)

        return self._select_action(obs)

    def _select_action(self, obs):
        raise NotImplementedError

    def _self_team(self, obs):
        return obs["self"]["team"]

    def _visible_ally_objects(self, obs):
        self_team = self._self_team(obs)
        return [
            obj
            for obj in obs["objects"]
            if obj["visible"]
            and obj["team"] == self_team
            and obj["status"]["alive"]
        ]

    def _visible_enemy_agents(self, obs):
        self_team = self._self_team(obs)
        return [
            agent
            for agent in obs["agents"]
            if agent["visible"]
            and agent["team"] != self_team
            and agent["status"]["alive"]
        ]

    def _visible_enemy_objects(self, obs):
        self_team = self._self_team(obs)
        return [
            obj
            for obj in obs["objects"]
            if obj["visible"]
            and obj["team"] != self_team
            and obj["status"]["alive"]
        ]

    def _visible_enemy_minions(self, obs):
        return [obj for obj in self._visible_enemy_objects(obs) if obj["type"] == "minion"]

    def _visible_ally_minions(self, obs):
        return [obj for obj in self._visible_ally_objects(obs) if obj["type"] == "minion"]

    def _visible_enemy_towers(self, obs):
        return [obj for obj in self._visible_enemy_objects(obs) if obj["type"] == "tower"]

    def _avoid_enemy_tower_without_ally_minions(self, obs):
        visible_enemy_towers = self._visible_enemy_towers(obs)
        if not visible_enemy_towers or self._visible_ally_minions(obs):
            return None

        return self.RETREAT

    def _select_observation_craving_action(self, obs):
        if len(self._visible_enemy_agents(obs)) <= 1:
            intent = self._sample_intent(
                [
                    ("move", 0.7),
                    ("attack_hero", 0.2),
                    ("attack_object", 0.1),
                ]
            )
        else:
            intent = self._sample_intent(
                [
                    ("move", 0.6),
                    ("attack_hero", 0.3),
                    ("attack_object", 0.1),
                ]
            )

        return self._resolve_intent(intent, obs)

    def _sample_intent(self, weighted_intents):
        """
        weighted_intents: list[tuple[str, float]]
        Probabilities are assumed to sum to 1.0 approximately.
        """
        prob = random.random()
        cumulative = 0.0
        for intent, weight in weighted_intents:
            cumulative += weight
            if prob < cumulative:
                return intent
        return weighted_intents[-1][0]

    def _resolve_intent(self, intent, obs):
        if intent == "attack_hero" and self._visible_enemy_agents(obs):
            return self.ATTACK_HERO

        if intent == "attack_object":
            if self._visible_enemy_minions(obs):
                return self.ATTACK_MINION
            if self._visible_enemy_towers(obs):
                return self.ATTACK_TOWER

        return self._sample_movement_action(obs)

    def _sample_movement_action(self, obs):
        if random.random() < 0.61:
            return self._move_toward_enemy_tower_action(obs)
        return random.choice(self.RANDOM_MOVE_ACTIONS)

    def _move_toward_enemy_tower_action(self, obs):
        visible_enemy_towers = self._visible_enemy_towers(obs)
        if visible_enemy_towers:
            nearest_tower = min(
                visible_enemy_towers,
                key=lambda tower: self._distance_sq(tower.get("relative_position")),
            )
            return self._vertical_move_toward(nearest_tower.get("relative_position"))

        self_team = self._self_team(obs)
        if self_team == "red":
            return self.MOVE_UP
        if self_team in {"blue", "enemy"}:
            return self.MOVE_DOWN

        self_position = obs.get("self", {}).get("position")
        if self_position is not None and len(self_position) >= 2:
            return self.MOVE_DOWN if self_position[1] < self.LANE_CENTER_Y else self.MOVE_UP

        return self.MOVE_UP

    def _distance_sq(self, relative_position):
        if relative_position is None:
            return float("inf")

        dx, dy = relative_position
        return dx * dx + dy * dy

    def _vertical_move_toward(self, relative_position):
        if relative_position is None:
            return self.MOVE_UP
        if relative_position[1] > 0:
            return self.MOVE_DOWN
        return self.MOVE_UP


class AggressiveStrategy(BaseStrategy):
    name = "aggressive"

    def _select_action(self, obs) -> int:
        """
        Aggressive strategy with controlled stochasticity.

        If a visible enemy hero exists:
        - 66% attack hero
        - 25% attack visible enemy object
        - 9% move / pressure forward

        Otherwise:
        - 61% attack visible enemy object
        - 39% move / pressure forward

        If a sampled attack intent has no valid target, fall back to movement.
        """
        if self._visible_enemy_agents(obs):
            intent = self._sample_intent(
                [
                    ("attack_hero", 0.65),
                    ("attack_object", 0.25),
                    ("move", 0.1),
                ]
            )
        else:
            intent = self._sample_intent(
                [
                    ("attack_object", 0.6),
                    ("move", 0.4),
                ]
            )

        return self._resolve_intent(intent, obs)


class FarmingStrategy(BaseStrategy):
    name = "farming"

    def _select_action(self, obs) -> int:
        if self._visible_enemy_minions(obs) or self._visible_enemy_towers(obs):
            intent = self._sample_intent(
                [
                    ("attack_object", 0.65),
                    ("attack_hero", 0.25),
                    ("move", 0.1),
                ]
            )
        else:
            intent = self._sample_intent(
                [
                    ("move", 0.6),
                    ("attack_hero", 0.4)
                ]
            )

        return self._resolve_intent(intent, obs)
    
class NeutralStrategy(BaseStrategy):
    name = "neutral"

    def _select_action(self, obs) -> int:
        if self._visible_enemy_agents(obs):
            intent = self._sample_intent(
                [
                    ("attack_hero", 0.45),
                    ("attack_object", 0.45),
                    ("move", 0.1),
                ]
            )
        else:
            intent = self._sample_intent(
                [
                    ("attack_object", 0.6),
                    ("move", 0.4)
                ]
            )

        return self._resolve_intent(intent, obs)

