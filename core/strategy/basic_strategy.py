import random


class BaseStrategy:
    name = "base"

    ATTACK_HERO = 9
    ATTACK_MINION = 10
    ATTACK_TOWER = 11
    MOVE_TOWARD_ENEMY_TOWER = 3
    RANDOM_MOVE_ACTIONS = [0, 1, 2, 4, 5, 6, 7, 8]

    def select_action(self, obs):
        raise NotImplementedError

    def _self_team(self, obs):
        return obs["self"]["team"]

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

    def _visible_enemy_towers(self, obs):
        return [obj for obj in self._visible_enemy_objects(obs) if obj["type"] == "tower"]

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

        return self._sample_movement_action()

    def _sample_movement_action(self):
        if random.random() < 0.61:
            return self.MOVE_TOWARD_ENEMY_TOWER
        return random.choice(self.RANDOM_MOVE_ACTIONS)


class AggressiveStrategy(BaseStrategy):
    name = "aggressive"

    def select_action(self, obs) -> int:
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

    def select_action(self, obs) -> int:
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

    def select_action(self, obs) -> int:
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


class ObservationCravingStrategy(BaseStrategy):
    name = "observation_craving"

    def select_action(self, obs) -> int:
        visible_enemy_agents = self._visible_enemy_agents(obs)
        visible_enemy_objects = self._visible_enemy_objects(obs)

        # When information is sparse, prioritize moving to reveal more entities.
        if len(visible_enemy_agents) <= 1:
            intent = self._sample_intent(
                [
                    ("move", 0.7),
                    ("attack_object", 0.2),
                    ("attack_hero", 0.1),
                ]
            )
        # Once enough enemy agents are visible, behave more like a neutral probe.
        elif visible_enemy_objects:
            intent = self._sample_intent(
                [
                    ("attack_hero", 0.35),
                    ("attack_object", 0.35),
                    ("move", 0.3),
                ]
            )
        else:
            intent = self._sample_intent(
                [
                    ("move", 0.6),
                    ("attack_hero", 0.25),
                    ("attack_object", 0.15),
                ]
            )

        return self._resolve_intent(intent, obs)
