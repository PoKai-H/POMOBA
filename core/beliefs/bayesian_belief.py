import jax.numpy as jnp
from collections import deque

from core.utils.obs_encoder import unwrap_obs


class BayesianBelief:
    """Bayesian opponent-strategy belief aligned with scripted strategies.

    The scripted opponents in ``basic_strategy.py`` choose an intent
    (attack_hero, attack_object, move) with strategy-specific probabilities and
    then resolve that intent into a concrete action. This belief model mirrors
    those intent probabilities and updates the posterior with the likelihood of
    the observed enemy action.

    The opponent strategy can switch during an episode, so every update mixes a
    small amount of uniform prior mass into the current posterior before applying
    the action likelihood. This keeps the belief able to recover after switches.
    """

    SCORE_ORDER = ["aggressive", "neutral", "farming"]

    ACTION_MOVE_UP = 0
    ACTION_MOVE_LEFT = 1
    ACTION_MOVE_RIGHT = 2
    ACTION_MOVE_DOWN = 3
    ACTION_MOVE_UP_LEFT = 4
    ACTION_MOVE_UP_RIGHT = 5
    ACTION_MOVE_DOWN_LEFT = 6
    ACTION_MOVE_DOWN_RIGHT = 7
    ACTION_HOLD = 8
    ACTION_ATTACK_HERO = 9
    ACTION_ATTACK_MINION = 10
    ACTION_ATTACK_TOWER = 11
    ACTION_RETREAT = 12

    MOVE_ACTIONS = {
        ACTION_MOVE_UP,
        ACTION_MOVE_LEFT,
        ACTION_MOVE_RIGHT,
        ACTION_MOVE_DOWN,
        ACTION_MOVE_UP_LEFT,
        ACTION_MOVE_UP_RIGHT,
        ACTION_MOVE_DOWN_LEFT,
        ACTION_MOVE_DOWN_RIGHT,
        ACTION_HOLD,
    }
    OBJECT_ATTACK_ACTIONS = {ACTION_ATTACK_MINION, ACTION_ATTACK_TOWER}

    EPS = 1e-4
    SWITCH_PRIOR = 0.025
    NO_SIGNAL_DECAY = 0.01
    ACTION_WINDOW = 90
    TRACKING_ALPHA = 0.65

    def __init__(self, num_strategies=3):
        self.num_strategies = num_strategies
        self.belief = self._uniform_belief()
        self.curr_obs = None
        self.prev_obs = None
        self.opponent_last_action = None
        self.opponent_last_target_type = None
        self.recent_action_categories = deque(maxlen=self.ACTION_WINDOW)

    def _uniform_belief(self):
        value = 1.0 / len(self.SCORE_ORDER)
        return {name: value for name in self.SCORE_ORDER}

    def update_observation(self, obs):
        self.curr_obs = unwrap_obs(obs)

    def _self_team(self, obs):
        return obs.get("self", {}).get("team")

    def _distance_sq(self, relative_position):
        if relative_position is None:
            return float("inf")
        dx, dy = relative_position
        return dx * dx + dy * dy

    def _visible_enemy_agents(self, obs):
        if obs is None:
            return []
        self_team = self._self_team(obs)
        return [
            agent
            for agent in obs.get("agents", [])
            if agent.get("team") != self_team
            and agent.get("visible") is True
            and agent.get("status", {}).get("alive")
        ]

    def _current_lane_enemy(self):
        visible_enemies = self._visible_enemy_agents(self.curr_obs)
        if not visible_enemies:
            return None
        return min(
            visible_enemies,
            key=lambda agent: self._distance_sq(agent.get("relative_position")),
        )

    def _current_observed_enemy_event(self):
        if self.curr_obs is None:
            return {"action": None, "target_type": None}

        lane_enemy = self._current_lane_enemy()
        if lane_enemy is None:
            return {"action": None, "target_type": None}

        observed_actions = self.curr_obs.get("extensions", {}).get(
            "observed_enemy_actions",
            {},
        )
        event = observed_actions.get(str(lane_enemy.get("id")))
        if isinstance(event, dict):
            action = event.get("action")
            return {
                "action": int(action) if action is not None else None,
                "target_type": event.get("target_type"),
            }

        return {
            "action": int(event) if event is not None else None,
            "target_type": None,
        }

    def _current_observed_enemy_action(self):
        return self._current_observed_enemy_event()["action"]

    def _visible_ally_objects(self, obs):
        if obs is None:
            return []
        self_team = self._self_team(obs)
        return [
            obj
            for obj in obs.get("objects", [])
            if obj.get("team") == self_team
            and obj.get("visible") is True
            and obj.get("status", {}).get("alive")
        ]

    def _visible_ally_minions(self, obs):
        return [
            obj for obj in self._visible_ally_objects(obs)
            if obj.get("type") == "minion"
        ]

    def _visible_ally_towers(self, obs):
        return [
            obj for obj in self._visible_ally_objects(obs)
            if obj.get("type") == "tower"
        ]

    def _visible_enemy_minions(self, obs):
        if obs is None:
            return []
        self_team = self._self_team(obs)
        return [
            obj
            for obj in obs.get("objects", [])
            if obj.get("team") != self_team
            and obj.get("type") == "minion"
            and obj.get("visible") is True
            and obj.get("status", {}).get("alive")
        ]

    def _ally_tower_hp_drop(self):
        if self.curr_obs is None or self.prev_obs is None:
            return False

        def ally_tower(obs):
            towers = self._visible_ally_towers(obs)
            return towers[0] if towers else None

        curr = ally_tower(self.curr_obs)
        prev = ally_tower(self.prev_obs)
        if curr is None or prev is None:
            return False

        curr_hp = curr.get("status", {}).get("hp")
        prev_hp = prev.get("status", {}).get("hp")
        return curr_hp is not None and prev_hp is not None and curr_hp < prev_hp

    def _ally_minion_hp_drop_count(self):
        if self.curr_obs is None or self.prev_obs is None:
            return 0

        def minion_hp_by_id(obs):
            return {
                obj.get("id"): obj.get("status", {}).get("hp")
                for obj in self._visible_ally_minions(obs)
                if obj.get("id") is not None
                and obj.get("status", {}).get("hp") is not None
            }

        curr_hp = minion_hp_by_id(self.curr_obs)
        prev_hp = minion_hp_by_id(self.prev_obs)
        drops = 0
        for obj_id, prev_value in prev_hp.items():
            curr_value = curr_hp.get(obj_id)
            if curr_value is not None and curr_value < prev_value:
                drops += 1
        return drops

    def _self_hp_drop(self):
        if self.curr_obs is None or self.prev_obs is None:
            return False
        curr_hp = self.curr_obs.get("self", {}).get("status", {}).get("hp")
        prev_hp = self.prev_obs.get("self", {}).get("status", {}).get("hp")
        return curr_hp is not None and prev_hp is not None and curr_hp < prev_hp

    def _ally_minion_pressure(self):
        if self.curr_obs is None:
            return False
        return len(self._visible_ally_minions(self.curr_obs)) < len(
            self._visible_enemy_minions(self.curr_obs)
        )

    def _action_category(self, action, target_type=None):
        if target_type == "hero":
            return "attack_hero"
        if target_type == "minion":
            return "attack_minion"
        if target_type == "tower":
            return "attack_tower"

        if action == self.ACTION_ATTACK_HERO:
            return "attack_hero"
        if action == self.ACTION_ATTACK_MINION:
            return "attack_minion"
        if action == self.ACTION_ATTACK_TOWER:
            return "attack_tower"
        if action == self.ACTION_RETREAT:
            return "retreat"
        if action in self.MOVE_ACTIONS:
            return "move"
        return "other"

    def _enemy_perspective_context(self):
        """Approximate the opponent's visible targets from our observation.

        If we can observe the opponent, we assume the opponent can observe our
        hero. Ally objects in our observation are enemy objects from the
        opponent's perspective, so they are the objects that its scripted policy
        would prefer to attack.
        """

        enemy_visible = self._current_lane_enemy() is not None
        ally_minions = self._visible_ally_minions(self.curr_obs)
        ally_towers = self._visible_ally_towers(self.curr_obs)
        return {
            "hero_available": enemy_visible,
            "minion_available": bool(ally_minions),
            "tower_available": bool(ally_towers),
            "object_available": bool(ally_minions or ally_towers),
            "object_action": (
                "attack_minion"
                if ally_minions
                else "attack_tower"
                if ally_towers
                else None
            ),
        }

    def _add_intent_probs(self, action_probs, context, attack_hero, attack_object, move):
        if context["hero_available"]:
            action_probs["attack_hero"] += attack_hero
        else:
            action_probs["move"] += attack_hero

        object_action = context["object_action"]
        if object_action is not None:
            action_probs[object_action] += attack_object
        else:
            action_probs["move"] += attack_object

        action_probs["move"] += move

    def _strategy_action_probs(self, strategy, context):
        action_probs = {
            "attack_hero": 0.0,
            "attack_minion": 0.0,
            "attack_tower": 0.0,
            "move": 0.0,
            "retreat": 0.0,
            "other": 0.0,
        }

        # Shared tower-avoidance behavior in BaseStrategy. It is not useful for
        # distinguishing strategies, but it prevents retreat observations from
        # being treated as impossible.
        action_probs["retreat"] = 0.02
        remaining_mass = 0.98

        if not context["object_available"]:
            # Shared observation-craving behavior before an enemy object is
            # visible to the scripted policy.
            self._add_intent_probs(
                action_probs,
                context,
                attack_hero=0.2 * remaining_mass,
                attack_object=0.1 * remaining_mass,
                move=0.7 * remaining_mass,
            )
            return self._normalize(action_probs)

        if strategy == "aggressive":
            if context["hero_available"]:
                self._add_intent_probs(
                    action_probs,
                    context,
                    attack_hero=0.65 * remaining_mass,
                    attack_object=0.25 * remaining_mass,
                    move=0.10 * remaining_mass,
                )
            else:
                self._add_intent_probs(
                    action_probs,
                    context,
                    attack_hero=0.0,
                    attack_object=0.60 * remaining_mass,
                    move=0.40 * remaining_mass,
                )
        elif strategy == "neutral":
            if context["hero_available"]:
                self._add_intent_probs(
                    action_probs,
                    context,
                    attack_hero=0.45 * remaining_mass,
                    attack_object=0.45 * remaining_mass,
                    move=0.10 * remaining_mass,
                )
            else:
                self._add_intent_probs(
                    action_probs,
                    context,
                    attack_hero=0.0,
                    attack_object=0.60 * remaining_mass,
                    move=0.40 * remaining_mass,
                )
        elif strategy == "farming":
            if context["object_available"]:
                self._add_intent_probs(
                    action_probs,
                    context,
                    attack_hero=0.25 * remaining_mass,
                    attack_object=0.65 * remaining_mass,
                    move=0.10 * remaining_mass,
                )
            else:
                self._add_intent_probs(
                    action_probs,
                    context,
                    attack_hero=0.40 * remaining_mass,
                    attack_object=0.0,
                    move=0.60 * remaining_mass,
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self._normalize(action_probs)

    def _normalize(self, values):
        total = sum(values.values())
        if total <= 0:
            return {key: 1.0 / len(values) for key in values}
        return {key: value / total for key, value in values.items()}

    def _context_likelihoods_without_action(self):
        likelihood = {name: 1.0 for name in self.SCORE_ORDER}

        if self._self_hp_drop():
            likelihood["aggressive"] *= 1.5
            likelihood["neutral"] *= 1.2

        if self._ally_tower_hp_drop() or self._ally_minion_pressure():
            likelihood["farming"] *= 1.5
            likelihood["neutral"] *= 1.2

        minion_drops = self._ally_minion_hp_drop_count()
        if minion_drops:
            likelihood["farming"] *= 1.0 + min(1.0, 0.25 * minion_drops)
            likelihood["neutral"] *= 1.1

        return likelihood

    def _action_likelihoods(self, action, target_type=None):
        category = self._action_category(action, target_type)
        context = self._enemy_perspective_context()
        likelihoods = {}
        for strategy in self.SCORE_ORDER:
            action_probs = self._strategy_action_probs(strategy, context)
            likelihoods[strategy] = max(action_probs.get(category, 0.0), self.EPS)

        if category in {"attack_minion", "attack_tower"} and self._self_hp_drop() is False:
            likelihoods["farming"] *= 1.35
            likelihoods["neutral"] *= 1.15
        if category == "attack_hero":
            likelihoods["aggressive"] *= 1.2

        return likelihoods

    def _temporal_likelihoods(self):
        attack_actions = [
            category
            for category in self.recent_action_categories
            if category in {"attack_hero", "attack_minion", "attack_tower"}
        ]
        if len(attack_actions) < 6:
            return {name: 1.0 for name in self.SCORE_ORDER}

        hero_count = attack_actions.count("attack_hero")
        hero_share = hero_count / len(attack_actions)

        if hero_share >= 0.50:
            likelihoods = {"aggressive": 0.66, "neutral": 0.26, "farming": 0.08}
        elif hero_share <= 0.38:
            likelihoods = {"aggressive": 0.08, "neutral": 0.26, "farming": 0.66}
        else:
            center_distance = abs(hero_share - 0.42)
            neutral_mass = max(0.56, 0.82 - 2.0 * center_distance)
            if hero_share > 0.42:
                aggressive_mass = 0.14 + (hero_share - 0.42) * 1.5
                farming_mass = 1.0 - neutral_mass - aggressive_mass
            else:
                farming_mass = 0.14 + (0.42 - hero_share) * 1.5
                aggressive_mass = 1.0 - neutral_mass - farming_mass
            likelihoods = {
                "aggressive": max(0.08, aggressive_mass),
                "neutral": max(0.56, neutral_mass),
                "farming": max(0.08, farming_mass),
            }

        return likelihoods

    def _recent_attack_count(self):
        return sum(
            1
            for category in self.recent_action_categories
            if category in {"attack_hero", "attack_minion", "attack_tower"}
        )

    def _mix_switch_prior(self):
        uniform = 1.0 / len(self.SCORE_ORDER)
        for strategy in self.SCORE_ORDER:
            self.belief[strategy] = (
                (1.0 - self.SWITCH_PRIOR) * self.belief[strategy]
                + self.SWITCH_PRIOR * uniform
            )

    def _apply_likelihoods(self, likelihoods):
        target = {}
        for strategy in self.SCORE_ORDER:
            target[strategy] = max(float(likelihoods.get(strategy, 1.0)), self.EPS)

        total = sum(target.values())
        if total <= 0:
            target = self._uniform_belief()
        else:
            target = {
                strategy: target[strategy] / total
                for strategy in self.SCORE_ORDER
            }

        for strategy in self.SCORE_ORDER:
            self.belief[strategy] = (
                (1.0 - self.TRACKING_ALPHA) * self.belief[strategy]
                + self.TRACKING_ALPHA * target[strategy]
            )

        belief_total = sum(self.belief.values())
        if belief_total <= 0:
            self.belief = self._uniform_belief()
            return

        for strategy in self.SCORE_ORDER:
            self.belief[strategy] = self.belief[strategy] / belief_total

    def _decay_toward_uniform(self):
        uniform = 1.0 / len(self.SCORE_ORDER)
        for strategy in self.SCORE_ORDER:
            self.belief[strategy] = (
                (1.0 - self.NO_SIGNAL_DECAY) * self.belief[strategy]
                + self.NO_SIGNAL_DECAY * uniform
            )

    def bayesian_update(self, obs):
        self.curr_obs = unwrap_obs(obs)
        enemy_event = self._current_observed_enemy_event()
        enemy_action = enemy_event["action"]
        target_type = enemy_event["target_type"]

        self._mix_switch_prior()
        if enemy_action is not None:
            action_category = self._action_category(enemy_action, target_type)
            self.recent_action_categories.append(action_category)
            if self._recent_attack_count() >= 12:
                likelihoods = self._temporal_likelihoods()
            else:
                likelihoods = self._action_likelihoods(enemy_action, target_type)
                likelihoods = {
                    strategy: 0.75 + 0.25 * likelihood
                    for strategy, likelihood in likelihoods.items()
                }

            pressure_likelihoods = self._context_likelihoods_without_action()
            for strategy in self.SCORE_ORDER:
                likelihoods[strategy] *= pressure_likelihoods[strategy]
            self._apply_likelihoods(likelihoods)
        else:
            likelihoods = self._context_likelihoods_without_action()
            if any(abs(value - 1.0) > 1e-8 for value in likelihoods.values()):
                self._apply_likelihoods(likelihoods)
            else:
                self._decay_toward_uniform()

        self.opponent_last_action = enemy_action
        self.opponent_last_target_type = target_type
        self.prev_obs = self.curr_obs
        return self.belief

    def update(self, obs):
        belief_dict = self.bayesian_update(obs)
        return jnp.asarray(
            [belief_dict[name] for name in self.SCORE_ORDER],
            dtype=jnp.float32,
        )
