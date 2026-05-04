from pathlib import Path
import sys
import socket

import jax
import jax.numpy as jnp
import numpy as np
import optax

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models.ppo_network import ActorCritic
from core.strategy.basic_strategy import (
    AggressiveStrategy,
    FarmingStrategy,
    NeutralStrategy,
    ObservationCravingStrategy,
)
from core.utils.obs_encoder import unwrap_obs

LEARNING_AGENT_ID = 0
NUM_ACTIONS = 13


STRATEGY_REGISTRY = {
    AggressiveStrategy.name: AggressiveStrategy,
    FarmingStrategy.name: FarmingStrategy,
    NeutralStrategy.name: NeutralStrategy,
    ObservationCravingStrategy.name: ObservationCravingStrategy,
}


class StrategyPolicyManager:
    def __init__(self, config):
        self._policy_cache = {}
        self.set_config(config)

    def set_config(self, config):
        self.config = config
        self._policy_cache.clear()

    def _core_config(self):
        return self.config.get("core", {})

    def _policy_id(self, agent_id, agent_obs):
        id_source = self.config.get("NPC_POLICY_ID_SOURCE", "index")
        if id_source == "self_id":
            return unwrap_obs(agent_obs).get("self", {}).get("id", agent_id)
        return agent_id

    def _strategy_name_for(self, policy_id, step):
        schedule = self.config.get("NPC_POLICY_SCHEDULE", {})
        schedule_entries = schedule.get(policy_id, schedule.get(str(policy_id), []))
        if schedule_entries:
            active_strategy = None
            for entry in schedule_entries:
                if step >= entry.get("start_step", 0):
                    active_strategy = entry.get("strategy")
            if active_strategy is not None:
                return active_strategy

        npc_policies = self.config.get("NPC_POLICIES", {})
        strategy_name = npc_policies.get(policy_id, npc_policies.get(str(policy_id)))
        if strategy_name is not None:
            return strategy_name

        return self.config.get(
            "DEFAULT_NPC_POLICY",
            self._core_config().get("opponent_strategy", "neutral"),
        )

    def policy_for(self, agent_id, agent_obs, step):
        policy_id = self._policy_id(agent_id, agent_obs)
        strategy_name = self._strategy_name_for(policy_id, step)
        strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
        if strategy_cls is None:
            available = ", ".join(sorted(STRATEGY_REGISTRY))
            raise ValueError(
                f"Unknown NPC strategy '{strategy_name}'. Available strategies: {available}"
            )

        if strategy_name not in self._policy_cache:
            self._policy_cache[strategy_name] = strategy_cls()
        return self._policy_cache[strategy_name]


class PPO:
    def __init__(self, env, encoder, belief , config):
        self.env = env
        self.encoder = encoder
        self.belief = belief
        seed = config.get("SEED", config.get("core", {}).get("random_seed", 0))
        self.rng = jax.random.PRNGKey(seed)
        self.np_rng = np.random.default_rng(seed)
        self.network = None
        self.params = None
        self._network_signature = None
        self.optimizer = None
        self.opt_state = None
        self._optimizer_signature = None
        self._optimizer_needs_reset = True
        self.strategy_manager = StrategyPolicyManager(config)
        self.episode_logs = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.gamma = config.get("GAMMA", 0.99)
        self.gae_lam = config.get("GAE_LAMBDA", 0.95)
        self.clip_eps = config.get("CLIP_EPS", config.get("PPO_CLIP_EPS", 0.2))
        self.value_coef = config.get("VF_COEF", config.get("VALUE_COEF", 0.5))
        self.entropy_coef = config.get("ENT_COEF", config.get("ENTROPY_COEF", 0.01))
        self.learning_rate = config.get("LR", config.get("LEARNING_RATE", 3e-4))
        self.max_grad_norm = config.get("MAX_GRAD_NORM", 0.5)
        self.update_epochs = config.get("UPDATE_EPOCHS", 4)
        self.num_minibatches = config.get("NUM_MINIBATCHES", 1)
        self.set_config(config)

    def set_config(self, config):
        self.config = config
        self.learning_agent_id = config.get(
            "LEARNING_AGENT_ID",
            LEARNING_AGENT_ID,
        )
        self.use_belief_input = config["USE_BELIEF_INPUT"]
        self.timestep_per_batch = config["TIMESTEP_PER_BATCH"]
        self.max_steps_per_episode = config["MAX_STEPS_PER_EPISODE"]
        self.gamma = config.get("GAMMA", getattr(self, "gamma", 0.99))
        self.gae_lam = config.get("GAE_LAMBDA", getattr(self, "gae_lam", 0.95))
        self.clip_eps = config.get("CLIP_EPS", config.get("PPO_CLIP_EPS", self.clip_eps))
        self.value_coef = config.get("VF_COEF", config.get("VALUE_COEF", self.value_coef))
        self.entropy_coef = config.get(
            "ENT_COEF",
            config.get("ENTROPY_COEF", self.entropy_coef),
        )
        self.learning_rate = config.get("LR", config.get("LEARNING_RATE", self.learning_rate))
        self.max_grad_norm = config.get("MAX_GRAD_NORM", self.max_grad_norm)
        self.update_epochs = config.get("UPDATE_EPOCHS", self.update_epochs)
        self.num_minibatches = config.get("NUM_MINIBATCHES", self.num_minibatches)
        self.strategy_manager.set_config(config)
        self._init_network_if_needed()
        self._init_optimizer_if_needed()
        

    def build_policy_input(self, obs_vec, belief_vec, use_belief_input=True):
        obs_vec = np.asarray(obs_vec, dtype=np.float32)
        if use_belief_input:
            belief_vec = np.asarray(belief_vec, dtype=np.float32)
            return np.concatenate([obs_vec, belief_vec], axis=0)
        return obs_vec
    
    def select_action(self, obs_vec, belief_vec, use_belief_input=None):
        if use_belief_input is None:
            use_belief_input = self.use_belief_input

        policy_input = self.build_policy_input(
            obs_vec,
            belief_vec,
            use_belief_input=use_belief_input,
        )
        self.rng, sample_key = jax.random.split(self.rng)
        pi, value = self.network.apply(
            self.params,
            jnp.asarray(policy_input, dtype=jnp.float32),
        )
        action = pi.sample(seed=sample_key)
        logprob = pi.log_prob(action)

        return int(action), float(logprob), float(value)

    def value(self, obs_vec, belief_vec, use_belief_input=None):
        if use_belief_input is None:
            use_belief_input = self.use_belief_input

        policy_input = self.build_policy_input(
            obs_vec,
            belief_vec,
            use_belief_input=use_belief_input,
        )
        _, value = self.network.apply(
            self.params,
            jnp.asarray(policy_input, dtype=jnp.float32),
        )
        return float(value)

    def _init_network_if_needed(self):
        action_dim = self.config.get("NUM_ACTIONS", NUM_ACTIONS)
        hidden_dim = self.config.get("HIDDEN_DIM", 64)
        activation = self.config.get("ACTIVATION", "tanh")
        input_dim = self.encoder.obs_dim
        if self.use_belief_input:
            input_dim += getattr(self.belief, "num_strategies", 0)

        signature = (action_dim, hidden_dim, activation, input_dim)
        if signature == self._network_signature:
            return

        self.network = ActorCritic(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        self.rng, init_key = jax.random.split(self.rng)
        dummy_x = jnp.zeros((input_dim,), dtype=jnp.float32)
        self.params = self.network.init(init_key, dummy_x)
        self._network_signature = signature
        self._optimizer_needs_reset = True

    def _init_optimizer_if_needed(self):
        signature = (self.learning_rate, self.max_grad_norm)
        if signature == self._optimizer_signature and not self._optimizer_needs_reset:
            return

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.learning_rate),
        )
        self.opt_state = self.optimizer.init(self.params)
        self._optimizer_signature = signature
        self._optimizer_needs_reset = False

    def _config_for_episode_reset(self):
        if not self.config.get("RANDOMIZE_NPC_POLICY_EACH_EPISODE", False):
            return self.config

        strategy_pool = list(self.config.get("NPC_STRATEGY_POOL", []))
        if len(strategy_pool) < 2:
            return self.config

        switch_min, switch_max = self.config.get("NPC_SWITCH_STEP_RANGE", [1, self.max_steps_per_episode - 1])
        switch_min = max(1, int(switch_min))
        switch_max = min(int(switch_max), self.max_steps_per_episode - 1)
        if switch_max < switch_min:
            switch_step = switch_min
        else:
            switch_step = int(self.np_rng.integers(switch_min, switch_max + 1))

        initial_strategy, next_strategy = self.np_rng.choice(
            strategy_pool,
            size=2,
            replace=False,
        ).tolist()
        npc_policy_ids = self.config.get("NPC_POLICY_IDS")
        if npc_policy_ids is None:
            npc_policy_ids = list(self.config.get("NPC_POLICIES", {}).keys())
        if not npc_policy_ids:
            npc_policy_ids = [1]

        npc_policies = {
            policy_id: initial_strategy
            for policy_id in npc_policy_ids
        }
        npc_schedule = {
            policy_id: [
                {"start_step": 0, "strategy": initial_strategy},
                {"start_step": switch_step, "strategy": next_strategy},
            ]
            for policy_id in npc_policy_ids
        }

        core_config = {
            **self.config.get("core", {}),
            "opponent_strategy": initial_strategy,
            "strategy_switch_mode": "random_time",
            "strategy_switch_step": switch_step,
            "next_opponent_strategy": next_strategy,
        }

        config = {
            **self.config,
            "core": core_config,
            "NPC_POLICIES": npc_policies,
            "NPC_POLICY_SCHEDULE": npc_schedule,
        }
        return config

    def collect_rollout(self, config=None):
        if config is not None:
            self.set_config(config)

        batch_t = 0
        trajectory = []
        while batch_t < self.timestep_per_batch:
            episode_runtime_config = self._config_for_episode_reset()
            self.set_config(episode_runtime_config)
            try:
                obs_list, _ = self.env.reset(episode_runtime_config)
                ep_reward = 0
                ep_length = 0
            except(ConnectionError, ConnectionResetError, BrokenPipeError, OSError, socket.error) as e:
                print("Godot connection lost during reset:", e)
                try:
                    self.env.close()
                except Exception:
                    pass
                sys.exit(1)

            for step in range(self.max_steps_per_episode):
                
                learning_obs = obs_list[self.learning_agent_id]
                obs_vec = self.encoder.encode(learning_obs)
                belief_vec = self.belief.update(obs_vec)
            
                learning_action, logprob, value = self.select_action(
                    obs_vec, 
                    belief_vec,
                    use_belief_input = self.use_belief_input

                )
                
                all_actions, all_policy_names = self.select_env_actions(
                    obs_list,
                    self.learning_agent_id,
                    learning_action,
                    step,
                )

                action_for_env = [np.asarray(all_actions, dtype=np.int32)]

                try:
                    obs_new_list, reward_list, done_list, truncated_list, _ = self.env.step(action_for_env)
                except (ConnectionError, ConnectionResetError, BrokenPipeError, OSError, socket.error) as e:
                    print("Godot connection lost during step():", e)
                    try:
                        self.env.close()
                    except Exception:
                        pass
                    sys.exit(1)

                ep_reward += reward_list[self.learning_agent_id] 
                ep_length += 1

                learning_reward = reward_list[self.learning_agent_id]
                learning_done = done_list[self.learning_agent_id]
                learning_truncated = truncated_list[self.learning_agent_id]

                trajectory.append(
                    {
                        # PPO training fields: build_ppo_batch 
                        "step": step,
                        "obs": obs_vec,
                        "belief": belief_vec,
                        "obs_belief": self.build_policy_input(
                            obs_vec,
                            belief_vec,
                            use_belief_input=True,
                        ),
                        "action": learning_action,
                        "reward": float(learning_reward),
                        "done": float(learning_done),
                        "truncated": float(learning_truncated),
                        "logprob": float(logprob),
                        "value": float(value),

                        # Debug / analysis fields
                        "raw_obs": learning_obs,
                        "all_raw_obs": obs_list,
                        "all_actions": all_actions,
                        "all_policy_names": all_policy_names,
                        "all_rewards": reward_list,
                        "all_dones": done_list,
                        "all_truncated": truncated_list,
                    }
                )

                obs_list = obs_new_list
                batch_t += 1


                if learning_done or learning_truncated:
                    self.episode_logs.append(
                        {
                            "episode_reward": float(ep_reward),
                            "episode_length": ep_length,
                            "terminated": bool(learning_done),
                            "truncated": bool(learning_truncated),
                            "core": episode_runtime_config.get("core", {}),
                            "npc_policy_schedule": episode_runtime_config.get("NPC_POLICY_SCHEDULE", {}),
                        }
                    )
                    break

                if batch_t >= self.timestep_per_batch:
                    break
                
                
        last_value = self.compute_last_value(trajectory, obs_list)
        return trajectory, last_value

    def compute_last_value(self, trajectory, obs_list):
        if not trajectory:
            return 0.0

        if trajectory[-1]["done"]:
            return 0.0

        learning_obs = obs_list[self.learning_agent_id]
        obs_vec = self.encoder.encode(learning_obs)
        belief_vec = self.belief.update(obs_vec)
        return self.value(
            obs_vec,
            belief_vec,
            use_belief_input=self.use_belief_input,
        )


    def select_env_actions(
        self,
        obs_list, 
        learning_agent_id,
        learning_action,
        step,
    ):
        all_actions = []
        all_policy_names = []
        
        for agent_id, agent_obs in enumerate(obs_list):
            if agent_id == learning_agent_id:
                action = learning_action
                policy_name = "ppo"
            else:
                policy = self.strategy_manager.policy_for(agent_id, agent_obs, step)
                action = policy.select_action(unwrap_obs(agent_obs))
                policy_name = policy.name

            all_actions.append(action)
            all_policy_names.append(policy_name)
        return all_actions, all_policy_names
    
    def compute_gae(self, trajectory, last_value=0.0):
        advantages = np.zeros(len(trajectory), dtype=np.float32)
        returns = np.zeros(len(trajectory), dtype=np.float32)
        values = [float(t["value"]) for t in trajectory] + [float(last_value)]

        gae = 0.0
        for t in reversed(range(len(trajectory))):
            reward = float(trajectory[t]["reward"])
            is_final_step = t == len(trajectory) - 1
            truncated_mid_batch = trajectory[t].get("truncated", 0.0) and not is_final_step
            episode_end = float(trajectory[t]["done"] or truncated_mid_batch)
            nonterminal = 1.0 - episode_end

            delta = reward + self.gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lam * nonterminal * gae

            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns
    
    def attach_returns_and_advantages(
        self,
        trajectory,
        last_value =0.0,
    ):
        advantages, returns = self.compute_gae(
            trajectory,
            last_value = last_value
        )

        for t in range(len(trajectory)):
            trajectory[t]["advantage"] = float(advantages[t])
            trajectory[t]["return"] = float(returns[t])

        return trajectory

    def build_ppo_batch(
        self,
        trajectory,
        use_belief_input=None,
        normalize_advantage=True,
    ):
        if use_belief_input is None:
            use_belief_input = self.use_belief_input

        obs_key = "obs_belief" if use_belief_input else "obs"
        batch = {
            "inputs": np.asarray([step[obs_key] for step in trajectory], dtype=np.float32),
            "obs": np.asarray([step["obs"] for step in trajectory], dtype=np.float32),
            "belief": np.asarray([step["belief"] for step in trajectory], dtype=np.float32),
            "actions": np.asarray([step["action"] for step in trajectory], dtype=np.int32),
            "old_logprobs": np.asarray([step["logprob"] for step in trajectory], dtype=np.float32),
            "values": np.asarray([step["value"] for step in trajectory], dtype=np.float32),
            "returns": np.asarray([step["return"] for step in trajectory], dtype=np.float32),
            "advantages": np.asarray([step["advantage"] for step in trajectory], dtype=np.float32),
            "rewards": np.asarray([step["reward"] for step in trajectory], dtype=np.float32),
            "dones": np.asarray([step["done"] for step in trajectory], dtype=np.float32),
            "truncated": np.asarray(
                [step.get("truncated", 0.0) for step in trajectory],
                dtype=np.float32,
            ),
        }

        if normalize_advantage and len(trajectory) > 1:
            adv = batch["advantages"]
            batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
       
        return {key: jnp.asarray(value) for key, value in batch.items()}
    
    def ppo_loss(self, params, batch):
        pi, value = self.network.apply(params, batch["inputs"])
        new_logprob = pi.log_prob(batch["actions"])
        ratio = jnp.exp(new_logprob - batch["old_logprobs"])

        unclipped = ratio * batch["advantages"]
        clipped = (
            jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            * batch["advantages"]
        )
        actor_loss = -jnp.minimum(unclipped, clipped)

        value_error = batch["returns"] - value
        critic_loss = 0.5 * jnp.square(value_error)
        entropy = pi.entropy()

        total_loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy
        )

        metrics = {
            "total_loss": jnp.mean(total_loss),
            "actor_loss": jnp.mean(actor_loss),
            "critic_loss": jnp.mean(critic_loss),
            "entropy": jnp.mean(entropy),
            "mean_ratio": jnp.mean(ratio),
            "mean_value": jnp.mean(value),
        }
        return metrics["total_loss"], metrics
    
    def update(self, batch):
        batch_size = batch["inputs"].shape[0]
        if batch_size == 0:
            raise ValueError("Cannot update PPO with an empty batch.")
        if batch_size % self.num_minibatches != 0:
            raise ValueError(
                "Batch size must be divisible by NUM_MINIBATCHES: "
                f"{batch_size} vs {self.num_minibatches}"
            )

        minibatch_size = batch_size // self.num_minibatches
        metric_history = []
        loss_history = []

        for _ in range(self.update_epochs):
            self.rng, shuffle_key = jax.random.split(self.rng)
            permutation = jax.random.permutation(shuffle_key, batch_size)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0),
                batch,
            )

            for mb_idx in range(self.num_minibatches):
                start = mb_idx * minibatch_size
                end = start + minibatch_size
                minibatch = jax.tree_util.tree_map(
                    lambda x: x[start:end],
                    shuffled_batch,
                )
                loss, metrics = self._update_minibatch(minibatch)
                loss_history.append(loss)
                metric_history.append(metrics)

        metrics = self._mean_metric_history(metric_history)
        loss = jnp.mean(jnp.asarray(loss_history))

        self.actor_losses.append(float(metrics["actor_loss"]))
        self.critic_losses.append(float(metrics["critic_loss"]))
        self.entropies.append(float(metrics["entropy"]))

        return loss, metrics

    def _update_minibatch(self, batch):
        def loss_fn(current_params):
            return self.ppo_loss(current_params, batch)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.params)
        updates, self.opt_state = self.optimizer.update(
            grads,
            self.opt_state,
            self.params,
        )
        self.params = optax.apply_updates(self.params, updates)

        return loss, metrics

    def _mean_metric_history(self, metric_history):
        return {
            key: jnp.mean(jnp.asarray([metrics[key] for metrics in metric_history]))
            for key in metric_history[0]
        }
