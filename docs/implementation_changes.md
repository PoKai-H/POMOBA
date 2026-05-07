# Implementation Changes

This document summarizes the current implementation changes, grouped by Python-side training code and Godot-side simulator code.

## Python

### Training Configuration

- Updated `config/basic_config.py` for the current 1v1 training setup.
- The action space now uses the 13-action Godot mapping:
  - `0`: move up
  - `1`: move left
  - `2`: move right
  - `3`: move down
  - `4`: move up-left
  - `5`: move up-right
  - `6`: move down-left
  - `7`: move down-right
  - `8`: hold
  - `9`: attack hero
  - `10`: attack nearest minion
  - `11`: attack tower
  - `12`: retreat
- Increased `TOTAL_TIMESTEPS` to `102_400`.
- Reduced `ENT_COEF` to `0.003` to make the PPO policy less random after early exploration.
- Set the default opponent strategy to `neutral`.
- Set the default NPC policy to `neutral`.
- Disabled per-episode random opponent strategy switching for the easier first training curriculum:
  - `RANDOMIZE_NPC_POLICY_EACH_EPISODE = False`
- Kept the strategy pool as:
  - `aggressive`
  - `farming`
  - `neutral`

### PPO Rollout And Episode Handling

- Updated `core/models/ppo.py` to use the raw learning observation for belief updates instead of the encoded observation vector.
- Fixed `done` and `truncated` reporting so terminal tower-destruction episodes are not also marked as truncated.
- The current truncation logic is:

```python
learning_truncated = (
    not learning_done
    and (
        truncated_list[self.learning_agent_id]
        or step == self.max_steps_per_episode - 1
    )
)
```

- This means:
  - tower destruction should produce `terminated=True, truncated=False`,
  - max-step timeout should produce `terminated=False, truncated=True`.
- `all_truncated` is updated with the corrected learning-agent truncation value before being stored in the trajectory.
- Rollout collection now preserves detailed debug fields in each trajectory step:
  - raw learning observation,
  - all raw observations,
  - all actions,
  - all policy names,
  - all rewards,
  - all done flags,
  - all truncated flags.

### Scripted Strategy Changes

- Reduced scripted strategies from four to three:
  - `aggressive`
  - `neutral`
  - `farming`
- Removed `observation_craving` as an independent strategy.
- Added shared observation-craving behavior to `BaseStrategy`.
- Before any enemy object is visible, all scripted strategies use exploration-focused behavior.
- Once an enemy object becomes visible, the strategy switches into its own style.
- Shared pre-object-visibility logic was added in `core/strategy/basic_strategy.py`:

```python
if not self._visible_enemy_objects(obs):
    return self._select_observation_craving_action(obs)
```

- Tower avoidance is shared across all scripted strategies:
  - if an enemy tower is visible,
  - and no allied minions are visible,
  - the scripted strategy returns `retreat`.
- Fixed team-direction logic for moving toward the enemy tower:
  - red-side agents move upward,
  - blue/enemy-side agents move downward,
  - visible tower relative position is used when available.
- Updated `core/models/ppo.py`:
  - removed `ObservationCravingStrategy` import,
  - removed it from `STRATEGY_REGISTRY`.
- Updated `config/basic_config.py`:
  - removed `"observation_craving"` from `NPC_STRATEGY_POOL`.
- Updated `core/test.py`:
  - replaced `ObservationCravingStrategy` with `NeutralStrategy`.

### Belief Model Changes

- Updated `DummyBelief` to use a fixed 3-way strategy belief.
- `DummyBelief` now defaults to `num_strategies=3`.
- Updated `BayesianBelief` to track:
  - `aggressive`
  - `neutral`
  - `farming`
- Removed the separate `observation_craving` belief category.
- Rebalanced belief rules that previously favored `observation_craving` into the remaining 3-strategy belief space.
- `BayesianBelief.update()` returns a JAX-compatible belief vector.

### Run Output And Training Diagnostics

- Updated `core/run.py` to replace the old `sampled_actions` output with a richer rollout summary.
- Added per-update terminal diagnostics for:
  - reward distribution,
  - value distribution,
  - return distribution,
  - advantage distribution,
  - positive and negative reward step counts,
  - action usage counts,
  - attack, movement, and retreat rates,
  - done and truncated counts,
  - recent done and truncated steps,
  - latest episode summary,
  - final-frame agent/object state summary.
- Added graceful handling for interrupted or disconnected training runs.
- If training stops early, collected history is still saved when possible.
- Added training history collection across updates.
- Added automatic analysis artifact export after training finishes or stops.
- Training analysis is saved under:

```text
outputs/training_analysis/<timestamp>/
```

- Generated files include:

```text
history.json
run_config.json
training_curves.png
action_diagnostics.png
reward_diagnostics.png
```

- `training_curves.png` shows:
  - episode reward,
  - episode length,
  - total loss,
  - critic loss,
  - actor loss,
  - policy entropy.
- `action_diagnostics.png` shows:
  - attack, movement, and retreat rate trends,
  - action usage heatmap across updates.
- `reward_diagnostics.png` shows:
  - reward mean and sum,
  - positive vs negative reward step counts,
  - advantage min, mean, and max,
  - return vs value.

### Training Belief Selection

- `core/run.py` currently initializes PPO with `DummyBelief`.
- This is useful for observation-only baseline training because `USE_BELIEF_INPUT` is currently `False`.
- `BayesianBelief` remains available for belief-conditioned experiments.

## Godot

### Environment Reset

- Reset now calls `_reset_world()` in `simulator/addons/godot_rl_agents/sync.gd`.
- `_reset_world()`:
  - resets `n_action_steps`,
  - calls `apply_sim_config()` in the main scene,
  - rebuilds the map and runtime agents,
  - reapplies simulator config,
  - calls `_get_agents()` to clear and rebuild the agent/controller lists,
  - sets training agents back to `"model"` when connected.
- `_get_agents()` now clears old cached agent lists before repopulating:
  - `all_agents`
  - `agents_training`
  - `agents_inference`
  - `agents_heuristic`
  - `agent_demo_record`

### Minion Wave Reset

- Fixed minion wave reset so old minions do not remain after a world reset.
- `main_scene.gd` now calls `spawner.reset_waves(minions_enabled)` during reset/config application.
- `minion_wave_spawner.gd` now clears old minions before spawning a new wave.
- Old minions are removed immediately with:

```gdscript
remove_child(child)
child.free()
```

- This replaces deferred `queue_free()` for reset cleanup.
- `_ready()` now uses `reset_waves(true)` instead of directly calling `spawn_wave()`.
- The wave timer is restarted during wave reset to avoid an immediate duplicate wave.
- If `minion_wave` is disabled, reset clears old minions and does not spawn new ones.

### Reward Reporting

- Fixed reward reporting to return per-step reward instead of accumulated reward totals.
- `controller.gd` `get_reward()` now:
  - stores the current controller reward,
  - clears the controller reward,
  - clears the player body reward,
  - returns only the current step reward.
- `character_body_2d.gd` `get_reward()` also uses read-and-clear behavior.
- Removed the per-frame reward `print()` from `character_body_2d.gd`.
- Main reason:
  - PPO should receive step rewards, not cumulative reward totals.
  - The old behavior could inflate returns and critic loss, making advantage estimates noisy.

### Agent Death And Respawn

- Agent death behavior was changed to immediate respawn.
- Player death no longer:
  - calls `game_over()`,
  - sets `done`,
  - sets `needs_reset`,
  - pauses the scene tree,
  - triggers a full episode/world reset.
- On death:
  - the dead agent receives the death penalty,
  - collision, hurtbox, and attack area are briefly disabled during the death handler,
  - the dead agent immediately respawns.
- On respawn:
  - HP is restored to `max_hp`,
  - position resets to the original spawn position,
  - velocity is cleared,
  - attack cooldown is reset,
  - collision, hurtbox, and attack area are re-enabled,
  - visibility is restored,
  - controller `done` and `needs_reset` flags are cleared.
- Only the dead agent respawns.
- The other agent is not reset when one agent dies.

### Tower And Minion Ranges

- Tower and minion combat/detection ranges were matched to the agent observation radius.
- Lane tower attack range now uses:

```gdscript
SimConfig.get_vision_radius_pixels()
```

- Lane minion detection/observation range now also uses:

```gdscript
SimConfig.get_vision_radius_pixels()
```

- With `vision_range = 5.0`, this becomes:

```text
5.0 * 30 = 150px
```

- Tower attack range is now `150px` instead of `170px`.
- Minion detection range is now `150px` instead of `180px`.
- Minion melee attack range remains unchanged at `26px`, so minions do not become long-range attackers.

### Tower Objective Reward And Episode Termination

- Added `tower_objective_reward_amount = 100.0` to `tower.gd`.
- When a tower is destroyed:
  - players on the destroyed tower's team receive `-100`,
  - players on the opposing team receive `+100`,
  - all player controllers are marked `done = true`,
  - Python receives `done`,
  - the episode terminates.
- If the PPO agent's own tower is destroyed:
  - PPO receives a large negative reward,
  - PPO episode ends with `done`.
- If the PPO agent destroys the enemy tower:
  - PPO receives a large positive reward,
  - PPO episode ends with `done`.
- Existing smaller rewards were kept:
  - `lane_tower_reward_amount`,
  - `minion_reward_amount`,
  - `agent_reward_amount`.
- The tower objective reward is currently the main terminal win/loss signal.

### Reward Shaping

- Added small reward-shaping signals in `controller.gd` to help PPO learn useful behavior faster.
- Added valid attack reward:
  - if the agent selects an attack action and a valid visible target exists, it receives `+0.02`.
- Added invalid attack penalty:
  - if the agent selects an attack action but no valid visible target exists, it receives `-0.02`.
- Added enemy-approach reward:
  - if the agent moves toward a visible enemy hero, it receives:

```gdscript
+0.02 * alignment
```

- This only applies when directional alignment is greater than `0.5`.
- Added lane-progress reward:
  - if the agent has visible allied minions and moves away from its allied tower, it receives:

```gdscript
+0.01 * alignment
```

- Added enemy-tower push reward:
  - if the agent has visible allied minions,
  - can see the enemy tower,
  - and moves toward the enemy tower,
  - it receives:

```gdscript
+0.03 * alignment
```

- Added enemy-tower danger penalty:
  - if the agent can see an enemy tower and there are no visible allied minions nearby, it receives `-0.05`.
  - This discourages the PPO agent from wandering under the enemy tower alone.
- Reduced the per-physics-tick living penalty in `character_body_2d.gd`:

```gdscript
reward_delta = -0.0001
```

- The shaping rewards are intentionally small so they do not override the main rewards:
  - damage reward,
  - kill reward,
  - death penalty,
  - tower objective reward.

### PPO Action Mapping

- Fixed PPO action mapping in `controller.gd`.
- `set_action()` now always uses the current 13-action mapping:

```gdscript
_action_index_to_type(action_index)
```

- Removed the old behavior where `action_index <= 8` used the legacy mapping.
- This ensures actions `4-7` are correctly interpreted as diagonal movement instead of old attack/retreat actions.

### Observation And Controller State

- `controller.gd` observations include:
  - self id, team, position, alive status, and HP,
  - visible enemy/ally agents,
  - visible objects such as minions and towers,
  - object relative positions when visible,
  - observed visible enemy actions in `extensions.observed_enemy_actions`.
- Controller state now tracks:
  - `current_action_type`,
  - `current_target_slot`,
  - `last_action_index`,
  - `last_action_type`.
- This supports both scripted strategies and PPO diagnostics.

