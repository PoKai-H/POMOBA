# Experimental Setup

We evaluate PPO agents in a simplified MOBA-like environment with partial
observations, scripted opponents, minion waves, and lane towers. The learning
agent observes nearby agents and objects, while the opponent strategy is hidden.

The scripted opponent strategy set is:

```text
aggressive
neutral
farming
```

The learning agent uses a 13-action discrete action space:

```text
move directions, hold, attack hero, attack nearest minion, attack tower, retreat
```

Training and evaluation are separated. Training uses PPO with optional expert
action mixing. Evaluation uses saved checkpoints only.

# Expert-Guided PPO Training

To make PPO learn basic lane behavior, we used expert action mixing. During
training, the learning agent executes a scripted expert action with a decaying
probability:

```text
EXPERT_MIX_INITIAL_RATIO = 0.9
EXPERT_MIX_FINAL_RATIO = 0.0
EXPERT_MIX_DECAY_UPDATES = 80
```

Expert-executed steps are excluded from PPO actor loss but still used for critic
learning. A behavior cloning loss is added:

```text
bc_loss = -log pi(expert_action | observation)
```

The expert curriculum is ranked with:

```text
score =
  final_reward
  + 0.5 * post_expert_reward
  - 5.0 * death_count
  + 2.0 * enemy_minion_takedowns
  + 10.0 * enemy_agent_takedowns
```

## Expert Ranking

| rank | config | expert | opponent | score | final_reward | post_expert_reward | death_count | enemy_minion_takedowns | enemy_agent_takedowns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | aggressiveExpert_farming | aggressive | farming | 343.2103 | 117.9463 | 108.2278 | 12.85 | 22.7 | 19.0 |
| 2 | aggressiveExpert_neutral | aggressive | neutral | 268.9132 | 82.7515 | 81.8236 | 14.65 | 18.5 | 18.15 |
| 3 | farmingExpert_neutral | farming | neutral | 237.4537 | 67.8168 | 67.5739 | 13.95 | 17.05 | 17.15 |
| 4 | aggressiveExpert_random_onlyOBS | aggressive | neutral | 235.0159 | 65.3244 | 67.083 | 14.35 | 18.7 | 17.05 |
| 5 | neutralExpert_neutral | neutral | neutral | 224.7529 | 63.1941 | 64.5175 | 13.4 | 15.65 | 16.5 |
| 6 | aggressiveExpert_aggressive | aggressive | aggressive | 220.6981 | 60.0673 | 59.1614 | 15.85 | 16.9 | 17.65 |
| 7 | ppo_neutral | none | neutral | -65.825 | -71.0209 | -86.2082 | 1.3 | 21.9 | 1.1 |
| 8 | ppo_farming | none | farming | -112.8581 | -94.5365 | -96.5431 | 0.75 | 8.35 | 1.7 |
| 9 | ppo_aggressive | none | aggressive | -124.8972 | -95.5164 | -101.6617 | 1.35 | 12.35 | 0.35 |

The strongest overall expert curriculum was aggressive expert guidance. We use
`aggressiveExpert_neutral` as the main observation-only baseline for the belief
comparison.

# Observation-Only Robustness

We evaluate saved checkpoints against fixed scripted opponents and against
opponents that switch strategy during the episode.

## Fixed-Opponent Robustness

| policy_checkpoint | avg_reward_across_opponents | worst_opponent_reward | robustness_gap | avg_win_rate | avg_deaths | avg_enemy_agent_takedowns | avg_enemy_minion_takedowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral | 140.953 | 109.8512 | 51.082 | 0.0133 | 14.34 | 22.6733 | 20.9333 |
| aggressiveExpert_random_onlyOBS | 133.5411 | 101.118 | 60.0585 | 0.0 | 14.52 | 22.3533 | 20.6867 |
| aggressiveExpert_farming | 114.1531 | 72.7278 | 73.3819 | 0.0 | 14.3333 | 20.1733 | 21.7267 |
| farmingExpert_neutral | 93.7264 | 67.2418 | 51.2219 | 0.0067 | 15.0133 | 21.0867 | 8.28 |
| neutralExpert_neutral | -1.3345 | -20.2767 | 28.6504 | 0.0 | 12.5533 | 14.42 | 0.0 |
| ppo_neutral | -88.0237 | -89.57 | 3.1506 | 0.0333 | 1.3667 | 0.0 | 10.8 |

## Strategy-Switching Robustness

| policy_checkpoint | eval_config | avg_reward_across_opponents | avg_win_rate | avg_deaths | avg_enemy_agent_takedowns | avg_enemy_minion_takedowns | avg_switch_reward_delta | avg_switch_reward_std_delta | avg_switch_action_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral | strategySwitching_onlyOBS | 139.3384 | 0.02 | 14.62 | 22.8 | 21.06 | -0.0148 | -0.1981 | 0.496 |
| aggressiveExpert_random_onlyOBS | strategySwitching_onlyOBS | 138.2989 | 0.0 | 14.36 | 22.42 | 21.82 | -0.0097 | -0.1384 | 0.5656 |
| aggressiveExpert_farming | strategySwitching_onlyOBS | 108.4256 | 0.0 | 14.36 | 19.76 | 20.82 | 0.0262 | -0.0334 | 0.4976 |
| ppo_neutral | strategySwitching_onlyOBS | -82.988 | 0.02 | 1.6 | 0.0 | 13.46 | -0.1487 | 0.6957 | 0.2013 |

Training directly against random switching opponents did not outperform fixed
neutral training for observation-only PPO. This suggests that switching
opponents increase training noise unless the agent has an explicit signal for
opponent strategy.

# Belief Model

The belief model estimates a distribution over:

```text
[aggressive, neutral, farming]
```

The first rule-based version failed because it used a hard-coded opponent id.
Godot uses runtime `instance_id` values, so the belief module often failed to
read the actual opponent action. After fixing dynamic enemy id lookup, the
belief model was rewritten as a likelihood-based tracker aligned with the
scripted strategies.

The update is:

```text
target(theta) = normalize(P(observed_action | theta, context))
belief_t(theta) = (1 - alpha) * belief_{t-1}(theta) + alpha * target(theta)
```

The likelihood uses:

- observed opponent action,
- rolling hero-attack vs object-attack ratio,
- target type (`hero`, `minion`, `tower`),
- ally minion HP drops,
- ally tower HP drops,
- lane pressure.

Godot observations were extended from:

```text
observed_enemy_actions[enemy_id] = action_index
```

to:

```text
observed_enemy_actions[enemy_id] = {
  action,
  action_type,
  target_type
}
```

# Belief Correctness

Before training a belief-conditioned policy, we tested whether the belief module
can infer scripted strategies from observations.

## Fixed-Opponent Belief Correctness

| strategy | accuracy |
| --- | --- |
| aggressive | 0.8419 |
| neutral | 0.2645 |
| farming | 0.5096 |

| true strategy | predicted aggressive | predicted neutral | predicted farming |
| --- | --- | --- | --- |
| aggressive | 16838 | 53 | 3109 |
| neutral | 10869 | 5290 | 3841 |
| farming | 2975 | 6832 | 10193 |

The belief model identifies aggressive behavior well and farming behavior
moderately well. Neutral remains difficult because its policy mixes hero
pressure and object pressure, making it behaviorally similar to both aggressive
and farming under partial observability.

## Switching-Opponent Belief Correctness

| metric | value |
| --- | --- |
| overall accuracy | 0.5106 |
| pre-switch accuracy | 0.4923 |
| post-switch accuracy | 0.4918 |
| mean switch detection delay | 34.8163 |
| mean entropy | 0.9371 |
| mean belief step shift | 0.0528 |

| true strategy | predicted aggressive | predicted neutral | predicted farming |
| --- | --- | --- | --- |
| aggressive | 13563 | 429 | 2664 |
| neutral | 9642 | 3138 | 3002 |
| farming | 3449 | 5283 | 8830 |

The belief is imperfect but informative: switching accuracy is above random
chance, and the mean detection delay is about 35 steps.

# Observation vs Belief-Conditioned PPO

We trained a belief-conditioned version of the strongest observation-only
curriculum:

```text
aggressiveExpert_neutral
aggressiveExpert_neutral_belief
```

The only intended policy-input difference is:

```text
observation-only: pi(a | o)
belief-conditioned: pi(a | o, b)
```

## Downstream Performance on Fixed Opponents

| policy | input | avg_reward | worst_reward | reward_gap | avg_win_rate | avg_deaths | avg_agent_takedowns | avg_minion_takedowns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral | obs | 140.953 | 109.8512 | 51.082 | 0.0133 | 14.34 | 22.6733 | 20.9333 |
| aggressiveExpert_neutral_belief | obs+belief | 119.9651 | 89.492 | 54.4178 | 0.0067 | 14.2467 | 21.5867 | 17.7533 |

![Fixed downstream performance](evaluation/paper_belief_analysis/downstream_fixed.png)

## Strategy-Switching Stability

| policy | input | eval_config | avg_reward | avg_win_rate | avg_deaths | avg_agent_takedowns | avg_minion_takedowns | switch_reward_delta | switch_reward_std_delta | switch_action_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral | obs | strategySwitching_onlyOBS | 139.3384 | 0.02 | 14.62 | 22.8 | 21.06 | -0.0148 | -0.1981 | 0.496 |
| aggressiveExpert_neutral_belief | obs+belief | strategySwitching_belief | 112.4393 | 0.0 | 14.0 | 20.86 | 17.48 | 0.0013 | 0.1022 | 0.4128 |

![Switching stability](evaluation/paper_belief_analysis/switching_stability.png)

## Belief Minus Observation Deltas

| comparison | reward_delta | death_delta | agent_takedown_delta | minion_takedown_delta | switch_reward_delta_delta | switch_std_delta_delta | action_shift_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| belief - observation | -20.9879 | -0.0933 | -1.0866 | -3.18 | 0.0 | 0.0 | 0.0 |
| belief - observation | -26.8991 | -0.62 | -1.94 | -3.58 | 0.0161 | 0.3003 | -0.0832 |

The belief-conditioned policy did not improve average reward. However, under
strategy switching it had slightly lower death count and lower action
distribution shift. This suggests the belief input may reduce some behavioral
instability, but the imperfect belief signal and added input complexity did not
translate into stronger downstream reward.

# Perturbed-Belief Ablation

To test whether the belief-conditioned policy uses the semantic content of the
belief vector, we evaluated the same trained belief policy under four belief
inputs:

```text
correct
random
shuffled
lagged
```

Training always used the normal Bayesian belief. Only evaluation-time belief
input is perturbed.

| policy | belief_mode | eval_config | avg_reward | avg_win_rate | avg_deaths | avg_agent_takedowns | avg_minion_takedowns | belief_accuracy | post_switch_accuracy | switch_detection_delay | belief_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral_belief | correct | strategySwitching_belief | 126.9331 | 0.0 | 14.96 | 22.34 | 16.98 | 0.5328 | 0.4826 | 40.625 | 0.9596 |
| aggressiveExpert_neutral_belief | lagged | strategySwitching_belief | 124.7269 | 0.0 | 14.42 | 21.88 | 18.24 | 0.5319 | 0.4908 | 36.6522 | 0.9605 |
| aggressiveExpert_neutral_belief | random | strategySwitching_belief | 127.2314 | 0.0 | 14.26 | 22.02 | 17.8 | 0.5183 | 0.474 | 40.1087 | 0.9632 |
| aggressiveExpert_neutral_belief | shuffled | strategySwitching_belief | 119.277 | 0.04 | 13.5 | 20.94 | 16.76 | 0.4907 | 0.4613 | 36.587 | 0.9688 |

![Belief ablation](evaluation/paper_belief_analysis/belief_ablation.png)

The shuffled-belief condition performs worse than correct belief, suggesting
that the ordering and semantics of the belief vector matter. However, random
belief performs similarly to correct belief, which indicates that the policy did
not consistently exploit the full correctness of the Bayesian estimate.

# Main Takeaways

1. Expert-guided PPO is necessary for learning useful lane behavior in this
   environment. Pure PPO performs poorly.

2. Fixed neutral training produced a stronger observation-only policy than
   training directly against random strategy switching.

3. The Bayesian belief model is informative but imperfect. It detects
   aggressive strategy well, farming moderately, and struggles with neutral
   because neutral is behaviorally mixture-like.

4. Belief-conditioned PPO did not improve average reward over the
   observation-only baseline in the current experiments.

5. Belief-conditioned PPO slightly reduced action distribution shift under
   strategy switching, suggesting some stabilizing effect, but this did not
   translate into higher reward.

6. Perturbed-belief ablation suggests that belief semantics matter somewhat
   because shuffled belief hurts performance, but random belief remaining close
   to correct belief suggests the learned policy does not rely strongly enough
   on belief correctness.

Overall, the experiments support a nuanced conclusion: explicit belief modeling
provides a measurable and interpretable opponent-strategy signal, but the
current handcrafted belief and policy training pipeline are not yet sufficient
to improve end-task reward over a strong observation-only expert-guided PPO
baseline.
