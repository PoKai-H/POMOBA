# Appendix: Metric and Field Definitions

This appendix collects the definitions of the metrics used in the evaluation
tables. All reported values are aggregated over evaluation episodes or final
training windows, depending on the table.

## Common Identifiers

| field | definition |
| --- | --- |
| `policy` | Name of the evaluated policy or checkpoint. |
| `policy_checkpoint` | Saved PPO checkpoint used for evaluation. |
| `config` | Training configuration name. |
| `eval_config` | Evaluation configuration name. For example, `fixed_opponents`, `strategySwitching_onlyOBS`, or `strategySwitching_belief`. |
| `input` | Policy input type. `obs` means observation-only input; `obs+belief` means the policy receives both environment observations and the belief vector. |
| `expert` | Scripted expert used during expert-mixing training. `none` means no expert action mixing. |
| `opponent` | Scripted opponent used during training. |
| `belief_mode` | Belief signal used during belief ablation. `correct` uses the normal Bayesian belief, `random` replaces it with random belief vectors, `shuffled` permutes belief dimensions, and `lagged` uses stale belief vectors. |

## Reward and Outcome Metrics

| field | definition | direction |
| --- | --- | --- |
| `avg_reward` | Mean episode reward over the evaluated episodes. | Higher is better. |
| `avg_reward_across_opponents` | Mean episode reward averaged across evaluated opponent strategies. | Higher is better. |
| `final_reward` | Mean episode reward over the final training/evaluation window. | Higher is better. |
| `post_expert_reward` | Mean reward after the expert mixing ratio has decayed below the active expert threshold. | Higher is better. |
| `worst_reward` | Lowest mean reward among evaluated opponent settings. | Higher is better. |
| `worst_opponent_reward` | Lowest mean reward against any fixed opponent strategy. | Higher is better. |
| `reward_gap` | Difference between best and worst opponent reward in a robustness evaluation. | Lower means more consistent. |
| `robustness_gap` | Same as `reward_gap`: best opponent reward minus worst opponent reward. | Lower means more consistent. |
| `avg_win_rate` | Fraction of episodes ending in a terminal win for the PPO agent. | Higher is better. |
| `death_count` | Mean number of PPO agent deaths in the final training/evaluation window. | Lower is usually better. |
| `avg_deaths` | Mean number of PPO agent deaths per episode or opponent setting. | Lower is usually better. |
| `enemy_agent_takedowns` | Mean number of enemy agent takedowns credited to the PPO agent in the final training window. | Higher is better. |
| `avg_enemy_agent_takedowns` | Mean enemy agent takedowns credited to the PPO agent across evaluation episodes. | Higher is better. |
| `avg_agent_takedowns` | Same as `avg_enemy_agent_takedowns`, shortened for paper tables. | Higher is better. |
| `enemy_minion_takedowns` | Mean number of enemy minion takedowns credited to the PPO agent in the final training window. | Higher is better. |
| `avg_enemy_minion_takedowns` | Mean enemy minion takedowns credited to the PPO agent across evaluation episodes. | Higher is better. |
| `avg_minion_takedowns` | Same as `avg_enemy_minion_takedowns`, shortened for paper tables. | Higher is better. |

## Expert Evaluation Metrics

Expert curricula are ranked with the following heuristic score:

```text
score =
  final_reward
  + 0.5 * post_expert_reward
  - 5.0 * death_count
  + 2.0 * enemy_minion_takedowns
  + 10.0 * enemy_agent_takedowns
```

| field | definition | direction |
| --- | --- | --- |
| `rank` | Rank of the training configuration according to `score`. | Lower rank number is better. |
| `score` | Heuristic expert-evaluation score combining reward, post-expert stability, deaths, minion takedowns, and agent takedowns. | Higher is better. |
| `post_expert_updates` | Number of updates included after expert mixing has mostly decayed. | Used for context. |
| `bc_loss_delta` | Change in behavior cloning loss across training. It measures imitation fit, not final task performance. | More negative usually means better imitation. |
| `active_reward_delta` | Reward change during the period where expert mixing is still active. | Higher suggests the expert phase improved reward faster. |
| `entropy` | Mean policy entropy in the final window. Higher entropy means a more stochastic policy. | Context-dependent. |

## Strategy-Switching Metrics

For switching evaluations, the opponent changes strategy during the episode. The
following metrics compare behavior before and after the switch.

| field | definition | direction |
| --- | --- | --- |
| `avg_switch_reward_delta` | Post-switch mean reward minus pre-switch mean reward, averaged across episodes. | Closer to zero or positive is usually better. |
| `switch_reward_delta` | Same as `avg_switch_reward_delta`, shortened for paper tables. | Closer to zero or positive is usually better. |
| `avg_switch_reward_std_delta` | Post-switch reward standard deviation minus pre-switch reward standard deviation. Positive values mean reward became more volatile after the switch. | Lower is usually more stable. |
| `switch_reward_std_delta` | Same as `avg_switch_reward_std_delta`, shortened for paper tables. | Lower is usually more stable. |
| `avg_switch_action_shift` | L1 distance between pre-switch and post-switch action distributions. Higher means the policy changed behavior more strongly after the opponent switched. | Context-dependent; lower means smoother behavior, higher may mean stronger adaptation. |
| `switch_action_shift` | Same as `avg_switch_action_shift`, shortened for paper tables. | Context-dependent. |

## Belief Correctness Metrics

The belief model estimates a probability distribution over the opponent strategy
set:

```text
[aggressive, neutral, farming]
```

The belief update used in the experiments is:

```text
target(theta) = normalize(P(observed_action | theta, context))
belief_t(theta) = (1 - alpha) * belief_{t-1}(theta) + alpha * target(theta)
```

| field | definition | direction |
| --- | --- | --- |
| `accuracy` | Fraction of steps where the most likely belief strategy matches the true opponent strategy. | Higher is better. |
| `overall_accuracy` | Belief classification accuracy across the whole episode. | Higher is better. |
| `pre_switch_accuracy` | Belief accuracy before the opponent strategy switch. | Higher is better. |
| `post_switch_accuracy` | Belief accuracy after the opponent strategy switch. | Higher is better. |
| `mean_switch_detection_delay` | Mean number of steps needed for the belief argmax to match the new strategy after a switch. | Lower is better. |
| `switch_detection_delay` | Same as `mean_switch_detection_delay`, shortened for paper tables. | Lower is better. |
| `mean_entropy` | Mean entropy of the belief distribution. Higher values mean more uncertainty. | Lower means more confident belief, but overconfidence can be harmful. |
| `belief_entropy` | Same as `mean_entropy`, shortened for paper tables. | Context-dependent. |
| `mean_belief_step_shift` | Mean step-to-step change in the belief vector. Larger values indicate more reactive belief updates. | Context-dependent. |
| `belief_accuracy` | Belief accuracy during robustness or ablation evaluation. | Higher is better. |
| `true strategy` | Ground-truth scripted opponent strategy in a confusion matrix. | Label. |
| `predicted aggressive` | Number of steps where the belief argmax predicted `aggressive`. | Count. |
| `predicted neutral` | Number of steps where the belief argmax predicted `neutral`. | Count. |
| `predicted farming` | Number of steps where the belief argmax predicted `farming`. | Count. |

## Belief-vs-Observation Delta Metrics

Delta rows compare the belief-conditioned policy against the observation-only
baseline:

```text
delta = belief_policy_metric - observation_policy_metric
```

| field | definition | direction |
| --- | --- | --- |
| `comparison` | Description of the compared policies, usually `belief - observation`. | Label. |
| `belief_policy` | Belief-conditioned policy used in the comparison. | Label. |
| `obs_baseline` | Observation-only baseline policy used in the comparison. | Label. |
| `eval_type` | Evaluation category, such as fixed opponents or strategy switching. | Label. |
| `reward_delta` | Difference in average reward. Negative means belief performed worse than observation-only. | Higher is better. |
| `death_delta` | Difference in average deaths. Negative means belief had fewer deaths. | Lower is better. |
| `agent_takedown_delta` | Difference in enemy agent takedowns. Positive means belief achieved more takedowns. | Higher is better. |
| `minion_takedown_delta` | Difference in enemy minion takedowns. Positive means belief achieved more minion takedowns. | Higher is better. |
| `switch_reward_delta_delta` | Difference in switch reward delta between belief and observation-only policies. Positive means belief lost less reward or gained more reward after switching. | Higher is usually better. |
| `switch_std_delta_delta` | Difference in post-switch reward volatility change. Negative means belief reduced volatility more. | Lower is usually better. |
| `action_shift_delta` | Difference in action distribution shift. Negative means belief changed actions less after the switch. | Context-dependent. |

## Belief Ablation Metrics

The perturbed-belief ablation tests whether the policy uses the semantic content
of the belief vector.

| field | definition | interpretation |
| --- | --- | --- |
| `correct` | Normal Bayesian belief vector. | Baseline for belief-conditioned evaluation. |
| `random` | Random belief vector at evaluation time. | Tests whether any extra input noise performs similarly to belief. |
| `shuffled` | Correct belief values with dimensions permuted. | Tests whether the meaning of each belief dimension matters. |
| `lagged` | Belief vector from earlier steps. | Tests whether current belief is more useful than stale belief. |

If `correct` outperforms `random`, the policy likely uses belief information
beyond extra input dimensions. If `correct` outperforms `shuffled`, the policy is
sensitive to the semantic ordering of belief dimensions. If `correct` outperforms
`lagged`, timely opponent-state inference is useful.

