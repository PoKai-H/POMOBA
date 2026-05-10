# Belief vs Observation Evaluation

This report compares downstream performance and strategy-switching stability between observation-only and belief-conditioned policies.

## Downstream Performance: Fixed Opponents

| policy | input | avg_reward | worst_reward | reward_gap | avg_win_rate | avg_deaths | avg_enemy_agent_takedowns | avg_enemy_minion_takedowns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral | obs | 140.953 | 109.8512 | 51.082 | 0.0133 | 14.34 | 22.6733 | 20.9333 |
| aggressiveExpert_neutral_belief | obs+belief | 119.9651 | 89.492 | 54.4178 | 0.0067 | 14.2467 | 21.5867 | 17.7533 |

## Switch Ability: Strategy-Switching Opponents

| policy | input | eval_config | avg_reward | avg_win_rate | avg_deaths | avg_enemy_agent_takedowns | avg_enemy_minion_takedowns | avg_switch_reward_delta | avg_switch_reward_std_delta | avg_switch_action_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral | obs | strategySwitching_onlyOBS | 139.3384 | 0.02 | 14.62 | 22.8 | 21.06 | -0.0148 | -0.1981 | 0.496 |
| aggressiveExpert_neutral_belief | obs+belief | strategySwitching_belief | 112.4393 | 0.0 | 14.0 | 20.86 | 17.48 | 0.0013 | 0.1022 | 0.4128 |

## Belief Minus Observation Deltas

`reward_delta > 0` means the belief-conditioned policy has higher average reward. `death_delta < 0` means fewer deaths. For switch metrics, positive `switch_reward_delta_delta` means the post-switch reward drop improved.

| belief_policy | obs_baseline | eval_type | eval_config | reward_delta | death_delta | agent_takedown_delta | minion_takedown_delta | switch_reward_delta_delta | switch_std_delta_delta | action_shift_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral_belief | aggressiveExpert_neutral | fixed | fixed_opponents | -20.9879 | -0.0933 | -1.0866 | -3.18 | 0.0 | 0.0 | 0.0 |

## Belief Ablation Results

These rows come from `robustness_belief_test.py` and compare correct, random, shuffled, and lagged belief inputs for the same belief-conditioned checkpoint.

_No results found._

## Interpretation Notes

- Fixed eval: `aggressiveExpert_neutral_belief` vs `aggressiveExpert_neutral` reward delta = `-20.9879`, death delta = `-0.0933`.

Interpret switch metrics jointly: lower action shift is not always better. A large action shift with stable reward can indicate adaptation, while a large action shift with worse reward indicates harmful instability.
