# Robustness Evaluation

- Checkpoint: `outputs/training_analysis/aggressiveExpert_random_onlyOBS_20260508_194836/checkpoints/final.pkl`
- Policy mode: `deterministic`
- Evaluation config: `fixed_opponents`

## Summary

| opponent | episodes | mean_reward | mean_length | win_rate | mean_deaths | mean_enemy_agent_takedowns | mean_enemy_minion_takedowns | initial_strategy_counts | next_strategy_counts | mean_switch_reward_delta | mean_switch_reward_std_delta | mean_switch_action_shift | mean_pre_switch_attack_rate | mean_post_switch_attack_rate | mean_pre_switch_retreat_rate | mean_post_switch_retreat_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive | 50 | 101.118 | 980.9 | 0.0 | 16.12 | 21.18 | 19.44 | {'aggressive': 50} | {'neutral': 50} | 0.097 | 0.04 | 0.4104 | 0.6912 | 0.7604 | 0.0356 | 0.0072 |
| neutral | 50 | 138.3288 | 993.04 | 0.0 | 14.24 | 22.5 | 20.56 | {'neutral': 50} | {'neutral': 50} | 0.0815 | -0.2436 | 0.468 | 0.6916 | 0.752 | 0.0488 | 0.022 |
| farming | 50 | 161.1765 | 982.62 | 0.0 | 13.2 | 23.38 | 22.06 | {'farming': 50} | {'neutral': 50} | 0.1376 | -0.4114 | 0.472 | 0.7344 | 0.8128 | 0.0672 | 0.0124 |
