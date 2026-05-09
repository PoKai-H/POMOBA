# Robustness Evaluation

- Checkpoint: `outputs/training_analysis/aggressiveExpert_farming_20260508_182624/checkpoints/final.pkl`
- Policy mode: `deterministic`
- Evaluation config: `strategySwitching_onlyOBS`

## Summary

| opponent | episodes | mean_reward | mean_length | win_rate | mean_deaths | mean_enemy_agent_takedowns | mean_enemy_minion_takedowns | initial_strategy_counts | next_strategy_counts | mean_switch_reward_delta | mean_switch_reward_std_delta | mean_switch_action_shift | mean_pre_switch_attack_rate | mean_post_switch_attack_rate | mean_pre_switch_retreat_rate | mean_post_switch_retreat_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strategySwitching_onlyOBS | 50 | 108.4256 | 979.2 | 0.0 | 14.36 | 19.76 | 20.82 | {'neutral': 21, 'farming': 15, 'aggressive': 14} | {'farming': 18, 'aggressive': 19, 'neutral': 13} | 0.0262 | -0.0334 | 0.4976 | 0.7172 | 0.7232 | 0.0 | 0.0 |
