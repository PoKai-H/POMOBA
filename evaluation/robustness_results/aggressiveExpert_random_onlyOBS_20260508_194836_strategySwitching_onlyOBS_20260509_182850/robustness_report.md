# Robustness Evaluation

- Checkpoint: `outputs/training_analysis/aggressiveExpert_random_onlyOBS_20260508_194836/checkpoints/final.pkl`
- Policy mode: `deterministic`
- Evaluation config: `strategySwitching_onlyOBS`

## Summary

| opponent | episodes | mean_reward | mean_length | win_rate | mean_deaths | mean_enemy_agent_takedowns | mean_enemy_minion_takedowns | initial_strategy_counts | next_strategy_counts | mean_switch_reward_delta | mean_switch_reward_std_delta | mean_switch_action_shift | mean_pre_switch_attack_rate | mean_post_switch_attack_rate | mean_pre_switch_retreat_rate | mean_post_switch_retreat_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strategySwitching_onlyOBS | 50 | 138.2989 | 992.76 | 0.0 | 14.36 | 22.42 | 21.82 | {'neutral': 21, 'farming': 15, 'aggressive': 14} | {'farming': 18, 'aggressive': 19, 'neutral': 13} | -0.0097 | -0.1384 | 0.5656 | 0.7056 | 0.7264 | 0.0284 | 0.0248 |
