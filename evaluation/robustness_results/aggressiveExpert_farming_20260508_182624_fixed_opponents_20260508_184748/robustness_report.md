# Robustness Evaluation

- Checkpoint: `outputs/training_analysis/aggressiveExpert_farming_20260508_182624/checkpoints/final.pkl`
- Policy mode: `deterministic`
- Evaluation config: `fixed_opponents`

## Summary

| opponent | episodes | mean_reward | mean_length | win_rate | mean_deaths | mean_enemy_agent_takedowns | mean_enemy_minion_takedowns | initial_strategy_counts | next_strategy_counts | mean_switch_reward_delta | mean_switch_reward_std_delta | mean_switch_action_shift | mean_pre_switch_attack_rate | mean_post_switch_attack_rate | mean_pre_switch_retreat_rate | mean_post_switch_retreat_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive | 50 | 72.7278 | 1000.0 | 0.0 | 16.84 | 18.56 | 21.16 | {'aggressive': 50} | {} | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neutral | 50 | 123.6218 | 986.76 | 0.0 | 13.98 | 20.48 | 22.64 | {'neutral': 50} | {} | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| farming | 50 | 146.1097 | 975.72 | 0.0 | 12.18 | 21.48 | 21.38 | {'farming': 50} | {} | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
