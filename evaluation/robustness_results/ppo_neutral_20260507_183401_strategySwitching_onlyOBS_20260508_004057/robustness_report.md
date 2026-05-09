# Robustness Evaluation

- Checkpoint: `outputs/training_analysis/ppo_neutral_20260507_183401/checkpoints/final.pkl`
- Policy mode: `deterministic`
- Evaluation config: `strategySwitching_onlyOBS`

## Summary

| opponent | episodes | mean_reward | mean_length | win_rate | mean_deaths | mean_enemy_agent_takedowns | mean_enemy_minion_takedowns | initial_strategy_counts | next_strategy_counts | mean_switch_reward_delta | mean_switch_reward_std_delta | mean_switch_action_shift | mean_pre_switch_attack_rate | mean_post_switch_attack_rate | mean_pre_switch_retreat_rate | mean_post_switch_retreat_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strategySwitching_onlyOBS | 50 | -82.988 | 523.94 | 0.02 | 1.6 | 0.0 | 13.46 | {'neutral': 21, 'farming': 15, 'aggressive': 14} | {'farming': 18, 'aggressive': 19, 'neutral': 13} | -0.1487 | 0.6957 | 0.2013 | 0.4452 | 0.4609 | 0.0948 | 0.0591 |
