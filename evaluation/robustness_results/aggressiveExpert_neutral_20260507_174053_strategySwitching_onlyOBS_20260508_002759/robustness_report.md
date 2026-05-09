# Robustness Evaluation

- Checkpoint: `outputs/training_analysis/aggressiveExpert_neutral_20260507_174053/checkpoints/final.pkl`
- Policy mode: `deterministic`
- Evaluation config: `strategySwitching_onlyOBS`

## Summary

| opponent | episodes | mean_reward | mean_length | win_rate | mean_deaths | mean_enemy_agent_takedowns | mean_enemy_minion_takedowns | initial_strategy_counts | next_strategy_counts | mean_switch_reward_delta | mean_switch_reward_std_delta | mean_switch_action_shift | mean_pre_switch_attack_rate | mean_post_switch_attack_rate | mean_pre_switch_retreat_rate | mean_post_switch_retreat_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strategySwitching_onlyOBS | 50 | 139.3384 | 995.92 | 0.02 | 14.62 | 22.8 | 21.06 | {'neutral': 21, 'farming': 15, 'aggressive': 14} | {'farming': 18, 'aggressive': 19, 'neutral': 13} | -0.0148 | -0.1981 | 0.496 | 0.7112 | 0.7408 | 0.0184 | 0.0128 |
