# Robustness Evaluation

- Checkpoint: `outputs/training_analysis/aggressiveExpert_neutral_belief_20260509_224537/checkpoints/final.pkl`
- Policy mode: `deterministic`
- Evaluation config: `strategySwitching_belief`

## Summary

| opponent | episodes | mean_reward | mean_length | win_rate | mean_deaths | mean_enemy_agent_takedowns | mean_enemy_minion_takedowns | initial_strategy_counts | next_strategy_counts | mean_switch_reward_delta | mean_switch_reward_std_delta | mean_switch_action_shift | mean_pre_switch_attack_rate | mean_post_switch_attack_rate | mean_pre_switch_retreat_rate | mean_post_switch_retreat_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strategySwitching_belief | 50 | 112.4393 | 964.64 | 0.0 | 14.0 | 20.86 | 17.48 | {'neutral': 21, 'farming': 15, 'aggressive': 14} | {'farming': 18, 'aggressive': 19, 'neutral': 13} | 0.0013 | 0.1022 | 0.4128 | 0.7212 | 0.7112 | 0.0 | 0.0 |
