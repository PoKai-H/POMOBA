# Robustness Evaluation

- Checkpoint: `outputs/training_analysis/aggressiveExpert_neutral_belief_20260509_224537/checkpoints/final.pkl`
- Policy mode: `deterministic`
- Evaluation config: `fixed_opponents`

## Summary

| opponent | episodes | mean_reward | mean_length | win_rate | mean_deaths | mean_enemy_agent_takedowns | mean_enemy_minion_takedowns | initial_strategy_counts | next_strategy_counts | mean_switch_reward_delta | mean_switch_reward_std_delta | mean_switch_action_shift | mean_pre_switch_attack_rate | mean_post_switch_attack_rate | mean_pre_switch_retreat_rate | mean_post_switch_retreat_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive | 50 | 89.492 | 970.66 | 0.0 | 15.46 | 20.24 | 17.56 | {'aggressive': 50} | {} | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neutral | 50 | 126.4935 | 983.9 | 0.0 | 14.44 | 22.02 | 17.88 | {'neutral': 50} | {} | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| farming | 50 | 143.9098 | 979.18 | 0.02 | 12.84 | 22.5 | 17.82 | {'farming': 50} | {} | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
