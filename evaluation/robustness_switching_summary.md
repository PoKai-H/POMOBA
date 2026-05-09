# Strategy-Switching Robustness Summary

This table compares each trained checkpoint when the opponent strategy changes during an episode.

- `avg_reward_across_opponents`: mean reward across switching evaluation configs.
- `avg_win_rate`: mean terminal-win rate.
- `avg_deaths`: mean deaths.
- `avg_enemy_agent_takedowns`: mean enemy agent takedowns credited directly to the evaluated PPO agent.
- `avg_enemy_minion_takedowns`: mean enemy minion takedowns credited directly to the evaluated PPO agent.
- `avg_switch_reward_delta`: post-switch mean reward minus pre-switch mean reward.
- `avg_switch_reward_std_delta`: post-switch reward volatility minus pre-switch volatility.
- `avg_switch_action_shift`: L1 distance between pre-switch and post-switch action distributions.

| policy_checkpoint | eval_config | avg_reward_across_opponents | avg_win_rate | avg_deaths | avg_enemy_agent_takedowns | avg_enemy_minion_takedowns | avg_switch_reward_delta | avg_switch_reward_std_delta | avg_switch_action_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral | strategySwitching_onlyOBS | 139.3384 | 0.02 | 14.62 | 22.8 | 21.06 | -0.0148 | -0.1981 | 0.496 |
| aggressiveExpert_farming | strategySwitching_onlyOBS | 108.4256 | 0.0 | 14.36 | 19.76 | 20.82 | 0.0262 | -0.0334 | 0.4976 |
| ppo_neutral | strategySwitching_onlyOBS | -82.988 | 0.02 | 1.6 | 0.0 | 13.46 | -0.1487 | 0.6957 | 0.2013 |
