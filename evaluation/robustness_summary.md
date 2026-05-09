# Robustness Summary

This table compares each trained checkpoint across all evaluated opponents.

- `avg_reward_across_opponents`: mean reward averaged over opponent strategies.
- `worst_opponent_reward`: lowest mean reward among evaluated opponents.
- `robustness_gap`: best opponent reward minus worst opponent reward. Lower means more consistent.
- `avg_win_rate`: mean terminal-win rate across opponents. New results use terminal-step reward; older result files may have used total episode reward.
- `avg_deaths`: mean deaths across opponents.
- `avg_enemy_agent_takedowns`: mean enemy agent takedowns credited directly to the evaluated PPO agent.
- `avg_enemy_minion_takedowns`: mean enemy minion takedowns credited directly to the evaluated PPO agent.

- `avg_switch_reward_delta`: post-switch mean reward minus pre-switch mean reward.
- `avg_switch_reward_std_delta`: post-switch reward volatility minus pre-switch volatility.
- `avg_switch_action_shift`: L1 distance between pre-switch and post-switch action distributions.

| policy_checkpoint | eval_config | avg_reward_across_opponents | worst_opponent_reward | robustness_gap | avg_win_rate | avg_deaths | avg_enemy_agent_takedowns | avg_enemy_minion_takedowns | avg_switch_reward_delta | avg_switch_reward_std_delta | avg_switch_action_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral | fixed_opponents | 140.953 | 109.8512 | 51.082 | 0.0133 | 14.34 | 22.6733 | 20.9333 | 0.0 | 0.0 | 0.0 |
| aggressiveExpert_neutral | strategySwitching_onlyOBS | 139.3384 | 139.3384 | 0.0 | 0.02 | 14.62 | 22.8 | 21.06 | -0.0148 | -0.1981 | 0.496 |
| farmingExpert_neutral | fixed_opponents | 93.7264 | 67.2418 | 51.2219 | 0.0067 | 15.0133 | 21.0867 | 8.28 | 0.0 | 0.0 | 0.0 |
| neutralExpert_neutral | fixed_opponents | -1.3345 | -20.2767 | 28.6504 | 0.0 | 12.5533 | 14.42 | 0.0 | 0.0 | 0.0 | 0.0 |
| ppo_neutral | fixed_opponents | -88.0237 | -89.57 | 3.1506 | 0.0333 | 1.3667 | 0.0 | 10.8 | 0.0 | 0.0 | 0.0 |
