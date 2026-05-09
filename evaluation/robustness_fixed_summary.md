# Fixed-Opponent Robustness Summary

This table compares each trained checkpoint against fixed scripted opponents.

- `avg_reward_across_opponents`: mean reward averaged over opponent strategies.
- `worst_opponent_reward`: lowest mean reward among evaluated opponents.
- `robustness_gap`: best opponent reward minus worst opponent reward. Lower means more consistent.
- `avg_win_rate`: mean terminal-win rate across opponents. New results use terminal-step reward; older result files may have used total episode reward.
- `avg_deaths`: mean deaths across opponents.
- `avg_enemy_agent_takedowns`: mean enemy agent takedowns credited directly to the evaluated PPO agent.
- `avg_enemy_minion_takedowns`: mean enemy minion takedowns credited directly to the evaluated PPO agent.

| policy_checkpoint | avg_reward_across_opponents | worst_opponent_reward | robustness_gap | avg_win_rate | avg_deaths | avg_enemy_agent_takedowns | avg_enemy_minion_takedowns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aggressiveExpert_neutral | 140.953 | 109.8512 | 51.082 | 0.0133 | 14.34 | 22.6733 | 20.9333 |
| aggressiveExpert_farming | 114.1531 | 72.7278 | 73.3819 | 0.0 | 14.3333 | 20.1733 | 21.7267 |
| farmingExpert_neutral | 93.7264 | 67.2418 | 51.2219 | 0.0067 | 15.0133 | 21.0867 | 8.28 |
| neutralExpert_neutral | -1.3345 | -20.2767 | 28.6504 | 0.0 | 12.5533 | 14.42 | 0.0 |
| ppo_neutral | -88.0237 | -89.57 | 3.1506 | 0.0333 | 1.3667 | 0.0 | 10.8 |
