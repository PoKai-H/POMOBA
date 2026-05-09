# Expert Evaluation Score

The score is a heuristic ranking metric for comparing which expert curriculum
helped PPO learn the strongest final policy.

Formula:

```text
score =
  final_reward
  + 0.5 * post_expert_reward
  - 5.0 * death_count
  + 2.0 * enemy_minion_takedowns
  + 10.0 * enemy_agent_takedowns
```

Term meanings:

- `final_reward`: mean episode reward over the final evaluation window.
- `post_expert_reward`: mean episode reward after the expert ratio has faded below the threshold.
- `death_count`: mean number of PPO agent deaths per update in the final window.
- `enemy_minion_takedowns`: mean number of enemy minion takedowns per update in the final window.
- `enemy_agent_takedowns`: mean number of enemy agent takedowns per update in the final window.

Interpretation:

- Higher score is better.
- The score rewards final performance, post-expert stability, lane progress, and combat success.
- The score penalizes deaths.
- Behavior-cloning diagnostics such as `bc_loss_delta` are reported separately and are not part of the score, because they measure imitation quality rather than final task performance.

## Ranking Table

| rank | config | expert | opponent | score | final_reward | post_expert_reward | post_expert_updates | death_count | enemy_minion_takedowns | enemy_agent_takedowns | bc_loss_delta | active_reward_delta | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | aggressiveExpert_farming | aggressive | farming | 343.2103 | 117.9463 | 108.2278 | 24 | 12.85 | 22.7 | 19.0 | -0.5347 | 18.289 | 1.47 |
| 2 | aggressiveExpert_neutral | aggressive | neutral | 268.9132 | 82.7515 | 81.8236 | 24 | 14.65 | 18.5 | 18.15 | -0.6011 | 47.6843 | 1.4803 |
| 3 | farmingExpert_neutral | farming | neutral | 237.4537 | 67.8168 | 67.5739 | 24 | 13.95 | 17.05 | 17.15 | -0.4393 | 63.3962 | 1.5087 |
| 4 | neutralExpert_neutral | neutral | neutral | 224.7529 | 63.1941 | 64.5175 | 24 | 13.4 | 15.65 | 16.5 | -0.399 | 44.3449 | 1.4182 |
| 5 | aggressiveExpert_aggressive | aggressive | aggressive | 220.6981 | 60.0673 | 59.1614 | 24 | 15.85 | 16.9 | 17.65 | -0.4555 | 28.1611 | 1.3679 |
| 6 | ppo_neutral | none | neutral | -65.825 | -71.0209 | -86.2082 | 100 | 1.3 | 21.9 | 1.1 | 0.0 | 0.0 | 2.0461 |
| 7 | ppo_farming | none | farming | -112.8581 | -94.5365 | -96.5431 | 100 | 0.75 | 8.35 | 1.7 | 0.0 | 0.0 | 2.4299 |
| 8 | ppo_aggressive | none | aggressive | -124.8972 | -95.5164 | -101.6617 | 100 | 1.35 | 12.35 | 0.35 | 0.0 | 0.0 | 2.4234 |
