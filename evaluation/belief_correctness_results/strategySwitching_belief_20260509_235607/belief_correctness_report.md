# Belief Correctness Evaluation

- Evaluation config: `strategySwitching_belief`
- Learning agent scripted strategy: `neutral`

## Summary

| label | episodes | overall_accuracy | pre_switch_accuracy | post_switch_accuracy | mean_switch_detection_delay | mean_entropy | mean_belief_step_shift | mean_episode_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strategySwitching_belief | 50 | 0.4943 | 0.4724 | 0.4928 | 38.3043 | 0.9352 | 0.0528 | 21.1066 |

## Confusion Matrices

### strategySwitching_belief

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 13869 | 455 | 2332 |
| neutral | 9836 | 2871 | 3075 |
| farming | 3503 | 6084 | 7975 |
