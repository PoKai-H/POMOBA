# Belief Correctness Evaluation

- Evaluation config: `strategySwitching_onlyOBS`
- Learning agent scripted strategy: `neutral`

## Summary

| label | episodes | overall_accuracy | pre_switch_accuracy | post_switch_accuracy | mean_switch_detection_delay | mean_entropy | mean_belief_step_shift | mean_episode_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strategySwitching_onlyOBS | 50 | 0.5106 | 0.4923 | 0.4918 | 34.8163 | 0.9371 | 0.0528 | 26.8261 |

## Confusion Matrices

### strategySwitching_onlyOBS

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 13563 | 429 | 2664 |
| neutral | 9642 | 3138 | 3002 |
| farming | 3449 | 5283 | 8830 |
