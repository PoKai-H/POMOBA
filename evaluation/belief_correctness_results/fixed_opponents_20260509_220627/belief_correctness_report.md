# Belief Correctness Evaluation

- Evaluation config: `fixed_opponents`
- Learning agent scripted strategy: `neutral`

## Summary

| label | episodes | overall_accuracy | pre_switch_accuracy | post_switch_accuracy | mean_switch_detection_delay | mean_entropy | mean_belief_step_shift | mean_episode_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive | 20 | 0.8419 | 0.8419 | 0.0 | None | 0.951 | 0.0425 | -5.1006 |
| neutral | 20 | 0.2645 | 0.2645 | 0.0 | None | 0.9385 | 0.0542 | 20.3508 |
| farming | 20 | 0.5096 | 0.5096 | 0.0 | None | 0.9124 | 0.0602 | 53.1323 |

## Confusion Matrices

### aggressive

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 16838 | 53 | 3109 |
| neutral | 0 | 0 | 0 |
| farming | 0 | 0 | 0 |

### neutral

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 10869 | 5290 | 3841 |
| farming | 0 | 0 | 0 |

### farming

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 |
| farming | 2975 | 6832 | 10193 |
