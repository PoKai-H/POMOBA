# Belief Correctness Evaluation

- Evaluation config: `fixed_opponents`
- Learning agent scripted strategy: `neutral`

## Summary

| label | episodes | overall_accuracy | pre_switch_accuracy | post_switch_accuracy | mean_switch_detection_delay | mean_entropy | mean_belief_step_shift | mean_episode_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive | 20 | 0.0748 | 0.0748 | 0.0 | None | 0.9 | 0.0172 | -12.0726 |
| neutral | 20 | 0.9257 | 0.9257 | 0.0 | None | 0.9153 | 0.0181 | 32.5856 |
| farming | 20 | 0.0 | 0.0 | 0.0 | None | 0.92 | 0.0189 | 63.5539 |

## Confusion Matrices

### aggressive

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 1495 | 18505 | 0 |
| neutral | 0 | 0 | 0 |
| farming | 0 | 0 | 0 |

### neutral

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 1487 | 18513 | 0 |
| farming | 0 | 0 | 0 |

### farming

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 |
| farming | 1082 | 18918 | 0 |
