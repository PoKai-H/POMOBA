# Belief Correctness Evaluation

- Evaluation config: `fixed_opponents`
- Learning agent scripted strategy: `neutral`

## Summary

| label | episodes | overall_accuracy | pre_switch_accuracy | post_switch_accuracy | mean_switch_detection_delay | mean_entropy | mean_belief_step_shift | mean_episode_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive | 20 | 0.2899 | 0.2899 | 0.0 | None | 0.7584 | 0.1094 | -8.0721 |
| neutral | 20 | 0.7204 | 0.7204 | 0.0 | None | 0.7137 | 0.1146 | 23.8113 |
| farming | 20 | 0.2212 | 0.2212 | 0.0 | None | 0.6413 | 0.1127 | 56.8379 |

## Confusion Matrices

### aggressive

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 5798 | 12952 | 1250 |
| neutral | 0 | 0 | 0 |
| farming | 0 | 0 | 0 |

### neutral

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 2441 | 14407 | 3152 |
| farming | 0 | 0 | 0 |

### farming

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 |
| farming | 1159 | 14418 | 4423 |
