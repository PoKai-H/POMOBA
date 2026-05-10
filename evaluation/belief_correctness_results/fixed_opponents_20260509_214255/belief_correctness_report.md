# Belief Correctness Evaluation

- Evaluation config: `fixed_opponents`
- Learning agent scripted strategy: `neutral`

## Summary

| label | episodes | overall_accuracy | pre_switch_accuracy | post_switch_accuracy | mean_switch_detection_delay | mean_entropy | mean_belief_step_shift | mean_episode_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive | 20 | 0.8909 | 0.8909 | 0.0 | None | 0.9205 | 0.0308 | -16.0853 |
| neutral | 20 | 0.1497 | 0.1497 | 0.0 | None | 0.9268 | 0.0441 | 30.1256 |
| farming | 20 | 0.4357 | 0.4357 | 0.0 | None | 0.9272 | 0.0576 | 61.1136 |

## Confusion Matrices

### aggressive

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 17818 | 21 | 2161 |
| neutral | 0 | 0 | 0 |
| farming | 0 | 0 | 0 |

### neutral

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 14111 | 2994 | 2895 |
| farming | 0 | 0 | 0 |

### farming

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 |
| farming | 4695 | 6592 | 8713 |
