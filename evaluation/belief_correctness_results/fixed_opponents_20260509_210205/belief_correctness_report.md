# Belief Correctness Evaluation

- Evaluation config: `fixed_opponents`
- Learning agent scripted strategy: `neutral`

## Summary

| label | episodes | overall_accuracy | pre_switch_accuracy | post_switch_accuracy | mean_switch_detection_delay | mean_entropy | mean_belief_step_shift | mean_episode_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive | 20 | 0.7211 | 0.7211 | 0.0 | None | 1.0815 | 0.014 | -3.687 |
| neutral | 20 | 0.1568 | 0.1568 | 0.0 | None | 1.0293 | 0.0375 | 16.7768 |
| farming | 20 | 0.4586 | 0.4586 | 0.0 | None | 1.0302 | 0.0424 | 37.3317 |

## Confusion Matrices

### aggressive

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 14421 | 67 | 5512 |
| neutral | 0 | 0 | 0 |
| farming | 0 | 0 | 0 |

### neutral

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 13127 | 3135 | 3738 |
| farming | 0 | 0 | 0 |

### farming

| true \ predicted | aggressive | neutral | farming |
| --- | --- | --- | --- |
| aggressive | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 |
| farming | 6207 | 4146 | 8876 |
