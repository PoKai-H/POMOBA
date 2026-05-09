### Set Up:

 1. Install Godot
 2. Pull the latest changes
 3. Create a Python virtual environment of your choice
 4. Run this command inside the environment: pip install -r rl_requirements.txt
 5. Activate vitrual environment: source <venv>/bin/activate

### Instructions to run with RL Agents:

 1. Open Godot
 2. In the main scene click the sync node and set the "Control Mode" to "Training".
 3. Open a terminal in the root directory of this repository
 4. Activate the Python environment
 5. Run core/run.py
 7. Press play in the Godot editor

# Experiment logs—Baseline

All graphs could be found under `outputs/training_analysis/*/`
All tables could be found under `evaluation/`
## Training
```bash
python core/run.py --config config/aggressiveExpert_aggressive.py
```


### Expert Guiding

- Added optional expert action mixing for PPO training.
- The learning agent can execute a scripted expert action with a decaying probability.
- The expert policy is configured by:  
```python

"EXPERT_MIX_STRATEGY": "neutral"

```

- The default expert ratio schedule is:
```python

"EXPERT_MIX_INITIAL_RATIO": 0.9

"EXPERT_MIX_FINAL_RATIO": 0.0

"EXPERT_MIX_DECAY_UPDATES": 80

```

- `core/run.py` computes `EXPERT_MIX_RATIO` for each update.
- During rollout:
	- with probability `EXPERT_MIX_RATIO`, the learning agent executes the expert action,
	- otherwise, it executes the PPO-sampled action.
- Expert-executed steps are marked with:
	- `expert_action`,
	- `expert_mask`
	- `ppo_actor_mask`.

- PPO actor loss is masked out on expert-executed steps to avoid treating expert actions as on-policy PPO samples.
- Critic/value learning still uses all rollout steps.
- Added behavior cloning loss:
```python

bc_loss = -log pi(expert_action | obs)

```
- `BC_COEF` controls the strength of this supervised imitation term.
- Training metrics now include:
	- `bc_loss`,
	- `expert_action_rate`.

- `run.py` rollout summaries and saved history include:
	- actual expert action rate,
	- configured expert ratio.
	- `action_diagnostics.png` plots both actual expert usage and the configured expert-ratio schedule.
## T1.1 PPO v.s neutral R0
pure ppo no expert
<img src="outputs/training_analysis/ppo_neutral_20260507_183401/action_diagnostics.png" width="720">
<img src="outputs/training_analysis/ppo_neutral_20260507_183401/combat_events.png" width="720">
<img src="outputs/training_analysis/ppo_neutral_20260507_183401/reward_diagnostics.png" width="720">
<img src="outputs/training_analysis/ppo_neutral_20260507_183401/training_curves.png" width="720">
## T1.2 neutralExpert v.s.neutral R0
with neutral expert guiding for 0.9 probability and decay to 0 on update 80
<img src="outputs/training_analysis/neutralExpert_neutral_20260507_205416/action_diagnostics.png" width="720">
<img src="outputs/training_analysis/neutralExpert_neutral_20260507_205416/combat_events.png" width="720">
<img src="outputs/training_analysis/neutralExpert_neutral_20260507_205416/reward_diagnostics.png" width="720">
<img src="outputs/training_analysis/neutralExpert_neutral_20260507_205416/training_curves.png" width="720">
## T1.3 farmingExpert v.s. neutral R0
<img src="outputs/training_analysis/farmingExpert_neutral_20260507_185120/action_diagnostics.png" width="720">
<img src="outputs/training_analysis/farmingExpert_neutral_20260507_185120/combat_events.png" width="720">
<img src="outputs/training_analysis/farmingExpert_neutral_20260507_185120/reward_diagnostics.png" width="720">
<img src="outputs/training_analysis/farmingExpert_neutral_20260507_185120/training_curves.png" width="720">
## T1.4 aggressiveExpert v.s. neutral R0
<img src="outputs/training_analysis/aggressiveExpert_neutral_20260507_174053/action_diagnostics.png" width="720">
<img src="outputs/training_analysis/aggressiveExpert_neutral_20260507_174053/combat_events.png" width="720">
<img src="outputs/training_analysis/aggressiveExpert_neutral_20260507_174053/reward_diagnostics.png" width="720">
<img src="outputs/training_analysis/aggressiveExpert_neutral_20260507_174053/training_curves.png" width="720">
> [!important]
>  T1.5 ~T1.8 comes from the thoughts of what if the enemies strategy affects the performance of our learning agent

## T1.5 aggressiveExpert v.s. aggressive
<img src="outputs/training_analysis/aggressiveExpert_aggressive_20260508_005238/action_diagnostics.png" width="720">
<img src="outputs/training_analysis/aggressiveExpert_aggressive_20260508_005238/combat_events.png" width="720">
<img src="outputs/training_analysis/aggressiveExpert_aggressive_20260508_005238/reward_diagnostics.png" width="720">
<img src="outputs/training_analysis/aggressiveExpert_aggressive_20260508_005238/training_curves.png" width="720">
## T1.6 aggressiveExpert v.s. farming 
<img src="outputs/training_analysis/aggressiveExpert_farming_20260508_182624/action_diagnostics.png" width="720">
<img src="outputs/training_analysis/aggressiveExpert_farming_20260508_182624/combat_events.png" width="720">
<img src="outputs/training_analysis/aggressiveExpert_farming_20260508_182624/reward_diagnostics.png" width="720">
<img src="outputs/training_analysis/aggressiveExpert_farming_20260508_182624/training_curves.png" width="720">
## T1.7 PPO v.s. aggressive
<img src="outputs/training_analysis/ppo_aggressive_20260508_013843/action_diagnostics.png" width="720">
<img src="outputs/training_analysis/ppo_aggressive_20260508_013843/combat_events.png" width="720">
<img src="outputs/training_analysis/ppo_aggressive_20260508_013843/reward_diagnostics.png" width="720">

<img src="outputs/training_analysis/ppo_aggressive_20260508_013843/training_curves.png" width="720">
## T1.8 PPO v.s. farming 
<img src="outputs/training_analysis/ppo_farming_20260508_022744/action_diagnostics.png" width="720">
<img src="outputs/training_analysis/ppo_farming_20260508_022744/combat_events.png" width="720">
<img src="outputs/training_analysis/ppo_farming_20260508_022744/reward_diagnostics.png" width="720">
<img src="outputs/training_analysis/ppo_farming_20260508_022744/training_curves.png" width="720">
## Eval 1 which expert performs the best?
```bash
python evaluation/evaluate_experts.py \                       
  outputs/training_analysis/aggressiveExpert_neutral_*/history.json \
  outputs/training_analysis/neutralExpert_neutral_*/history.json \
  outputs/training_analysis/farmingExpert_neutral_*/history.json
```

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

| rank | config                      | expert     | opponent   | score     | final_reward | post_expert_reward | post_expert_updates | death_count | enemy_minion_takedowns | enemy_agent_takedowns | bc_loss_delta | active_reward_delta | entropy |
| ---- | --------------------------- | ---------- | ---------- | --------- | ------------ | ------------------ | ------------------- | ----------- | ---------------------- | --------------------- | ------------- | ------------------- | ------- |
| 1    | aggressiveExpert_farming    | aggressive | farming    | 343.2103  | 117.9463     | 108.2278           | 24                  | 12.85       | 22.7                   | 19.0                  | -0.5347       | 18.289              | 1.47    |
| 2    | aggressiveExpert_neutral    | aggressive | neutral    | 268.9132  | 82.7515      | 81.8236            | 24                  | 14.65       | 18.5                   | 18.15                 | -0.6011       | 47.6843             | 1.4803  |
| 3    | farmingExpert_neutral       | farming    | neutral    | 237.4537  | 67.8168      | 67.5739            | 24                  | 13.95       | 17.05                  | 17.15                 | -0.4393       | 63.3962             | 1.5087  |
| 4    | neutralExpert_neutral       | neutral    | neutral    | 224.7529  | 63.1941      | 64.5175            | 24                  | 13.4        | 15.65                  | 16.5                  | -0.399        | 44.3449             | 1.4182  |
| 5    | aggressiveExpert_aggressive | aggressive | aggressive | 220.6981  | 60.0673      | 59.1614            | 24                  | 15.85       | 16.9                   | 17.65                 | -0.4555       | 28.1611             | 1.3679  |
| 6    | ppo_neutral                 | none       | neutral    | -65.825   | -71.0209     | -86.2082           | 100                 | 1.3         | 21.9                   | 1.1                   | 0.0           | 0.0                 | 2.0461  |
| 7    | ppo_farming                 | none       | farming    | -112.8581 | -94.5365     | -96.5431           | 100                 | 0.75        | 8.35                   | 1.7                   | 0.0           | 0.0                 | 2.4299  |
| 8    | ppo_aggressive              | none       | aggressive | -124.8972 | -95.5164     | -101.6617          | 100                 | 1.35        | 12.35                  | 0.35                  | 0.0           | 0.0                 | 2.4234  |

The result showing it against farming strategy opponent does perform a lot better

## Eval 2 Fixed-Opponent Robustness Summary
This table compares each trained checkpoint against fixed scripted opponents.
- `avg_reward_across_opponents`: mean reward averaged over opponent strategies.
- `worst_opponent_reward`: lowest mean reward among evaluated opponents.
- `robustness_gap`: best opponent reward minus worst opponent reward. Lower means more consistent.
- `avg_win_rate`: mean terminal-win rate across opponents. New results use terminal-step reward; older result files may have used total episode reward.
- `avg_deaths`: mean deaths across opponents.
- `avg_enemy_agent_takedowns`: mean enemy agent takedowns credited directly to the evaluated PPO agent.
- `avg_enemy_minion_takedowns`: mean enemy minion takedowns credited directly to the evaluated PPO agent.

only evaluate using the weight save on experiment 1s no training in this stage
### Eval 2.1 aggressiveExpert_neutral v.s. all 3 strategy
```bash
python evaluation/robustness_test.py \
  --checkpoint outputs/training_analysis/aggressiveExpert_neutral_<time>/checkpoints/final.pkl
```
### Eval 2.2 neutralExpert_neutral v.s. all 3 strategy
### Eval 2.3 farmingExpert_neutral v.s. all 3 strategy
### Eval 2.4 ppo_neutral v.s. all 3 strategy

### Eval 2.5 aggressiveExpert_farming v.s. all 3 strategy(not done)

| policy_checkpoint        | avg_reward_across_opponents | worst_opponent_reward | robustness_gap | avg_win_rate | avg_deaths | avg_enemy_agent_takedowns | avg_enemy_minion_takedowns |
| ------------------------ | --------------------------- | --------------------- | -------------- | ------------ | ---------- | ------------------------- | -------------------------- |
| aggressiveExpert_neutral | 140.953                     | 109.8512              | 51.082         | 0.0133       | 14.34      | 22.6733                   | 20.9333                    |
| aggressiveExpert_farming | 114.1531                    | 72.7278               | 73.3819        | 0.0          | 14.3333    | 20.1733                   | 21.7267                    |
| farmingExpert_neutral    | 93.7264                     | 67.2418               | 51.2219        | 0.0067       | 15.0133    | 21.0867                   | 8.28                       |
| neutralExpert_neutral    | -1.3345                     | -20.2767              | 28.6504        | 0.0          | 12.5533    | 14.42                     | 0.0                        |
| ppo_neutral              | -88.0237                    | -89.57                | 3.1506         | 0.0333       | 1.3667     | 0.0                       | 10.8                       |
## Ex3 Switching Strategy
```bash
python evaluation/robustness_test.py \
  --checkpoint outputs/training_analysis/ppo_neutral_20260507_183401/checkpoints/final.pkl \
  --eval-config strategySwitching_onlyOBS \
  --episodes 50
```

This table compares each trained checkpoint when the opponent strategy changes during an episode.

- `avg_reward_across_opponents`: mean reward across switching evaluation configs.
- `avg_win_rate`: mean terminal-win rate.
- `avg_deaths`: mean deaths.
- `avg_enemy_agent_takedowns`: mean enemy agent takedowns credited directly to the evaluated PPO agent.
- `avg_enemy_minion_takedowns`: mean enemy minion takedowns credited directly to the evaluated PPO agent.
- `avg_switch_reward_delta`: post-switch mean reward minus pre-switch mean reward.
- `avg_switch_reward_std_delta`: post-switch reward volatility minus pre-switch volatility.
- `avg_switch_action_shift`: L1 distance between pre-switch and post-switch action distributions.
### Eval 3.1 aggressive Expert_neutral
### Eval 3.2 ppo_neutral
### Eval 3.3 aggressiveExpert_farming(not done)

| policy_checkpoint        | eval_config               | avg_reward_across_opponents | avg_win_rate | avg_deaths | avg_enemy_agent_takedowns | avg_enemy_minion_takedowns | avg_switch_reward_delta | avg_switch_reward_std_delta | avg_switch_action_shift |
| ------------------------ | ------------------------- | --------------------------- | ------------ | ---------- | ------------------------- | -------------------------- | ----------------------- | --------------------------- | ----------------------- |
| aggressiveExpert_neutral | strategySwitching_onlyOBS | 139.3384                    | 0.02         | 14.62      | 22.8                      | 21.06                      | -0.0148                 | -0.1981                     | 0.496                   |
| aggressiveExpert_farming | strategySwitching_onlyOBS | 108.4256                    | 0.0          | 14.36      | 19.76                     | 20.82                      | 0.0262                  | -0.0334                     | 0.4976                  |
| ppo_neutral              | strategySwitching_onlyOBS | -82.988                     | 0.02         | 1.6        | 0.0                       | 13.46                      | -0.1487                 | 0.6957                      | 0.2013                  |

## Summarize all robustness test in one file
```bash
python evaluation/summarize_robustness.py 
```
will save to  2 md files, one fore fixed opponent strategy one for strategy switching
