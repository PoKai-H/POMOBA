# Reproduction Instructions

This file lists the terminal commands needed to reproduce the current training,
evaluation, robustness, belief, and plotting results.

Most commands assume the Godot simulator is available. If no exported game
binary is configured, start the Godot editor and press Play before running the
Python command. The Python process will wait for the Godot connection.

# Setting Up:

 1. Pull the latest changes
 2. Create a Python virtual environment of your choice
 3. Run this command inside the environment: pip install -r rl_requirements.txt
 4. Activate vitrual environment: source <venv>/bin/activate

## 1. Instructions to run with RL Agents:

Run these from the project root. The godot executable works on MacOS and Windows.

### Observation-only baselines

```bash
python core/run.py --config config/ppo_neutral.py
python core/run.py --config config/ppo_aggressive.py
python core/run.py --config config/ppo_farming.py
```

### Expert-guided observation-only policies

```bash
python core/run.py --config config/aggressiveExpert_neutral.py
python core/run.py --config config/aggressiveExpert_aggressive.py
python core/run.py --config config/aggressiveExpert_farming.py
python core/run.py --config config/farmingExpert_neutral.py
python core/run.py --config config/neutralExpert_neutral.py
python core/run.py --config config/aggressiveExpert_random_onlyOBS.py
```

### Belief-conditioned policy

```bash
python core/run.py --config config/aggressiveExpert_neutral_belief.py
```

Each run creates a timestamped folder:

```text
outputs/training_analysis/<config_name>_<timestamp>/
```

The folder contains:

```text
history.json
run_config.json
checkpoints/final.pkl
```



## 2. Generate Training Plots

Use `plot.py` after a training run finishes.

```bash
python plot.py --folder outputs/training_analysis/aggressiveExpert_neutral_20260507_174053
```

Equivalent forms:

```bash
python plot.py outputs/training_analysis/aggressiveExpert_neutral_20260507_174053
python plot.py --aggressiveExpert_neutral_20260507_174053
```

Generated files:

```text
training_curves.png
action_diagnostics.png
reward_diagnostics.png
combat_events.png
```

To regenerate plots for the current saved runs:

```bash
python plot.py --ppo_neutral_20260507_183401
python plot.py --ppo_aggressive_20260508_013843
python plot.py --ppo_farming_20260508_022744
python plot.py --aggressiveExpert_neutral_20260507_174053
python plot.py --aggressiveExpert_aggressive_20260508_005238
python plot.py --aggressiveExpert_farming_20260508_182624
python plot.py --farmingExpert_neutral_20260507_185120
python plot.py --neutralExpert_neutral_20260507_205416
python plot.py --aggressiveExpert_random_onlyOBS_20260508_194836
python plot.py --aggressiveExpert_neutral_belief_20260509_224537
```

## 3. Rank Expert Curricula

This prints the expert ranking table to the terminal.

```bash
python evaluation/evaluate_experts.py
```

Optional JSON/CSV output:

```bash
python evaluation/evaluate_experts.py \
  --save-json evaluation/expert_evaluation_results.json \
  --save-csv evaluation/expert_evaluation_results.csv
```

## 4. Evaluate Fixed-Opponent Robustness

Set checkpoint variables for readability:

```bash
OBS_CKPT=outputs/training_analysis/aggressiveExpert_neutral_20260507_174053/checkpoints/final.pkl
RANDOM_OBS_CKPT=outputs/training_analysis/aggressiveExpert_random_onlyOBS_20260508_194836/checkpoints/final.pkl
FARMING_EXPERT_CKPT=outputs/training_analysis/aggressiveExpert_farming_20260508_182624/checkpoints/final.pkl
NEUTRAL_EXPERT_CKPT=outputs/training_analysis/neutralExpert_neutral_20260507_205416/checkpoints/final.pkl
FARMING_NEUTRAL_CKPT=outputs/training_analysis/farmingExpert_neutral_20260507_185120/checkpoints/final.pkl
PPO_NEUTRAL_CKPT=outputs/training_analysis/ppo_neutral_20260507_183401/checkpoints/final.pkl
BELIEF_CKPT=outputs/training_analysis/aggressiveExpert_neutral_belief_20260509_224537/checkpoints/final.pkl
```

Run fixed-opponent robustness:

```bash
python evaluation/robustness_test.py --checkpoint "$OBS_CKPT" --episodes 50
python evaluation/robustness_test.py --checkpoint "$RANDOM_OBS_CKPT" --episodes 50
python evaluation/robustness_test.py --checkpoint "$FARMING_EXPERT_CKPT" --episodes 50
python evaluation/robustness_test.py --checkpoint "$NEUTRAL_EXPERT_CKPT" --episodes 50
python evaluation/robustness_test.py --checkpoint "$FARMING_NEUTRAL_CKPT" --episodes 50
python evaluation/robustness_test.py --checkpoint "$PPO_NEUTRAL_CKPT" --episodes 50
python evaluation/robustness_belief_test.py --checkpoint "$BELIEF_CKPT" --episodes 50 --belief-mode correct
```

Each command saves a JSON result under:

```text
evaluation/robustness_results/
evaluation/belief_results/
```

and prints the markdown-style table to the terminal.

## 5. Evaluate Strategy-Switching Robustness

Observation-only switching evaluation:

```bash
python evaluation/robustness_test.py \
  --checkpoint "$OBS_CKPT" \
  --eval-config strategySwitching_onlyOBS \
  --episodes 50

python evaluation/robustness_test.py \
  --checkpoint "$RANDOM_OBS_CKPT" \
  --eval-config strategySwitching_onlyOBS \
  --episodes 50

python evaluation/robustness_test.py \
  --checkpoint "$FARMING_EXPERT_CKPT" \
  --eval-config strategySwitching_onlyOBS \
  --episodes 50

python evaluation/robustness_test.py \
  --checkpoint "$PPO_NEUTRAL_CKPT" \
  --eval-config strategySwitching_onlyOBS \
  --episodes 50
```

Belief-conditioned switching evaluation:

```bash
python evaluation/robustness_belief_test.py \
  --checkpoint "$BELIEF_CKPT" \
  --eval-config strategySwitching_belief \
  --episodes 50 \
  --belief-mode correct
```

## 6. Summarize Robustness Results

Print fixed-opponent and strategy-switching summary tables:

```bash
python evaluation/summarize_robustness.py
```

Optional JSON/CSV output:

```bash
python evaluation/summarize_robustness.py \
  --save-json evaluation/robustness_summary.json \
  --save-csv evaluation/robustness_summary.csv
```

## 7. Evaluate Belief Correctness

### Fixed opponents

```bash
python evaluation/evaluate_belief_correctness.py \
  --base-config aggressiveExpert_neutral_belief \
  --episodes 20 \
  --learning-strategy aggressive
```

### Strategy-switching opponent

```bash
python evaluation/evaluate_belief_correctness.py \
  --eval-config strategySwitching_onlyOBS \
  --episodes 50 \
  --learning-strategy aggressive
```

These commands save JSON under:

```text
evaluation/belief_correctness_results/
```

and print the belief correctness tables to the terminal.

## 8. Run Belief Ablations

These commands evaluate the same belief-trained checkpoint while changing the
belief vector fed into the policy.

```bash
python evaluation/robustness_belief_test.py \
  --checkpoint "$BELIEF_CKPT" \
  --eval-config strategySwitching_belief \
  --episodes 50 \
  --belief-mode correct

python evaluation/robustness_belief_test.py \
  --checkpoint "$BELIEF_CKPT" \
  --eval-config strategySwitching_belief \
  --episodes 50 \
  --belief-mode random

python evaluation/robustness_belief_test.py \
  --checkpoint "$BELIEF_CKPT" \
  --eval-config strategySwitching_belief \
  --episodes 50 \
  --belief-mode shuffled

python evaluation/robustness_belief_test.py \
  --checkpoint "$BELIEF_CKPT" \
  --eval-config strategySwitching_belief \
  --episodes 50 \
  --belief-mode lagged
```

## 9. Compare Belief vs Observation

Print downstream fixed-opponent, switching, delta, and belief-ablation tables:

```bash
python evaluation/compare_belief_vs_observation.py \
  --include aggressiveExpert_neutral aggressiveExpert_neutral_belief
```


## 10. Current Saved Run Folders

The current saved training folders used for the reported results are:

```text
outputs/training_analysis/ppo_neutral_20260507_183401
outputs/training_analysis/ppo_aggressive_20260508_013843
outputs/training_analysis/ppo_farming_20260508_022744
outputs/training_analysis/aggressiveExpert_neutral_20260507_174053
outputs/training_analysis/aggressiveExpert_aggressive_20260508_005238
outputs/training_analysis/aggressiveExpert_farming_20260508_182624
outputs/training_analysis/farmingExpert_neutral_20260507_185120
outputs/training_analysis/neutralExpert_neutral_20260507_205416
outputs/training_analysis/aggressiveExpert_random_onlyOBS_20260508_194836
outputs/training_analysis/aggressiveExpert_neutral_belief_20260509_224537
```

