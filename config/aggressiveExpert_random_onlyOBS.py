from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if sys.platform == "win32":
    godot_env_path = ROOT / "exports" / "windows" / "pomoba"
elif sys.platform == "darwin":
    godot_env_path = ROOT / "exports" / "mac" / "pomoba"
else:
    godot_env_path = None

basic_config = {
    "GODOT_ENV_PATH": str(godot_env_path) if godot_env_path else None,
    "GODOT_PORT": 11008,
    "SHOW_WINDOW": True,
    "core": {
        "map_name": "arena",
        "random_seed": 42,
        "max_steps": 1000,
        "agents_per_team": 1,
        "vision_range": 5.0,
    },
    "extensions": {
        "minion_wave": True,
        "tower": True,
        "jungle": True,
        "wards": False,
        "leveling": False,
        "skills": False,
        "mode_2v2": False,
    },
}


training_config = {
    "SEED": 42,
    "NUM_ACTIONS": 13,
    "USE_BELIEF_INPUT": False,
    "TIMESTEP_PER_BATCH": 1024,
    "MAX_STEPS_PER_EPISODE": 1000,
    "TOTAL_TIMESTEPS": 102_400,
    "LEARNING_AGENT_ID": 0,
    "LR": 3e-4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "VF_COEF": 0.5,
    "ENT_COEF": 0.003,
    "BC_COEF": 0.1,
    "EXPERT_MIX_STRATEGY": "aggressive",
    "EXPERT_MIX_INITIAL_RATIO": 0.9,
    "EXPERT_MIX_FINAL_RATIO": 0.0,
    "EXPERT_MIX_DECAY_UPDATES": 80,
    "CHECKPOINT_EVERY_UPDATES": 0,
    "MAX_GRAD_NORM": 0.5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "HIDDEN_DIM": 64,
    "ACTIVATION": "tanh",
    "NPC_POLICY_ID_SOURCE": "index",
    "opponent_strategy": "neutral",
    "strategy_switch_mode": "random_time",
    "DEFAULT_NPC_POLICY": "neutral",
    "NPC_POLICIES": {
        1: "neutral",
    },
    "RANDOMIZE_NPC_POLICY_EACH_EPISODE": True,
    "NPC_POLICY_IDS": [1],
    "NPC_STRATEGY_POOL": [
        "aggressive",
        "farming",
        "neutral",
    ],
    "NPC_SWITCH_STEP_RANGE": [150, 850],
}
