from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.beliefs.dummy_belief import DummyBelief
from core.envs.dummy_env import DummyEnv
from core.models.ppo_netowork import ActorCritic
from core.utils.obs_encoder import ObservationEncoder
from config.basic_config import basic_config
from godot_rl.core.godot_env import GodotEnv


import jax
import jax.numpy as jnp
import optax


NUM_ACTIONS = 13
USE_BELIEF_INPUT = False
PPO_CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LEARNING_RATE = 3e-4

def main():
    GodotEnv.DEFAULT_TIMEOUT = 180

    env = DummyEnv()


    obs_list, info = env.reset(basic_config)
    print(obs_list)

if __name__ == "__main__":
    main()