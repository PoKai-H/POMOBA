"""
Simple scripted strategy example for Godot RL Agents.
This demonstrates basic Python-Godot communication without using ML frameworks.
"""

import numpy as np
from godot_rl.core.godot_env import GodotEnv


def simple_scripted_policy(observation):
    """
    A simple scripted policy that makes decisions based on observations.
    Replace this with your custom logic.
    
    Args:
        observation (dict): Dictionary containing observation data from Godot
        
    Returns:
        action: Action to send to the environment
    """
    # Example: Access observation data
    # observation is a dict with keys like "obs", "image", etc.
    actions = []
    for env_obs in observation:
        # Minimal scripted movement: drift right if x <= 0, else drift left.
        x = env_obs.get("obs", [0.0, 0.0])[0]
        move_x = 1.0 if x <= 0.0 else -1.0
        actions.append([move_x, 0.0])
    return actions


def main():
    # Give the editor more time to start and connect when using Play.
    GodotEnv.DEFAULT_TIMEOUT = 180

    # Initialize the environment
    # If env_path is None, it will wait for you to press PLAY in the Godot editor
    env = GodotEnv(
        env_path=None,  # Set to path of exported Godot binary, or None for in-editor training
        port=11008,
        show_window=False,
    )
    
    # Reset the environment
    obs, info = env.reset()
    
    done = [False] * env.num_envs
    episode_rewards = [0.0] * env.num_envs
    steps = 0
    max_steps = 1000
    
    print(f"Starting scripted strategy training with {env.num_envs} environment(s)")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Main loop
    while not all(done) and steps < max_steps:
        # Get action from scripted policy
        move_actions = simple_scripted_policy(obs)

        # GodotEnv expects actions grouped by action branch, then by env index.
        # Here we have one branch ("move") with 2 continuous values.
        action = [np.asarray(move_actions, dtype=np.float32)]
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Track rewards
        for i in range(env.num_envs):
            episode_rewards[i] += reward[i]
            if done[i] or steps >= max_steps - 1:
                print(f"Episode {steps}: Agent {i} - Reward: {episode_rewards[i]:.2f}")
                episode_rewards[i] = 0.0
        
        steps += 1
        
        # Log periodically
        if steps % 100 == 0:
            print(f"Step {steps}")
    
    # Close the environment
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()