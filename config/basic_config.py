env_config = {
    "map_name": "arena",
    "random_seed": 42,
    "max_steps": 1000,
    "agents_per_team": 1,
    "vision_range": 5.0,
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

episode_config = {
    "opponent_strategy": "aggressive",
    "strategy_switch_mode": "time_based",
}

basic_config = {
    "env_config": env_config,
    "episode_config": episode_config,
}

legacy_basic_config = {
    "core": {
        "map_name": env_config["map_name"],
        "random_seed": env_config["random_seed"],
        "max_steps": env_config["max_steps"],
        "agents_per_team": env_config["agents_per_team"],
        "vision_range": env_config["vision_range"],
        "opponent_strategy": episode_config["opponent_strategy"],
        "strategy_switch_mode": episode_config["strategy_switch_mode"],
    },
    "extensions": env_config["extensions"],
}
