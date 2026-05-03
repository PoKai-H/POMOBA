extends Node

const DEFAULT_ENV_CONFIG := {
	"map_name": "arena",
	"random_seed": 42,
	"max_steps": 300,
	"agents_per_team": 1,
	"vision_range": 5.0,
	"extensions": {
		"minion_wave": true,
		"tower": true,
		"jungle": true,
		"wards": false,
		"leveling": false,
		"skills": false,
		"mode_2v2": false
	}
}

const DEFAULT_EPISODE_CONFIG := {
	"opponent_strategy": "aggressive",
	"strategy_switch_mode": "time_based"
}

var env_config: Dictionary = DEFAULT_ENV_CONFIG.duplicate(true)
var episode_config: Dictionary = DEFAULT_EPISODE_CONFIG.duplicate(true)
var config: Dictionary = {}


func _ready() -> void:
	_refresh_merged_config()


func reset_to_defaults() -> void:
	env_config = DEFAULT_ENV_CONFIG.duplicate(true)
	episode_config = DEFAULT_EPISODE_CONFIG.duplicate(true)
	_refresh_merged_config()


func apply_config_message(message: Dictionary) -> void:
	if message.has("env_config") or message.has("episode_config"):
		_apply_split_config(message)
	else:
		_apply_legacy_config(message)
	_refresh_merged_config()


func get_env_value(key: String, fallback = null):
	return env_config.get(key, fallback)


func get_episode_value(key: String, fallback = null):
	return episode_config.get(key, fallback)


func get_extension_value(key: String, fallback = null):
	var extensions: Dictionary = env_config.get("extensions", {})
	return extensions.get(key, fallback)


func set_env_value(key: String, value) -> void:
	env_config[key] = value
	_refresh_merged_config()


func set_episode_value(key: String, value) -> void:
	episode_config[key] = value
	_refresh_merged_config()


func set_extension_value(key: String, value) -> void:
	var extensions: Dictionary = env_config.get("extensions", {}).duplicate(true)
	extensions[key] = value
	env_config["extensions"] = extensions
	_refresh_merged_config()


func get_vision_radius_pixels() -> float:
	return float(get_env_value("vision_range", 5.0)) * 30.0


func get_agents_per_team() -> int:
	var count := int(get_env_value("agents_per_team", 1))
	if bool(get_extension_value("mode_2v2", false)):
		return maxi(count, 2)
	return maxi(count, 1)


func _apply_split_config(message: Dictionary) -> void:
	var env_update: Dictionary = message.get("env_config", {})
	var episode_update: Dictionary = message.get("episode_config", {})

	env_config = _deep_merge(env_config, env_update)
	episode_config = _deep_merge(episode_config, episode_update)


func _apply_legacy_config(message: Dictionary) -> void:
	var core: Dictionary = message.get("core", {})
	var extensions: Dictionary = message.get("extensions", {})

	env_config = _deep_merge(DEFAULT_ENV_CONFIG, env_config)
	episode_config = _deep_merge(DEFAULT_EPISODE_CONFIG, episode_config)

	for key in ["map_name", "random_seed", "max_steps", "agents_per_team", "vision_range"]:
		if core.has(key):
			env_config[key] = core[key]

	var merged_extensions: Dictionary = env_config.get("extensions", {}).duplicate(true)
	for key in extensions.keys():
		merged_extensions[key] = extensions[key]
	env_config["extensions"] = merged_extensions

	for key in ["opponent_strategy", "strategy_switch_mode"]:
		if core.has(key):
			episode_config[key] = core[key]


func _refresh_merged_config() -> void:
	config = {
		"core": {
			"map_name": env_config.get("map_name", DEFAULT_ENV_CONFIG["map_name"]),
			"random_seed": env_config.get("random_seed", DEFAULT_ENV_CONFIG["random_seed"]),
			"max_steps": env_config.get("max_steps", DEFAULT_ENV_CONFIG["max_steps"]),
			"agents_per_team": env_config.get("agents_per_team", DEFAULT_ENV_CONFIG["agents_per_team"]),
			"vision_range": env_config.get("vision_range", DEFAULT_ENV_CONFIG["vision_range"]),
			"opponent_strategy": episode_config.get("opponent_strategy", DEFAULT_EPISODE_CONFIG["opponent_strategy"]),
			"strategy_switch_mode": episode_config.get("strategy_switch_mode", DEFAULT_EPISODE_CONFIG["strategy_switch_mode"])
		},
		"extensions": env_config.get("extensions", {}).duplicate(true)
	}


func _deep_merge(base: Dictionary, override: Dictionary) -> Dictionary:
	var merged := base.duplicate(true)
	for key in override.keys():
		if merged.get(key) is Dictionary and override[key] is Dictionary:
			merged[key] = _deep_merge(merged[key], override[key])
		else:
			merged[key] = override[key]
	return merged
