extends Node

const DEFAULT_CONFIG := {
	"core": {
		"map_name": "basic_arena",
		"random_seed": 0,
		"max_steps": 0,
		"agents_per_team": 0,
		"vision_range": 0,
		"opponent_strategy": "",
		"strategy_switch_mode": ""
	},
	"extensions": {
		"minion_wave": false,
		"tower": false,
		"jungle": false,
		"wards": false,
		"leveling": false,
		"skills": false,
		"mode_2v2": false
	}
}

var config: Dictionary = DEFAULT_CONFIG.duplicate(true)
