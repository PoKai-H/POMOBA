extends AIController2D

# Stores the action sampled for the agent's policy, running in python
var move_action : Vector2 = Vector2.ZERO

func get_obs() -> Dictionary:
	if _player == null:
		return {"obs": [0.0, 0.0]}
	var pos: Vector2 = _player.global_position
	return {"obs": [pos.x, pos.y]}

func get_reward() -> float:	
	return reward
	
func get_action_space() -> Dictionary:
	return {
		"move" : {
			"size": 2,
			"action_type": "continuous"
		},
	}
	
func set_action(action) -> void:	
	if not action.has("move"):
		move_action = Vector2.ZERO
		return
	move_action.x = clamp(float(action["move"][0]), -1.0, 1.0)
	move_action.y = clamp(float(action["move"][1]), -1.0, 1.0)
	#var move := action["move"]
	#if move.size() < 2:
		#move_action = Vector2.ZERO
		#return
#
	#move_action.x = clamp(float(move[0]), -1.0, 1.0)
	#move_action.y = clamp(float(move[1]), -1.0, 1.0)
