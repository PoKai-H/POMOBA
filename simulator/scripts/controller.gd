extends AIController2D

# Stores the action sampled for the agent's policy, running in python
var move_action : Vector2 = Vector2.ZERO

func get_obs() -> Dictionary:
	if _player == null:
		return {"obs": [0.0, 0.0]}
	var pos: Vector2 = _player.global_position
	
	var obs = {
		"timestep": 1, #int
		"self": {
			"id": 1, #int
			"team": "blue", #str,
			"hp": get_info()["hp"], #float
			"position": [pos.x, pos.y],
			"status": { "alive": get_info()["alive"] } 
		},
		"agents": [ # only other agents 
		{	"id": 2, #int
			"team": "red", #str,
			"visible": false, #bool,
			"relative_position": [0., 40.], # [dx, dy] | None,
			"hp": 50., #float | None,
			"status": {
				"alive": true #bool
				}
		} ],
		"objects": [
		{
			"id": 3, #int
			"type": "tower", #str,
			"team": "blue", #str | None,
			"visible": true, #bool,
			"relative_position": [59., 43.],
			"status":{
				"alive": true #bool
			},
		} ], 
		"extensions": {}
	}
	
	return {"obs": obs}

func get_reward() -> float:	
	return reward


func get_info() -> Dictionary:
	return {
		"hp": _player.hp if _player != null and is_instance_valid(_player) else 0.0,
		"alive": _player != null and is_instance_valid(_player) and _player.is_alive()
	}
	
func get_action_space() -> Dictionary:
	return {
		"action" : {
			"size": 13,
			"action_type": "discrete"
		},
	}

func _action_index_to_move(action_index: int) -> Vector2:
	"""Convert discrete action index (0-8) to movement direction.
	
	Actions:
	0: move up      		  -> [0, -1]
	1: move left              -> [-1, 0]
	2: move right             -> [1, 0]
	3: move down		      -> [0, 1]
	4: move up-left           -> [-1, -1]
	5: move up-right          -> [1, -1]
	6: move down-left         -> [-1, 1]
	7: move down-right        -> [1, 1]
	8: hold (no movement)     -> [0, 0]
	"""
	match action_index:
		0:
			return Vector2(0, -1)  # forward
		1:
			return Vector2(-1, 0)  # left
		2:
			return Vector2(1, 0)   # right
		3:
			return Vector2(0, 1)   # back
		4:
			return Vector2(-1, -1) # up-left
		5:
			return Vector2(1, -1)  # up-right
		6:
			return Vector2(-1, 1)  # down-left
		7:
			return Vector2(1, 1)   # down-right
		_:
			return Vector2.ZERO    # hold / default
	
func set_action(action) -> void:	
	if not action.has("action"):
		move_action = Vector2.ZERO
		return
	var action_index = action["action"]
	var direction = _action_index_to_move(int(action_index))
	move_action = direction.normalized() if direction != Vector2.ZERO else Vector2.ZERO
	#var move := action["move"]
	#if move.size() < 2:
		#move_action = Vector2.ZERO
		#return
#
	#move_action.x = clamp(float(move[0]), -1.0, 1.0)
	#move_action.y = clamp(float(move[1]), -1.0, 1.0)
