extends AIController2D

const ACTION_MOVE_UP := "move_up"
const ACTION_MOVE_LEFT := "move_left"
const ACTION_MOVE_RIGHT := "move_right"
const ACTION_MOVE_DOWN := "move_down"
const ACTION_MOVE_UP_LEFT := "move_up_left"
const ACTION_MOVE_UP_RIGHT := "move_up_right"
const ACTION_MOVE_DOWN_LEFT := "move_down_left"
const ACTION_MOVE_DOWN_RIGHT := "move_down_right"
const ACTION_HOLD := "hold"
const ACTION_ATTACK_HERO := "attack_hero"
const ACTION_ATTACK_NEAREST_MINION := "attack_nearest_minion"
const ACTION_ATTACK_TOWER := "attack_tower"
const ACTION_RETREAT := "retreat"

var move_action: Vector2 = Vector2.ZERO
var current_action_type: String = ACTION_HOLD
var current_target_slot: Variant = null
var last_action_index: int = 8
var last_action_type: String = ACTION_HOLD


func get_obs() -> Dictionary:
	if _player == null or not is_instance_valid(_player):
		return {
			"timestep": n_steps,
			"self": {
				"id": -1,
				"team": "ally",
				"position": [0.0, 0.0],
				"status": {"alive": false, "hp": 0.0}
			},
			"agents": [],
			"objects": [],
			"extensions": {
				"observed_enemy_actions": {}
			}
		}

	var self_position: Vector2 = _player.global_position
	return {
		"timestep": n_steps,
		"self": {
			"id": _get_actor_id(_player),
			"team": "ally",
			"position": [self_position.x, self_position.y],
			"status": {
				"alive": _player.is_alive(),
				"hp": _player.hp
			}
		},
		"agents": _build_other_agents(self_position),
		"objects": _build_objects(self_position),
		"extensions": {
			"observed_enemy_actions": _build_observed_enemy_actions()
		}
	}


func get_obs_space() -> Dictionary:
	return {
		"obs": {"size": [1], "space": "box"}
	}


func get_reward() -> float:
	return reward


func get_info() -> Dictionary:
	return {
		"hp": _player.hp if _player != null and is_instance_valid(_player) else 0.0,
		"alive": _player != null and is_instance_valid(_player) and _player.is_alive(),
		"env_config": SimConfig.env_config.duplicate(true),
		"episode_config": SimConfig.episode_config.duplicate(true)
	}


func get_action_space() -> Dictionary:
	return {
		"action": {
			"size": 13,
			"action_type": "discrete"
		},
	}


func _action_index_to_move(action_index: int) -> Vector2:
	match action_index:
		0:
			return Vector2(0, -1)
		1:
			return Vector2(-1, 0)
		2:
			return Vector2(1, 0)
		3:
			return Vector2(0, 1)
		4:
			return Vector2(-1, -1)
		5:
			return Vector2(1, -1)
		6:
			return Vector2(-1, 1)
		7:
			return Vector2(1, 1)
		_:
			return Vector2.ZERO


func _action_index_to_type(action_index: int) -> String:
	match action_index:
		0:
			return ACTION_MOVE_UP
		1:
			return ACTION_MOVE_LEFT
		2:
			return ACTION_MOVE_RIGHT
		3:
			return ACTION_MOVE_DOWN
		4:
			return ACTION_MOVE_UP_LEFT
		5:
			return ACTION_MOVE_UP_RIGHT
		6:
			return ACTION_MOVE_DOWN_LEFT
		7:
			return ACTION_MOVE_DOWN_RIGHT
		8:
			return ACTION_HOLD
		9:
			return ACTION_ATTACK_HERO
		10:
			return ACTION_ATTACK_NEAREST_MINION
		11:
			return ACTION_ATTACK_TOWER
		12:
			return ACTION_RETREAT
		_:
			return ACTION_HOLD


func _legacy_action_index_to_type(action_index: int) -> String:
	match action_index:
		0:
			return ACTION_MOVE_UP
		1:
			return ACTION_MOVE_LEFT
		2:
			return ACTION_MOVE_RIGHT
		3:
			return ACTION_MOVE_DOWN
		4:
			return ACTION_ATTACK_HERO
		5:
			return ACTION_ATTACK_NEAREST_MINION
		6:
			return ACTION_RETREAT
		7:
			return ACTION_ATTACK_TOWER
		8:
			return ACTION_HOLD
		_:
			return ACTION_HOLD


func set_action(action) -> void:
	current_target_slot = null

	if action.has("type"):
		current_action_type = String(action.get("type", ACTION_HOLD))
		current_target_slot = action.get("target_slot", null)
		last_action_type = current_action_type
		last_action_index = _action_type_to_index(current_action_type)
		move_action = _resolve_move_action()
		return

	if not action.has("action"):
		current_action_type = ACTION_HOLD
		last_action_type = current_action_type
		last_action_index = _action_type_to_index(current_action_type)
		move_action = Vector2.ZERO
		return

	var action_index := int(action["action"])
	if action_index <= 8:
		current_action_type = _legacy_action_index_to_type(action_index)
	else:
		current_action_type = _action_index_to_type(action_index)

	last_action_index = action_index
	last_action_type = current_action_type
	current_target_slot = action.get("target_slot", null)
	move_action = _resolve_move_action()


func _build_other_agents(self_position: Vector2) -> Array[Dictionary]:
	var agents: Array[Dictionary] = []

	for node in get_tree().get_nodes_in_group("combat_actor"):
		if not (node is CombatActor):
			continue

		var actor := node as CombatActor
		if actor == _player or actor.actor_kind != &"player":
			continue

		var visible := _is_visible_actor(actor)
		var relative_position = null
		var hp_value = null
		if visible:
			relative_position = [
				actor.global_position.x - self_position.x,
				actor.global_position.y - self_position.y
			]
			hp_value = actor.hp

		agents.append(
			{
				"id": _get_actor_id(actor),
				"team": _relative_team(actor),
				"visible": visible,
				"relative_position": relative_position,
				"status": {
					"alive": actor.is_alive(),
					"hp": hp_value
				}
			}
		)

	return agents


func _build_objects(self_position: Vector2) -> Array[Dictionary]:
	var objects: Array[Dictionary] = []

	for node in get_tree().get_nodes_in_group("combat_actor"):
		if not (node is CombatActor):
			continue

		var actor := node as CombatActor
		if actor == _player or actor.actor_kind == &"player":
			continue

		var visible := _is_visible_actor(actor)
		var relative_position = null
		if visible:
			relative_position = [
				actor.global_position.x - self_position.x,
				actor.global_position.y - self_position.y
			]

		objects.append(
			{
				"id": _get_actor_id(actor),
				"type": _object_type(actor),
				"team": _relative_team(actor),
				"visible": visible,
				"relative_position": relative_position,
				"status": {
					"alive": actor.is_alive(),
					"hp": actor.hp if visible else null
				}
			}
		)

	return objects


func _is_visible_actor(actor: CombatActor) -> bool:
	if actor == null or not is_instance_valid(actor):
		return false
	if _player == null or not is_instance_valid(_player):
		return false
	if not actor.is_alive():
		return false
	if _player.global_position.distance_to(actor.global_position) > SimConfig.get_vision_radius_pixels():
		return false
	if actor.is_hidden_from(_player):
		return false
	if actor.actor_kind == &"tower":
		return true
	return _has_line_of_sight(actor)


func _has_line_of_sight(actor: CombatActor) -> bool:
	var query := PhysicsRayQueryParameters2D.create(_player.global_position, actor.global_position)
	var excluded_rids: Array[RID] = []
	for node in get_tree().get_nodes_in_group("combat_actor"):
		if node is CollisionObject2D:
			excluded_rids.append((node as CollisionObject2D).get_rid())
	query.exclude = excluded_rids
	query.collision_mask = 1

	var result := get_world_2d().direct_space_state.intersect_ray(query)
	return result.is_empty()


func _relative_team(actor: CombatActor) -> String:
	if actor.team_id == _player.team_id:
		return "ally"
	if actor.team_id == &"neutral":
		return "neutral"
	return "enemy"


func _object_type(actor: CombatActor) -> String:
	if actor.actor_kind == &"tower":
		return "tower"
	if actor.actor_kind == &"neutral_minion":
		return "monster"
	return "minion"


func _get_actor_id(actor: CombatActor) -> int:
	if actor == null or not is_instance_valid(actor):
		return -1
	return int(actor.get_instance_id())


func _build_observed_enemy_actions() -> Dictionary:
	var observed_actions := {}

	for node in get_tree().get_nodes_in_group("combat_actor"):
		if not (node is CombatActor):
			continue

		var actor := node as CombatActor
		if actor == _player or actor.actor_kind != &"player":
			continue
		if not _is_visible_actor(actor):
			continue

		var actor_controller := actor.get_node_or_null("AIController2D")
		if actor_controller == null:
			continue

		var observed_action = actor_controller.get("last_action_index")
		observed_actions[str(_get_actor_id(actor))] = observed_action if observed_action != null else null

	return observed_actions


func _resolve_move_action() -> Vector2:
	if _is_movement_action(current_action_type):
		return _action_index_to_move(_action_type_to_index(current_action_type)).normalized()

	match current_action_type:
		ACTION_ATTACK_HERO:
			return _get_seek_direction(&"player")
		ACTION_ATTACK_NEAREST_MINION:
			return _get_seek_direction(&"minion")
		ACTION_ATTACK_TOWER:
			return _get_seek_direction(&"tower")
		ACTION_RETREAT:
			return _get_retreat_direction()
		_:
			return Vector2.ZERO


func _action_type_to_index(action_type: String) -> int:
	match action_type:
		ACTION_MOVE_UP:
			return 0
		ACTION_MOVE_LEFT:
			return 1
		ACTION_MOVE_RIGHT:
			return 2
		ACTION_MOVE_DOWN:
			return 3
		ACTION_MOVE_UP_LEFT:
			return 4
		ACTION_MOVE_UP_RIGHT:
			return 5
		ACTION_MOVE_DOWN_LEFT:
			return 6
		ACTION_MOVE_DOWN_RIGHT:
			return 7
		ACTION_HOLD:
			return 8
		ACTION_ATTACK_HERO:
			return 9
		ACTION_ATTACK_NEAREST_MINION:
			return 10
		ACTION_ATTACK_TOWER:
			return 11
		ACTION_RETREAT:
			return 12
		_:
			return 8


func _is_movement_action(action_type: String) -> bool:
	return action_type in [
		ACTION_MOVE_UP,
		ACTION_MOVE_LEFT,
		ACTION_MOVE_RIGHT,
		ACTION_MOVE_DOWN,
		ACTION_MOVE_UP_LEFT,
		ACTION_MOVE_UP_RIGHT,
		ACTION_MOVE_DOWN_LEFT,
		ACTION_MOVE_DOWN_RIGHT
	]


func _get_seek_direction(actor_kind: StringName) -> Vector2:
	var target := _get_visible_target(actor_kind)
	if target == null or _player == null or not is_instance_valid(_player):
		return Vector2.ZERO

	var offset := target.global_position - _player.global_position
	if offset == Vector2.ZERO:
		return Vector2.ZERO
	return offset.normalized()


func _get_visible_target(actor_kind: StringName) -> CombatActor:
	var candidates: Array[CombatActor] = []

	for node in get_tree().get_nodes_in_group("combat_actor"):
		if not (node is CombatActor):
			continue

		var actor := node as CombatActor
		if actor == _player or actor.actor_kind != actor_kind:
			continue
		if _player == null or not _player.is_enemy(actor):
			continue
		if not _is_visible_actor(actor):
			continue

		candidates.append(actor)

	if candidates.is_empty():
		return null

	candidates.sort_custom(func(a: CombatActor, b: CombatActor) -> bool:
		return _player.global_position.distance_squared_to(a.global_position) < _player.global_position.distance_squared_to(b.global_position)
	)

	if current_target_slot is int:
		var slot := clampi(int(current_target_slot), 0, candidates.size() - 1)
		return candidates[slot]

	return candidates[0]


func _get_retreat_direction() -> Vector2:
	if _player == null or not is_instance_valid(_player):
		return Vector2.ZERO

	var nearest_enemy := _get_nearest_visible_enemy()
	if nearest_enemy != null:
		var flee_vector := _player.global_position - nearest_enemy.global_position
		if flee_vector != Vector2.ZERO:
			return flee_vector.normalized()

	var fallback_direction := Vector2.UP if _player.team_id == &"blue" else Vector2.DOWN
	var allied_tower := _get_nearest_allied_tower()
	if allied_tower != null:
		fallback_direction = allied_tower.global_position - _player.global_position
	return fallback_direction.normalized() if fallback_direction != Vector2.ZERO else Vector2.ZERO


func _get_nearest_visible_enemy() -> CombatActor:
	var closest: CombatActor = null
	var best_distance := INF

	for node in get_tree().get_nodes_in_group("combat_actor"):
		if not (node is CombatActor):
			continue

		var actor := node as CombatActor
		if actor == _player or not _player.is_enemy(actor):
			continue
		if not _is_visible_actor(actor):
			continue

		var distance := _player.global_position.distance_squared_to(actor.global_position)
		if distance < best_distance:
			best_distance = distance
			closest = actor

	return closest


func _get_nearest_allied_tower() -> CombatActor:
	var closest: CombatActor = null
	var best_distance := INF

	for node in get_tree().get_nodes_in_group("combat_actor"):
		if not (node is CombatActor):
			continue

		var actor := node as CombatActor
		if actor.actor_kind != &"tower" or actor.team_id != _player.team_id or not actor.is_alive():
			continue

		var distance := _player.global_position.distance_squared_to(actor.global_position)
		if distance < best_distance:
			best_distance = distance
			closest = actor

	return closest


func get_requested_attack_target() -> CombatActor:
	match current_action_type:
		ACTION_ATTACK_HERO:
			return _get_visible_target(&"player")
		ACTION_ATTACK_NEAREST_MINION:
			return _get_visible_target(&"minion")
		ACTION_ATTACK_TOWER:
			return _get_visible_target(&"tower")
		_:
			return null
