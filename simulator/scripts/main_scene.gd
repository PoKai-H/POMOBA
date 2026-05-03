extends Node2D

const ARENA_SCENE := preload("res://scenes/arena.tscn")
const BASIC_ARENA_SCENE := preload("res://scenes/basic_arena.tscn")
const AGENT_SCENE := preload("res://scenes/agent.tscn")

const ARENA_LANE_CENTER_X := 576.0
const BASIC_LANE_CENTER_X := 556.0


func _enter_tree() -> void:
	apply_sim_config()


func apply_sim_config() -> void:
	var canvas_layer := get_node("CanvasLayer")
	var map_node := _replace_map_scene(canvas_layer)
	_remove_baked_agents(map_node)
	_configure_map_features(map_node)
	_spawn_agents(canvas_layer)
	_configure_minion_spawner(canvas_layer)


func _replace_map_scene(canvas_layer: Node) -> Node:
	var existing_map := canvas_layer.get_node_or_null("arena")
	if existing_map != null:
		canvas_layer.remove_child(existing_map)
		existing_map.free()

	var map_scene := _get_map_scene()
	var map_instance := map_scene.instantiate()
	map_instance.name = "arena"
	canvas_layer.add_child(map_instance)
	canvas_layer.move_child(map_instance, 0)
	return map_instance


func _get_map_scene() -> PackedScene:
	match String(SimConfig.get_env_value("map_name", "arena")):
		"basic_arena":
			return BASIC_ARENA_SCENE
		_:
			return ARENA_SCENE


func _remove_baked_agents(map_node: Node) -> void:
	var baked_agent := map_node.get_node_or_null("Agent")
	if baked_agent != null:
		map_node.remove_child(baked_agent)
		baked_agent.free()


func _configure_map_features(map_node: Node) -> void:
	var towers_enabled := bool(SimConfig.get_extension_value("tower", true))
	var jungle_enabled := bool(SimConfig.get_extension_value("jungle", true))

	for tower_name in ["BlueTower", "RedTower"]:
		var tower := map_node.get_node_or_null(tower_name)
		if tower != null:
			tower.visible = towers_enabled
			tower.set_physics_process(towers_enabled)
			if tower.has_method("refresh_visuals"):
				tower.refresh_visuals()

	var tower_blockers := map_node.get_node_or_null("StaticBody2D")
	if tower_blockers != null:
		tower_blockers.process_mode = Node.PROCESS_MODE_INHERIT if towers_enabled else Node.PROCESS_MODE_DISABLED
		for child in tower_blockers.get_children():
			if child is CollisionShape2D:
				(child as CollisionShape2D).disabled = not towers_enabled

	for neutral_name in [
		"NeutralCampTopLeft",
		"NeutralCampTopRight",
		"NeutralCampBottomLeft",
		"NeutralCampBottomRight"
	]:
		var neutral := map_node.get_node_or_null(neutral_name)
		if neutral != null:
			neutral.visible = jungle_enabled
			neutral.process_mode = Node.PROCESS_MODE_INHERIT if jungle_enabled else Node.PROCESS_MODE_DISABLED


func _spawn_agents(canvas_layer: Node) -> void:
	var stale_agents: Array[Node] = []
	for child in canvas_layer.get_children():
		if child.name.begins_with("RuntimeAgent"):
			stale_agents.append(child)
	for child in stale_agents:
		canvas_layer.remove_child(child)
		child.free()

	var agents_per_team := SimConfig.get_agents_per_team()
	var map_name := String(SimConfig.get_env_value("map_name", "arena"))
	var lane_center_x := ARENA_LANE_CENTER_X if map_name == "arena" else BASIC_LANE_CENTER_X
	var x_offsets := _build_spawn_offsets(agents_per_team)
	var top_y := 40.0
	var bottom_y := 610.0
	var agent_index := 0

	for slot in range(agents_per_team):
		_spawn_agent(
			canvas_layer,
			"RuntimeAgentBlue%d" % slot,
			Vector2(lane_center_x + x_offsets[slot], top_y),
			&"blue",
			Color(0.0, 0.35, 1.0, 1.0),
			agent_index,
			"blue_policy"
		)
		agent_index += 1

	for slot in range(agents_per_team):
		_spawn_agent(
			canvas_layer,
			"RuntimeAgentRed%d" % slot,
			Vector2(lane_center_x + x_offsets[slot], bottom_y),
			&"red",
			Color(0.95, 0.2, 0.2, 1.0),
			agent_index,
			"red_policy"
		)
		agent_index += 1


func _spawn_agent(
	canvas_layer: Node,
	node_name: String,
	spawn_position: Vector2,
	team_id: StringName,
	color: Color,
	agent_index: int,
	policy_name: String
) -> void:
	var agent_root := AGENT_SCENE.instantiate() as Node2D
	if agent_root == null:
		return

	agent_root.name = node_name
	agent_root.position = spawn_position
	canvas_layer.add_child(agent_root)

	var body := agent_root.get_node("CharacterBody2D") as CombatActor
	body.team_id = team_id
	body.body_color = color
	body.refresh_visuals()

	var controller := agent_root.get_node("CharacterBody2D/AIController2D") as AIController2D
	controller.policy_name = policy_name
	controller.set_meta("agent_index", agent_index)


func _build_spawn_offsets(agents_per_team: int) -> Array[float]:
	var offsets: Array[float] = []
	if agents_per_team <= 1:
		offsets.append(0.0)
		return offsets

	var spacing := 34.0
	for slot in range(agents_per_team):
		offsets.append((float(slot) - float(agents_per_team - 1) * 0.5) * spacing)
	return offsets


func _configure_minion_spawner(canvas_layer: Node) -> void:
	var spawner := canvas_layer.get_node_or_null("MinionWaveSpawner")
	if spawner == null:
		return

	var minions_enabled := bool(SimConfig.get_extension_value("minion_wave", true))
	spawner.visible = minions_enabled
	spawner.process_mode = Node.PROCESS_MODE_INHERIT if minions_enabled else Node.PROCESS_MODE_DISABLED
	spawner.set_physics_process(minions_enabled)
	spawner.set_process(minions_enabled)
	spawner.lane_center_x = ARENA_LANE_CENTER_X if String(SimConfig.get_env_value("map_name", "arena")) == "arena" else BASIC_LANE_CENTER_X
	spawner.side_offset = 0.0
