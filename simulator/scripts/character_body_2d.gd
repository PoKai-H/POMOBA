extends CombatActor

@export var click_attack_range: float = 110.0
@export var click_attack_damage: float = 20.0
@export var click_attack_cooldown: float = 0.45
@export var attack_ring_color: Color = Color(0.35, 0.7, 1.0, 0.08)
@export var attack_ring_outline_color: Color = Color(0.35, 0.7, 1.0, 0.35)

@export var reward_delta: float = -0.001
@export var reward: float = 0.0
@export var agent_reward_amount: float = 10.0
@export var agent_death_penalty_amount: float = 10.0
@export var damage_reward: float = 0.01

var _click_attack_cooldown_remaining: float = 0.0

@onready var ai_controller: Node2D = $AIController2D
@onready var attack_area: Area2D = $AttackArea
@onready var attack_area_shape: CollisionShape2D = $AttackArea/CollisionShape2D

func _ready() -> void:
	super._ready()
	reward_amount = agent_reward_amount
	ai_controller.init(self)
	_configure_attack_area()
	queue_redraw()

func game_over() -> void:
	ai_controller.done = true
	ai_controller.needs_reset = true

func _physics_process(delta: float) -> void:
	if not is_alive():
		velocity = Vector2.ZERO
		return
	_click_attack_cooldown_remaining = maxf(0.0, _click_attack_cooldown_remaining - delta)
	if ai_controller.needs_reset:
		ai_controller.reset()
		return
	var input_vector : Vector2
	if ai_controller.heuristic == "human":
		input_vector = Vector2(
			Input.get_axis("key_a", "key_d"),
			Input.get_axis("key_w", "key_s")
		)
	else:
		input_vector = ai_controller.move_action
	
	set_move_direction(input_vector)
	move_and_slide()
	if ai_controller.heuristic != "human":
		_process_agent_attack()
	reward += reward_delta
	update_reward()


func set_reward(amount) -> void:
	reward = amount


func get_reward() -> float:
	return reward


func reward_penalty_on_death() -> void:
	set_reward(get_reward() - agent_death_penalty_amount)
	update_reward()

func get_damage_reward() -> float:
	return damage_reward


func get_reward_amount() -> float:
	return reward_amount


func update_reward() -> void:
	ai_controller.set_reward(reward)
	print("%0.3f" % reward)


func _handle_death() -> void:
	emit_signal("died", self)
	velocity = Vector2.ZERO
	if collision_shape != null:
		collision_shape.disabled = true
	if hurtbox_shape != null:
		hurtbox_shape.disabled = true
	if attack_area_shape != null:
		attack_area_shape.disabled = true
	
	reward_penalty_on_death()
	game_over()
	get_tree().paused = true


func _process_agent_attack() -> void:
	if _click_attack_cooldown_remaining > 0.0:
		return

	if not ai_controller.has_method("get_requested_attack_target"):
		return

	var action_type = ai_controller.get("current_action_type")
	if action_type == null:
		return

	var target: CombatActor = ai_controller.get_requested_attack_target()

	if target == null or not _is_actor_in_attack_area(target):
		return

	target.take_damage(click_attack_damage, self)
	_click_attack_cooldown_remaining = click_attack_cooldown


func _input(event: InputEvent) -> void:
	if not is_alive():
		return
	if ai_controller.heuristic != "human":
		return
	if not (event is InputEventMouseButton):
		return

	var mouse_event := event as InputEventMouseButton
	if mouse_event.button_index != MOUSE_BUTTON_LEFT or not mouse_event.pressed:
		return
	if _click_attack_cooldown_remaining > 0.0:
		return

	var target := _get_clicked_enemy(mouse_event.position)
	if target == null:
		return

	target.take_damage(click_attack_damage, self)
	_click_attack_cooldown_remaining = click_attack_cooldown


func _configure_attack_area() -> void:
	attack_area.collision_layer = 0
	attack_area.collision_mask = 4
	attack_area.monitoring = true
	attack_area.monitorable = false

	var shape := attack_area_shape.shape as CircleShape2D
	if shape == null:
		shape = CircleShape2D.new()
		attack_area_shape.shape = shape
	shape.radius = click_attack_range


func _draw() -> void:
	super._draw()
	draw_circle(Vector2.ZERO, click_attack_range, attack_ring_color)
	draw_arc(Vector2.ZERO, click_attack_range,0.0,TAU,48,attack_ring_outline_color,1.5)


func _get_clicked_enemy(_screen_position: Vector2) -> CombatActor:
	var clicked_world_pos := get_global_mouse_position()
	var best_target: CombatActor = null
	var best_distance := INF

	for area in attack_area.get_overlapping_areas():
		var owner = area.get_meta("owner_actor", null)
		if not (owner is CombatActor):
			continue

		var actor := owner as CombatActor
		if not actor.is_alive() or not is_enemy(actor):
			continue
		if actor.is_hidden_from(self):
			continue
		if actor.actor_kind != &"tower" and not _has_line_of_sight(actor):
			continue

		var click_distance := clicked_world_pos.distance_to(actor.global_position)
		if click_distance > actor.body_radius*1.5:
			continue
		if click_distance >= best_distance:
			continue

		best_distance = click_distance
		best_target = actor

	return best_target


func _get_nearest_enemy_by_kind(actor_kind: StringName) -> CombatActor:
	var best_target: CombatActor = null
	var best_distance := INF

	for area in attack_area.get_overlapping_areas():
		var owner = area.get_meta("owner_actor", null)
		if not (owner is CombatActor):
			continue

		var actor := owner as CombatActor
		if actor.actor_kind != actor_kind:
			continue
		if not actor.is_alive() or not is_enemy(actor):
			continue
		if actor.is_hidden_from(self):
			continue
		if actor.actor_kind != &"tower" and not _has_line_of_sight(actor):
			continue

		var distance := global_position.distance_squared_to(actor.global_position)
		if distance >= best_distance:
			continue

		best_distance = distance
		best_target = actor

	return best_target


func _is_actor_in_attack_area(actor: CombatActor) -> bool:
	if actor == null or not is_instance_valid(actor):
		return false

	for area in attack_area.get_overlapping_areas():
		if area.get_meta("owner_actor", null) == actor:
			return true
	return false


func _has_line_of_sight(actor: CombatActor) -> bool:
	var query := PhysicsRayQueryParameters2D.create(global_position, actor.global_position)
	var excluded_rids: Array[RID] = []
	for node in get_tree().get_nodes_in_group("combat_actor"):
		if node is CollisionObject2D:
			excluded_rids.append((node as CollisionObject2D).get_rid())
	query.exclude = excluded_rids
	query.collision_mask = 1

	var result := get_world_2d().direct_space_state.intersect_ray(query)
	return result.is_empty()
