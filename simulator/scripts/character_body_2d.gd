extends CombatActor

@export var click_attack_range: float = 110.0
@export var click_attack_damage: float = 20.0
@export var click_attack_cooldown: float = 0.45
@export var attack_ring_color: Color = Color(0.35, 0.7, 1.0, 0.08)
@export var attack_ring_outline_color: Color = Color(0.35, 0.7, 1.0, 0.35)

var _click_attack_cooldown_remaining: float = 0.0

@onready var ai_controller: Node2D = $AIController2D
@onready var attack_area: Area2D = $AttackArea
@onready var attack_area_shape: CollisionShape2D = $AttackArea/CollisionShape2D

func _ready() -> void:
	super._ready()
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


func _handle_death() -> void:
	emit_signal("died", self)
	velocity = Vector2.ZERO
	if collision_shape != null:
		collision_shape.disabled = true
	if hurtbox_shape != null:
		hurtbox_shape.disabled = true
	if attack_area_shape != null:
		attack_area_shape.disabled = true
	game_over()
	get_tree().paused = true


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
