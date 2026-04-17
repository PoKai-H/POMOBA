extends CombatActor
class_name LaneMinion

@export var forward_direction: Vector2 = Vector2.DOWN
@export var lane_center_x: float = 576.0
@export var lane_lookahead: float = 120.0
@export var detection_range: float = 180.0
@export var attack_range: float = 26.0
@export var attack_damage: float = 8.0
@export var attack_cooldown: float = 0.8
@export var target_refresh_interval: float = 0.2
@export var observation_ring_color: Color = Color(0.4, 0.95, 0.7, 0.08)
@export var observation_outline_color: Color = Color(0.4, 0.95, 0.7, 0.28)

var _attack_cooldown_remaining: float = 0.0
var _target_refresh_remaining: float = 0.0
var _target: Variant = null

@onready var attack_range_area: Area2D = $AttackRange
@onready var attack_range_shape: CollisionShape2D = $AttackRange/CollisionShape2D
@onready var observation_area: Area2D = $ObservationArea
@onready var observation_shape: CollisionShape2D = $ObservationArea/CollisionShape2D


func _ready() -> void:
	actor_kind = &"minion"
	super._ready()
	_configure_attack_range()
	_configure_observation_area()
	queue_redraw()


func _physics_process(delta: float) -> void:
	if not is_alive():
		return

	_attack_cooldown_remaining = maxf(0.0, _attack_cooldown_remaining - delta)
	_target_refresh_remaining -= delta

	if not _is_target_valid(_target):
		_target = null

	if _target_refresh_remaining <= 0.0 and not _is_target_locked():
		_target = _choose_target()
		_target_refresh_remaining = target_refresh_interval

	var desired_direction := _get_lane_direction()
	var target_actor := _get_target_actor()
	if target_actor != null:
		var to_target := target_actor.global_position - global_position
		if to_target.length() > attack_range*0.8:
			desired_direction = to_target.normalized()
		else:
			desired_direction = Vector2.ZERO

	set_move_direction(desired_direction)
	move_and_slide()
	_try_attack_target()


func _get_lane_direction() -> Vector2:
	if forward_direction == Vector2.ZERO:
		return Vector2.ZERO

	var forward := forward_direction.normalized()
	var lane_target := Vector2(lane_center_x, global_position.y + forward.y*lane_lookahead)
	return (lane_target - global_position).normalized()


func _configure_attack_range() -> void:
	attack_range_area.collision_layer = 0
	attack_range_area.collision_mask = 4
	attack_range_area.monitoring = true
	attack_range_area.monitorable = false

	var shape := attack_range_shape.shape as CircleShape2D
	if shape == null:
		shape = CircleShape2D.new()
		attack_range_shape.shape = shape
	shape.radius = attack_range


func _configure_observation_area() -> void:
	observation_area.collision_layer = 0
	observation_area.collision_mask = 0
	observation_area.monitoring = false
	observation_area.monitorable = false

	var shape := observation_shape.shape as CircleShape2D
	if shape == null:
		shape = CircleShape2D.new()
		observation_shape.shape = shape
	shape.radius = detection_range


func _choose_target() -> CombatActor:
	var visible_enemies: Array[CombatActor] = []

	for node in get_tree().get_nodes_in_group("combat_actor"):
		if not (node is CombatActor):
			continue

		var actor := node as CombatActor
		if actor == self or not actor.is_alive() or not is_enemy(actor):
			continue
		if not actor.lane_targetable:
			continue
		if global_position.distance_to(actor.global_position) > detection_range:
			continue
		if not actor.is_visible_to(self):
			continue
		if actor.actor_kind != &"tower" and not _has_line_of_sight(actor):
			continue

		visible_enemies.append(actor)

	if visible_enemies.is_empty():
		return null

	visible_enemies.sort_custom(_sort_targets)
	return visible_enemies[0]


func _sort_targets(a: CombatActor, b: CombatActor) -> bool:
	var priority_a := _target_priority(a)
	var priority_b := _target_priority(b)
	if priority_a != priority_b:
		return priority_a < priority_b

	var distance_a := global_position.distance_squared_to(a.global_position)
	var distance_b := global_position.distance_squared_to(b.global_position)
	return distance_a < distance_b


func _target_priority(actor: CombatActor) -> int:
	if actor.actor_kind == &"minion":
		return 0
	if actor.actor_kind == &"tower":
		return 1
	return 2


func _is_target_valid(actor: Variant) -> bool:
	if actor == null or not is_instance_valid(actor):
		return false
	if not (actor is CombatActor):
		return false
	var combat_actor := actor as CombatActor
	if not combat_actor.is_alive() or not is_enemy(combat_actor):
		return false
	if not combat_actor.lane_targetable:
		return false
	if global_position.distance_to(combat_actor.global_position) > detection_range:
		return false
	if not combat_actor.is_visible_to(self):
		return false
	if combat_actor.actor_kind == &"tower":
		return true
	return _has_line_of_sight(combat_actor)


func _get_target_actor() -> CombatActor:
	if not _is_target_valid(_target):
		return null
	return _target as CombatActor


func _is_target_locked() -> bool:
	var target_actor := _get_target_actor()
	if target_actor == null:
		return false
	return _is_actor_in_melee_area(target_actor)


func _try_attack_target() -> void:
	var target_actor := _get_target_actor()
	if target_actor == null or _attack_cooldown_remaining > 0.0:
		return

	for area in attack_range_area.get_overlapping_areas():
		var owner = area.get_meta("owner_actor", null)
		if owner == target_actor:
			target_actor.take_damage(attack_damage, self)
			_attack_cooldown_remaining = attack_cooldown
			return


func _is_actor_in_melee_area(actor: CombatActor) -> bool:
	if actor == null or not is_instance_valid(actor):
		return false

	for area in attack_range_area.get_overlapping_areas():
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


func _draw() -> void:
	super._draw()
	draw_circle(Vector2.ZERO, detection_range, observation_ring_color)
	draw_arc(Vector2.ZERO,detection_range,0.0,TAU,48,observation_outline_color,1.5)
