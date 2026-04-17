extends LaneMinion
class_name NeutralMinion

@export var leash_radius: float = 120.0
@export var home_tolerance: float = 8.0

var _spawn_position: Vector2
var _retaliation_target: Variant = null


func _ready() -> void:
	team_id = &"neutral"
	actor_kind = &"neutral_minion"
	lane_targetable = false
	forward_direction = Vector2.ZERO
	super._ready()
	_spawn_position = global_position


func _physics_process(delta: float) -> void:
	if not is_alive():
		return

	_attack_cooldown_remaining = maxf(0.0, _attack_cooldown_remaining - delta)

	if not _is_target_valid(_retaliation_target):
		_retaliation_target = null

	var target_actor: CombatActor = null
	if _retaliation_target != null and is_instance_valid(_retaliation_target) and _retaliation_target is CombatActor:
		target_actor = _retaliation_target as CombatActor

	if target_actor != null:
		var leash_distance := target_actor.global_position.distance_to(_spawn_position)
		if leash_distance > leash_radius:
			_retaliation_target = null
			target_actor = null

	var desired_direction := Vector2.ZERO
	if target_actor != null:
		var to_target := target_actor.global_position - global_position
		if to_target.length() > attack_range * 0.8:
			desired_direction = to_target.normalized()
	else:
		var home_offset := _spawn_position - global_position
		if home_offset.length() > home_tolerance:
			desired_direction = home_offset.normalized()

	set_move_direction(desired_direction)
	move_and_slide()

	if target_actor != null:
		_try_attack_retaliation_target(target_actor)


func _after_damage(attacker: CombatActor) -> void:
	if attacker == null or not is_instance_valid(attacker):
		return
	if not is_enemy(attacker):
		return
	if attacker.global_position.distance_to(_spawn_position) > leash_radius:
		return
	_retaliation_target = attacker


func _is_target_valid(actor: Variant) -> bool:
	if actor == null or not is_instance_valid(actor):
		return false
	if not (actor is CombatActor):
		return false

	var combat_actor := actor as CombatActor
	if not combat_actor.is_alive() or not is_enemy(combat_actor):
		return false
	if combat_actor.global_position.distance_to(_spawn_position) > leash_radius:
		return false
	if combat_actor.is_hidden_from(self):
		return false
	return _has_line_of_sight(combat_actor)


func _choose_target() -> CombatActor:
	return null


func _try_attack_retaliation_target(target_actor: CombatActor) -> void:
	if target_actor == null or _attack_cooldown_remaining > 0.0:
		return

	for area in attack_range_area.get_overlapping_areas():
		var owner = area.get_meta("owner_actor", null)
		if owner == target_actor:
			target_actor.take_damage(attack_damage, self)
			_attack_cooldown_remaining = attack_cooldown
			return
