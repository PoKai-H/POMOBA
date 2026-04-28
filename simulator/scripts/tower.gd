extends CombatActor
class_name LaneTower

@export var attack_range: float = 170.0
@export var attack_damage: float = 14.0
@export var attack_cooldown: float = 1.8
@export var range_ring_color: Color = Color(1.0, 0.9, 0.35, 0.18)
@export var range_outline_color: Color = Color(1.0, 0.95, 0.6, 0.45)

@export var lane_tower_reward_amount: float = 10.0

var _cooldown_remaining: float = 0.0


func _ready() -> void:
	actor_kind = &"tower"
	super._ready()
	reward_amount = lane_tower_reward_amount

func _physics_process(delta: float) -> void:
	if not is_alive():
		return

	_cooldown_remaining = maxf(0.0, _cooldown_remaining - delta)
	if _cooldown_remaining > 0.0:
		return

	var target := _choose_target()
	if target == null:
		return

	target.take_damage(attack_damage)
	_cooldown_remaining = attack_cooldown


func _choose_target() -> CombatActor:
	var candidates: Array[CombatActor] = []

	for node in get_tree().get_nodes_in_group("combat_actor"):
		if not (node is CombatActor):
			continue

		var actor := node as CombatActor
		if not actor.is_alive() or actor.team_id == team_id:
			continue
		if not actor.lane_targetable:
			continue
		if actor.is_hidden_from(null):
			continue
		if global_position.distance_to(actor.global_position) > attack_range:
			continue

		candidates.append(actor)

	if candidates.is_empty():
		return null

	candidates.sort_custom(_sort_targets)
	return candidates[0]


func _sort_targets(a: CombatActor, b: CombatActor) -> bool:
	if a.actor_kind != b.actor_kind:
		return a.actor_kind == &"minion"
	return global_position.distance_squared_to(a.global_position) < global_position.distance_squared_to(
		b.global_position
	)


func _draw() -> void:
	super._draw()
	draw_circle(Vector2.ZERO, attack_range, range_ring_color)
	draw_arc(Vector2.ZERO, attack_range, 0.0, TAU, 64, range_outline_color, 2.0)
