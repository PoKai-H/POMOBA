extends CharacterBody2D
class_name CombatActor

signal died(actor: CombatActor)

@export var team_id: StringName = &"blue"
@export var actor_kind: StringName = &"player"
@export var max_hp: float = 100.0
@export var move_speed: float = 100.0
@export var body_radius: float = 12.0
@export var body_color: Color = Color(0.0, 0.35, 1.0, 1.0)
@export var lane_targetable: bool = true
@export var show_health_bar: bool = false
@export var health_bar_size: Vector2 = Vector2(26.0, 4.0)
@export var health_bar_offset: Vector2 = Vector2(-13.0, -22.0)
@export var health_bar_bg_color: Color = Color(0.12, 0.12, 0.12, 0.9)
@export var health_bar_fill_color: Color = Color(0.3, 0.9, 0.35, 1.0)
@export var reward_amount: float

var hp: float = 0.0
var _active_bushes: Dictionary = {}

@onready var collision_shape: CollisionShape2D = $CollisionShape2D
@onready var mesh_instance: MeshInstance2D = $MeshInstance2D
@onready var hurtbox: Area2D = $Hurtbox
@onready var hurtbox_shape: CollisionShape2D = $Hurtbox/CollisionShape2D


func _ready() -> void:
	hp = max_hp
	add_to_group("combat_actor")
	add_to_group(String(actor_kind))
	_configure_body()
	queue_redraw()
	if hurtbox != null:
		hurtbox.collision_layer = 4
		hurtbox.collision_mask = 0
		hurtbox.monitoring = false
		hurtbox.monitorable = true
		hurtbox.set_meta("owner_actor", self)


func _configure_body() -> void:
	var body_shape := collision_shape.shape as CircleShape2D
	if body_shape == null:
		body_shape = CircleShape2D.new()
		collision_shape.shape = body_shape
	body_shape.radius = body_radius

	var hurt_shape := hurtbox_shape.shape as CircleShape2D
	if hurt_shape == null:
		hurt_shape = CircleShape2D.new()
		hurtbox_shape.shape = hurt_shape
	hurt_shape.radius = body_radius

	mesh_instance.modulate = body_color


func refresh_visuals() -> void:
	_configure_body()
	queue_redraw()


func set_move_direction(direction: Vector2) -> void:
	if direction == Vector2.ZERO:
		velocity = Vector2.ZERO
	else:
		velocity = direction.normalized() * move_speed


func take_damage(amount: float, _attacker: CombatActor = null) -> void:
	if hp <= 0.0:
		return
	
	if self.is_in_group(&"player"):
		_handle_reward_on_damage_taken(self, hp, maxf(0.0, hp - amount))
	hp = maxf(0.0, hp - amount)
	_after_damage(_attacker)
	_handle_reward_on_attack(_attacker, amount)
	queue_redraw()
	if hp > 0.0:
		return
	
	_handle_reward_on_death(_attacker)
	_handle_death()


func is_alive() -> bool:
	return hp > 0.0 and is_inside_tree()


func is_enemy(other: CombatActor) -> bool:
	return other != null and other.team_id != team_id


func register_bush(bush: Area2D) -> void:
	_active_bushes[bush.get_instance_id()] = bush


func unregister_bush(bush: Area2D) -> void:
	_active_bushes.erase(bush.get_instance_id())


func is_in_bush() -> bool:
	return not _active_bushes.is_empty()


func shares_bush_with(other: CombatActor) -> bool:
	if other == null:
		return false

	for bush_id in _active_bushes.keys():
		if other._active_bushes.has(bush_id):
			return true
	return false


func is_hidden_from(observer: CombatActor) -> bool:
	if not is_in_bush():
		return false
	if observer == null:
		return true
	return not shares_bush_with(observer)


func is_visible_to(observer: CombatActor) -> bool:
	return not is_hidden_from(observer)


func _draw() -> void:
	if not show_health_bar or max_hp <= 0.0:
		return

	draw_rect(Rect2(health_bar_offset, health_bar_size), health_bar_bg_color, true)

	var hp_ratio := clampf(hp / max_hp, 0.0, 1.0)
	if hp_ratio <= 0.0:
		return

	var fill_size := Vector2(health_bar_size.x * hp_ratio, health_bar_size.y)
	draw_rect(Rect2(health_bar_offset, fill_size), health_bar_fill_color, true)


func _handle_death() -> void:
	emit_signal("died", self)
	queue_free()


func _handle_reward_on_damage_taken(agent, starting_hp: float, ending_hp: float) -> void:
	agent.set_reward(agent.get_reward() - (agent.get_damage_reward() * (starting_hp - ending_hp)))


func _handle_reward_on_attack(attacker: CombatActor, damage: int) -> void:
	if attacker == null:
		return
	if attacker.is_in_group(&"player"):
		attacker.set_reward(attacker.get_reward() + (attacker.get_damage_reward() * damage))


func _handle_reward_on_death(attacker: CombatActor) -> void:
	if attacker == null:
		return
	if attacker.is_in_group(&"player"):
		attacker.set_reward(attacker.get_reward() + reward_amount)
	

func _after_damage(_attacker: CombatActor) -> void:
	pass
