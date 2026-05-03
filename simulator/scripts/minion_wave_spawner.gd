extends Node2D

@export var minion_scene: PackedScene
@export var minions_per_team: int = 3
@export var lane_center_x: float = 576.0
@export var side_offset: float = 0.0
@export var minion_spacing: float = 26.0
@export var top_spawn_y: float = 120.0
@export var bottom_spawn_y: float = 530.0
@export var wave_interval_seconds: float = 8.0
@export var spawn_jitter_x: float = 4.0
@export var spawn_jitter_y: float = 3.0

var _wave_timer: Timer
var _rng := RandomNumberGenerator.new()


func _ready() -> void:
	_rng.randomize()
	if not visible:
		return

	spawn_wave()

	if wave_interval_seconds <= 0.0:
		return

	_wave_timer = Timer.new()
	_wave_timer.wait_time = wave_interval_seconds
	_wave_timer.autostart = true
	_wave_timer.timeout.connect(spawn_wave)
	add_child(_wave_timer)


func reset_waves(respawn_immediately: bool = false) -> void:
	for child in get_children():
		if child is LaneMinion:
			child.queue_free()

	if respawn_immediately and visible:
		spawn_wave()


func spawn_wave() -> void:
	if minion_scene == null:
		return

	for index in range(minions_per_team):
		var x_offset := (float(index) - float(minions_per_team - 1) * 0.5) * minion_spacing
		var blue_jitter := _get_spawn_jitter()
		var red_jitter := _get_spawn_jitter()
		_spawn_minion(
			&"blue",
			Vector2(lane_center_x + x_offset - side_offset, top_spawn_y) + blue_jitter,
			Vector2.DOWN,
			Color(0.1, 0.45, 1.0, 1.0)
		)
		_spawn_minion(
			&"red",
			Vector2(lane_center_x + x_offset + side_offset, bottom_spawn_y) + red_jitter,
			Vector2.UP,
			Color(0.95, 0.2, 0.2, 1.0)
		)


func _spawn_minion(team_id: StringName, spawn_position: Vector2, direction: Vector2, color: Color) -> void:
	var minion := minion_scene.instantiate() as LaneMinion
	if minion == null:
		return

	minion.team_id = team_id
	minion.forward_direction = direction
	minion.lane_center_x = lane_center_x
	minion.body_color = color
	minion.global_position = spawn_position
	add_child(minion)
	minion.refresh_visuals()


func _get_spawn_jitter() -> Vector2:
	return Vector2(
		_rng.randf_range(-spawn_jitter_x, spawn_jitter_x),
		_rng.randf_range(-spawn_jitter_y, spawn_jitter_y)
	)
