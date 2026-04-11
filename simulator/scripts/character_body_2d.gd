extends CharacterBody2D


const SPEED: float = 100.0

@onready var ai_controller: Node2D = $AIController2D

func _ready():
	ai_controller.init(self)

func game_over():
	ai_controller.done = true
	ai_controller.needs_reset = true

func _physics_process(delta: float) -> void:
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
	
	if input_vector != Vector2.ZERO:
		velocity = input_vector.normalized() * SPEED
	else:
		velocity = Vector2.ZERO

	move_and_slide()
