extends CharacterBody2D


const SPEED: float = 100.0

func _physics_process(delta: float) -> void:
	var input_vector = Vector2(
		Input.get_axis("key_a", "key_d"),
		Input.get_axis("key_w", "key_s")
	)

	if input_vector != Vector2.ZERO:
		velocity = input_vector.normalized() * SPEED
	else:
		velocity = Vector2.ZERO

	move_and_slide()
