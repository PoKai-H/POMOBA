extends Area2D
class_name BushArea


func _ready() -> void:
	body_entered.connect(_on_body_entered)
	body_exited.connect(_on_body_exited)


func _on_body_entered(body: Node2D) -> void:
	if body is CombatActor:
		(body as CombatActor).register_bush(self)


func _on_body_exited(body: Node2D) -> void:
	if body is CombatActor:
		(body as CombatActor).unregister_bush(self)
