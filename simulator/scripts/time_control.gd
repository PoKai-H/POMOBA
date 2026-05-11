extends Node

@export var time_scale: float = 1.0:
	set(value):
		time_scale = value
		Engine.time_scale = value
