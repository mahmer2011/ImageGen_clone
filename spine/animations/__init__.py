"""
Pre-defined animation templates for common character actions.
"""

from .walk import create_walk_animation
from .jump import create_jump_animation
from .wave import create_wave_animation
from .idle import create_idle_animation

__all__ = [
    "create_walk_animation",
    "create_jump_animation",
    "create_wave_animation",
    "create_idle_animation",
]

