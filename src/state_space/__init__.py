"""状态空间模型：建模、离散化、极点/可控性/可观性分析。

    from src.state_space import StateSpace, mass_spring_damper, dc_motor
"""

from .state_space import (
    StateSpace,
    canonical_from_transfer,
    dc_motor,
    mass_spring_damper,
)

__all__ = [
    "StateSpace",
    "mass_spring_damper",
    "dc_motor",
    "canonical_from_transfer",
]
