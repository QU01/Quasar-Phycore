"""Layers 2-4 of the family, generic over :class:`phycore.plugin.PhyPlugin`."""

from .dominance import (Individual, crowding, dominates, fast_nondominated_sort,
                        front_comparison, joint_front, optimistic, pessimistic,
                        scalar_fitness, value_of_reducing_uncertainty)
from .ensemble import EnsembleSurrogate, RESIDUAL_MIN_SAMPLES, residual_matrix
from .hypervolume import hypervolume, hv_curve, reference_point
from .nsga2 import NSGA2
from .acquisition import select_acquisition_batch
from .sampling import lhs
from .designer import Designer, DesignResult, design, design_sequence

__all__ = [
    "Individual", "crowding", "dominates", "fast_nondominated_sort",
    "front_comparison", "joint_front", "optimistic", "pessimistic",
    "scalar_fitness", "value_of_reducing_uncertainty",
    "EnsembleSurrogate", "RESIDUAL_MIN_SAMPLES", "residual_matrix",
    "hypervolume", "hv_curve", "reference_point", "NSGA2",
    "select_acquisition_batch", "lhs",
    "Designer", "DesignResult", "design", "design_sequence",
]
