"""phycore - the engine of the QUASAR Phy family, written once.

Every Phy model (CC, AC, AT, CT, CB, HX, Wing, RP) carried its own copy
of the same layers 2-4: a residual ensemble with split conformal, NSGA-II
on Deb's constrained dominance, an exploit/explore/frontier acquisition
batch and a front made only of evaluations that really happened. This
package is that engine, plus the newer layer 3 (a learned, amortised
Pareto front) behind the SAME plugin contract, so a model chooses its
search with one argument and keeps its physics to itself.

    Phy-1  ``search="nsga2"``  ensemble + NSGA-II (the family's loop, verbatim)
    Phy-2  ``search="psl"``    actor pi(theta | spec, lambda) + anchored critic

Layout::

    phycore.plugin      the Protocol every model implements, and its validator
    phycore.engine      mlp, ensemble, dominance, nsga2, acquisition, psl,
                        hypervolume, sampling, designer
    phycore.uq          conformal / isotonic / Mondrian, declared bands, the
                        AR1 discrepancy GP of the expensive ladder
    phycore.contracts   schema helpers and the [V]/[S]/[M]/[?] marks
    phycore.vvuq        acceptance campaign: criteria, semaphore, gate
    phycore.product     run metadata, cache keys, constants fingerprint
    phycore.adapters    Phy-HX and Phy-Bench problems as plugins

Rules: the core stays pure NumPy (torch only inside ``engine.psl``, and
only when asked for); a plugin never imports another plugin; and
``ENGINE_REVISION`` enters every cache key and checkpoint next to the
plugin's ``model_revision``.
"""

from __future__ import annotations

__version__ = "0.1.0"

#: Bump when a change in the engine can move a number a plugin reports.
#: It travels with ``model_revision`` in cache keys and run metadata, so
#: a front produced by an older engine can never be mistaken for a fresh
#: one.
ENGINE_REVISION = 2

from .plugin import (                                   # noqa: E402
    ContractError, InfeasibleDesign, PhyError, PhyPlugin, repair,
    validate_plugin, violation,
)
from .engine.designer import Designer, DesignResult, design, design_sequence  # noqa: E402

__all__ = [
    "ENGINE_REVISION", "__version__",
    "PhyPlugin", "PhyError", "InfeasibleDesign", "ContractError",
    "repair", "violation", "validate_plugin",
    "Designer", "DesignResult", "design", "design_sequence",
]
