"""Declared bands: the uncertainty that is NOT the ensemble spread.

Five bootstrapped networks agreeing tells you the surrogate is
self-consistent; it says nothing about whether the correlation agrees
with the machine. That second number is declared per closure, from the
source that published it, and it is the dominant term.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def combined_sigma(sigma_model, mu, declared_band: float):
    """Model disagreement and declared band, in quadrature.

    The declared term is heteroscedastic the way a correlation band is:
    proportional to the predicted value, because a band is a percentage.
    """
    sm = np.asarray(sigma_model, dtype=float)
    return np.sqrt(sm * sm + (float(declared_band)
                              * np.abs(np.asarray(mu, dtype=float))) ** 2)


class BandNotDeclared(KeyError):
    """A closure without a declared band gets no band, not a default."""


@dataclass(frozen=True)
class DeclaredBand:
    """Relative half-width of one closure on one quantity, with provenance."""

    closure_id: str
    quantity: str
    relative: float                 # e.g. 0.076 for +-7.6 %
    source: str                     # who measured it
    mark: str = "[V]"               # [V] measured, [S] scaled, [M] modelled, [?] unknown
    scope: dict = field(default_factory=dict)   # e.g. {"re": (300, 10000)}

    def sigma(self, mu) -> np.ndarray:
        return self.relative * np.abs(np.asarray(mu, dtype=float))


class BandTable:
    """The frozen table a plugin ships. Lookup is exact: no fallbacks."""

    def __init__(self, bands=()):
        self._t: dict = {}
        for b in bands:
            self.add(b)

    def add(self, band: DeclaredBand) -> None:
        self._t[(band.closure_id, band.quantity)] = band

    def band_for(self, closure_id: str, quantity: str) -> DeclaredBand:
        try:
            return self._t[(closure_id, quantity)]
        except KeyError:
            raise BandNotDeclared(
                f"sin banda declarada para ({closure_id!r}, {quantity!r}); "
                "un cierre sin banda no recibe una por defecto") from None

    def __contains__(self, key) -> bool:
        return tuple(key) in self._t

    def __len__(self) -> int:
        return len(self._t)

    def as_dict(self) -> dict:
        return {f"{c}:{q}": {"relative": b.relative, "source": b.source,
                             "mark": b.mark, "scope": dict(b.scope)}
                for (c, q), b in sorted(self._t.items())}
