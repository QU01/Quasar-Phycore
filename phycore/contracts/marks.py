"""The evidence marks: [V] measured, [S] scaled, [M] modelled, [?] unknown.

Every constant, closure and criterion of the family carries one. Here
they become data: a plugin declares the mark of each hook and the
hierarchical calibration (report section 6.4) reads the prior width from
it - ``[V]`` narrow, ``[M]`` wide, ``[?]`` no prior at all, which means
the hook is BLOCKED rather than free.
"""

from __future__ import annotations

import re
from enum import Enum


class Mark(str, Enum):
    V = "[V]"      # measured on the machine or published with its error
    S = "[S]"      # scaled from a validated sibling
    M = "[M]"      # modelled: a correlation or an engineering rule
    Q = "[?]"      # nobody knows yet; blocks whatever rests on it

    @property
    def letter(self) -> str:
        return self.value[1]


MARKS = tuple(Mark)
_RX = re.compile(r"\[(V|S|M|\?)\]")

#: Log-normal prior half-width (in relative units) per mark, for the
#: hierarchical calibration. ``None`` = no prior = blocked.
PRIOR_WIDTH = {Mark.V: 0.05, Mark.S: 0.15, Mark.M: 0.30, Mark.Q: None}


def parse_mark(text: str) -> Mark | None:
    """The FIRST mark in a string, or None."""
    m = _RX.search(str(text))
    return Mark(f"[{m.group(1)}]") if m else None


def count_marks(text: str) -> dict:
    out = {m: 0 for m in MARKS}
    for m in _RX.findall(str(text)):
        out[Mark(f"[{m}]")] += 1
    return {m.value: n for m, n in out.items()}


def strip_marks(text: str) -> str:
    return _RX.sub("", str(text)).strip()


def prior_width(mark: Mark | str | None) -> float | None:
    if mark is None:
        return None
    if isinstance(mark, str):
        mark = parse_mark(mark) if not mark.startswith("[") or mark not in {m.value for m in MARKS} \
            else Mark(mark)
        if mark is None:
            return None
    return PRIOR_WIDTH[mark]
