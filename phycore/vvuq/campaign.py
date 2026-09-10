"""The acceptance campaign: criteria with anchors, a semaphore, a gate.

Ported from Phy-HX ``validation/campaign.py`` with the measurements
taken out: a plugin registers a measurement function per criterion and
the campaign does the rest - blocking what has no data, subordinating
what may not approve on its own, tallying, and refusing to release on a
red or a blocked criterion with the reason named.

    A criterion with no data behind it is BLOCKED, never green.
    Silence is not a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

GREEN, AMBER, RED, BLOCKED, INFO = "GREEN", "AMBER", "RED", "BLOCKED", "INFO"
VERDICTS = (GREEN, AMBER, RED, BLOCKED, INFO)


@dataclass
class Criterion:
    """One acceptance criterion, with the evidence for its threshold."""

    cid: str
    what: str
    threshold: str
    anchor: str              # the published number the threshold rests on
    #: Criteria that may never be declared green while another is red.
    subordinate_to: tuple = ()
    #: True for criteria that report rather than approve.
    informational: bool = False
    mark: str = "[V]"        # how good the anchor is


@dataclass
class Score:
    criterion: Criterion
    verdict: str
    measured: str = ""
    cause: str = ""
    waits_on: tuple = ()
    detail: dict = field(default_factory=dict)


class Campaign:
    def __init__(self, name: str, criteria=()):
        self.name = name
        self.criteria: list = list(criteria)
        self.by_id = {c.cid: c for c in self.criteria}
        self._measure: dict = {}
        self._blocked: dict = {}

    def add(self, criterion: Criterion) -> Criterion:
        self.criteria.append(criterion)
        self.by_id[criterion.cid] = criterion
        return criterion

    def measure(self, cid: str):
        """Decorator: ``fn(criterion) -> Score``."""
        if cid not in self.by_id:
            raise KeyError(f"criterio desconocido: {cid}")

        def deco(fn: Callable):
            self._measure[cid] = fn
            return fn
        return deco

    def block(self, cid: str, waits_on) -> None:
        """Declare a criterion blocked on named missing data."""
        self._blocked[cid] = tuple(waits_on)

    def score_all(self) -> list:
        scores: list = []
        for crit in self.criteria:
            if crit.cid in self._blocked:
                scores.append(Score(crit, BLOCKED, waits_on=self._blocked[crit.cid]))
                continue
            fn = self._measure.get(crit.cid)
            if fn is None:
                scores.append(Score(crit, BLOCKED,
                                    cause="sin medida registrada",
                                    waits_on=("measurement",)))
                continue
            try:
                s = fn(crit)
            except Exception as exc:                         # noqa: BLE001
                s = Score(crit, BLOCKED, cause=f"la medida falló: {exc!r}")
            if s.verdict not in VERDICTS:
                raise ValueError(f"{crit.cid}: veredicto inválido {s.verdict!r}")
            if crit.informational and s.verdict == GREEN:
                s.verdict = INFO
            scores.append(s)
        # Subordination: a criterion may not be green while a parent is not.
        verdicts = {s.criterion.cid: s.verdict for s in scores}
        for score in scores:
            for parent in score.criterion.subordinate_to:
                if verdicts.get(parent) in (RED, AMBER, BLOCKED):
                    if score.verdict == GREEN:
                        score.verdict = INFO
                        score.cause = (f"reported only: {parent} is "
                                       f"{verdicts[parent]}, and this criterion "
                                       "may not approve on its own")
        return scores

    def report(self, scores=None) -> str:
        return report(self.name, scores if scores is not None else self.score_all())

    def gate(self, scores=None, block_on=(RED, BLOCKED)) -> dict:
        return gate(scores if scores is not None else self.score_all(), block_on)


def tally(scores: list) -> dict:
    out = {v: 0 for v in VERDICTS}
    for s in scores:
        out[s.verdict] += 1
    return out


def gate(scores: list, block_on=(RED, BLOCKED)) -> dict:
    """The release gate: blocks on any verdict in ``block_on`` and names
    the criterion and the cause. An aggregate never hides a red."""
    blocking = [s for s in scores if s.verdict in block_on]
    return {
        "passed": not blocking,
        "tally": tally(scores),
        "blocking": [{"cid": s.criterion.cid, "verdict": s.verdict,
                      "cause": s.cause or s.measured,
                      "waits_on": list(s.waits_on)} for s in blocking],
        "note": ("a criterion with no data behind it is BLOCKED, never "
                 "green; silence is not a pass"),
    }


def report(name: str, scores: list) -> str:
    lines = [f"{name} acceptance campaign", "=" * 78]
    for s in scores:
        lines.append(f"[{s.verdict:7s}] {s.criterion.cid:4s} "
                     f"{s.criterion.what}")
        if s.measured:
            lines.append(f"            measured: {s.measured}")
        if s.waits_on:
            lines.append(f"            waits on: {', '.join(s.waits_on)}")
        if s.cause:
            lines.append(f"            cause:    {s.cause}")
        lines.append(f"            threshold: {s.criterion.threshold}")
        lines.append(f"            anchor:    {s.criterion.anchor} {s.criterion.mark}")
        lines.append("")
    t = tally(scores)
    lines.append("  ".join(f"{v}: {t.get(v, 0)}" for v in VERDICTS))
    lines.append("")
    lines.append("A criterion with no data behind it is BLOCKED, never "
                 "green. Silence is not a pass.")
    return "\n".join(lines)
