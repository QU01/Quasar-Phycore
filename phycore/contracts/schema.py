"""Schema helpers shared by every ``contract_schema.py`` of the family.

Phy-AT and Phy-CT carry the same twelve lines of ``_is_num``,
``_check_keys``, ``_num``, ``_check_polygon`` and ``_monotone_z``; Phy-HX
its own. They live here once, with the same messages, and a plugin's
schema becomes a :class:`ContractSchema` with its id, version and list
of section validators. No dependency beyond the standard library.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable

from ..plugin import ContractError


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(float(x))


def check_keys(obj, keys, where: str, errs: list) -> bool:
    if not isinstance(obj, dict):
        errs.append(f"{where}: se esperaba un objeto, hay "
                    f"{type(obj).__name__}")
        return False
    for k in keys:
        if k not in obj:
            errs.append(f"{where}: falta la clave obligatoria '{k}'")
    return True


def num(obj, key, where, errs, lo=None, hi=None):
    """Read a finite number and optionally check its range."""
    if not isinstance(obj, dict) or key not in obj:
        return None
    v = obj[key]
    if not is_num(v):
        errs.append(f"{where}.{key}: no es un número finito ({v!r})")
        return None
    v = float(v)
    if lo is not None and v < lo:
        errs.append(f"{where}.{key} = {v:.6g} < {lo:.6g}")
    if hi is not None and v > hi:
        errs.append(f"{where}.{key} = {v:.6g} > {hi:.6g}")
    return v


def monotone_z(line, where: str, errs: list, n_min: int = 8) -> bool:
    """A meridional contour [[z, r], ...] with non-decreasing z."""
    if not isinstance(line, list) or len(line) < n_min:
        errs.append(f"{where}: se esperaban >= {n_min} puntos [z, r]")
        return False
    for i, p in enumerate(line):
        if not (isinstance(p, (list, tuple)) and len(p) == 2 and
                is_num(p[0]) and is_num(p[1])):
            errs.append(f"{where}[{i}]: se esperaba [z, r] finito")
            return False
    for i in range(1, len(line)):
        if line[i][0] < line[i - 1][0] - 1e-9:
            errs.append(f"{where}: z retrocede en el índice {i} "
                        f"({line[i][0]:.6g} < {line[i - 1][0]:.6g}); el "
                        "contorno meridional se emite de LE a TE")
            return False
    return True


def check_polygon(pts, where: str, errs: list, n_expect=None, n_min: int = 20):
    """A 3-D section polygon [[x, y, z], ...]: no repeated closing point
    and the SAME count along a row (the loft of layer 5c demands it).
    Returns the point count, or None."""
    if not isinstance(pts, list) or len(pts) < n_min:
        errs.append(f"{where}: se esperaban >= {n_min} puntos")
        return None
    if n_expect is not None and len(pts) != n_expect:
        errs.append(f"{where}: {len(pts)} puntos, pero la fila usa "
                    f"{n_expect} -- el loft de la capa 5c exige el MISMO "
                    "conteo en todas las secciones")
    for i in (0, 1, len(pts) // 2, len(pts) - 1):
        p = pts[i]
        if not (isinstance(p, (list, tuple)) and len(p) == 3 and
                all(is_num(c) for c in p)):
            errs.append(f"{where}[{i}]: se esperaba [x, y, z] finito")
            return None
    if pts[0] == pts[-1]:
        errs.append(f"{where}: el polígono NO debe repetir el primer punto "
                    "al final (se cierra implícitamente)")
    return len(pts)


@dataclass
class ContractSchema:
    """A versioned contract: id, major.minor, required top keys and the
    section validators, each ``fn(doc, errs) -> None``."""

    schema_id: str
    major: int
    minor: int
    required_top: tuple = ()
    validators: list = field(default_factory=list)

    @property
    def version(self) -> str:
        return f"{self.major}.{self.minor}"

    def add(self, fn: Callable) -> Callable:
        self.validators.append(fn)
        return fn

    def validate(self, doc) -> list:
        errs: list = []
        if not check_keys(doc, ("schema", "version") + tuple(self.required_top),
                          "raíz", errs):
            return errs
        if doc.get("schema") != self.schema_id:
            errs.append(f"schema = {doc.get('schema')!r}, se esperaba "
                        f"{self.schema_id!r}")
        v = str(doc.get("version", ""))
        try:
            maj = int(v.split(".")[0])
            if maj != self.major:
                errs.append(f"version {v}: major {maj} != {self.major} "
                            "(un cambio de major no es compatible)")
        except ValueError:
            errs.append(f"version {v!r} no es 'major.minor'")
        for fn in self.validators:
            fn(doc, errs)
        return errs

    def validate_or_raise(self, doc) -> None:
        validate_or_raise(self.schema_id, self.validate(doc))

    def stamp(self, doc: dict) -> dict:
        """Write id and version into a document being emitted."""
        doc["schema"] = self.schema_id
        doc["version"] = self.version
        return doc

    def main(self, argv) -> int:
        """The CLI every ``contract_schema.py`` had: validate JSON files."""
        if len(argv) < 2:
            print(f"uso: python {argv[0]} contrato.json [...]")
            return 2
        ok_all = True
        for path in argv[1:]:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            errs = self.validate(doc)
            print(f"{path}: {'válido' if not errs else f'{len(errs)} errores'}")
            for e in errs:
                print(f"  - {e}")
            ok_all &= not errs
        return 0 if ok_all else 1


def validate_or_raise(schema_id: str, errs: list) -> None:
    if errs:
        raise ContractError(
            f"contrato {schema_id} inválido ({len(errs)} errores):\n  - " +
            "\n  - ".join(errs))
