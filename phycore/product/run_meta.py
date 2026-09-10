"""Run metadata and cache keys.

Every result the family emits carries the plugin's ``MODEL_REVISION``
and a fingerprint of its constants; the report and the cache key are
built from them. This module adds the engine's revision next to the
plugin's, so a change in the engine invalidates the same caches a change
in the physics does.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import platform
import sys
from types import ModuleType

import numpy as np

from .. import ENGINE_REVISION, __version__


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if dataclasses.is_dataclass(v):
        return _jsonable(dataclasses.asdict(v))
    return repr(v)


def constants_fingerprint(source) -> str:
    """SHA-256 of every UPPER_CASE numeric/tuple constant of a module or
    dict, in sorted order. Two plugins with the same physics and one
    constant moved get two fingerprints."""
    if isinstance(source, ModuleType):
        items = {k: getattr(source, k) for k in dir(source)
                 if k.isupper() and not k.startswith("_")}
    else:
        items = {k: v for k, v in dict(source).items() if str(k).isupper()}
    keep = {}
    for k, v in items.items():
        if isinstance(v, (bool, int, float, str, tuple, list, np.ndarray, dict)):
            keep[k] = _jsonable(v)
    blob = json.dumps(keep, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def theta_digest(theta) -> str:
    a = np.ascontiguousarray(np.asarray(theta, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:24]


def spec_digest(spec) -> str:
    blob = json.dumps(_jsonable(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def cache_key(plugin, theta, spec, fidelity: int = 0,
              constants: str | None = None) -> str:
    """The key an evaluation cache must use: plugin, its revision, the
    engine's revision, the constants fingerprint, the spec, theta and
    the fidelity. Nothing else may be omitted."""
    parts = (str(plugin.plugin_id), int(getattr(plugin, "model_revision", 0)),
             int(ENGINE_REVISION), constants or "", spec_digest(spec),
             theta_digest(theta), int(fidelity))
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


def run_meta(plugin, constants_source=None, extra: dict | None = None) -> dict:
    """What goes at the top of every report and checkpoint."""
    from ..backend import device_info
    meta = {
        "plugin_id": str(plugin.plugin_id),
        "model_revision": int(getattr(plugin, "model_revision", 0)),
        "engine_revision": int(ENGINE_REVISION),
        "phycore_version": __version__,
        "constants_fingerprint": (constants_fingerprint(constants_source)
                                  if constants_source is not None else None),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "compute": device_info(),
    }
    if extra:
        meta.update(_jsonable(extra))
    return meta
