"""``python -m phycore``: what this machine can run, and the engine's revision."""
import json

from . import ENGINE_REVISION, __version__
from .backend import device_info

if __name__ == "__main__":
    print(json.dumps({"phycore": __version__, "engine_revision": ENGINE_REVISION,
                      "compute": device_info()}, indent=2))
