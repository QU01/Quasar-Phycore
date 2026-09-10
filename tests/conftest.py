import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # phycore/
PROYECTOS = ROOT.parent
for p in (ROOT, PROYECTOS / "Phy-Bench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
