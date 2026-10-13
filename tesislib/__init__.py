"""Librería común de la base de código de la tesis."""
import os
import sys

# Permite `import config` desde cualquier script de scripts/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
