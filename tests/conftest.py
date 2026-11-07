import sys
from pathlib import Path

# Make `longloom` importable when pytest is invoked from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))