import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AOD = ROOT / "pipeline" / "aod" / "ai_agent"
SITE = ROOT / "pipeline" / "aod" / "harness_site"

pytest.importorskip("openai")
# The vendored package imports its modules as top-level packages (from tools.tool import Tool).
sys.path.insert(0, str(AOD))
