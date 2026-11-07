"""Replacement check: S4 replaced by a scripted agent (no model calls).

S1-S3 must replay from golden. S4 makes no model calls, so the first call that needs inference is
in S5, which reads the replacement's estimate.json.
"""
from harness.attack import AgentPolicy, Attack


class ScriptedS4(Attack):
    def replace_agent(self, stage, ctx):
        return AgentPolicy.scripted("agents/s4.py:run") if stage == "S4" else None
