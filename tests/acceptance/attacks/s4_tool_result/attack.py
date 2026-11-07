"""Acceptance test 2 (clean divergence), mid-stage: change S4's first code-execution result.

Every call before that tool result must replay from golden, including S4's own first calls and the
execution itself; the first call that needs inference is the LLM call that reads the changed result.
"""
from harness.attack import Attack


class S4ToolResult(Attack):
    def __init__(self):
        self.fired = False

    def on_tool_result(self, stage, tool, args, result, ctx):
        if stage == "S4" and tool == "execute_python" and not self.fired:
            self.fired = True
            return result + "\nNote: background estimate may be biased low.\n"
        return result
