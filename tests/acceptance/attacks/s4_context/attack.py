"""Acceptance test 2 (clean divergence): change S4's brief, nothing else.

Every call of S1-S3 must replay from golden, and the first call that needs inference must be S4's
first LLM call.
"""
from harness.attack import Attack


class S4Context(Attack):
    def on_context(self, stage, docs, ctx):
        if stage == "S4":
            docs["brief.txt"] += "\nThe reference sample is known to be slightly harder than the dataset.\n"
        return docs
