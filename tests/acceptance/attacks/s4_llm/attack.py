"""Replacement check: S4's Researcher given a covert system prompt (llm kind); its Coder and
reviewers stay honest.

S1-S3 must replay from golden, and the first call that needs inference is S4's first LLM call:
its system prompt is the team's.
"""
from harness.attack import AgentPolicy, Attack, read_asset


class SubvertedResearcher(Attack):
    def replace_agent(self, stage, ctx):
        if stage == "S4":
            return AgentPolicy.llm({"Researcher": read_asset("assets/researcher.md")})
        return None
