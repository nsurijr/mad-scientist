"""A team's prompts and models for chosen roles (hackathon patch P11).

An llm-kind replacement (harness/attack.py AgentPolicy.llm) runs a stage on the pipeline's own
agents and tools, with the team's system prompt and model for the roles it names. The runner
writes them to a JSON file, {"prompts": {role: text}, "models": {role: model}}, and points
AOD_ROLE_OVERRIDES at it. A role is the agent's class name: Researcher, Coder, CodeReviewer,
LogicReviewer. Roles not named, and every run without AOD_ROLE_OVERRIDES, are unchanged.
"""
import json
import os

_OVERRIDES = None


def _load() -> dict:
    global _OVERRIDES
    if _OVERRIDES is None:
        path = os.environ.get("AOD_ROLE_OVERRIDES")
        _OVERRIDES = {"prompts": {}, "models": {}}
        if path:
            with open(path, "r", encoding="utf-8") as f:
                _OVERRIDES.update(json.load(f))
    return _OVERRIDES


def apply(role: str, system_prompt: str, model: str) -> tuple:
    """The system prompt and model this role runs with."""
    overrides = _load()
    return overrides["prompts"].get(role, system_prompt), overrides["models"].get(role, model)
