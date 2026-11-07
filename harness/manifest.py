"""A submission's manifest.yaml.

    team: team-07
    name: attack-3
    attack_class: input-poisoning   # injection | input-poisoning | agent-code | collusion | other
    stages_touched: [S1, S4]        # every stage a hook may change or replace
    hooks: [on_input]               # every hook the attack implements
    intended_direction: "+"         # the declared shift of the target
    target: n_excess                # n_excess, or significance (Z from S5's p value)
    shared: {plan_id: 7}            # optional: the static dict every hook and replacement gets as ctx.shared
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness.attack import HOOKS

MANIFEST = "manifest.yaml"
ATTACK_CLASSES = ("injection", "input-poisoning", "agent-code", "collusion", "other")
TARGETS = ("n_excess", "significance")
STAGE_NAMES = ("S1", "S2", "S3", "S4", "S5")


@dataclass(frozen=True)
class Manifest:
    team: str
    name: str
    attack_class: str
    stages_touched: tuple[str, ...]
    hooks: tuple[str, ...]
    intended_direction: str
    target: str
    shared: dict[str, Any] = field(default_factory=dict)

    @property
    def sign(self) -> int:
        return 1 if self.intended_direction == "+" else -1


def parse(data: Any) -> tuple[Manifest | None, list[str]]:
    """The manifest, or None and what is wrong with it."""
    if not isinstance(data, dict):
        return None, [f"{MANIFEST} must be a mapping of fields"]
    problems = []
    for key in ("team", "name"):
        if not isinstance(data.get(key), str) or not data.get(key):
            problems.append(f"{MANIFEST}: {key} must be a non-empty string")
    if data.get("attack_class") not in ATTACK_CLASSES:
        problems.append(f"{MANIFEST}: attack_class must be one of {list(ATTACK_CLASSES)}")
    stages, hooks = data.get("stages_touched"), data.get("hooks")
    if not isinstance(stages, list) or not stages or not set(stages) <= set(STAGE_NAMES):
        problems.append(f"{MANIFEST}: stages_touched must be a non-empty list of {list(STAGE_NAMES)}")
    if not isinstance(hooks, list) or not hooks or not set(hooks) <= set(HOOKS):
        problems.append(f"{MANIFEST}: hooks must be a non-empty list of {list(HOOKS)}")
    if str(data.get("intended_direction")) not in ("+", "-"):
        problems.append(f"{MANIFEST}: intended_direction must be \"+\" or \"-\"")
    if data.get("target") not in TARGETS:
        problems.append(f"{MANIFEST}: target must be one of {list(TARGETS)}")
    if not isinstance(data.get("shared", {}), dict):
        problems.append(f"{MANIFEST}: shared must be a mapping")
    unknown = sorted(set(data) - {"team", "name", "attack_class", "stages_touched", "hooks", "intended_direction",
                                  "target", "shared"})
    if unknown:
        problems.append(f"{MANIFEST}: unknown fields {unknown}")
    if problems:
        return None, problems
    return Manifest(data["team"], data["name"], data["attack_class"], tuple(stages), tuple(hooks),
                    str(data["intended_direction"]), data["target"], dict(data.get("shared") or {})), []


def load(submission: Path) -> tuple[Manifest | None, list[str]]:
    path = Path(submission) / MANIFEST
    if not path.is_file():
        return None, [f"{MANIFEST} is missing"]
    try:
        return parse(yaml.safe_load(path.read_text(encoding="utf-8")))
    except yaml.YAMLError as e:
        return None, [f"{MANIFEST} is not valid YAML: {e}"]
