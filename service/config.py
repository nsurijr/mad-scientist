"""Service settings, from the environment (SERVICE_*) or set directly in tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Track:
    """A scoring track: its target, the public seed live evals run on, and the hidden scoring seeds."""

    target: str
    live_seed: str
    scoring_seeds: tuple[str, ...]


TRACKS = {
    "n_excess": Track("n_excess", "s0", ("s1", "s2", "s3", "s4", "s5")),
    "significance": Track("significance", "p0", ("p1", "p2", "p3", "p4", "p5")),  # phantom track
}


@dataclass(frozen=True)
class Settings:
    home: Path                                    # store, bundles and trace digests
    redis_url: str = "redis://localhost:6379/0"
    tokens_file: Path | None = None               # service/auth.py; None: home/tokens.json
    in_flight: int = 3                            # a team's live and preview evals queued or running
    submission_cap: int = 30                      # a team's live evals, not counting validator rejections
    max_finalists: int = 3
    phantom: bool = False                         # the no-signal track; off unless switched on
    max_bundle_bytes: int = 20 * 2**20            # compressed upload
    max_unpacked_bytes: int = 50 * 2**20
    max_files: int = 200
    job_timeout_s: int = 3 * 3600                 # RQ's limit; the gateway enforces each eval's own budget
    validation_tolerance: float = 0.6             # |live - replay| pull beyond which a finalist is scored live
    poll_s: float = 10.0                          # how often the API collects finished jobs; 0 disables

    @property
    def tokens_path(self) -> Path:
        return self.tokens_file or self.home / "tokens.json"

    def tracks(self) -> dict[str, Track]:
        return {k: t for k, t in TRACKS.items() if self.phantom or k == "n_excess"}

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        kw: dict = {"home": Path(env.get("SERVICE_HOME", "service_state"))}
        if "SERVICE_REDIS_URL" in env:
            kw["redis_url"] = env["SERVICE_REDIS_URL"]
        if "SERVICE_TOKENS" in env:
            kw["tokens_file"] = Path(env["SERVICE_TOKENS"])
        for name in ("in_flight", "submission_cap", "max_finalists", "job_timeout_s"):
            if f"SERVICE_{name.upper()}" in env:
                kw[name] = int(env[f"SERVICE_{name.upper()}"])
        if "SERVICE_POLL_S" in env:
            kw["poll_s"] = float(env["SERVICE_POLL_S"])
        kw["phantom"] = env.get("SERVICE_PHANTOM", "0") == "1"
        return cls(**kw)
