"""Bearer tokens: one per team, one for the organizers.

    python -m service.auth mint --out secrets/service_tokens.json team-01 team-02 ...

The file keeps only SHA-256 hashes; mint prints each token once. Keep the file out of git
(secrets/ is gitignored) and hand each team its token. JupyterHub tokens can replace this later:
Tokens.lookup is the only thing the API asks.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

ORGANIZERS = "organizers"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class Who:
    team: str

    @property
    def organizer(self) -> bool:
        return self.team == ORGANIZERS


class Tokens:
    def __init__(self, hashes: dict[str, str]):
        self.hashes = dict(hashes)  # team (or "organizers") -> sha256 of its token

    @classmethod
    def load(cls, path: Path) -> "Tokens":
        return cls(json.loads(Path(path).read_text()) if Path(path).is_file() else {})

    @property
    def teams(self) -> list[str]:
        return sorted(t for t in self.hashes if t != ORGANIZERS)

    def lookup(self, token: str | None) -> Who | None:
        if not token:
            return None
        h = _hash(token)
        for team, expected in self.hashes.items():
            if hmac.compare_digest(h, expected):
                return Who(team)
        return None


def mint(teams: list[str], out: Path) -> dict[str, str]:
    """New tokens for the teams and the organizers; writes the hashes (mode 600), returns the tokens."""
    tokens = {t: secrets.token_urlsafe(24) for t in [*teams, ORGANIZERS]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({t: _hash(tok) for t, tok in tokens.items()}, indent=2) + "\n")
    out.chmod(0o600)
    return tokens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mint")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("teams", nargs="+")
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"{args.out} exists; remove it first to replace every token", file=sys.stderr)
        return 1
    for team, token in mint(args.teams, args.out).items():
        print(f"{team}\t{token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
