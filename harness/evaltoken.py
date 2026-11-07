"""Eval tokens: what one eval may do at the gateway.

    python -m harness.evaltoken new-secret secrets/gateway.key
    python -m harness.evaltoken mint --secret secrets/gateway.key --team team-03 --eval-id e17 --seed s0

The worker mints one token per eval and hands it to the eval's agents as OPENAI_API_KEY, which AoD
already sends as a bearer token. The gateway takes the eval's namespace, budget and allowed models
from the token, never from the request. A token is

    ms1.<base64url(claims JSON)>.<base64url(HMAC-SHA256 of the first two parts)>

signed with a secret only the worker and the gateway hold.

Roles:
- pipeline: the honest agents of an eval.
- replacement: a team's replacement agents, restricted to the models in allowed_models.
- control: the runner, which marks an eval as diverged and reads its state. Never given to agents.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

from harness.cas import GOLDEN

PREFIX = "ms1"
ROLES = ("pipeline", "replacement", "control")
PER_EVAL_TOKENS = 3_000_000  # live tokens per eval
PER_EVAL_WALL_S = 45 * 60


class TokenError(Exception):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(secret: bytes, signed: str) -> str:
    return _b64(hmac.new(secret, signed.encode(), hashlib.sha256).digest())


def mint(secret: bytes, *, team: str, eval_id: str, seed: str, role: str = "pipeline",
         namespace: str | None = None, submission: str = "", allowed_models: tuple[str, ...] = ("gpt-oss-120b",),
         token_cap: int = PER_EVAL_TOKENS, wall_s: int = PER_EVAL_WALL_S, live: bool = True,
         now: float | None = None) -> str:
    """A signed token. namespace defaults to the team's; live=False is a preview, which stops at
    the first call that would need inference instead of running it."""
    if role not in ROLES:
        raise TokenError(f"role must be one of {ROLES}")
    namespace = namespace or team
    if namespace == GOLDEN:
        raise TokenError("an eval cannot write the golden namespace")
    iat = int(time.time() if now is None else now)
    claims = {"team": team, "submission": submission, "eval_id": eval_id, "seed": seed, "role": role,
              "namespace": namespace, "allowed_models": list(allowed_models), "token_cap": token_cap,
              "live": live, "iat": iat, "exp": iat + wall_s}
    signed = f"{PREFIX}.{_b64(json.dumps(claims, sort_keys=True).encode())}"
    return f"{signed}.{_sign(secret, signed)}"


def verify(secret: bytes, token: str) -> dict[str, Any]:
    """The claims of a correctly signed token. Expiry is the caller's business: past exp the eval
    is over its wall-time budget, which the gateway reports as such."""
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != PREFIX:
        raise TokenError("not an eval token")
    signed = f"{parts[0]}.{parts[1]}"
    if not hmac.compare_digest(_sign(secret, signed), parts[2]):
        raise TokenError("bad signature")
    try:
        return json.loads(_unb64(parts[1]))
    except ValueError as e:
        raise TokenError(f"unreadable claims: {e}") from None


def read_secret(path: Path | str) -> bytes:
    data = Path(path).read_bytes().strip()
    if len(data) < 32:
        raise TokenError(f"{path}: secret too short")
    return data


def new_secret(path: Path | str) -> None:
    """A fresh random secret, readable by its owner only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(32) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("new-secret")
    s.add_argument("path", type=Path)
    m = sub.add_parser("mint")
    m.add_argument("--secret", type=Path, required=True)
    m.add_argument("--team", required=True)
    m.add_argument("--eval-id", required=True)
    m.add_argument("--seed", required=True)
    m.add_argument("--role", choices=ROLES, default="pipeline")
    m.add_argument("--namespace")
    m.add_argument("--submission", default="")
    m.add_argument("--allowed-models", default="gpt-oss-120b", help="comma-separated")
    m.add_argument("--token-cap", type=int, default=PER_EVAL_TOKENS)
    m.add_argument("--wall-s", type=int, default=PER_EVAL_WALL_S)
    m.add_argument("--preview", action="store_true", help="stop at the first call that needs inference")
    args = ap.parse_args(argv)

    if args.cmd == "new-secret":
        new_secret(args.path)
        print(f"wrote {args.path}")
        return 0
    print(mint(read_secret(args.secret), team=args.team, eval_id=args.eval_id, seed=args.seed, role=args.role,
               namespace=args.namespace, submission=args.submission,
               allowed_models=tuple(m for m in args.allowed_models.split(",") if m),
               token_cap=args.token_cap, wall_s=args.wall_s, live=not args.preview))
    return 0


if __name__ == "__main__":
    sys.exit(main())
