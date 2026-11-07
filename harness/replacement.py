"""A scripted replacement of one stage (replace_agent with AgentPolicy.scripted), in its own sandbox.

    python -m harness.replacement          # inside the sandbox, started by the runner

The runner starts a fresh process per replaced stage, in the pipeline image with no
network and only these mounts:

    /submission  the team's submission, read-only
    /task        replacement.json: stage, seed, entrypoint, shared, the input and output names (ro)
    /in, /data   the stage's inputs, as the honest stage would get them (ro)
    /out         the stage's working folder, where the outputs are written

The entrypoint's run(inputs, ctx) returns {output name: bytes or str}. Every required output must
be there and nothing else; the runner then checks them like an honest stage's. Exit status 0 when
the outputs are written, 1 when the team's code failed (traceback on stderr, the stage's
agent.log).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

from harness.attack import Ctx, call_rng

DATA_FILES = ("c_data.csv", "c_bkg_data.csv")


class Inputs(Mapping):
    """The stage's input files by name, read when first used."""

    def __init__(self, paths: dict[str, Path]):
        self._paths, self._cache = paths, {}

    def __getitem__(self, name: str) -> bytes:
        if name not in self._paths:
            raise KeyError(f"{name} is not an input of this stage (inputs: {sorted(self._paths)})")
        if name not in self._cache:
            self._cache[name] = self._paths[name].read_bytes()
        return self._cache[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


def load_entrypoint(submission: Path, entrypoint: str):
    path, _, func = entrypoint.partition(":")
    sys.path.insert(0, str(submission))
    spec = importlib.util.spec_from_file_location("team_replacement", submission / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func)


def write_outputs(result: Any, outputs: list[str], out: Path) -> None:
    if not isinstance(result, dict):
        raise TypeError(f"run() must return a dict of output files, not {type(result).__name__}")
    missing, extra = sorted(set(outputs) - set(result)), sorted(set(result) - set(outputs))
    if missing or extra:
        raise ValueError(f"run() must return exactly {outputs}; missing {missing}, unexpected {extra}")
    for name in outputs:
        data = result[name]
        if isinstance(data, str):
            data = data.encode()
        if not isinstance(data, bytes):
            raise TypeError(f"{name}: return bytes or str, not {type(data).__name__}")
        (out / name).write_bytes(data)


def run(submission: Path, task: Path, inp: Path, data: Path, out: Path) -> int:
    spec = json.loads(task.read_text())
    paths = {name: data / name for name in DATA_FILES} | {name: inp / name for name in spec["inputs"]}
    inputs = Inputs(paths)
    ctx = Ctx(seed=spec["seed"], stage=spec["stage"], shared=spec.get("shared") or {},
              rng=call_rng(spec["seed"], spec["stage"], "replace_agent", "run#1", spec.get("harness_seed", 0)),
              _reader=inputs.__getitem__)
    try:
        func = load_entrypoint(submission, spec["entrypoint"])
        write_outputs(func(inputs, ctx), spec["outputs"], out)
    except Exception:
        traceback.print_exc(limit=8)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--submission", type=Path, default=Path("/submission"))
    ap.add_argument("--task", type=Path, default=Path("/task/replacement.json"))
    ap.add_argument("--in", dest="inp", type=Path, default=Path("/in"))
    ap.add_argument("--data", type=Path, default=Path("/data"))
    ap.add_argument("--out", type=Path, default=Path("/out"))
    args = ap.parse_args(argv)
    return run(args.submission, args.task, args.inp, args.data, args.out)


if __name__ == "__main__":
    sys.exit(main())
