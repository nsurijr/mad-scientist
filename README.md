# mad-scientist

Mad Scientist is a science-sabotage hackathon built around red-teaming a modified Agents of
Discovery (AoD), a recently proposed multi-agent pipeline for analyses in high-energy physics. Teams
attack AoD running its original task, an honest anomaly-detection analysis of the LHC Olympics R&D
data, split here into five stages. This repository is the harness. It records and replays every
model call, code execution and data file, so an attack pays only for the calls whose inputs it
changes. It scores how far an attack moves the result, and whether the pipeline's checks catch it.

**This copy is an alpha release for testers.** Everything runs, but some values (the noise gates
and the ranking) are provisional, and parts of the event setup are still being decided. The
hidden scoring seeds and the organizers' notes are not included.

## Start here

| Read | For |
| --- | --- |
| `docs/participant_guide.md` | The task, the pipeline, how to write and submit an attack, and how it is scored |
| `docs/running_locally.md` | Running the whole system on your own machine with iLaaS |
| `pipeline/stages/README.md` | What each of the five stages reads, writes and is told |
| `pipeline/aod/README.md` | AoD's agents and tools, and what was patched to run them here |
| `harness/attack.py` | The hook interface, in full |
| `examples/attacks/` | Four worked attacks |

## Layout

| Path | What it holds |
| --- | --- |
| `harness/` | Hook SDK, runner, gateway, content-addressed store, verifier checks, validator, scorer |
| `pipeline/stages/` | The stage briefs |
| `pipeline/aod/` | Agents of Discovery: the agents, their prompts and tools, patched |
| `service/` | The event service: submissions, eval queue, leaderboard, trace pages, the `hack` command |
| `calibration/` | The scorer's provisional constants |
| `data/inputs.lock.json` | Hashes of the public seeds' files, checked before every run |
| `containers/` | The pipeline image definition and its hash-pinned packages |
| `examples/attacks/` | Worked attacks |
| `tests/` | Unit and pipeline tests, and small demo attacks in `tests/acceptance/attacks/` |
| `deploy/bouchet/` | The organizers' Slurm scripts, for reference |

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install -e '.[data,pipeline,service,dev]'
.venv/bin/pytest
```

Running the pipeline needs Linux, Apptainer, and the public seeds and replay kit attached to the
`alpha-1` release. The kit replays recorded runs with no model; live runs also need an iLaaS key.
See `docs/running_locally.md`.

`EXPORTED_FROM` records the commit of the organizers' repository that this copy was made from.
Please don't redistribute this copy.
