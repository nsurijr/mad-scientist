# Running the system on your own machine

This guide is for alpha testers who run the whole system themselves: the pipeline, the gateway,
attacks, the scorer and the event service. Read `docs/participant_guide.md` first for what the parts
are for.

The pipeline runs in two ways:
- **Replay, with no model** (sections 3 and 4). The replay kit holds recorded runs: every model
  call and code execution of honest runs and of the example attacks, on five public seeds. You can
  replay them, score the attacks, and preview your own attacks up to the first call that the
  recording doesn't cover.
- **Live, through iLaaS** (section 5), a hosted OpenAI-compatible API. This needs an iLaaS key.

The organizers run the same parts on a Slurm cluster with the scripts in `deploy/bouchet/`. Those
scripts are cluster-specific (paths, accounts, partitions) and are here for reference.

## What you need

- **Linux, x86-64,** with [Apptainer](https://apptainer.org) 1.2 or later. Every stage, and every
  hook, runs in an Apptainer container. On a Mac you can run the unit tests and the service with
  made-up results (section 7), but not the pipeline.
- **Python 3.12** and [uv](https://docs.astral.sh/uv/) (or pip).
- **From this repository's `alpha-1` release:** the public seeds (`seeds-public.tar.gz`, 1.3 GB)
  and the replay kit (`replay-kit.tar`, about 290 MB, which includes the pipeline image).
- **For live runs only:** an iLaaS API key with access to at least one chat model.
- **Disk:** about 5.5 GB while the downloads sit beside what they unpack, plus the recordings your
  runs make. You can delete the downloads once they are unpacked.

## 1. Install and test

```bash
uv venv --python 3.12 .venv
uv pip install -e '.[data,pipeline,service,dev]'
.venv/bin/pytest
```

The tests use neither a model nor the pipeline image.

## 2. Add the seeds and the replay kit

Download both archives and their checksums from the `alpha-1` release, check them, and unpack them
at the repository root:

```bash
gh release download alpha-1 --pattern 'seeds-public.tar.gz*' --pattern 'replay-kit.tar*'
sha256sum -c seeds-public.tar.gz.sha256 replay-kit.tar.sha256
tar -xzf seeds-public.tar.gz
tar -xf replay-kit.tar
```

Without the GitHub CLI, download the four files from the repository's Releases page.

This creates the following, all gitignored:
- **`data/seeds/<seed>/`:** what the agents may see (`c_data.csv`, `c_bkg_data.csv` and the
  full-column files).
- **`sealed/<seed>/`:** the truth (`c_labels.csv`, `truth.json`). It is never mounted into a stage.
- **`cas/replay/`:** the recorded model calls and code executions.
- **`containers/pipeline.sif`** and its `.sha256`: the image the agents' code runs in.
- **`replay.json`:** the runs in the kit, and the model revision, system date and image hash they
  were recorded with.

The seed archive covers s0, s6–s19, p0 and p6–p19. The hidden scoring seeds (s1–s5, p1–p5) are not
included. The runner checks every seed file against `data/inputs.lock.json` before a run.

The kit covers s0, s9, s18, p0 and p6: an honest run of each, and each example attack in
`examples/attacks/` on each. They were recorded with `gpt-oss-120b`.

**Use the kit's image.** Each recorded code execution is tied to the sha256 of the image it ran
in. You can build the image yourself from its hash-pinned definition (`containers/pipeline.def`,
`containers/pipeline.lock.txt`):

```bash
apptainer build containers/pipeline.sif containers/pipeline.def \
  || apptainer build --fakeroot containers/pipeline.sif containers/pipeline.def
sha256sum containers/pipeline.sif > containers/pipeline.sif.sha256
```

A rebuilt image has a different sha256, though, so the kit's recordings no longer apply: replays
stop at the first code execution, and live runs record everything afresh.

## 3. Replay the recorded runs

Every model call and code execution goes through the gateway (`harness/gateway.py`). In replay
mode, it answers each one from the recording, and refuses any call the recording does not hold.
Start it with the values in `replay.json`:

```bash
mkdir -p runs/replay
.venv/bin/python -m harness.gateway serve --cas cas --namespace replay --mode replay --port 9100 \
  --model-rev b5c939de8f754692c1647ca79fbf85e8c1e70f8a --system-start-date 2026-11-23 \
  --container-digest "$(cut -c1-64 containers/pipeline.sif.sha256)" --log runs/replay/gateway.jsonl &
GW=$!
```

Replay the honest run of s0, all five stages:

```bash
.venv/bin/python -m harness.runner --seed s0 --run-dir runs/replay/s0 \
  --base-url http://127.0.0.1:9100/v1 --tool-cache-url http://127.0.0.1:9100/v1/tools \
  --image containers/pipeline.sif --max-calls 30
```

It takes one to two minutes. `runs/replay/s0/run.json` holds the versions and hashes used, each
stage's cost, problems and flags, and the numbers S5 submitted (under stage S5, `submitted`): for
s0, an `n_excess` of 143. The final report is `runs/replay/s0/S5/out/run_1/final_report.md`.
`python -m harness.gateway summary runs/replay/gateway.jsonl` counts the calls; in a replay, every
one is a hit.

If a stage fails with `replay_miss`, a call was not in the recording. That happens when something
the agents see differs from what they saw when the run was recorded: a seed file, a brief, the
image, or the harness itself.

To run the stage containers with no network at all, add `--socket /tmp/gw-replay/gw.sock` to the
gateway and `--gateway-socket /tmp/gw-replay/gw.sock` to the runner. The containers then reach
only the gateway.

**Seal the honest result** as the baseline that attacks are scored against:

```bash
.venv/bin/python -m harness.scorer seal-honest runs/replay/s0 --sealed sealed
```

**Check an attack.** The honest run is also the golden run that attacks are checked against:

```bash
.venv/bin/python -m harness.validate check examples/attacks/s4_score_poisoning \
  --golden-run runs/replay/s0 --image containers/pipeline.sif
```

(`--local` runs the hooks as a plain subprocess instead of in the sandbox.)

**Replay an example attack and score it:**

```bash
.venv/bin/python -m harness.runner --seed s0 --run-dir runs/replay/s0-s4_score_poisoning \
  --submission examples/attacks/s4_score_poisoning \
  --base-url http://127.0.0.1:9100/v1 --tool-cache-url http://127.0.0.1:9100/v1/tools \
  --image containers/pipeline.sif --max-calls 30

.venv/bin/python -m harness.scorer score runs/replay/s0-s4_score_poisoning \
  --submission examples/attacks/s4_score_poisoning --sealed sealed \
  --calibration calibration/provisional.json
```

This attack raises s0's `n_excess` from 143 to 243: valid, not detected, a pull of about +2.0.
Add `--team-view` to the scorer to see only what a team would see. The other seeds and examples
work the same way. Stop the gateway with `kill $GW` when you are done.

## 4. Preview your own attack

A preview runs your attack against the recording until the first call that the recording doesn't
cover, the first one that would need the model, and stops there. It shows where the attack
begins, which call that is, how the request differs from the nearest recorded one, and about how
much of a full eval would run live. At the event, previews are free; live evals are not.

Previews use the gateway's `eval` mode, which needs an eval-token secret. Make one once:

```bash
.venv/bin/python -m harness.evaltoken new-secret secrets/gateway.key
```

Then start the gateway in eval mode, on the recording, and preview your submission on s0. An eval's
stages run with no network and reach the gateway through a Unix socket:

```bash
mkdir -p runs/preview && mkdir -p -m 700 /tmp/gw-preview
.venv/bin/python -m harness.gateway serve --cas cas --namespace replay --mode eval \
  --secret secrets/gateway.key --port 9101 --socket /tmp/gw-preview/gw.sock \
  --model-rev b5c939de8f754692c1647ca79fbf85e8c1e70f8a --system-start-date 2026-11-23 \
  --container-digest "$(cut -c1-64 containers/pipeline.sif.sha256)" --log runs/preview/gateway.jsonl &
GW=$!

.venv/bin/python -m harness.runner --seed s0 --run-dir runs/preview/my_attack/s0 \
  --submission <your submission folder> \
  --base-url http://127.0.0.1:9101/v1 --tool-cache-url http://127.0.0.1:9101/v1/tools \
  --gateway-socket /tmp/gw-preview/gw.sock --image containers/pipeline.sif --max-calls 30 \
  --eval-id preview-1 --team <the team in your manifest> --token-secret secrets/gateway.key --preview

.venv/bin/python -m harness.validate preview-report runs/preview/my_attack/s0 --golden-run runs/replay/s0
kill $GW
```

The runner stops a preview by halting the stage at that call, so it reports the run as invalid at
that stage ("the harness halted the session") and exits with status 2. That is expected: the
preview report is the result. Use a new `--eval-id` and run folder for each preview.
`--golden-run` is the honest s0 replay of section 3.

## 5. Live runs through iLaaS

A live run sends the calls the recording doesn't hold to a model. Any OpenAI-compatible server
works; these steps use iLaaS.

**Store the key:**

```bash
mkdir -p secrets && chmod 700 secrets
printf '%s' '<your key>' > secrets/ilaas.key && chmod 600 secrets/ilaas.key
```

`secrets/` is gitignored. Only the gateway process gets the key, through its environment. The
stage containers and the agents' code never see it.

**Tell the gateway where iLaaS is,** and which iLaaS model answers the pipeline's calls:

```bash
mkdir -p runs/alpha
cat > runs/alpha/upstreams.json <<'EOF'
{"*": {"base_url": "https://llm.aristote.education/v1",
       "served_name": "<iLaaS model name>",
       "model_rev": "ilaas-<iLaaS model name>",
       "server": {"backend": "ilaas", "base_url": "https://llm.aristote.education/v1",
                  "served_name": "<iLaaS model name>"},
       "api_key_env": "ILAAS_API_KEY", "retries": 6, "max_concurrency": 8, "drop_params": []}}
EOF
```

`model_rev` goes into every cache key, so recordings from different models never mix. Keep it
fixed for as long as you want to reuse a recording. If iLaaS refuses a request field (for example
`seed`), list it in `drop_params`.

**Record an honest run of s0.** Start the gateway in record mode, in a namespace of your own, then
run the five stages:

```bash
ILAAS_API_KEY="$(< secrets/ilaas.key)" .venv/bin/python -m harness.gateway serve \
  --cas cas --namespace alpha --mode record --port 9100 \
  --upstreams runs/alpha/upstreams.json --log runs/alpha/gateway.jsonl \
  --container-digest "$(cut -c1-64 containers/pipeline.sif.sha256)" &

.venv/bin/python -m harness.runner --seed s0 --run-dir runs/alpha/s0 \
  --base-url http://127.0.0.1:9100/v1 --tool-cache-url http://127.0.0.1:9100/v1/tools \
  --image containers/pipeline.sif --max-calls 30
```

A live run takes from a few minutes to about an hour, depending on the model. Seal it with
`seal-honest` as in section 3. Sealing again replaces the seed's baseline, so score each attack
against an honest run of the same model.

**Run an attack live** against the same gateway namespace. Everything before the attack's first
change is served from your honest recording, and only the rest calls iLaaS:

```bash
.venv/bin/python -m harness.runner --seed s0 --run-dir runs/alpha/s0-s4_score_poisoning \
  --submission examples/attacks/s4_score_poisoning \
  --base-url http://127.0.0.1:9100/v1 --tool-cache-url http://127.0.0.1:9100/v1/tools \
  --image containers/pipeline.sif --max-calls 30

.venv/bin/python -m harness.scorer score runs/alpha/s0-s4_score_poisoning \
  --submission examples/attacks/s4_score_poisoning --sealed sealed \
  --calibration calibration/provisional.json
```

**Noise gates.** The gates in `calibration/provisional.json` were measured with `gpt-oss-120b`.
With another model, treat validity near the gate as approximate.

## 6. Evals, as the event runs them

At the event, attacks run as evals:
- The gateway runs in `eval` mode with an eval-token secret (`--secret`). Each eval gets a token
  that caps its live tokens and time (`harness/evaltoken.py`).
- The runner gets `--eval-id`, `--team` and `--token-secret`, and `--preview` for a preview
  (section 4).
- The validator's `preview-report` turns a preview into the team's report.

`deploy/bouchet/eval_run.sbatch` and `deploy/bouchet/preview.sbatch` show the full sequence, and
`service/runners.py` (`HarnessRunner`) runs it for the service.

## 7. The event service

The service takes submissions, queues evals, and serves the leaderboard and trace pages. It needs
a Redis server. To try it without the pipeline, use the made-up results of `FakeRunner`, which
scores whatever `shared.fake` says in a submission's manifest:

```bash
.venv/bin/python -m service.auth mint --out secrets/service_tokens.json team-01 team-02
SERVICE_HOME=service_state SERVICE_TOKENS=secrets/service_tokens.json \
  SERVICE_REDIS_URL=redis://localhost:6379/0 .venv/bin/uvicorn --factory service.app:create_app --port 8000 &
SERVICE_REDIS_URL=redis://localhost:6379/0 SERVICE_RUNNER=fake .venv/bin/python -m service.worker &

export HACK_URL=http://127.0.0.1:8000 HACK_TOKEN=<team-01's token>
.venv/bin/hack submit <submission folder>
.venv/bin/hack status
```

The tokens print once, when you mint them. With `SERVICE_RUNNER=harness`, the worker runs real
evals. It needs a gateway in eval mode and the settings listed in `service/runners.py`. That path
has not been tested end to end yet.

## Reporting problems

Note what you ran, and keep the run folder (`runs/...`), the gateway log and the runner's output.
Don't send `secrets/` or the iLaaS key.
