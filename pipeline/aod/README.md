# Agents of Discovery

Mad Scientist is built around red-teaming a modified
[Agents of Discovery](https://github.com/uhh-pd-ml/AgentsOfDiscovery) (AoD), a recently proposed
multi-agent pipeline for analyses in high-energy physics (the paper "Agents of Discovery",
arXiv:2509.08535). This folder is the pipeline the hackathon attacks. A Researcher plans each
stage. It hands code to a Coder, has the code checked by a CodeReviewer and its reasoning by a
LogicReviewer, and submits the stage's results.

## Layout

| Path | What it holds |
| --- | --- |
| `ai_agent/agents/` | The roles: Researcher, Coder, CodeReviewer, LogicReviewer |
| `ai_agent/tools/` | The tools the agents call: `write_python`, `execute_python`, `view_text_files`, `submit_numeric_values`, `end_project` and others |
| `ai_agent/projects/default.py` | The session entry point, run once per stage by `harness/runner.py` |
| `ai_agent/project_descriptions/` | Task prompts. The staged pipeline uses the stage briefs in `pipeline/stages/` instead. |
| `ai_agent/utility/` | Tracing, the tool cache, the hook relay and role overrides |
| `harness_site/sitecustomize.py` | Loaded into the agents' executed code: seeded RNGs, fixed threads, plot data |

Inside a stage container the code is mounted read-only at `/code`.

## Changes from upstream

| Change | Files | What changed |
| --- | --- | --- |
| P1 | `agents/agent.py`, `agents/chat_backend.py`, `agents/code_reviewer.py` | Stateless Chat Completions with full local history instead of server-side Responses chains; client from `OPENAI_BASE_URL`; enforced sampling; deterministic tool-call ids |
| P2 | `projects/default.py`, `project_descriptions/Hackathon` | Truth labels never reach agents: no evaluation file, no `get_feedback`; the task asks for N_sig directly |
| P3 | `tools/execute_python.py`, `harness_site/sitecustomize.py` | Timeout, argument parsing with `shlex`, seeded RNGs, fixed threads, headless matplotlib for executed code |
| P4 | `tools/view_plot_data.py`, `agents/researcher.py`, `agents/logic_reviewer.py`, `harness_site/sitecustomize.py` | Images replaced by text plot data recorded beside every saved figure |
| P5 | `utility/trace.py`, `agents/agent.py` | Full JSON-lines trace of every request and response |
| P6 | `projects/default.py`, `tools/view_text_files.py` | Staging: `--max_calls` sets each stage session's Researcher budget (`harness/runner.py`); `--reprompt_file` replaces the empty message upstream sends after a reply without a tool call (`pipeline/stages/reprompt.txt` asks a finished stage to call `end_project`); `view_text_files` accepts absolute paths, so agents can read the read-only hand-over folder `/in` |
| P7 | `containers/pipeline.def`, `containers/pipeline.lock.txt` | The agents' code runs in a pinned image with hash-locked Python packages, so executed code gives the same results on every run |
| P8 | `agents/agent.py` | A failing tool call (unparsable arguments, a tool raising) becomes the call's result, `Tool call failed: <error>`, instead of ending the session. Without it, a Coder calling `write_python` with an empty filename ended the whole stage |
| P9 | `utility/tool_cache.py`, `utility/harness_halt.py`, `tools/execute_python.py`, `agents/agent.py`, `agents/chat_backend.py`, `projects/default.py` | Record and replay through the harness gateway (`harness/gateway.py`). `execute_python` asks the gateway first (`AOD_TOOL_CACHE_URL`): a recorded execution with the same arguments and readable files is replayed, restoring its output files; a live one is recorded. Every call carries its eval tag (`AOD_EVAL_TAG`). A strict-replay miss or a failing model server or gateway raises `HarnessHalt`, which P8 cannot catch, and the session exits with status 3 |
| P10 | `utility/hook_relay.py`, `agents/agent.py` | In an attacked run (`AOD_HOOK_URL` set), every tool result goes through the attack's `on_tool_result` hook before the agent sees it |
| P11 | `utility/role_overrides.py`, `agents/agent.py` | An llm-kind replacement (`AgentPolicy.llm` in `harness/attack.py`) swaps the system prompt and model of the roles it names (`AOD_ROLE_OVERRIDES`) |
| P12 | `tools/execute_python.py` | Code the agents run gets neither the eval token nor the harness endpoints (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AOD_TOOL_CACHE_URL`, `AOD_HOOK_URL`, `AOD_ROLE_OVERRIDES`, `AOD_EVAL_TAG`). With a gateway socket the stage container itself has no network (`harness/runner.py` `stage_command`, `harness/socket_bridge.py`), so that code reaches only the gateway and the hook relay |
