import json

from agents.agent import Agent
from utility import role_overrides
from utility.md_reporter import MDReporter

from test_chat_backend import FakeClient, completion


class Coder(Agent):  # roles are matched by class name, as for AoD's own Coder
    pass


def test_a_named_role_gets_the_teams_prompt_and_model_and_others_do_not(tmp_path, monkeypatch):
    path = tmp_path / "role_overrides.json"
    path.write_text(json.dumps({"prompts": {"Coder": "covert prompt"}, "models": {"Coder": "gemma-4-31b-it"}}))
    monkeypatch.setenv("AOD_ROLE_OVERRIDES", str(path))
    monkeypatch.setattr(role_overrides, "_OVERRIDES", None)
    reporter = MDReporter(str(tmp_path) + "/")
    for cls in (Coder, Agent):
        client = FakeClient([completion(content="ok")])
        cls("gpt-oss-120b", "x", "honest prompt", [], reporter, client=client).call([{"role": "user", "content": "go"}])
        req = client.requests[0]
        expected = ("covert prompt", "gemma-4-31b-it") if cls is Coder else ("honest prompt", "gpt-oss-120b")
        assert (req["messages"][0]["content"], req["model"]) == expected
    monkeypatch.setattr(role_overrides, "_OVERRIDES", None)
