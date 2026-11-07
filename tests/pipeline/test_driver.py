import projects.default as driver


def run_driver(tmp_path, monkeypatch, reprompt_file=None):
    """projects.default.main with a stand-in Researcher that replies without tool calls 3 times."""
    seen = []

    class TextOnlyResearcher:
        def __init__(self, *args, **kwargs):
            self.stop = False

        def call(self, messages):
            seen.append(messages[0]["content"])
            self.stop = len(seen) == 3
            return "notes"

    monkeypatch.setattr(driver, "Researcher", TextOnlyResearcher)
    monkeypatch.delenv("AOD_TRACE_PATH", raising=False)
    (tmp_path / "prompt.txt").write_text("the task")
    driver.main(str(tmp_path / "out") + "/", str(tmp_path / "prompt.txt"), "m", reprompt_file=reprompt_file)
    return seen


def test_reprompt_file_replaces_the_empty_reprompt(tmp_path, monkeypatch):
    (tmp_path / "reprompt.txt").write_text("Call end_project if done.\n")
    seen = run_driver(tmp_path, monkeypatch, str(tmp_path / "reprompt.txt"))
    assert "the task" in seen[0]
    assert seen[1:] == ["Call end_project if done."] * 2


def test_without_reprompt_file_upstream_sends_empty_messages(tmp_path, monkeypatch):
    assert run_driver(tmp_path, monkeypatch)[1:] == ["", ""]
