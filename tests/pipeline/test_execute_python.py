from tools import execute_python as ep
from tools.execute_python import ExecutePython
from utility.md_reporter import MDReporter


def reporter(tmp_path):
    """Outside the working folder: the reporter writes a metrics file of its own."""
    (tmp_path / "report").mkdir(exist_ok=True)
    return MDReporter(str(tmp_path / "report") + "/")


def tool(tmp_path, monkeypatch, script, text):
    monkeypatch.setenv("SAVE_EXECUTION_ENV", "True")
    monkeypatch.delenv("AOD_TOOL_CACHE_URL", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    (work / script).write_text(text)
    rep = reporter(tmp_path)
    return ExecutePython(str(work) + "/", rep), rep


def test_output_up_to_the_cap_is_unchanged():
    text = "x" * ep.MAX_OUTPUT_CHARS
    assert ep._cap(text) == text
    assert ep._cap("") == ""


def test_longer_output_keeps_its_start_and_end():
    text = "".join(f"data,{i},0,0.5\n" for i in range(100_000))  # a printed score file
    capped = ep._cap(text)
    assert len(capped) < ep.MAX_OUTPUT_CHARS + 200
    assert capped.startswith("data,0,0,0.5\n") and capped.endswith("data,99999,0,0.5\n")
    assert f"of {len(text)} characters not shown" in capped


def test_a_script_printing_a_score_file_is_capped(tmp_path, monkeypatch):
    dump = "import sys\nfor i in range(200_000):\n    print(f'data,{i},0,0.5')\nprint('e' * 50_000, file=sys.stderr)\n"
    t, rep = tool(tmp_path, monkeypatch, "dump.py", dump)
    message = t.execute_python("dump.py", "")
    assert len(message) < 2 * ep.MAX_OUTPUT_CHARS + 1000
    assert "data,0,0,0.5" in message and "data,199999,0,0.5" in message and "Exit status: 0" in message
    assert rep.metrics["execute_python_output_capped"] == 1


def test_short_output_reaches_the_model_unchanged(tmp_path, monkeypatch):
    t, rep = tool(tmp_path, monkeypatch, "hello.py", "print('n_excess 184.0')\n")
    assert t.execute_python("hello.py", "") == (
        "The following files were created: []\n Program output: \nn_excess 184.0\n"
        "\n Error output: \n\n Exit status: 0")
    assert "execute_python_output_capped" not in rep.metrics
