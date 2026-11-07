import pytest

from tools.end_project import EndProject
from tools.submit_numeric_values import SubmitNumericValues
from tools.write_final_report import WriteFinalReport
from utility import review_gate
from utility.md_reporter import MDReporter


@pytest.fixture
def tools(tmp_path):
    review_gate.reset()
    questions = tmp_path / "q.json"
    questions.write_text('{"questions": [{"question_identifier": "n_excess", "question": "N"}]}')
    reporter = MDReporter(str(tmp_path) + "/")
    return (EndProject(reporter), WriteFinalReport([], str(tmp_path) + "/", reporter),
            SubmitNumericValues(str(questions), reporter), tmp_path)


def test_gate_off_by_default(tools, monkeypatch):
    monkeypatch.delenv("AOD_REQUIRE_LOGIC_REVIEW", raising=False)
    end, report, _, tmp = tools
    assert report.write_final_report("r").startswith("Final report written")
    assert end.end_project("done").startswith("The project was finished") and end.stop


def test_gate_refuses_until_a_review(tools, monkeypatch):
    monkeypatch.setenv("AOD_REQUIRE_LOGIC_REVIEW", "1")
    end, report, _, tmp = tools
    assert report.write_final_report("r").startswith("Refused: a logic review is required before write_final_report")
    assert not (tmp / "final_report.md").exists()
    assert end.end_project("done").startswith("Refused") and not end.stop

    review_gate.note_review()
    assert report.write_final_report("r").startswith("Final report written")
    assert end.end_project("done").startswith("The project was finished") and end.stop


def test_new_submission_needs_a_new_review(tools, monkeypatch):
    monkeypatch.setenv("AOD_REQUIRE_LOGIC_REVIEW", "1")
    end, _, submit, _ = tools
    review_gate.note_review()
    submit.run({"n_excess": 140.0})
    assert end.end_project("done").startswith("Refused")
    review_gate.note_review()
    assert end.end_project("done").startswith("The project was finished")
