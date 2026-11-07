"""
Optional gate that makes the logic review mandatory (hackathon experiment, off by default).

Upstream's Researcher prompt says a logic review is required before writing reports or ending
the project, but nothing enforces it. With AOD_REQUIRE_LOGIC_REVIEW=1, write_final_report and
end_project refuse until logic_review has been called since the most recent
submit_numeric_values, so the review covers the numbers actually submitted.
"""
import os

_reviewed = False


def required() -> bool:
    return os.environ.get("AOD_REQUIRE_LOGIC_REVIEW", "0") == "1"


def note_review():
    global _reviewed
    _reviewed = True


def note_submission():
    global _reviewed
    _reviewed = False


def refusal(tool_name: str):
    """None if the tool may proceed, else the message returned to the agent instead."""
    if not required() or _reviewed:
        return None
    return (
        f"Refused: a logic review is required before {tool_name}. Call logic_review with a "
        "statement summarizing your results, including every number you submitted and the files "
        f"it came from. Act on the feedback, then call {tool_name} again."
    )


def reset():
    global _reviewed
    _reviewed = False
