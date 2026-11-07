import pytest

from harness.keys import (
    Normalizer,
    canonical_json,
    llm_key,
    manifest_hash,
    tool_key,
    workspace_manifest,
)

SAMPLING = {"temperature": 0, "top_p": 1, "seed": 7, "max_tokens": 4096}
MESSAGES = [
    {"role": "system", "content": "You are the S1 data-prep agent."},
    {"role": "user", "content": "Load /data/c_data.csv and report the event count."},
]


def key(**overrides):
    args = dict(model_id="gpt-oss-120b", model_rev="abc123", sampling=SAMPLING, messages=MESSAGES, tools=[])
    args.update(overrides)
    return llm_key(**args)[0]


def test_canonical_json_ignores_dict_order():
    assert canonical_json({"b": 1, "a": [1, {"y": 2, "x": 3}]}) == canonical_json({"a": [1, {"x": 3, "y": 2}], "b": 1})


def test_canonical_json_refuses_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_llm_key_is_stable_and_hex():
    k = key()
    assert k == key()
    assert len(k) == 64 and all(c in "0123456789abcdef" for c in k)


@pytest.mark.parametrize(
    "override",
    [
        {"model_id": "gpt-oss-20b"},
        {"model_rev": "def456"},
        {"sampling": {**SAMPLING, "seed": 8}},
        {"messages": MESSAGES + [{"role": "assistant", "content": "ok"}]},
        {"messages": [MESSAGES[0], {"role": "user", "content": MESSAGES[1]["content"] + " "}]},
        {"tools": [{"type": "function", "function": {"name": "execute_python"}}]},
        {"server": {"system_start_date": "2026-10-21"}},
        {"harness_version": "9.9.9"},
    ],
)
def test_any_input_change_changes_llm_key(override):
    assert key(**override) != key()


def test_normalizer_is_noop_on_clean_messages():
    norm, report = Normalizer(("/work/s1",))(MESSAGES)
    assert norm == MESSAGES
    assert not report.changed


def test_normalizer_replaces_timestamps_uuids_and_workdirs():
    msgs = [{"role": "user", "content": "run 3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90 at 2026-10-20T18:20:00Z in /tmp/xyz/s1/out"}]
    norm, report = Normalizer(("/tmp/xyz",))(msgs)
    assert norm[0]["content"] == "run <UUID> at <TIMESTAMP> in <WORKDIR>/s1/out"
    assert len(report.replacements) == 3


def test_normalization_makes_volatile_requests_share_a_key():
    a = [{"role": "user", "content": "started 2026-10-20 18:20:00 in /tmp/run-a/s1"}]
    b = [{"role": "user", "content": "started 2026-10-21 09:01:59 in /tmp/run-b/s1"}]
    ka = llm_key(model_id="m", model_rev="r", sampling=SAMPLING, messages=a, normalizer=Normalizer(("/tmp/run-a",)))[0]
    kb = llm_key(model_id="m", model_rev="r", sampling=SAMPLING, messages=b, normalizer=Normalizer(("/tmp/run-b",)))[0]
    assert ka == kb


def test_workspace_manifest_tracks_content_and_paths(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "c_data.csv").write_text("mj1,delta_mj\n1,2\n")
    (tmp_path / "notes.txt").write_text("hello")
    m1 = workspace_manifest(tmp_path)
    assert [p for p, _ in m1] == ["data/c_data.csv", "notes.txt"]

    (tmp_path / "notes.txt").write_text("hello!")
    assert manifest_hash(workspace_manifest(tmp_path)) != manifest_hash(m1)


def test_tool_key_depends_on_every_field():
    base = dict(
        tool_name="execute_python",
        tool_version="1",
        args={"filename": "fit.py"},
        workspace_manifest_hash="0" * 64,
        container_digest="sha256:aaa",
        harness_seed=1,
    )
    k = tool_key(**base)
    for field, value in [
        ("tool_name", "view_text_files"),
        ("tool_version", "2"),
        ("args", {"filename": "fit2.py"}),
        ("workspace_manifest_hash", "1" * 64),
        ("container_digest", "sha256:bbb"),
        ("harness_seed", 2),
    ]:
        assert tool_key(**{**base, field: value}) != k
