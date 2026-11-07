from harness.cas import Policy, Store
from harness.keys import canonical_json, sha256_bytes
from harness.missdiff import describe, miss_report

SYSTEM = {"role": "system", "content": "You are the Researcher."}


def request(*contents, **extra):
    return {"model": "gpt-oss-120b", "messages": [SYSTEM] + [{"role": "user", "content": c} for c in contents], **extra}


def test_the_diff_starts_at_the_first_differing_message():
    report = describe(request("line 1\nline 2\nline 3"), request("line 1\nline TWO\nline 3"), "k" * 64)
    assert report["first_difference"] == 1 and report["messages"] == [2, 2]
    assert "-line TWO" in report["diff"] and "+line 2" in report["diff"]


def test_identical_messages_report_the_other_fields_that_differ():
    report = describe(request("hi", tools=[{"name": "a"}]), request("hi", tools=[{"name": "b"}]), "k" * 64)
    assert report["other_fields"] == ["tools"] and "diff" not in report


def test_the_nearest_request_prefers_the_same_eval_tag_then_the_longest_shared_prefix(tmp_path):
    store = Store(tmp_path, Policy.recording())
    for i, (req, tag) in enumerate([(request("a", "b", "c"), "s0/S3"), (request("a", "b"), "s0/S2"),
                                    (request("a", "x"), "s0/S3")]):
        store.put(sha256_bytes(str(i).encode()), b"reply", meta={"eval": tag}, files={"request.json": canonical_json(req)})
    probe = request("a", "b", "z")
    assert miss_report(store, "golden", probe, "s0/S3")["nearest_key"] == sha256_bytes(b"0")
    assert miss_report(store, "golden", probe, "s1/S3")["nearest_key"] == sha256_bytes(b"0")  # no s1 tag: all
    assert miss_report(Store(tmp_path, Policy.recording("empty")), "empty", probe)["nearest_key"] is None
