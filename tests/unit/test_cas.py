import pytest

from harness.cas import (
    GOLDEN,
    CASError,
    IntegrityError,
    NamespaceSealed,
    Policy,
    Store,
    WriteNotAllowed,
)
from harness.keys import sha256_bytes

K1 = sha256_bytes(b"call-1")
K2 = sha256_bytes(b"call-2")


def test_put_get_roundtrip_with_files(tmp_path):
    rec = Store(tmp_path, Policy.recording())
    rec.put(K1, b'{"reply": "hi"}', meta={"stage": "S1"}, files={"out/prep.npz": b"\x00\x01"})
    e = Store(tmp_path, Policy.replay_only()).get(K1)
    assert e.value == b'{"reply": "hi"}'
    assert e.namespace == GOLDEN
    assert e.meta == {"stage": "S1"}
    assert Store(tmp_path, Policy.replay_only()).get_file(e, "out/prep.npz") == b"\x00\x01"


def test_miss_returns_none(tmp_path):
    assert Store(tmp_path, Policy.replay_only()).get(K1) is None


def test_lookup_order_is_golden_then_team(tmp_path):
    Store(tmp_path, Policy.recording()).put(K1, b"golden")
    team = Store(tmp_path, Policy.for_team("team-03"))
    team.put(K1, b"team")  # recorded in team-03, but golden wins on lookup
    team.put(K2, b"team-only")
    assert team.get(K1).value == b"golden"
    assert team.get(K2).namespace == "team-03"


def test_teams_cannot_see_each_other(tmp_path):
    Store(tmp_path, Policy.for_team("team-01")).put(K1, b"secret")
    other = Store(tmp_path, Policy.for_team("team-02"))
    assert other.get(K1) is None
    with pytest.raises(CASError):
        other.get_from("team-01", K1)


def test_team_policy_never_writes_golden(tmp_path):
    with pytest.raises(CASError):
        Policy.for_team(GOLDEN)
    Store(tmp_path, Policy.for_team("team-01")).put(K1, b"x")
    assert Store(tmp_path, Policy.replay_only()).get(K1) is None


def test_replay_only_refuses_writes(tmp_path):
    with pytest.raises(WriteNotAllowed):
        Store(tmp_path, Policy.replay_only()).put(K1, b"x")


def test_sealed_namespace_refuses_writes(tmp_path):
    rec = Store(tmp_path, Policy.recording())
    rec.put(K1, b"golden")
    rec.seal(GOLDEN)
    with pytest.raises(NamespaceSealed):
        rec.put(K2, b"late")
    assert Store(tmp_path, Policy.replay_only()).get(K1).value == b"golden"


def test_first_writer_wins(tmp_path):
    s = Store(tmp_path, Policy.for_team("team-01"))
    s.put(K1, b"first")
    assert s.put(K1, b"second").value == b"first"
    assert s.get(K1).value == b"first"


def test_corrupted_blob_is_detected(tmp_path):
    rec = Store(tmp_path, Policy.recording())
    e = rec.put(K1, b"payload")
    blob = next((tmp_path / GOLDEN / "objects").rglob("*"))
    while blob.is_dir():
        blob = next(blob.iterdir())
    blob.write_bytes(rec._cctx.compress(b"tampered"))
    with pytest.raises(IntegrityError):
        Store(tmp_path, Policy.replay_only()).get(e.key)


@pytest.mark.parametrize("bad", ["ABC", "x" * 64, K1[:-1]])
def test_bad_keys_rejected(tmp_path, bad):
    with pytest.raises(CASError):
        Store(tmp_path, Policy.replay_only()).get(bad)


@pytest.mark.parametrize("bad", ["Golden", "../etc", "", "a/b"])
def test_bad_namespaces_rejected(bad):
    with pytest.raises(CASError):
        Policy(read=(bad,))


def test_one_store_is_safe_across_threads(tmp_path):
    """The gateway serves requests from many threads through one Store; shared zstandard contexts
    crashed it."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    store = Store(tmp_path, Policy.recording())
    blobs = {sha256_bytes(f"k{i}".encode()): os.urandom(200_000) for i in range(64)}

    def roundtrip(item):
        key, value = item
        store.put(key, value, files={"out.bin": value[::-1]})
        entry = store.get(key)
        return entry.value == value and store.get_file(entry, "out.bin") == value[::-1]

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert all(pool.map(roundtrip, blobs.items()))


def test_scan_lists_a_namespace_with_meta_and_files_but_no_values(tmp_path):
    rec = Store(tmp_path, Policy.recording())
    rec.put(K2, b"two", meta={"eval": "s0/S1"}, files={"request.json": b"{}"})
    rec.put(K1, b"one")
    entries = list(Store(tmp_path, Policy.replay_only()).scan(GOLDEN))
    assert [e.key for e in entries] == sorted([K1, K2])
    assert all(e.value == b"" for e in entries)
    assert next(e for e in entries if e.key == K2).meta == {"eval": "s0/S1"}
    with pytest.raises(CASError):
        list(Store(tmp_path, Policy.for_team("team-03")).scan("team-04"))
