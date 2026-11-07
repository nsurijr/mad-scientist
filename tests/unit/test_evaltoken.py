import pytest

from harness.evaltoken import TokenError, mint, new_secret, read_secret, verify

SECRET = b"k" * 64


def test_a_minted_token_verifies_to_its_claims():
    claims = verify(SECRET, mint(SECRET, team="team-03", eval_id="e1", seed="s2", token_cap=100, wall_s=60, now=1000))
    assert claims["team"] == claims["namespace"] == "team-03"
    assert claims["seed"] == "s2" and claims["token_cap"] == 100 and claims["exp"] == 1060
    assert claims["role"] == "pipeline" and claims["live"] is True


def test_tampered_or_foreign_tokens_are_refused():
    token = mint(SECRET, team="team-03", eval_id="e1", seed="s0")
    prefix, claims, sig = token.split(".")
    forged = mint(SECRET, team="team-04", eval_id="e1", seed="s0").split(".")[1]
    with pytest.raises(TokenError):
        verify(SECRET, f"{prefix}.{forged}.{sig}")  # another team's claims under this signature
    with pytest.raises(TokenError):
        verify(b"x" * 64, token)
    with pytest.raises(TokenError):
        verify(SECRET, "EMPTY")


def test_no_token_can_name_golden_or_an_unknown_role():
    with pytest.raises(TokenError):
        mint(SECRET, team="team-03", eval_id="e1", seed="s0", namespace="golden")
    with pytest.raises(TokenError):
        mint(SECRET, team="team-03", eval_id="e1", seed="s0", role="admin")


def test_secret_files_are_private_and_not_overwritten(tmp_path):
    path = tmp_path / "gateway.key"
    new_secret(path)
    assert path.stat().st_mode & 0o777 == 0o600 and len(read_secret(path)) == 64
    with pytest.raises(FileExistsError):
        new_secret(path)
