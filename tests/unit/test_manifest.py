from harness.manifest import load, parse

GOOD = {"team": "team-07", "name": "attack-3", "attack_class": "collusion", "stages_touched": ["S2", "S4"],
        "hooks": ["replace_agent"], "intended_direction": "+", "target": "n_excess", "shared": {"plan_id": 7}}


def test_a_good_manifest_parses():
    m, problems = parse(GOOD)
    assert problems == [] and m.stages_touched == ("S2", "S4") and m.sign == 1 and m.shared == {"plan_id": 7}


def test_every_field_is_checked():
    bad = {**GOOD, "attack_class": "theft", "stages_touched": ["S6"], "hooks": ["on_everything"],
           "intended_direction": "up", "target": "mass", "colour": "red"}
    m, problems = parse(bad)
    assert m is None and len(problems) == 6


def test_yaml_files_load_and_bad_ones_are_reported(tmp_path):
    (tmp_path / "manifest.yaml").write_text("team: t\nname: n\nattack_class: other\nstages_touched: [S1]\n"
                                            "hooks: [on_input]\nintended_direction: '-'\ntarget: significance\n")
    m, _ = load(tmp_path)
    assert m.sign == -1 and m.target == "significance"
    (tmp_path / "manifest.yaml").write_text("team: [unclosed\n")
    assert "not valid YAML" in load(tmp_path)[1][0]
    assert load(tmp_path / "nowhere")[1] == ["manifest.yaml is missing"]
