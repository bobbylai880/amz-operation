from scpc.etl.scene_pipeline import parse_args


def test_parse_args_strips_leading_space_tokens(monkeypatch):
    monkeypatch.setenv("SCENE_TOPN", "7")
    args = parse_args([" --scene", "浴室袋", " --mk", "US", "--weeks-back", "12", " --write"])
    assert args.scene == "浴室袋"
    assert args.mk == "US"
    assert args.weeks_back == 12
    assert args.write is True
    assert args.topn == 7
