from pathlib import Path

from lip.generator.reinvent import ReinventWrapper, StageConfig


def test_reinvent_toml_writes_one_step_checkpoint_target(tmp_path: Path) -> None:
    wrapper = ReinventWrapper(
        {
            "prior_model": ".reinvent",
            "agent_model": ".reinvent",
            "work_dir": str(tmp_path),
        }
    )
    stages = [
        StageConfig(
            max_steps=1,
            min_steps=1,
            chkpt_file=str(tmp_path / "agent_step1.chkpt"),
        )
    ]

    config_path = wrapper._build_toml_config(stages, str(tmp_path))
    config_text = config_path.read_text(encoding="utf-8")

    assert config_text.count("[[stage]]") == 1
    assert "max_steps = 1" in config_text
    assert "min_steps = 1" in config_text
    assert "agent_step1.chkpt" in config_text
