from pathlib import Path

from lip.generator.reinvent import ReinventWrapper, StageConfig


def test_reinvent_toml_writes_one_checkpoint_per_stage(tmp_path: Path) -> None:
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
            chkpt_file=str(tmp_path / f"agent_step{step}.chkpt"),
        )
        for step in range(1, 4)
    ]

    config_path = wrapper._build_toml_config(stages, str(tmp_path))
    config_text = config_path.read_text(encoding="utf-8")

    assert config_text.count("[[stage]]") == 3
    assert config_text.count("max_steps = 1") == 3
    assert config_text.count("min_steps = 1") == 3
    assert "agent_step1.chkpt" in config_text
    assert "agent_step2.chkpt" in config_text
    assert "agent_step3.chkpt" in config_text
