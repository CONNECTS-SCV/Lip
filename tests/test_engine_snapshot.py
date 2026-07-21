"""snapshot 번들 감지/복원 검증 (torch 필요; REINVENT4는 일부만 필요).

REINVENT4/torch가 없는 환경에서는 자동 스킵된다.
"""

import pytest

torch = pytest.importorskip("torch")

from lip.engine import snapshot


def test_is_lip_bundle_true_for_bundle(tmp_path):
    snap = {
        "version": snapshot.BUNDLE_VERSION,
        "agent_save_dict": {"network": {}},
        "optimizer_state": {},
        "diversity_filter": None,
        "inception": None,
    }
    path = tmp_path / "best.chkpt"
    snapshot.save_bundle(snap, str(path))
    assert snapshot.is_lip_bundle(str(path)) is True


def test_is_lip_bundle_false_for_plain_savedict(tmp_path):
    # REINVENT model save-dict 흉내(BUNDLE_KEY 없음)
    path = tmp_path / "plain.model"
    torch.save({"network": {}, "model_type": "Reinvent"}, str(path))
    assert snapshot.is_lip_bundle(str(path)) is False


def test_is_lip_bundle_false_for_garbage(tmp_path):
    path = tmp_path / "garbage.bin"
    path.write_bytes(b"not a torch file")
    assert snapshot.is_lip_bundle(str(path)) is False


def test_extract_agent_save_dict_roundtrip(tmp_path):
    snap = {
        "version": snapshot.BUNDLE_VERSION,
        "agent_save_dict": {"network": {"w": torch.zeros(3)}, "model_type": "Reinvent"},
        "optimizer_state": {},
        "diversity_filter": None,
        "inception": None,
    }
    path = tmp_path / "b.chkpt"
    snapshot.save_bundle(snap, str(path))
    extracted = snapshot.extract_agent_save_dict(str(path))
    assert extracted["model_type"] == "Reinvent"
    assert "network" in extracted


def test_version_mismatch_raises(tmp_path):
    path = tmp_path / "old.chkpt"
    torch.save(
        {snapshot.BUNDLE_KEY: {"version": 999, "agent_save_dict": {}}}, str(path)
    )
    with pytest.raises(RuntimeError, match="Unsupported lip bundle version"):
        snapshot.extract_agent_save_dict(str(path))
