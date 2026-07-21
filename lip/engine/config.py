"""EngineConfig — LipConfig에서 in-process RL 엔진 파라미터를 뽑아낸다.

기존 subprocess 래퍼(ReinventWrapper)는 dict로 파라미터를 받았지만, 엔진은
LipConfig의 하위 설정(GeneratorConfig/OptimizationConfig 등)에서 필요한 값만
평평한 dataclass로 정리해 assembly/step_loop가 REINVENT4 객체를 조립할 때 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 순환 import 회피 — 런타임엔 필요 없음
    from lip.config import LipConfig


@dataclass
class EngineConfig:
    """REINVENT4 in-process 학습에 필요한 파라미터 묶음."""

    # 모델 경로
    prior_file: str
    agent_file: str
    device: str

    # 학습 파라미터
    n_steps: int
    batch_size: int
    sigma: float
    learning_rate: float
    patience: int

    # diversity filter
    diversity_filter: str

    # inception
    inception_memory_size: int
    inception_sample_size: int
    inception_smiles_file: str | None

    # 작업 디렉토리 (temp 파일·복원용)
    work_dir: str

    @classmethod
    def from_lip_config(
        cls,
        config: "LipConfig",
        *,
        inception_smiles_file: str | None = None,
    ) -> "EngineConfig":
        """LipConfig → EngineConfig 매핑.

        agent_file은 비어 있으면(새 실행) prior_file을 시작점으로 삼는다.
        REINVENT4는 prior와 agent를 각각 로드하는데, 새 실행은 둘 다 동일한
        prior 가중치에서 출발하는 것이 관례다(run_staged_learning과 동일).
        """
        gen = config.generator
        opt = config.optimization

        prior_file = gen.prior_model
        agent_file = gen.agent_model or prior_file

        return cls(
            prior_file=prior_file,
            agent_file=agent_file,
            device=gen.device,
            n_steps=opt.n_steps,
            batch_size=gen.batch_size,
            sigma=gen.sigma,
            learning_rate=gen.learning_rate,
            patience=opt.early_stop_patience,
            diversity_filter=gen.diversity_filter,
            inception_memory_size=opt.inception.memory_size,
            inception_sample_size=opt.inception.sample_size,
            inception_smiles_file=inception_smiles_file,
            work_dir=gen.work_dir or config.output_dir,
        )
