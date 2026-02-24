"""REINVENT4 subprocess wrapper for molecular generation via RL."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lip.generator.base import BaseGenerator, GenerationResult

log = logging.getLogger(__name__)

DEFAULT_PRIOR = ".reinvent"


def _toml_path(path: str) -> str:
    """Normalize path for TOML: convert backslashes to forward slashes."""
    return path.replace("\\", "/")


# ---------------------------------------------------------------------------
# REINVENT4 scoring component / stage config
# ---------------------------------------------------------------------------

@dataclass
class ScoringComponent:
    """A single scoring component for REINVENT4 staged_learning."""
    type: str
    name: str = ""
    weight: float = 1.0
    transform: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


@dataclass
class StageConfig:
    """Configuration for one REINVENT4 stage."""
    max_steps: int = 100
    min_steps: int = 10
    max_score: float = 0.85
    scoring_components: list[ScoringComponent] = field(default_factory=list)
    chkpt_file: str = ""


# ---------------------------------------------------------------------------
# Built-in scoring component factories
# ---------------------------------------------------------------------------

from lip.utils.math import (
    reinvent_double_sigmoid as _double_sigmoid,
    reinvent_reverse_sigmoid as _reverse_sigmoid,
    reinvent_sigmoid as _sigmoid,
)


def mw_component(low: float, high: float, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("MolecularWeight", "mw", weight, _double_sigmoid(low, high))


def logp_component(low: float, high: float, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("SlogP", "logp", weight, _double_sigmoid(low, high))


def qed_component(weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("QED", "qed", weight)


def sa_component(weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("SAScore", "sa", weight)


def tpsa_component(low: float, high: float, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("TPSA", "tpsa", weight, _double_sigmoid(low, high))


def similarity_component(
    ref_smiles: str, weight: float = 1.0
) -> ScoringComponent:
    return ScoringComponent(
        type="TanimotoDistance",
        name="Similarity",
        weight=weight,
        params={"smiles": [ref_smiles], "radius": 3, "use_counts": True, "use_features": True},
    )


def hbd_component(low: int, high: int, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("HBondDonors", "hbd", weight, _double_sigmoid(low, high))


def hba_component(low: int, high: int, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("HBondAcceptors", "hba", weight, _double_sigmoid(low, high))


def rotbond_component(low: int, high: int, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("NumRotBond", "rotbond", weight, _double_sigmoid(low, high))


def num_rings_component(low: int, high: int, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("NumRings", "rings", weight, _double_sigmoid(low, high))


def aromatic_rings_component(low: int, high: int, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("NumAromaticRings", "aromatic", weight, _double_sigmoid(low, high))


def heavy_atoms_component(low: int, high: int, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("NumHeavyAtoms", "heavy", weight, _double_sigmoid(low, high))


def csp3_component(min_val: float, weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("Csp3", "csp3", weight, _sigmoid(min_val))


def alerts_component(smarts: list[str], weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("custom_alerts", "alerts", weight, params={"smarts": smarts})


def matching_substructure_component(smarts: list[str], weight: float = 1.0) -> ScoringComponent:
    return ScoringComponent("MatchingSubstructure", "substruct", weight, params={"smarts": smarts})


def external_process_component(
    executable: str,
    args: str,
    property_name: str,
    weight: float = 1.0,
    transform: dict | None = None,
) -> ScoringComponent:
    """Create ExternalProcess component for custom scoring."""
    return ScoringComponent(
        type="ExternalProcess",
        name=property_name,
        weight=weight,
        transform=transform or {},
        params={"executable": executable, "args": args, "property": property_name},
    )


def shape_similarity_component(
    reference_smiles: str = "",
    reference_sdf: str = "",
    weight: float = 1.0,
    n_ref_conformers: int = 10,
) -> ScoringComponent:
    """Create shape similarity ExternalProcess component."""
    script = str(Path(__file__).parent.parent / "scoring" / "shape.py")
    args_parts = [sys.executable, script]
    if reference_sdf:
        args_parts.extend(["--reference-sdf", reference_sdf])
    elif reference_smiles:
        args_parts.extend(["--reference-smiles", reference_smiles])
    args_parts.extend(["--n-ref-conformers", str(n_ref_conformers)])

    return external_process_component(
        executable=args_parts[0],
        args=" ".join(args_parts[1:]),
        property_name="shape_similarity",
        weight=weight,
        transform={"type": "sigmoid", "low": 0.2, "high": 0.8, "k": 0.4},
    )


# ---------------------------------------------------------------------------
# REINVENT4 Wrapper
# ---------------------------------------------------------------------------

class ReinventWrapper(BaseGenerator):
    """Wrapper for REINVENT4 staged_learning subprocess."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.prior_model = config.get("prior_model", DEFAULT_PRIOR)
        self.agent_model = config.get("agent_model", "")
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 50)
        self.sigma = config.get("sigma", 128)
        self.learning_rate = config.get("learning_rate", 0.0001)
        self.diversity_filter = config.get("diversity_filter", "IdenticalMurckoScaffold")
        self.inception_memory_size = config.get("inception_memory_size", 100)
        self.inception_sample_size = config.get("inception_sample_size", 20)
        self.inception_retention_mode = config.get("inception_retention_mode", "top")
        self.work_dir = config.get("work_dir", "") or tempfile.mkdtemp(prefix="lip_")

        self._process: subprocess.Popen | None = None

        # Ensure agent model exists
        if not self.agent_model:
            agent_path = Path(self.work_dir) / "agent.model"
            if self.prior_model and Path(self.prior_model).exists():
                shutil.copy2(self.prior_model, agent_path)
                self.agent_model = str(agent_path)
            else:
                # prior가 특수값(.reinvent 등)이면 REINVENT4가 내부 resolve하므로
                # agent에도 동일값 사용
                self.agent_model = self.prior_model

    def sample(self, n: int) -> GenerationResult:
        """Sample n molecules from the current agent via REINVENT4 sampling mode."""
        import csv

        output_file = str(Path(self.work_dir) / "sampled.csv")
        model = self.agent_model if Path(self.agent_model).exists() else self.prior_model

        # Build sampling TOML config
        toml_str = (
            f'run_type = "sampling"\n'
            f'device = "{self.device}"\n'
            f"\n"
            f"[parameters]\n"
            f'model_file = "{_toml_path(model)}"\n'
            f'output_file = "{_toml_path(output_file)}"\n'
            f"num_smiles = {n}\n"
            f"unique_molecules = true\n"
        )

        config_path = Path(self.work_dir) / "sample_config.toml"
        config_path.write_text(toml_str)

        cmd = [sys.executable, "-m", "reinvent", str(config_path)]
        timeout = min(max(60, n // 2), 600)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=self.work_dir,
            )
        except subprocess.TimeoutExpired:
            log.error("REINVENT4 sampling timed out")
            return GenerationResult(smiles=[], nll=[], valid_ratio=0.0)

        if result.returncode != 0:
            log.error(f"REINVENT4 sampling failed: {result.stderr[-500:]}")
            return GenerationResult(smiles=[], nll=[], valid_ratio=0.0)

        # Parse output CSV: columns SMILES, SMILES_state, NLL
        smiles_list: list[str] = []
        nll_list: list[float] = []
        total = 0

        if Path(output_file).exists():
            with open(output_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    if row.get("SMILES_state", "0") == "1":
                        smiles_list.append(row["SMILES"])
                        try:
                            nll_list.append(float(row.get("NLL", 0)))
                        except (ValueError, TypeError):
                            nll_list.append(0.0)

        valid_ratio = len(smiles_list) / total if total > 0 else 0.0
        return GenerationResult(smiles=smiles_list, nll=nll_list, valid_ratio=valid_ratio)

    def start_rl(self, config: dict) -> None:
        """Start REINVENT4 staged_learning as subprocess."""
        raise NotImplementedError("Use run_staged_learning() directly")

    def run_staged_learning(
        self,
        stages: list[StageConfig],
        output_dir: str | None = None,
        inception_smiles_file: str | None = None,
    ) -> subprocess.Popen:
        """Launch REINVENT4 staged_learning subprocess.

        Args:
            stages: List of stage configurations.
            output_dir: Directory for REINVENT4 output.
            inception_smiles_file: CSV file with seed SMILES for inception.

        Returns:
            Running subprocess.Popen object.
        """
        output_dir = output_dir or self.work_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        toml_path = self._build_toml_config(stages, output_dir, inception_smiles_file)

        cmd = [
            sys.executable, "-m", "reinvent",
            str(toml_path),
        ]

        log.info(f"Starting REINVENT4: {' '.join(cmd)}")

        # Use CREATE_NEW_PROCESS_GROUP on Windows, setsid on Unix
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=output_dir,
            **kwargs,
        )

        return self._process

    def stop(self) -> None:
        """Stop the running REINVENT4 subprocess."""
        if self._process is None:
            return

        try:
            if sys.platform == "win32":
                self._process.terminate()
            else:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()

        self._process = None

    def save_checkpoint(self, path: str) -> None:
        """Copy current agent model to checkpoint path."""
        if Path(self.agent_model).exists():
            shutil.copy2(self.agent_model, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent model from checkpoint."""
        if Path(path).exists():
            shutil.copy2(path, self.agent_model)

    # -----------------------------------------------------------------------
    # TOML config generation
    # -----------------------------------------------------------------------

    @staticmethod
    def _emit_toml_value(lines: list[str], prefix: str, key: str, value):
        """Emit a single TOML key-value under a dotted prefix."""
        if isinstance(value, str):
            lines.append(f'{prefix}.{key} = "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{prefix}.{key} = {'true' if value else 'false'}")
        elif isinstance(value, list):
            items = ", ".join(
                f'"{x}"' if isinstance(x, str) else str(x) for x in value
            )
            lines.append(f"{prefix}.{key} = [{items}]")
        else:
            lines.append(f"{prefix}.{key} = {value}")

    @staticmethod
    def _emit_endpoint(lines: list[str], comp, comp_type: str):
        """Emit a single endpoint block for a scoring component."""
        lines.append(f"[[stage.scoring.component.{comp_type}.endpoint]]")
        name = comp.name or comp_type
        lines.append(f'name = "{name}"')
        lines.append(f"weight = {comp.weight}")
        if comp.transform:
            for k, v in comp.transform.items():
                ReinventWrapper._emit_toml_value(lines, "transform", k, v)
        if comp.params:
            for k, v in comp.params.items():
                ReinventWrapper._emit_toml_value(lines, "params", k, v)

    def _build_toml_config(
        self,
        stages: list[StageConfig],
        output_dir: str,
        inception_smiles_file: str | None = None,
    ) -> Path:
        """Generate REINVENT4 TOML config file."""
        toml_path = Path(output_dir) / "reinvent_config.toml"

        lines = [
            f'run_type = "staged_learning"',
            f'device = "{self.device}"',
            f'tb_logdir = "{_toml_path(output_dir)}/tb_logs"',
            "",
            "[parameters]",
            f"summary_csv_prefix = \"{_toml_path(output_dir)}/staged_learning\"",
            "use_checkpoint = false",
            "purge_memories = false",
            f'prior_file = "{_toml_path(self.prior_model)}"',
            f'agent_file = "{_toml_path(self.agent_model)}"',
            f"batch_size = {self.batch_size}",
            "unique_sequences = true",
            "randomize_smiles = true",
            "",
            "[learning_strategy]",
            'type = "dap"',
            f"sigma = {self.sigma}",
            f"rate = {self.learning_rate}",
            "",
        ]

        # Diversity filter
        if self.diversity_filter:
            lines.extend([
                "[diversity_filter]",
                f'type = "{self.diversity_filter}"',
                "bucket_size = 25",
                "minscore = 0.4",
                "",
            ])

        # Inception
        lines.extend([
            "[inception]",
            f"memory_size = {self.inception_memory_size}",
            f"sample_size = {self.inception_sample_size}",
        ])
        if inception_smiles_file:
            lines.append(f'smiles_file = "{_toml_path(inception_smiles_file)}"')
        lines.append("")

        # Stages
        for i, stage in enumerate(stages):
            chkpt = stage.chkpt_file or f"{_toml_path(output_dir)}/agent_stage{i+1}.chkpt"
            lines.extend([
                f"[[stage]]",
                f'chkpt_file = "{_toml_path(chkpt)}"',
                'termination = "simple"',
                f"max_score = {stage.max_score}",
                f"min_steps = {stage.min_steps}",
                f"max_steps = {stage.max_steps}",
            ])
            lines.append("")

            # Scoring type (required by REINVENT4 ScorerConfig)
            lines.append("[stage.scoring]")
            lines.append('type = "geometric_mean"')
            lines.append("")

            # Group ExternalProcess components by (executable, args) for multi-endpoint
            ext_groups: dict[tuple, list] = {}
            other_components = []

            for comp in stage.scoring_components:
                if comp.type == "ExternalProcess":
                    key = (comp.params.get("executable", ""), comp.params.get("args", ""))
                    ext_groups.setdefault(key, []).append(comp)
                else:
                    other_components.append(comp)

            # Emit non-ExternalProcess components (one block each)
            for comp in other_components:
                lines.append("[[stage.scoring.component]]")
                lines.append(f"[stage.scoring.component.{comp.type}]")
                self._emit_endpoint(lines, comp, comp.type)
                lines.append("")

            # Emit ExternalProcess groups (one block per unique executable+args)
            for (exe, args_str), comps in ext_groups.items():
                lines.append("[[stage.scoring.component]]")
                lines.append("[stage.scoring.component.ExternalProcess]")
                for comp in comps:
                    self._emit_endpoint(lines, comp, "ExternalProcess")
                lines.append("")

        toml_path.write_text("\n".join(lines))
        log.debug(f"TOML config written to {toml_path}")
        return toml_path
