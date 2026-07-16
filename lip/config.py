"""Configuration system for Lip.

Supports three sources (priority: CLI args > YAML config > defaults):
  1. Built-in defaults (dataclass field defaults)
  2. YAML config file (--config path.yaml)
  3. CLI arguments (--batch-size 100, etc.)
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    prior_model: str = ".reinvent"
    agent_model: str = ""
    device: str = "cpu"
    batch_size: int = 50
    sigma: int = 128
    learning_rate: float = 0.0001
    diversity_filter: str = "IdenticalMurckoScaffold"
    work_dir: str = ""


@dataclass
class InceptionConfig:
    memory_size: int = 100
    sample_size: int = 10000
    retention: str = "top"  # "top" | "random"


@dataclass
class OptimizationConfig:
    mode: str = "managed"  # "managed" | "manual"
    n_steps: int = 10
    batch_size: int = 100
    reinvent_timeout: int = 43200
    max_score: float = 0.85
    early_stop_patience: int = 5
    inception: InceptionConfig = field(default_factory=InceptionConfig)
    # Manual mode
    n_molecules_per_round: int = 500
    n_rounds: int = 50
    diversity_threshold: float = 0.4
    checkpoint_every: int = 1
    early_stop_threshold: float = 0.001


@dataclass
class DockingTransform:
    high: float = -6.0
    low: float = -12.0


@dataclass
class DockingConfig:
    enabled: bool = True
    method: str = "auto"  # "vina" | "unidock" | "auto"
    exhaustiveness: int = 8
    box_size: int = 25
    weight: float = 0.5
    transform: DockingTransform = field(default_factory=DockingTransform)
    # Interaction analysis
    analyze_interactions: bool = True
    interaction_weight: float = 0.3
    interaction_norm_max: float = 10.0


@dataclass
class ConstraintConfig:
    type: str = ""
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class FilterConfig:
    lipinski: bool = True
    pains: bool = True


@dataclass
class Pocket2MolConfig:
    enabled: bool = False
    n_samples: int = 3
    bbox_size: float = 23.0
    shape_weight: float = 0.6
    conda_env: str = "Pocket2Mol"
    timeout: int = 600


@dataclass
class PathsConfig:
    pocket2mol_dir: str = ""
    aizynthfinder_config: str = ""


@dataclass
class SynthesisConfig:
    enabled: bool = True
    top_n: int = 10
    time_limit: int = 120
    iteration_limit: int = 100
    max_routes: int = 5


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class LipConfig:
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    constraints: list[ConstraintConfig] = field(default_factory=list)
    filter: FilterConfig = field(default_factory=FilterConfig)
    pocket2mol: Pocket2MolConfig = field(default_factory=Pocket2MolConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)

    receptor_pdb: str = ""
    pocket_center: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    is_docked: bool = True              # True: PDB에 리간드 포함, False: PDB+SDF 별도
    ligand_sdf: str = ""                # is_docked=False일 때 리간드 SDF 경로
    ligand_id: str = ""                  # 타겟 리간드 지정 "chain:resnum" (e.g. "A:300")
    final_docking: bool = False          # True: 최종 top molecule만 재도킹해 complex PDB 저장
    output_dir: str = "results/"
    scoring_method: str = "weighted_sum"  # "weighted_sum" | "pareto"

    # --- loaders ---

    @classmethod
    def from_yaml(cls, path: str | Path) -> LipConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> LipConfig:
        """Recursively build config from a nested dict."""
        cfg = cls()

        # Generator
        if "generator" in data:
            cfg.generator = _merge_dataclass(GeneratorConfig, data["generator"])

        # Optimization
        if "optimization" in data:
            opt_data = dict(data["optimization"])
            inception_data = opt_data.pop("inception", None)
            cfg.optimization = _merge_dataclass(OptimizationConfig, opt_data)
            if inception_data:
                cfg.optimization.inception = _merge_dataclass(
                    InceptionConfig, inception_data
                )

        # Docking
        if "docking" in data:
            dock_data = dict(data["docking"])
            transform_data = dock_data.pop("transform", None)
            cfg.docking = _merge_dataclass(DockingConfig, dock_data)
            if transform_data:
                cfg.docking.transform = _merge_dataclass(
                    DockingTransform, transform_data
                )

        # Constraints
        if "constraints" in data:
            cfg.constraints = [
                _merge_dataclass(ConstraintConfig, c) for c in data["constraints"]
            ]

        # Filter
        if "filter" in data:
            cfg.filter = _merge_dataclass(FilterConfig, data["filter"])

        # Pocket2Mol
        if "pocket2mol" in data:
            cfg.pocket2mol = _merge_dataclass(Pocket2MolConfig, data["pocket2mol"])

        # Paths
        if "paths" in data:
            cfg.paths = _merge_dataclass(PathsConfig, data["paths"])

        # Synthesis
        if "synthesis" in data:
            cfg.synthesis = _merge_dataclass(SynthesisConfig, data["synthesis"])

        # Top-level scalars
        for key in ("receptor_pdb", "pocket_center", "is_docked", "ligand_sdf", "ligand_id", "final_docking", "output_dir", "scoring_method"):
            if key in data:
                setattr(cfg, key, data[key])

        return cfg

    def merge_cli(self, cli_args: dict[str, Any]) -> None:
        """Override config values with CLI arguments (non-None only)."""
        mapping = {
            # Generator
            "prior_model": ("generator", "prior_model"),
            "agent_model": ("generator", "agent_model"),
            "device": ("generator", "device"),
            "batch_size": ("generator", "batch_size"),
            "sigma": ("generator", "sigma"),
            "learning_rate": ("generator", "learning_rate"),
            "diversity_filter": ("generator", "diversity_filter"),
            # Optimization
            "mode": ("optimization", "mode"),
            "n_steps": ("optimization", "n_steps"),
            "opt_batch_size": ("optimization", "batch_size"),
            "max_score": ("optimization", "max_score"),
            "early_stop_patience": ("optimization", "early_stop_patience"),
            "inception_memory_size": ("optimization.inception", "memory_size"),
            "inception_sample_size": ("optimization.inception", "sample_size"),
            "inception_retention": ("optimization.inception", "retention"),
            "n_molecules_per_round": ("optimization", "n_molecules_per_round"),
            "n_rounds": ("optimization", "n_rounds"),
            "diversity_threshold": ("optimization", "diversity_threshold"),
            "checkpoint_every": ("optimization", "checkpoint_every"),
            # Docking
            "docking_method": ("docking", "method"),
            "exhaustiveness": ("docking", "exhaustiveness"),
            "box_size": ("docking", "box_size"),
            "docking_weight": ("docking", "weight"),
            "transform_high": ("docking.transform", "high"),
            "transform_low": ("docking.transform", "low"),
            "analyze_interactions": ("docking", "analyze_interactions"),
            "interaction_weight": ("docking", "interaction_weight"),
            # Filter
            "filter_lipinski": ("filter", "lipinski"),
            "filter_pains": ("filter", "pains"),
            # Pocket2Mol
            "pocket2mol": ("pocket2mol", "enabled"),
            "pocket2mol_n_samples": ("pocket2mol", "n_samples"),
            "pocket2mol_shape_weight": ("pocket2mol", "shape_weight"),
            # Paths
            "pocket2mol_dir": ("paths", "pocket2mol_dir"),
            "aizynthfinder_config": ("paths", "aizynthfinder_config"),
            # Synthesis
            "synthesis_enabled": ("synthesis", "enabled"),
            "synthesis_top_n": ("synthesis", "top_n"),
            "synthesis_time_limit": ("synthesis", "time_limit"),
        }

        for cli_key, (path, attr) in mapping.items():
            value = cli_args.get(cli_key)
            if value is None:
                continue
            obj = self._resolve_path(path)
            setattr(obj, attr, value)

        # Top-level scalars
        for key in ("receptor_pdb", "pocket_center", "is_docked", "ligand_sdf", "ligand_id", "final_docking", "output_dir", "scoring_method"):
            value = cli_args.get(key)
            if value is not None:
                setattr(self, key, value)

        # CLI constraints (parsed from --constraint flags)
        cli_constraints = cli_args.get("_cli_constraints")
        if cli_constraints:
            self.constraints.extend(cli_constraints)

    def _resolve_path(self, dotted: str) -> Any:
        """Resolve 'docking.transform' → self.docking.transform."""
        obj = self
        for part in dotted.split("."):
            obj = getattr(obj, part)
        return obj

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for saving config.yaml snapshots)."""
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """Save current config to YAML."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_dataclass(cls, data: dict[str, Any]):
    """Create a dataclass instance, ignoring unknown keys."""
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)
