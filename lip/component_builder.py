"""Build REINVENT4 ScoringComponents from constraints and docking config.

Separated from OptimizationLoop for SRP: the loop orchestrates,
the builder translates constraints into REINVENT4 components.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any

from lip.config import LipConfig
from lip.generator.reinvent import (
    ScoringComponent,
    mw_component,
    logp_component,
    qed_component,
    sa_component,
    similarity_component,
    external_process_component,
    shape_similarity_component,
)
from lip.utils.math import reinvent_reverse_sigmoid

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint -> REINVENT4 component mapping
# ---------------------------------------------------------------------------

# Each entry: constraint_type_name -> factory(params, weight) -> ScoringComponent | None
COMPONENT_FACTORIES: dict[str, Any] = {}


def _register_component(name: str):
    """Decorator to register a constraint-to-component factory."""
    def decorator(func):
        COMPONENT_FACTORIES[name] = func
        return func
    return decorator


@_register_component("property_range")
def _property_range_to_component(params: dict, weight: float) -> ScoringComponent | None:
    prop = params.get("property", "molecular_weight")
    lo = params.get("min", 0)
    hi = params.get("max", 500)
    prop_map = {
        "molecular_weight": lambda: mw_component(lo, hi, weight),
        "logp": lambda: logp_component(lo, hi, weight),
    }
    factory = prop_map.get(prop)
    return factory() if factory else None


@_register_component("qed")
def _qed_to_component(params: dict, weight: float) -> ScoringComponent | None:
    return qed_component(weight)


@_register_component("sa")
def _sa_to_component(params: dict, weight: float) -> ScoringComponent | None:
    return sa_component(weight)


@_register_component("similarity")
def _similarity_to_component(params: dict, weight: float) -> ScoringComponent | None:
    ref = params.get("reference_smiles", "")
    return similarity_component(ref, weight) if ref else None


@_register_component("shape")
def _shape_to_component(params: dict, weight: float) -> ScoringComponent | None:
    return shape_similarity_component(
        reference_smiles=params.get("reference_smiles", ""),
        reference_sdf=params.get("reference_sdf", ""),
        weight=weight,
    )


# ---------------------------------------------------------------------------
# ComponentBuilder
# ---------------------------------------------------------------------------

class ComponentBuilder:
    """Translates Lip constraints + docking config into REINVENT4 ScoringComponents."""

    def __init__(self, config: LipConfig):
        self.config = config

    def build_all(self, constraints: list[tuple[str, float, Any]]) -> list[ScoringComponent]:
        """Build all ScoringComponents from constraints + docking.

        Args:
            constraints: List of (type_name, weight, constraint_instance) tuples.

        Returns:
            List of ScoringComponents for REINVENT4 staged_learning.
        """
        components = []

        for name, weight, constraint in constraints:
            comp = self.constraint_to_component(name, weight, constraint)
            if comp is not None:
                components.append(comp)

        docking_comp = self.build_docking_component()
        if docking_comp is not None:
            components.append(docking_comp)

        return components

    def constraint_to_component(
        self, name: str, weight: float, constraint: Any,
    ) -> ScoringComponent | None:
        """Map a constraint to a REINVENT4 ScoringComponent via registry."""
        factory = COMPONENT_FACTORIES.get(name)
        if factory is None:
            log.warning(f"No REINVENT4 component mapping for constraint '{name}'")
            return None
        return factory(constraint.params, weight)

    def build_docking_component(self) -> ScoringComponent | None:
        """Build docking ExternalProcess component if docking is enabled."""
        cfg = self.config
        if not cfg.docking.enabled or not cfg.receptor_pdb:
            return None

        script = str(Path(__file__).parent / "scoring" / "docking.py")
        center = ",".join(str(c) for c in cfg.pocket_center)
        box_sz = cfg.docking.box_size
        box = f"{box_sz},{box_sz},{box_sz}"

        args = (
            f"{script} "
            f"--receptor {cfg.receptor_pdb} "
            f"--center {center} "
            f"--box-size {box} "
            f"--exhaustiveness {cfg.docking.exhaustiveness}"
        )

        return external_process_component(
            executable=sys.executable,
            args=args,
            property_name="docking_score",
            weight=cfg.docking.weight,
            transform=reinvent_reverse_sigmoid(
                high=cfg.docking.transform.high,
                low=cfg.docking.transform.low,
            ),
        )
