"""Base generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    smiles: list[str]
    nll: list[float]  # Negative log-likelihood from the model
    valid_ratio: float


class BaseGenerator(ABC):
    """Abstract base for molecular generators."""

    @abstractmethod
    def sample(self, n: int) -> GenerationResult:
        """Generate n molecules."""

    @abstractmethod
    def start_rl(self, config: dict) -> None:
        """Start RL training loop (managed mode)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop any running subprocess."""

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save current agent model checkpoint."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load agent model from checkpoint."""
