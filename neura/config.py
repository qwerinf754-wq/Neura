"""
Neura Configuration
"""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class Config:

    project_name: str = "Neura"

    version: str = "11.0.0"

    debug: bool = True

    host: str = "127.0.0.1"

    port: int = 8000

    models = ROOT / "models"

    datasets = ROOT / "datasets"

    checkpoints = ROOT / "checkpoints"


CONFIG = Config()
