"""Safety policy for Neura Evolution Engine.

Phase 1 is observation-only.
No automatic source, model, checkpoint, or configuration mutation is allowed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionPolicy:
    observation_only: bool = True

    allow_source_modification: bool = False
    allow_checkpoint_modification: bool = False
    allow_config_modification: bool = False
    allow_dependency_installation: bool = False
    allow_file_deletion: bool = False
    allow_shell_execution: bool = False

    require_validation_before_apply: bool = True
    require_backup_before_apply: bool = True


DEFAULT_POLICY = EvolutionPolicy()
