"""
neura.core.constants

Global constants for the Neura AI Platform.
"""

from __future__ import annotations


# ---------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------

PROJECT_NAME = "Neura"
ENGINE_NAME = "Neura Core"

DEFAULT_STAGE = "Alpha"


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = "."

CONFIG_DIR = "configs"
DATASET_DIR = "datasets"
CHECKPOINT_DIR = "checkpoints"
DOCS_DIR = "docs"
TESTS_DIR = "tests"


# ---------------------------------------------------------------------
# Model Defaults
# ---------------------------------------------------------------------

DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_CONTEXT_SIZE = DEFAULT_CONTEXT_LENGTH

DEFAULT_VOCAB_SIZE = 32768
DEFAULT_EMBED_DIM = 768

DEFAULT_NUM_HEADS = 12
DEFAULT_NUM_LAYERS = 12


# ---------------------------------------------------------------------
# Training Defaults
# ---------------------------------------------------------------------

DEFAULT_SEED = 42

DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.1


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

DEFAULT_LOGGER_NAME = "neura"
DEFAULT_LOG_LEVEL = "INFO"


# ---------------------------------------------------------------------
# File Extensions
# ---------------------------------------------------------------------

CONFIG_EXTENSION = ".yaml"
CHECKPOINT_EXTENSION = ".pt"
TOKENIZER_EXTENSION = ".json"
