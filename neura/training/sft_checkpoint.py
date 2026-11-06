"""
Neura SFT Checkpoint Runtime

Persistent checkpoint support for conversational SFT.

Stores:
    - model state
    - optimizer state
    - training progress
    - loss history
    - model configuration
    - SFT configuration
    - masking configuration
    - tokenizer vocabulary
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import torch


class SFTCheckpointError(Exception):
    """Raised when an SFT checkpoint is invalid."""


class SFTCheckpoint:

    VERSION = 3

    def __init__(self, runtime=None):
        self.runtime = runtime

    # ---------------------------------------------------------
    # VOCABULARY
    # ---------------------------------------------------------

    @staticmethod
    def _capture_vocabulary(tokenizer=None):
        if tokenizer is None:
            from neura.tokenizer import TOKENIZER
            tokenizer = TOKENIZER

        # Production BPE tokenizer exposes vocabulary as a dict.
        if hasattr(tokenizer, "vocab_size") and isinstance(
            getattr(tokenizer, "vocab", None), dict
        ):
            token_to_id = dict(tokenizer.vocab)
            id_to_token = {
                str(token_id): token
                for token, token_id in token_to_id.items()
            }

            vocabulary = {
                "token_to_id": token_to_id,
                "id_to_token": id_to_token,
                "size": int(tokenizer.vocab_size),
            }

            if vocabulary["size"] != len(token_to_id):
                raise SFTCheckpointError(
                    "tokenizer vocabulary size does not match vocabulary entries: "
                    f"size={vocabulary['size']}, entries={len(token_to_id)}"
                )

            return vocabulary

        # Legacy tokenizer compatibility.
        vocabulary = tokenizer.vocab

        return {
            "token_to_id": dict(vocabulary.token_to_id),
            "id_to_token": {
                str(key): value
                for key, value in vocabulary.id_to_token.items()
            },
            "size": int(vocabulary.size()),
        }

    @staticmethod
    def _restore_vocabulary(vocabulary, tokenizer=None):
        if vocabulary is None:
            raise SFTCheckpointError("checkpoint vocabulary cannot be None")

        if tokenizer is None:
            from neura.tokenizer import TOKENIZER
            tokenizer = TOKENIZER

        expected_size = int(vocabulary["size"])
        token_to_id = dict(vocabulary["token_to_id"])

        if expected_size != len(token_to_id):
            raise SFTCheckpointError(
                "checkpoint vocabulary size does not match entries: "
                f"size={expected_size}, entries={len(token_to_id)}"
            )

        # Production BPE tokenizer.
        # The tokenizer artifact is authoritative, so do not mutate it.
        if (
            hasattr(tokenizer, "vocab_size")
            and isinstance(getattr(tokenizer, "vocab", None), dict)
        ):
            if int(tokenizer.vocab_size) != expected_size:
                raise SFTCheckpointError(
                    "checkpoint/tokenizer vocabulary size mismatch: "
                    f"checkpoint={expected_size}, "
                    f"tokenizer={tokenizer.vocab_size}"
                )

            if dict(tokenizer.vocab) != token_to_id:
                raise SFTCheckpointError(
                    "checkpoint/tokenizer vocabulary entries mismatch"
                )

            return

        # Legacy tokenizer compatibility.
        legacy_vocab = tokenizer.vocab

        legacy_vocab.token_to_id.clear()
        legacy_vocab.token_to_id.update(token_to_id)

        legacy_vocab.id_to_token.clear()
        legacy_vocab.id_to_token.update(
            {
                int(token_id): token
                for token_id, token in vocabulary["id_to_token"].items()
            }
        )

        if hasattr(tokenizer, "freeze"):
            tokenizer.freeze()

    # ---------------------------------------------------------
    # TOKENIZER CONTRACT
    # ---------------------------------------------------------

    @staticmethod
    def _tokenizer_contract(tokenizer, artifact_path=None):
        if tokenizer is None:
            raise SFTCheckpointError("tokenizer cannot be None")

        required = (
            "vocab_size",
            "PAD_ID",
            "UNK_ID",
            "BOS_ID",
            "EOS_ID",
        )

        for name in required:
            if not hasattr(tokenizer, name):
                raise SFTCheckpointError(
                    f"tokenizer missing required attribute: {name}"
                )

        vocab_size = int(tokenizer.vocab_size)

        if vocab_size <= 0:
            raise SFTCheckpointError(
                "tokenizer vocabulary size must be positive"
            )

        special_ids = {
            "PAD_ID": int(tokenizer.PAD_ID),
            "UNK_ID": int(tokenizer.UNK_ID),
            "BOS_ID": int(tokenizer.BOS_ID),
            "EOS_ID": int(tokenizer.EOS_ID),
        }

        for name, token_id in special_ids.items():
            if not 0 <= token_id < vocab_size:
                raise SFTCheckpointError(
                    f"tokenizer {name}={token_id} outside "
                    f"vocabulary size {vocab_size}"
                )

        contract = {
            "type": type(tokenizer).__name__,
            "vocab_size": vocab_size,
            "special_ids": special_ids,
        }

        if artifact_path is not None:
            artifact = Path(artifact_path)

            if not artifact.exists():
                raise SFTCheckpointError(
                    f"tokenizer artifact not found: {artifact}"
                )

            digest = hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest()

            contract["artifact"] = str(artifact)
            contract["sha256"] = digest

        return contract

    @staticmethod
    def _validate_tokenizer_contract(
        checkpoint_contract,
        tokenizer,
        artifact_path=None,
    ):
        if not isinstance(checkpoint_contract, dict):
            raise SFTCheckpointError(
                "invalid tokenizer contract"
            )

        current = SFTCheckpoint._tokenizer_contract(
            tokenizer,
            artifact_path=artifact_path,
        )

        for key in (
            "vocab_size",
            "special_ids",
        ):
            if checkpoint_contract.get(key) != current.get(key):
                raise SFTCheckpointError(
                    "tokenizer contract mismatch: "
                    f"{key}: "
                    f"checkpoint={checkpoint_contract.get(key)!r}, "
                    f"current={current.get(key)!r}"
                )

        checkpoint_sha = checkpoint_contract.get("sha256")

        if checkpoint_sha is not None:
            if current.get("sha256") != checkpoint_sha:
                raise SFTCheckpointError(
                    "tokenizer artifact SHA256 mismatch"
                )

    # ---------------------------------------------------------
    # STATE
    # ---------------------------------------------------------

    def build_state(
        self,
        model,
        optimizer,
        *,
        step,
        loss,
        model_config,
        sft_config,
        loss_history=None,
        tokenizer=None,
        tokenizer_artifact=None,
    ):

        if model is None:
            raise SFTCheckpointError(
                "model cannot be None"
            )

        if optimizer is None:
            raise SFTCheckpointError(
                "optimizer cannot be None"
            )

        vocabulary = self._capture_vocabulary(tokenizer)

        tokenizer_contract = None

        if tokenizer is not None:
            tokenizer_contract = self._tokenizer_contract(
                tokenizer,
                artifact_path=tokenizer_artifact,
            )

        model_vocab_size = int(
            model_config["vocab_size"]
        )

        if vocabulary["size"] != model_vocab_size:
            raise SFTCheckpointError(
                "model/tokenizer vocabulary size mismatch: "
                f"model={model_vocab_size}, "
                f"tokenizer={vocabulary['size']}"
            )

        state = {
            "version": self.VERSION,

            "training": {
                "step": int(step),
                "loss": float(loss),
                "loss_history": (
                    list(loss_history)
                    if loss_history is not None
                    else []
                ),
            },

            "model_config": dict(
                model_config
            ),

            "sft_config": dict(
                sft_config
            ),

            "masking": {
                "ignore_index": -100,
                "policy": "assistant_only",
                "system": "ignore",
                "user": "ignore",
                "assistant_marker": "ignore",
                "assistant_content": "train",
                "assistant_eos": "train",
            },

            "vocabulary": vocabulary,

            "tokenizer_contract": tokenizer_contract,

            "model_state_dict": (
                model.state_dict()
                if hasattr(model, "state_dict")
                else self._capture_model_state()
            ),

            "optimizer_state_dict": (
                optimizer.state_dict()
                if hasattr(optimizer, "state_dict")
                else self._capture_optimizer_state(
                    optimizer
                )
            ),
        }

        return state

    # ---------------------------------------------------------
    # MODEL STATE
    # ---------------------------------------------------------

    @staticmethod
    def _capture_model_state():
        from neura.model.parameters import PARAMETERS

        return PARAMETERS.state_dict()

    @staticmethod
    def _restore_model_state(state):
        from neura.model.parameters import PARAMETERS

        PARAMETERS.load_state_dict(state)

    # ---------------------------------------------------------
    # OPTIMIZER STATE
    # ---------------------------------------------------------

    @staticmethod
    def _capture_optimizer_state(optimizer):
        import copy

        required = (
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "step_count",
            "m",
            "v",
        )

        missing = [
            name
            for name in required
            if not hasattr(optimizer, name)
        ]

        if missing:
            raise SFTCheckpointError(
                "optimizer does not expose required state: "
                + ", ".join(missing)
            )

        return {
            "learning_rate": float(
                optimizer.learning_rate
            ),
            "weight_decay": float(
                optimizer.weight_decay
            ),
            "beta1": float(
                optimizer.beta1
            ),
            "beta2": float(
                optimizer.beta2
            ),
            "epsilon": float(
                optimizer.epsilon
            ),
            "step_count": int(
                optimizer.step_count
            ),
            "m": copy.deepcopy(
                optimizer.m
            ),
            "v": copy.deepcopy(
                optimizer.v
            ),
        }

    @staticmethod
    def _restore_optimizer_state(
        optimizer,
        state,
    ):
        import copy

        if not isinstance(state, dict):
            raise SFTCheckpointError(
                "optimizer state must be a dictionary"
            )

        required = (
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "step_count",
            "m",
            "v",
        )

        missing = [
            name
            for name in required
            if name not in state
        ]

        if missing:
            raise SFTCheckpointError(
                "optimizer checkpoint is missing state: "
                + ", ".join(missing)
            )

        optimizer.learning_rate = float(
            state["learning_rate"]
        )
        optimizer.weight_decay = float(
            state["weight_decay"]
        )
        optimizer.beta1 = float(
            state["beta1"]
        )
        optimizer.beta2 = float(
            state["beta2"]
        )
        optimizer.epsilon = float(
            state["epsilon"]
        )
        optimizer.step_count = int(
            state["step_count"]
        )
        optimizer.m = copy.deepcopy(
            state["m"]
        )
        optimizer.v = copy.deepcopy(
            state["v"]
        )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    def save_state(
        self,
        path,
        state,
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            state,
            path,
        )

        return path

    def save(
        self,
        path,
        model,
        optimizer,
        *,
        step,
        loss,
        model_config,
        sft_config,
        loss_history=None,
        tokenizer=None,
        tokenizer_artifact=None,
    ):

        state = self.build_state(
            model,
            optimizer,
            step=step,
            loss=loss,
            model_config=model_config,
            sft_config=sft_config,
            loss_history=loss_history,
            tokenizer=tokenizer,
            tokenizer_artifact=tokenizer_artifact,
        )

        return self.save_state(
            path,
            state,
        )

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    def load_state(
        self,
        path,
        *,
        map_location="cpu",
    ):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"SFT checkpoint not found: {path}"
            )

        try:

            state = torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )

        except TypeError:

            state = torch.load(
                path,
                map_location=map_location,
            )

        except Exception as exc:

            raise SFTCheckpointError(
                f"failed to load checkpoint: {exc}"
            ) from exc

        if not isinstance(
            state,
            dict,
        ):
            raise SFTCheckpointError(
                "checkpoint must contain a dictionary"
            )

        version = state.get(
            "version"
        )

        if version != self.VERSION:
            raise SFTCheckpointError(
                "unsupported SFT checkpoint version: "
                f"{version}; expected {self.VERSION}"
            )

        required = (
            "training",
            "model_config",
            "sft_config",
            "masking",
            "vocabulary",
            "model_state_dict",
            "optimizer_state_dict",
        )

        for key in required:

            if key not in state:
                raise SFTCheckpointError(
                    f"missing checkpoint field: {key}"
                )

        return state

    # ---------------------------------------------------------
    # RESTORE
    # ---------------------------------------------------------

    def restore(
        self,
        path,
        model,
        optimizer,
        *,
        tokenizer=None,
        tokenizer_artifact=None,
        map_location="cpu",
    ):

        state = self.load_state(
            path,
            map_location=map_location,
        )

        model_config = state[
            "model_config"
        ]

        checkpoint_vocab_size = int(
            model_config["vocab_size"]
        )

        vocabulary = state[
            "vocabulary"
        ]

        if int(vocabulary["size"]) != checkpoint_vocab_size:
            raise SFTCheckpointError(
                "checkpoint vocabulary/model mismatch: "
                f"vocabulary={vocabulary['size']}, "
                f"model={checkpoint_vocab_size}"
            )

        tokenizer_contract = state.get(
            "tokenizer_contract"
        )

        if tokenizer_contract is not None:
            if tokenizer is None:
                raise SFTCheckpointError(
                    "checkpoint requires an explicit tokenizer"
                )

            self._validate_tokenizer_contract(
                tokenizer_contract,
                tokenizer,
                artifact_path=tokenizer_artifact,
            )

        self._restore_vocabulary(
            vocabulary,
            tokenizer=tokenizer,
        )

        try:

            if hasattr(model, "load_state_dict"):
                model.load_state_dict(
                    state["model_state_dict"],
                    strict=True,
                )
            else:
                self._restore_model_state(
                    state["model_state_dict"]
                )

            if hasattr(optimizer, "load_state_dict"):
                optimizer.load_state_dict(
                    state["optimizer_state_dict"]
                )
            else:
                self._restore_optimizer_state(
                    optimizer,
                    state["optimizer_state_dict"],
                )

        except Exception as exc:

            raise SFTCheckpointError(
                f"failed to restore training state: {exc}"
            ) from exc

        training = state[
            "training"
        ]

        return {
            "step": int(
                training["step"]
            ),

            "loss": float(
                training["loss"]
            ),

            "loss_history": list(
                training.get(
                    "loss_history",
                    [],
                )
            ),

            "model_config": dict(
                state["model_config"]
            ),

            "sft_config": dict(
                state["sft_config"]
            ),

            "masking": dict(
                state["masking"]
            ),

            "vocabulary": dict(
                state["vocabulary"]
            ),
            "tokenizer_contract": (
                dict(tokenizer_contract)
                if isinstance(tokenizer_contract, dict)
                else None
            ),
        }


__all__ = [
    "SFTCheckpoint",
    "SFTCheckpointError",
]
