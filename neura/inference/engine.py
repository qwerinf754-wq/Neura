"""
Neura Inference Engine
======================

Single-request inference runtime.

Lifecycle:

    CREATED
       |
       v
    PREFILL
       |
       v
    DECODING
       |
       +----> FINISHED
       |
       +----> FAILED

The engine owns the lifecycle of one active request.

Important cache contract:

    prefill()
        -> creates model KV cache
        -> keeps KV cache alive

    decode()
        -> reuses the existing model KV cache
        -> appends one generated token to the KV cache

    request finish/failure
        -> clears model inference/training caches

The model's CausalLM implementation is the source of truth for
incremental KV-cache behavior.
"""

from dataclasses import dataclass

from neura.tokenizer import TOKENIZER, VOCAB
from neura.generation.sampler import Sampler

from .state import InferenceState


class InferenceError(RuntimeError):
    """Base exception for inference runtime failures."""


@dataclass(slots=True)
class InferenceRequest:
    """Configuration for one inference request."""

    prompt_tokens: list[int]

    max_tokens: int = 20
    temperature: float = 1.0
    top_k: int | None = None
    repetition_penalty: float | None = None
    eos_token: int = 3


@dataclass(slots=True)
class InferenceResponse:
    """Final result of an inference request."""

    prompt_tokens: list[int]
    generated_tokens: list[int]
    finish_reason: str

    @property
    def tokens(self) -> list[int]:
        """Return prompt + generated tokens."""
        return self.prompt_tokens + self.generated_tokens


# Backward-compatible public name.
InferenceResult = InferenceResponse


class InferenceEngine:
    """
    Single-request inference engine.

    The engine intentionally supports only one active request at a time.
    This keeps request-scoped KV ownership explicit and correct.

    For each request:

        _begin_request()
            -> clear stale model state

        prefill()
            -> populate KV cache

        decode()
            -> reuse KV cache

        _finish_request()
            -> release model state
    """

    def __init__(self, model, sampler=None):
        self.model = model

        self.sampler = sampler or Sampler(
            blocked_tokens=[
                0,  # PAD
                1,  # UNK
                2,  # BOS
                *VOCAB.blocked_token_ids(),
            ]
        )

        self._active = False
        self._state = InferenceState.CREATED

        self._prompt_tokens: list[int] = []
        self._generated_tokens: list[int] = []

        self._logits = None
        self._request = None

    # =========================================================
    # Runtime state
    # =========================================================

    @property
    def state(self) -> InferenceState:
        """Current inference lifecycle state."""
        return self._state

    @property
    def active(self) -> bool:
        """Whether a request is currently active."""
        return self._active

    @property
    def prompt_tokens(self) -> list[int]:
        """Copy of the active request prompt tokens."""
        return list(self._prompt_tokens)

    @property
    def generated_tokens(self) -> list[int]:
        """Copy of generated tokens for the active request."""
        return list(self._generated_tokens)

    @property
    def position(self) -> int:
        """
        Current logical sequence length.

        Before decoding:
            prompt length + generated length

        This corresponds to the next absolute position that would be
        inserted into the model KV cache.
        """
        return len(self._prompt_tokens) + len(self._generated_tokens)

    @property
    def logits(self):
        """Latest model logits."""
        return self._logits

    @property
    def kv_cache_size(self) -> int:
        """
        Current model KV-cache sequence length.

        Returns zero when the model does not expose the KV-cache API.
        """
        getter = getattr(self.model, "kv_cache_size", None)

        if getter is None:
            return 0

        return int(getter())

    # =========================================================
    # Model cache helpers
    # =========================================================

    def _clear_model_cache(self) -> None:
        """
        Clear model caches.

        CausalLM.clear_cache() is the preferred API.

        The fallback exists only for compatibility with older model
        implementations.
        """
        clear_cache = getattr(self.model, "clear_cache", None)

        if clear_cache is None:
            raise InferenceError(
                "Model does not provide clear_cache()."
            )

        clear_cache()

    def _clear_model_kv_cache(self) -> None:
        """
        Clear only the inference KV cache when supported.

        This method is intentionally separate from clear_cache().
        It allows future runtimes to release inference state without
        touching training state.
        """
        clear_kv_cache = getattr(
            self.model,
            "clear_kv_cache",
            None,
        )

        if clear_kv_cache is not None:
            clear_kv_cache()
            return

        # Compatibility fallback.
        self._clear_model_cache()

    # =========================================================
    # Request lifecycle
    # =========================================================

    def _begin_request(self, request: InferenceRequest) -> None:
        """
        Begin a new request.

        Any stale model state from a previous request is removed before
        the new request starts.
        """
        if self._active:
            raise InferenceError(
                "InferenceEngine already has an active request."
            )

        self._clear_model_cache()

        self._active = True
        self._request = request

        self._prompt_tokens = []
        self._generated_tokens = []

        self._logits = None
        self._state = InferenceState.CREATED

    def _finish_request(
        self,
        state: InferenceState = InferenceState.FINISHED,
    ) -> None:
        """
        Finish the current request and release model state.

        Runtime metadata remains available through the engine properties,
        while model caches are released.
        """
        try:
            self._clear_model_cache()
        finally:
            self._state = state
            self._active = False
            self._request = None
            self._logits = None

    def clear(self) -> None:
        """
        Explicitly clear the engine and model state.

        This is useful for callers that need to abort a request.
        """
        self._clear_model_cache()

        self._active = False
        self._state = InferenceState.CREATED

        self._prompt_tokens = []
        self._generated_tokens = []

        self._logits = None
        self._request = None

    # =========================================================
    # Validation
    # =========================================================

    def _validate_request(
        self,
        request: InferenceRequest,
    ) -> None:
        """Validate an inference request before execution."""

        if not isinstance(request, InferenceRequest):
            raise TypeError(
                "request must be InferenceRequest"
            )

        if not request.prompt_tokens:
            raise ValueError(
                "prompt_tokens cannot be empty"
            )

        if request.max_tokens < 0:
            raise ValueError(
                "max_tokens must be non-negative"
            )

        if request.temperature <= 0:
            raise ValueError(
                "temperature must be positive"
            )

        if (
            request.top_k is not None
            and request.top_k <= 0
        ):
            raise ValueError(
                "top_k must be positive"
            )

        if (
            request.repetition_penalty is not None
            and request.repetition_penalty <= 0
        ):
            raise ValueError(
                "repetition_penalty must be positive"
            )

        model_vocab_size = getattr(
            self.model,
            "vocab_size",
            None,
        )

        if model_vocab_size is None:
            raise InferenceError(
                "Model does not expose vocab_size."
            )

        model_vocab_size = int(model_vocab_size)

        if model_vocab_size <= 0:
            raise InferenceError(
                f"Invalid model vocab_size: {model_vocab_size}"
            )

        if (
            not isinstance(request.eos_token, int)
            or request.eos_token < 0
            or request.eos_token >= model_vocab_size
        ):
            raise ValueError(
                f"Invalid eos_token: {request.eos_token} "
                f"for model vocab_size={model_vocab_size}"
            )

        for token in request.prompt_tokens:
            if not isinstance(token, int):
                raise TypeError(
                    "prompt tokens must be integers"
                )

            if (
                token < 0
                or token >= model_vocab_size
            ):
                raise ValueError(
                    f"Invalid prompt token: {token} "
                    f"for model vocab_size={model_vocab_size}"
                )

    # =========================================================
    # Model vocabulary
    # =========================================================

    def _model_vocab_size(self) -> int:
        """
        Return the vocabulary size exposed by the active model.

        Inference validation must use the model vocabulary, not
        VOCAB.capacity, because tokenizer capacity is optional.
        """
        value = getattr(
            self.model,
            "vocab_size",
            None,
        )

        if value is None:
            raise InferenceError(
                "Model does not expose vocab_size."
            )

        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise InferenceError(
                f"Invalid model vocab_size: {value!r}"
            ) from exc

        if value <= 0:
            raise InferenceError(
                f"Invalid model vocab_size: {value}"
            )

        return value

    # =========================================================
    # Sampling
    # =========================================================

    def _sample(self, logits):
        """Sample one token from the latest logits."""

        request = self._request

        if request is None:
            raise InferenceError(
                "No active request."
            )

        scores = list(logits)

        history = (
            self._prompt_tokens
            + self._generated_tokens
        )

        if request.repetition_penalty:
            scores = self.sampler.apply_repetition_penalty(
                scores,
                history,
                request.repetition_penalty,
            )

        token = self.sampler.select(
            scores,
            len(scores),
            request.temperature,
            request.top_k,
        )

        vocab_size = self._model_vocab_size()

        if (
            token < 0
            or token >= vocab_size
        ):
            raise InferenceError(
                f"Sampler produced invalid token: {token} "
                f"for model vocab_size={vocab_size}"
            )

        return token

    # =========================================================
    # PREFILL
    # =========================================================

    def prefill(
        self,
        request: InferenceRequest,
    ):
        """
        Execute prompt prefill.

        The prompt is processed once.

        The model KV cache is intentionally retained after this method
        returns so decode() can continue incrementally.
        """
        self._validate_request(request)
        self._begin_request(request)

        try:
            self._state = InferenceState.PREFILL

            prompt = list(request.prompt_tokens)

            self._prompt_tokens = prompt

            logits = self.model.forward(
                prompt,
                use_cache=True,
            )

            if len(logits) != len(prompt):
                raise InferenceError(
                    "Prefill returned invalid sequence length."
                )

            self._logits = logits

            # The model must contain exactly the prompt positions.
            if self.kv_cache_size != len(prompt):
                raise InferenceError(
                    "KV cache size does not match prompt length: "
                    f"{self.kv_cache_size} != {len(prompt)}"
                )

            if request.max_tokens == 0:
                return logits

            self._state = InferenceState.DECODING

            return logits

        except Exception:
            if self._active:
                self._finish_request(
                    InferenceState.FAILED
                )
            raise

    # =========================================================
    # DECODE
    # =========================================================

    def decode(self):
        """
        Generate exactly one token.

        The current model KV cache is reused.

        Before generating the next token:

            cache_size == len(prompt) + len(generated)

        The sampled token is then forwarded through the model, which
        appends its K/V at that absolute position.

        If the token immediately finishes the request (EOS or length),
        no additional forward is required because the request is ending.
        """
        if not self._active:
            raise InferenceError(
                "No active inference request."
            )

        if self._state not in (
            InferenceState.PREFILL,
            InferenceState.DECODING,
        ):
            raise InferenceError(
                "Cannot decode from state "
                f"{self._state.value}."
            )

        request = self._request

        if request is None:
            raise InferenceError(
                "Active request metadata is missing."
            )

        if (
            len(self._generated_tokens)
            >= request.max_tokens
        ):
            raise InferenceError(
                "Maximum generation length reached."
            )

        if self._logits is None:
            raise InferenceError(
                "Prefill must complete before decode."
            )

        self._state = InferenceState.DECODING

        # -----------------------------------------------------
        # Validate cache position before sampling.
        # -----------------------------------------------------

        expected_position = self.position

        if self.kv_cache_size != expected_position:
            raise InferenceError(
                "KV cache position mismatch before decode: "
                f"cache={self.kv_cache_size}, "
                f"expected={expected_position}"
            )

        # -----------------------------------------------------
        # Sample next token.
        # -----------------------------------------------------

        token = self._sample(
            self._logits[-1]
        )

        self._generated_tokens.append(token)

        # -----------------------------------------------------
        # EOS termination.
        #
        # The sampled token does not need another model forward
        # when the request is already finished.
        # -----------------------------------------------------

        if token == request.eos_token:
            response = InferenceResponse(
                prompt_tokens=list(
                    self._prompt_tokens
                ),
                generated_tokens=list(
                    self._generated_tokens
                ),
                finish_reason="eos",
            )

            self._finish_request(
                InferenceState.FINISHED
            )

            return token, response

        # -----------------------------------------------------
        # Length termination.
        # -----------------------------------------------------

        if (
            len(self._generated_tokens)
            >= request.max_tokens
        ):
            response = InferenceResponse(
                prompt_tokens=list(
                    self._prompt_tokens
                ),
                generated_tokens=list(
                    self._generated_tokens
                ),
                finish_reason="length",
            )

            self._finish_request(
                InferenceState.FINISHED
            )

            return token, response

        # -----------------------------------------------------
        # Incremental model forward.
        #
        # CausalLM automatically derives cache_position from
        # the existing KV cache when cache_position is omitted.
        # -----------------------------------------------------

        self._logits = self.model.forward(
            [token],
            use_cache=True,
        )

        if len(self._logits) != 1:
            raise InferenceError(
                "Incremental forward must return one token."
            )

        expected_after = self.position

        if self.kv_cache_size != expected_after:
            raise InferenceError(
                "KV cache position mismatch after decode: "
                f"cache={self.kv_cache_size}, "
                f"expected={expected_after}"
            )

        return token, None

    # =========================================================
    # COMPLETE REQUEST
    # =========================================================

    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """
        Generate a complete response using prefill + decode.
        """
        self.prefill(request)

        # Zero-token generation finishes immediately.
        if request.max_tokens == 0:
            response = InferenceResponse(
                prompt_tokens=list(
                    request.prompt_tokens
                ),
                generated_tokens=[],
                finish_reason="length",
            )

            self._finish_request(
                InferenceState.FINISHED
            )

            return response

        try:
            while self._active:
                _, response = self.decode()

                if response is not None:
                    return response

            raise InferenceError(
                "Inference terminated without a response."
            )

        except Exception:
            if self._active:
                self._finish_request(
                    InferenceState.FAILED
                )
            raise

    # =========================================================
    # Text convenience API
    # =========================================================

    def generate_text(
        self,
        text: str,
        max_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        eos_token: int = 3,
    ) -> InferenceResponse:
        """
        Tokenize text and generate a response.

        This method is intentionally a thin convenience wrapper around
        the token-based inference API.
        """
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        tokens = TOKENIZER.encode(text)

        request = InferenceRequest(
            prompt_tokens=list(tokens),
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token=eos_token,
        )

        return self.generate(request)


# Public exports.
__all__ = [
    "InferenceError",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceResult",
    "InferenceEngine",
]
