"""
Neura Batch Executor
====================

Executes validated inference batches against the Neura model.

Version 2 guarantees:

- Prefill execution
- Incremental decode execution
- Request ordering preservation
- Per-request logits
- Sequential request isolation
- Persistent KV cache across prefill/decode
- Request state transitions
- Failure isolation
- Deterministic cleanup

Important
---------
The current CausalLM owns a model-level KV cache.

Therefore this executor intentionally executes requests sequentially.

A request owns the model cache for its complete lifecycle:

    CREATED
        |
        v
    PREFILL
        |
        v
    DECODING
        |
        v
    FINISHED / FAILED
        |
        v
    cache cleared

The cache MUST NOT be cleared between prefill and decode.

True concurrent request-scoped KV isolation will be introduced later
through a dedicated cache abstraction.
"""

from dataclasses import dataclass

from .batch_builder import InferenceBatch, BatchError
from .request import RequestContext
from .state import InferenceState


class BatchExecutionError(RuntimeError):
    """Base exception for batch execution failures."""


@dataclass(slots=True)
class ExecutionResult:
    request_id: str
    mode: str
    logits: list
    cache_size: int


@dataclass(slots=True)
class ExecutorStep:
    """
    Compatibility representation of one executor step.
    """

    request_id: str
    mode: str
    logits: list
    token: int | None = None
    finished: bool = False
    finish_reason: str | None = None

    @property
    def generated_token(self):
        return self.token


class BatchExecutor:
    """
    Sequential executor for request-scoped inference.

    The model cache remains alive from prefill through incremental
    decode. It is cleared only when the request finishes or fails.
    """

    def __init__(self, model):
        self.model = model

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_batch(self, batch, expected_mode):
        if not isinstance(batch, InferenceBatch):
            raise TypeError(
                "batch must be InferenceBatch"
            )

        if batch.mode != expected_mode:
            raise BatchError(
                f"Expected {expected_mode} batch, "
                f"got {batch.mode}"
            )

        for item in batch.items:
            if not isinstance(
                item.request,
                RequestContext,
            ):
                raise TypeError(
                    "batch item must contain RequestContext"
                )

    # ---------------------------------------------------------
    # Model cache helpers
    # ---------------------------------------------------------

    def _clear_model_cache(self):
        """
        Clear both training and inference model caches.

        This is used only at request boundaries.
        """
        self.model.clear_cache()

    def _kv_cache_size(self) -> int:
        """
        Return the current model KV-cache size.

        CausalLM exposes kv_cache_size() as the authoritative
        inference-cache API.
        """
        value = getattr(
            self.model,
            "kv_cache_size",
            None,
        )

        if not callable(value):
            raise BatchExecutionError(
                "Model does not expose kv_cache_size()."
            )

        try:
            size = int(value())
        except (TypeError, ValueError) as exc:
            raise BatchExecutionError(
                "Model returned an invalid KV-cache size."
            ) from exc

        if size < 0:
            raise BatchExecutionError(
                f"Invalid KV-cache size: {size}"
            )

        return size

    # ---------------------------------------------------------
    # Request cleanup
    # ---------------------------------------------------------

    def _finish_request_cleanup(self, request):
        """
        Clear model state after a request reaches a terminal state.
        """
        try:
            self._clear_model_cache()
        except Exception as exc:
            request.state = InferenceState.FAILED
            raise BatchExecutionError(
                "Failed to clear model cache after "
                f"request {request.request_id}: {exc}"
            ) from exc

    # ---------------------------------------------------------
    # Prefill
    # ---------------------------------------------------------

    def execute_prefill(self, batch):
        """
        Execute prefill for a sequential batch.

        The KV cache remains alive after this method returns so that
        execute_decode() can continue the same request.

        Because the model cache is shared, this executor supports
        only one active request at a time.
        """
        self._validate_batch(
            batch,
            "prefill",
        )

        results = []

        if len(batch.items) > 1:
            raise BatchExecutionError(
                "Current model-level KV cache supports only "
                "one active request at a time."
            )

        for item in batch.items:
            request = item.request

            if request.state != InferenceState.CREATED:
                raise BatchExecutionError(
                    "Prefill request must be CREATED: "
                    f"{request.request_id}"
                )

            # A new request must start with a clean model state.
            self._clear_model_cache()

            try:
                request.state = InferenceState.PREFILL

                prompt_tokens = list(
                    request.prompt_tokens
                )

                logits = self.model.forward(
                    prompt_tokens,
                    use_cache=True,
                )

                if len(logits) != len(prompt_tokens):
                    raise BatchExecutionError(
                        "Prefill returned invalid sequence "
                        "length for "
                        f"{request.request_id}"
                    )

                cache_size = self._kv_cache_size()

                if cache_size != len(prompt_tokens):
                    raise BatchExecutionError(
                        "Prefill KV-cache size mismatch for "
                        f"{request.request_id}: "
                        f"expected {len(prompt_tokens)}, "
                        f"got {cache_size}"
                    )

                request.state = InferenceState.DECODING

                results.append(
                    ExecutionResult(
                        request_id=request.request_id,
                        mode="prefill",
                        logits=logits,
                        cache_size=cache_size,
                    )
                )

            except Exception as exc:
                request.state = InferenceState.FAILED

                try:
                    self._clear_model_cache()
                except Exception:
                    pass

                if isinstance(
                    exc,
                    BatchExecutionError,
                ):
                    raise

                raise BatchExecutionError(
                    "Prefill failed for "
                    f"{request.request_id}: {exc}"
                ) from exc

        return results

    # ---------------------------------------------------------
    # Decode
    # ---------------------------------------------------------

    def execute_decode(
        self,
        batch,
        tokens_by_request,
    ):
        """
        Execute one incremental decode step.

        The model KV cache from prefill MUST remain alive.

        The supplied token is appended to the request only after
        the model has successfully processed it.

        Cache invariant:

            before decode:
                KV size == request.position

            after decode:
                KV size == request.position + 1
        """
        self._validate_batch(
            batch,
            "decode",
        )

        if not isinstance(
            tokens_by_request,
            dict,
        ):
            raise TypeError(
                "tokens_by_request must be dict"
            )

        if len(batch.items) > 1:
            raise BatchExecutionError(
                "Current model-level KV cache supports only "
                "one active request at a time."
            )

        results = []

        for item in batch.items:
            request = item.request

            if request.state not in (
                InferenceState.PREFILL,
                InferenceState.DECODING,
            ):
                raise BatchExecutionError(
                    "Decode request is not active: "
                    f"{request.request_id}"
                )

            if request.request_id not in tokens_by_request:
                raise BatchExecutionError(
                    "Missing decode token for "
                    f"{request.request_id}"
                )

            token = tokens_by_request[
                request.request_id
            ]

            if not isinstance(token, int):
                raise TypeError(
                    "decode token must be int"
                )

            expected_cache_size = request.position
            actual_cache_size = self._kv_cache_size()

            if actual_cache_size != expected_cache_size:
                raise BatchExecutionError(
                    "Decode KV-cache position mismatch for "
                    f"{request.request_id}: "
                    f"expected {expected_cache_size}, "
                    f"got {actual_cache_size}"
                )

            try:
                request.state = InferenceState.DECODING

                logits = self.model.forward(
                    [token],
                    use_cache=True,
                )

                if len(logits) != 1:
                    raise BatchExecutionError(
                        "Incremental decode must return "
                        "exactly one logit vector for "
                        f"{request.request_id}"
                    )

                request.append_token(token)

                cache_size = self._kv_cache_size()

                if cache_size != request.position:
                    raise BatchExecutionError(
                        "Decode KV-cache size mismatch for "
                        f"{request.request_id}: "
                        f"expected {request.position}, "
                        f"got {cache_size}"
                    )

                results.append(
                    ExecutionResult(
                        request_id=request.request_id,
                        mode="decode",
                        logits=logits,
                        cache_size=cache_size,
                    )
                )

            except Exception as exc:
                request.state = InferenceState.FAILED

                try:
                    self._clear_model_cache()
                except Exception:
                    pass

                if isinstance(
                    exc,
                    BatchExecutionError,
                ):
                    raise

                raise BatchExecutionError(
                    "Decode failed for "
                    f"{request.request_id}: {exc}"
                ) from exc

        return results

    # ---------------------------------------------------------
    # Request completion
    # ---------------------------------------------------------

    def finish_request(
        self,
        request,
        reason="length",
    ):
        """
        Mark a request as finished and clear its model cache.
        """
        if request.state not in (
            InferenceState.PREFILL,
            InferenceState.DECODING,
        ):
            raise BatchExecutionError(
                "Cannot finish inactive request: "
                f"{request.request_id}"
            )

        request.finish(reason)

        self._finish_request_cleanup(request)

    def fail_request(self, request):
        """
        Mark a request as failed and clear its model cache.
        """
        request.fail()
        self._clear_model_cache()

    # ---------------------------------------------------------
    # Single request helpers
    # ---------------------------------------------------------

    def prefill_request(self, request):
        from .batch_builder import BatchItem

        batch = InferenceBatch(
            batch_id="single-prefill",
            mode="prefill",
            items=[
                BatchItem(
                    request=request,
                    slot=0,
                )
            ],
        )

        return self.execute_prefill(batch)[0]

    def decode_request(
        self,
        request,
        token,
    ):
        from .batch_builder import BatchItem

        batch = InferenceBatch(
            batch_id="single-decode",
            mode="decode",
            items=[
                BatchItem(
                    request=request,
                    slot=0,
                )
            ],
        )

        return self.execute_decode(
            batch,
            {
                request.request_id: token
            },
        )[0]


__all__ = [
    "BatchExecutionError",
    "ExecutionResult",
    "ExecutorStep",
    "BatchExecutor",
]
