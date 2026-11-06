"""
Neura Batch Runtime

Step-based execution layer for multiple inference requests.

This is the bridge between the logical BatchExecutor and future
continuous/dynamic batching.

Current guarantees:
- one logical decode step per active request
- independent request state
- independent KV-cache snapshots
- prefill/decode lifecycle
- EOS / max-token completion
- failed-request isolation
- batch completion detection
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .request import RequestContext
from .state import InferenceState
from .scheduler import InferenceScheduler, SchedulerStateError
from .batch import BatchError, BatchStateError


@dataclass(slots=True)
class RuntimeStepResult:
    request_id: str
    token: int | None
    state: InferenceState
    finish_reason: str | None


class BatchRuntime:
    """
    Step-based runtime for a group of scheduled requests.

    The model remains the execution backend. Runtime state belongs
    to individual RequestContext objects and private runtime maps.
    """

    def __init__(self, model, scheduler: InferenceScheduler):
        if model is None:
            raise TypeError("model must not be None")

        if not isinstance(scheduler, InferenceScheduler):
            raise TypeError(
                "scheduler must be InferenceScheduler"
            )

        self.model = model
        self.scheduler = scheduler

        self._active = {}
        self._logits = {}
        self._cache_snapshots = {}

        self._started = False
        self._active_requests = {}
        self._completed = False

    # ---------------------------------------------------------
    # Runtime state
    # ---------------------------------------------------------

    @property
    def started(self):
        return self._started

    @property
    def completed(self):
        return self._completed

    @property
    def active_size(self):
        return len(self._active_requests)

    def active(self):
        return list(self._active.values())

    def get(self, request_id):
        return self._active.get(request_id)

    # ---------------------------------------------------------
    # Cache isolation
    # ---------------------------------------------------------

    def _capture_cache(self):
        cache = getattr(self.model, "kv_cache", None)

        if cache is None:
            return None

        layers = getattr(cache, "_layers", None)

        if layers is None:
            raise BatchError(
                "Model KV cache does not expose layer storage."
            )

        return deepcopy(layers)

    def _restore_cache(self, snapshot):
        cache = getattr(self.model, "kv_cache", None)

        if cache is None:
            return

        if snapshot is None:
            cache.clear()
            return

        cache._layers = deepcopy(snapshot)

    def _clear_model_cache(self):
        self.model.clear_cache()

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------

    def _sampler(self):
        from neura.generation.sampler import Sampler

        return Sampler(
            blocked_tokens=[
                0,
                1,
                2,
            ]
        )

    def _sample(self, request, logits):
        sampler = self._sampler()

        scores = list(logits)

        if request.repetition_penalty:
            scores = sampler.apply_repetition_penalty(
                scores,
                request.tokens,
                request.repetition_penalty,
            )

        return sampler.select(
            scores,
            len(scores),
            request.temperature,
            request.top_k,
        )

    # ---------------------------------------------------------
    # Start
    # ---------------------------------------------------------

    def start(self, requests=None):
        """
        Start runtime execution.

        If requests are omitted, CREATED requests are imported from
        RequestManager through the scheduler.
        """

        if self._started:
            raise BatchStateError(
                "BatchRuntime already started."
            )

        if requests is not None:
            requests = list(requests)

            if not requests:
                raise ValueError(
                    "runtime requires at least one request"
                )

            for request in requests:
                if not isinstance(request, RequestContext):
                    raise TypeError(
                        "all requests must be RequestContext"
                    )

                if (
                    self.scheduler.manager.get(
                        request.request_id
                    )
                    is not request
                ):
                    raise BatchStateError(
                        "Request must be registered with RequestManager."
                    )

                if request.state != InferenceState.CREATED:
                    raise BatchStateError(
                        f"Request is not CREATED: "
                        f"{request.request_id}"
                    )

                self.scheduler.enqueue(request)

        else:
            self.scheduler.enqueue_pending()

        self._started = True
        self._completed = False

        self._activate_next_requests()

        # Runtime ownership mirror.
        self._active_requests = dict(self._active)

        if not self._active:
            self._completed = True

        return self

    # ---------------------------------------------------------
    # Activation / prefill
    # ---------------------------------------------------------

    def _activate_next_requests(self):
        """
        Move all currently queued requests into runtime ownership.

        Current model architecture executes each request's prefill
        independently, while runtime state remains batch-oriented.
        """

        while self.scheduler.queued_size() > 0:
            request = self.scheduler.schedule_next()

            if request is None:
                break

            self._prefill(request)

    def _prefill(self, request):
        request.state = InferenceState.PREFILL

        # Runtime now owns this request.
        self._active[request.request_id] = request
        self._active_requests[request.request_id] = request

        # Restore only this request's cache.
        self._restore_cache(
            self._cache_snapshots.get(
                request.request_id
            )
        )

        try:
            logits = self.model.forward(
                list(request.prompt_tokens),
                use_cache=True,
            )

            if len(logits) != len(request.prompt_tokens):
                raise BatchError(
                    "Prefill returned invalid sequence length."
                )

            self._logits[
                request.request_id
            ] = logits[-1]

            self._cache_snapshots[
                request.request_id
            ] = self._capture_cache()

            request.state = InferenceState.DECODING

        except Exception:
            request.fail()

            try:
                self.scheduler.fail(
                    request,
                    RuntimeError(
                        "request prefill failed"
                    ),
                )
            except SchedulerStateError:
                pass

            self._logits.pop(
                request.request_id,
                None,
            )

            self._cache_snapshots.pop(
                request.request_id,
                None,
            )

        finally:
            self._clear_model_cache()

    # ---------------------------------------------------------
    # Decode one request
    # ---------------------------------------------------------

    def _decode_one(self, request):
        request_id = request.request_id

        if request.state != InferenceState.DECODING:
            raise BatchStateError(
                f"Request is not decoding: {request_id}"
            )

        if request.position >= (
            len(request.prompt_tokens)
            + request.max_tokens
        ):
            self._active.pop(
                request.request_id,
                None,
            )
            self._active_requests.pop(
                request.request_id,
                None,
            )

            request.finish("length")

            self._active.pop(
                request.request_id,
                None,
            )
            self._active_requests.pop(
                request.request_id,
                None,
            )

            self.scheduler.finish(
                request,
                "length",
            )

            return RuntimeStepResult(
                request_id=request_id,
                token=None,
                state=request.state,
                finish_reason=request.finish_reason,
            )

        snapshot = self._cache_snapshots.get(
            request_id
        )

        self._restore_cache(snapshot)

        try:
            logits = self._logits[request_id]

            token = self._sample(
                request,
                logits,
            )

            if (
                not isinstance(token, int)
                or token < 0
            ):
                raise BatchError(
                    f"Invalid generated token: {token}"
                )

            request.append_token(token)

            if token == request.eos_token:
                request.finish("eos")

                self.scheduler.finish(
                    request,
                    "eos",
                )

                return RuntimeStepResult(
                    request_id=request_id,
                    token=token,
                    state=request.state,
                    finish_reason=request.finish_reason,
                )

            # Incremental forward for exactly one token.
            next_logits = self.model.forward(
                [token],
                use_cache=True,
            )

            if len(next_logits) != 1:
                raise BatchError(
                    "Incremental forward must return one token."
                )

            self._logits[
                request_id
            ] = next_logits[-1]

            self._cache_snapshots[
                request_id
            ] = self._capture_cache()

            if len(request.generated_tokens) >= request.max_tokens:
                request.finish("length")

                self.scheduler.finish(
                    request,
                    "length",
                )

            return RuntimeStepResult(
                request_id=request_id,
                token=token,
                state=request.state,
                finish_reason=request.finish_reason,
            )

        except Exception as exc:
            request.fail()

            try:
                self.scheduler.fail(
                    request,
                    exc,
                )
            except SchedulerStateError:
                pass

            return RuntimeStepResult(
                request_id=request_id,
                token=None,
                state=InferenceState.FAILED,
                finish_reason=None,
            )

        finally:
            self._clear_model_cache()

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------

    def step(self):
        """
        Execute exactly one decode step for every active request.
        """

        if not self._started:
            raise BatchStateError(
                "BatchRuntime has not been started."
            )

        if self._completed:
            raise BatchStateError(
                "BatchRuntime is already completed."
            )

        results = []

        for request in list(
            self._active.values()
        ):
            result = self._decode_one(
                request
            )

            results.append(result)

            if request.state in (
                InferenceState.FINISHED,
                InferenceState.FAILED,
            ):
                self._active.pop(
                    request.request_id,
                    None,
                )

                self._logits.pop(
                    request.request_id,
                    None,
                )

                self._cache_snapshots.pop(
                    request.request_id,
                    None,
                )

        # Pull newly queued requests into the runtime.
        self._activate_next_requests()

        if not self._active:
            self._completed = True

        return results

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    def run(self, max_steps=None):
        """
        Run step-by-step until all requests finish.
        """

        if not self._started:
            self.start()

        results = []

        steps = 0

        while not self._completed:
            if (
                max_steps is not None
                and steps >= max_steps
            ):
                break

            results.extend(
                self.step()
            )

            steps += 1

        return results

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def clear(self):
        self._clear_model_cache()

        self._active.clear()
        self._active_requests.clear()
        self._logits.clear()
        self._cache_snapshots.clear()

        self._started = False
        self._completed = False


__all__ = [
    "BatchRuntime",
    "RuntimeStepResult",
]
