"""
Neura BatchExecutor Request Isolation Contract

Verifies that:

1. Request A owns the model KV cache during its lifecycle.
2. Finishing Request A clears the KV cache completely.
3. Request B starts from an empty cache.
4. Request B's initial KV size equals only its own prompt length.
"""

from neura.core.seed import set_seed
from neura.model.parameters import PARAMETERS
from neura.model.causal_lm import CausalLM

from neura.inference.batch_builder import (
    InferenceBatch,
    BatchItem,
)
from neura.inference.batch_executor import BatchExecutor
from neura.inference.request import RequestContext
from neura.inference.state import InferenceState


def build_batch(request, mode):
    return InferenceBatch(
        batch_id=f"isolation-{request.request_id}-{mode}",
        mode=mode,
        items=[
            BatchItem(
                request=request,
                slot=0,
            )
        ],
    )


def main():
    print("===== BATCH EXECUTOR REQUEST ISOLATION =====")

    set_seed(42)

    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    model = CausalLM(
        vocab_size=32,
        hidden_size=16,
        heads=4,
        num_layers=2,
        intermediate_size=32,
        context_length=32,
    )

    executor = BatchExecutor(model)

    # ---------------------------------------------------------
    # Request A
    # ---------------------------------------------------------

    request_a = RequestContext(
        request_id="request-A",
        prompt_tokens=[3, 7, 2, 9],
        max_tokens=1,
        eos_token=3,
    )

    prompt_a_length = len(request_a.prompt_tokens)

    result_a = executor.execute_prefill(
        build_batch(request_a, "prefill")
    )[0]

    print("Request A after prefill:")
    print("  State :", request_a.state.value)
    print("  KV    :", result_a.cache_size)
    print("  Pos   :", request_a.position)

    assert request_a.state == InferenceState.DECODING
    assert result_a.cache_size == prompt_a_length
    assert model.kv_cache_size() == prompt_a_length

    result_a = executor.execute_decode(
        build_batch(request_a, "decode"),
        {
            request_a.request_id: 5,
        },
    )[0]

    print("Request A after decode:")
    print("  State :", request_a.state.value)
    print("  KV    :", result_a.cache_size)
    print("  Pos   :", request_a.position)

    assert result_a.cache_size == prompt_a_length + 1
    assert model.kv_cache_size() == prompt_a_length + 1

    # ---------------------------------------------------------
    # Finish Request A
    # ---------------------------------------------------------

    executor.finish_request(
        request_a,
        reason="length",
    )

    print("Request A after finish:")
    print("  State :", request_a.state.value)
    print("  KV    :", model.kv_cache_size())

    assert request_a.state == InferenceState.FINISHED
    assert model.kv_cache_size() == 0

    # Critical isolation assertion:
    # Request B must not inherit Request A's KV.
    assert model.kv_cache_size() != prompt_a_length + 1

    # ---------------------------------------------------------
    # Request B
    # ---------------------------------------------------------

    request_b = RequestContext(
        request_id="request-B",
        prompt_tokens=[11, 4],
        max_tokens=1,
        eos_token=3,
    )

    prompt_b_length = len(request_b.prompt_tokens)

    assert model.kv_cache_size() == 0

    result_b = executor.execute_prefill(
        build_batch(request_b, "prefill")
    )[0]

    print("Request B after prefill:")
    print("  State :", request_b.state.value)
    print("  KV    :", result_b.cache_size)
    print("  Pos   :", request_b.position)

    assert request_b.state == InferenceState.DECODING

    # The critical assertion:
    # B starts with exactly its own prompt length.
    assert result_b.cache_size == prompt_b_length
    assert model.kv_cache_size() == prompt_b_length
    assert request_b.position == prompt_b_length

    # B must not contain A's tokens.
    assert request_b.tokens == request_b.prompt_tokens
    assert request_b.tokens != request_a.tokens

    # ---------------------------------------------------------
    # Finish Request B
    # ---------------------------------------------------------

    executor.finish_request(
        request_b,
        reason="length",
    )

    print("Request B after finish:")
    print("  State :", request_b.state.value)
    print("  KV    :", model.kv_cache_size())

    assert request_b.state == InferenceState.FINISHED
    assert model.kv_cache_size() == 0

    print()
    print("PASS")


if __name__ == "__main__":
    main()
