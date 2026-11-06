"""
Neura BatchExecutor KV Lifecycle Contract

Verifies:

    prefill
        KV == prompt length

    decode #1
        KV == prompt length + 1

    decode #2
        KV == prompt length + 2

    finish
        KV == 0
"""

from neura.core.seed import set_seed
from neura.model.parameters import PARAMETERS
from neura.model.causal_lm import CausalLM

from neura.inference.batch_builder import (
    InferenceBatch,
    BatchItem,
)
from neura.inference.batch_executor import (
    BatchExecutor,
)
from neura.inference.request import RequestContext
from neura.inference.state import InferenceState


def build_batch(request, mode):
    return InferenceBatch(
        batch_id=f"kv-{mode}",
        mode=mode,
        items=[
            BatchItem(
                request=request,
                slot=0,
            )
        ],
    )


def main():
    print("===== BATCH EXECUTOR KV LIFECYCLE =====")

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

    request = RequestContext(
        request_id="kv-lifecycle-1",
        prompt_tokens=[3, 7, 2, 9],
        max_tokens=2,
        eos_token=3,
    )

    prompt_length = len(request.prompt_tokens)

    # ---------------------------------------------------------
    # Prefill
    # ---------------------------------------------------------

    result = executor.execute_prefill(
        build_batch(request, "prefill")
    )[0]

    print("After prefill:")
    print("  State :", request.state.value)
    print("  KV    :", result.cache_size)
    print("  Pos   :", request.position)

    assert request.state == InferenceState.DECODING
    assert result.cache_size == prompt_length
    assert model.kv_cache_size() == prompt_length
    assert request.position == prompt_length

    # ---------------------------------------------------------
    # Decode #1
    # ---------------------------------------------------------

    result = executor.execute_decode(
        build_batch(request, "decode"),
        {
            request.request_id: 5,
        },
    )[0]

    print("After decode #1:")
    print("  State :", request.state.value)
    print("  KV    :", result.cache_size)
    print("  Pos   :", request.position)

    assert request.state == InferenceState.DECODING
    assert result.cache_size == prompt_length + 1
    assert model.kv_cache_size() == prompt_length + 1
    assert request.position == prompt_length + 1
    assert request.generated_tokens == [5]

    # ---------------------------------------------------------
    # Decode #2
    # ---------------------------------------------------------

    result = executor.execute_decode(
        build_batch(request, "decode"),
        {
            request.request_id: 6,
        },
    )[0]

    print("After decode #2:")
    print("  State :", request.state.value)
    print("  KV    :", result.cache_size)
    print("  Pos   :", request.position)

    assert request.state == InferenceState.DECODING
    assert result.cache_size == prompt_length + 2
    assert model.kv_cache_size() == prompt_length + 2
    assert request.position == prompt_length + 2
    assert request.generated_tokens == [5, 6]

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------

    executor.finish_request(
        request,
        reason="length",
    )

    print("After finish:")
    print("  State :", request.state.value)
    print("  KV    :", model.kv_cache_size())
    print("  Pos   :", request.position)
    print("  Reason:", request.finish_reason)

    assert request.state == InferenceState.FINISHED
    assert request.finish_reason == "length"
    assert model.kv_cache_size() == 0

    print()
    print("PASS")


if __name__ == "__main__":
    main()
