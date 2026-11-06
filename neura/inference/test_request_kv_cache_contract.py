from neura.inference.request_kv_cache import (
    RequestKVCache,
)


print("=" * 60)
print("NEURA REQUEST-SCOPED KV CACHE CONTRACT")
print("=" * 60)


cache = RequestKVCache()


# ---------------------------------------------------------
# Initial state
# ---------------------------------------------------------

assert cache.num_requests() == 0
assert cache.request_ids() == []

print("Initial cache state: PASS")


# ---------------------------------------------------------
# Request creation
# ---------------------------------------------------------

a = cache.create("req-a")
b = cache.create("req-b")

assert a is not b
assert cache.num_requests() == 2

print("Request cache creation: PASS")


# ---------------------------------------------------------
# Request isolation
# ---------------------------------------------------------

cache.update(
    "req-a",
    ["ka0", "ka1"],
    ["va0", "va1"],
    layer_id=0,
)

cache.update(
    "req-b",
    ["kb0"],
    ["vb0"],
    layer_id=0,
)

assert cache.size("req-a", 0) == 2
assert cache.size("req-b", 0) == 1

assert cache.total_size("req-a") == 2
assert cache.total_size("req-b") == 1

print("Request isolation: PASS")


# ---------------------------------------------------------
# Layer isolation
# ---------------------------------------------------------

cache.update(
    "req-a",
    ["ka2", "ka3", "ka4"],
    ["va2", "va3", "va4"],
    layer_id=1,
)

assert cache.size("req-a", 0) == 2
assert cache.size("req-a", 1) == 3
assert cache.total_size("req-a") == 5

print("Per-layer isolation: PASS")


# ---------------------------------------------------------
# Layer IDs
# ---------------------------------------------------------

assert cache.layer_ids("req-a") == [0, 1]
assert cache.layer_ids("req-b") == [0]

print("Layer accounting: PASS")


# ---------------------------------------------------------
# Independent growth
# ---------------------------------------------------------

cache.update(
    "req-b",
    ["kb1", "kb2"],
    ["vb1", "vb2"],
    layer_id=2,
)

assert cache.total_size("req-a") == 5
assert cache.total_size("req-b") == 3

print("Independent request growth: PASS")


# ---------------------------------------------------------
# Missing request
# ---------------------------------------------------------

assert cache.get("missing") is None
assert cache.size("missing") == 0
assert cache.total_size("missing") == 0
assert cache.layer_ids("missing") == []

print("Missing request handling: PASS")


# ---------------------------------------------------------
# Duplicate creation
# ---------------------------------------------------------

try:
    cache.create("req-a")
    raise AssertionError(
        "Duplicate request cache accepted."
    )
except ValueError:
    pass

print("Duplicate request protection: PASS")


# ---------------------------------------------------------
# Request-specific clear
# ---------------------------------------------------------

cache.clear("req-a")

assert cache.total_size("req-a") == 0
assert cache.total_size("req-b") == 3

assert cache.has("req-a")
assert cache.has("req-b")

print("Per-request clear isolation: PASS")


# ---------------------------------------------------------
# Request removal
# ---------------------------------------------------------

removed = cache.remove("req-b")

assert removed is b
assert cache.get("req-b") is None
assert not cache.has("req-b")

assert cache.has("req-a")

print("Request removal isolation: PASS")


# ---------------------------------------------------------
# Recreate request after removal
# ---------------------------------------------------------

new_b = cache.create("req-b")

assert new_b is not b
assert cache.total_size("req-b") == 0

print("Request recreation: PASS")


# ---------------------------------------------------------
# Global accounting
# ---------------------------------------------------------

cache.update(
    "req-a",
    ["x"],
    ["x"],
    layer_id=0,
)

cache.update(
    "req-b",
    ["y", "z"],
    ["y", "z"],
    layer_id=0,
)

assert cache.total_size_all() == 3

print("Global KV accounting: PASS")


# ---------------------------------------------------------
# Global clear
# ---------------------------------------------------------

cache.clear()

assert cache.num_requests() == 0
assert cache.request_ids() == []
assert cache.total_size_all() == 0

print("Global clear: PASS")


# ---------------------------------------------------------
# Final state
# ---------------------------------------------------------

print("=" * 60)
print("RESULT: REQUEST-SCOPED KV CACHE CONTRACT PASSED")
print("=" * 60)
