"""
Neura Request-Scoped KV Cache
==============================

Provides an isolated KV-cache namespace for every inference request.

Architecture:

    Request
       |
       +--> KV[request_id]
                |
                +--> layer 0
                +--> layer 1
                +--> layer N

The underlying layer-aware KVCache remains unchanged.
"""

from .kv_cache import KVCache


class RequestKVCacheError(RuntimeError):
    """Base error for request-scoped KV cache."""


class RequestKVCache:

    def __init__(self):
        self._requests = {}

    # ---------------------------------------------------------
    # Internal
    # ---------------------------------------------------------

    def _get_cache(self, request_id):
        if request_id not in self._requests:
            self._requests[request_id] = KVCache()

        return self._requests[request_id]

    # ---------------------------------------------------------
    # Request lifecycle
    # ---------------------------------------------------------

    def create(self, request_id):
        if not request_id:
            raise ValueError(
                "request_id cannot be empty"
            )

        if request_id in self._requests:
            raise ValueError(
                f"Request cache already exists: {request_id}"
            )

        self._requests[request_id] = KVCache()

        return self._requests[request_id]

    def get(self, request_id):
        return self._requests.get(request_id)

    def has(self, request_id):
        return request_id in self._requests

    def remove(self, request_id):
        cache = self._requests.pop(
            request_id,
            None,
        )

        if cache is not None:
            cache.clear()

        return cache

    def clear(self, request_id=None):
        if request_id is None:
            for cache in self._requests.values():
                cache.clear()

            self._requests.clear()
            return

        cache = self._requests.get(request_id)

        if cache is not None:
            cache.clear()

    # ---------------------------------------------------------
    # KV operations
    # ---------------------------------------------------------

    def update(
        self,
        request_id,
        keys,
        values,
        layer_id=0,
    ):
        cache = self._get_cache(request_id)

        cache.update(
            keys,
            values,
            layer_id=layer_id,
        )

    def get_layer(
        self,
        request_id,
        layer_id=0,
    ):
        cache = self._requests.get(request_id)

        if cache is None:
            return None

        return cache.get(layer_id)

    def layer_ids(self, request_id):
        cache = self._requests.get(request_id)

        if cache is None:
            return []

        return cache.layer_ids()

    # ---------------------------------------------------------
    # Accounting
    # ---------------------------------------------------------

    def size(
        self,
        request_id,
        layer_id=0,
    ):
        cache = self._requests.get(request_id)

        if cache is None:
            return 0

        return cache.size(layer_id)

    def total_size(self, request_id):
        cache = self._requests.get(request_id)

        if cache is None:
            return 0

        return cache.total_size()

    def request_ids(self):
        return list(self._requests.keys())

    def num_requests(self):
        return len(self._requests)

    def total_size_all(self):
        return sum(
            cache.total_size()
            for cache in self._requests.values()
        )


__all__ = [
    "RequestKVCache",
    "RequestKVCacheError",
]
