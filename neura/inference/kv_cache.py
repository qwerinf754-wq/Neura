"""
Neura KV Cache

Layer-aware Key/Value cache for autoregressive generation.

Each transformer layer owns an independent K/V sequence.
"""

class KVCache:

    def __init__(self):
        self._layers = {}

    def _get_layer(self, layer_id):
        if layer_id not in self._layers:
            self._layers[layer_id] = {
                "keys": [],
                "values": [],
            }

        return self._layers[layer_id]

    def update(
        self,
        keys,
        values,
        layer_id=0,
    ):
        if len(keys) != len(values):
            raise ValueError(
                "keys and values must have "
                "the same sequence length"
            )

        layer = self._get_layer(layer_id)

        layer["keys"].extend(keys)
        layer["values"].extend(values)

    def get(self, layer_id=0):

        layer = self._get_layer(layer_id)

        return (
            layer["keys"],
            layer["values"],
        )

    def clear(self, layer_id=None):

        if layer_id is None:
            self._layers.clear()
            return

        self._layers.pop(
            layer_id,
            None,
        )

    def size(self, layer_id=0):

        layer = self._get_layer(layer_id)

        return len(
            layer["keys"]
        )

    def layer_ids(self):

        return list(
            self._layers.keys()
        )

    def num_layers(self):

        return len(
            self._layers
        )

    def total_size(self):

        return sum(
            len(layer["keys"])
            for layer in self._layers.values()
        )
