"""
Neura KV Cache
"""


class KVCache:

    def __init__(self):
        self.clear()

    def add(self, token, hidden):
        self.tokens.append(token)
        self.hidden_states.append(hidden)

    def token_count(self):
        return len(self.tokens)

    def last_token(self):
        if not self.tokens:
            return None
        return self.tokens[-1]

    def last_hidden(self):
        if not self.hidden_states:
            return None
        return self.hidden_states[-1]

    def clear(self):
        self.tokens = []
        self.hidden_states = []

    def state_dict(self):
        return {
            "tokens": list(self.tokens),
            "hidden": list(self.hidden_states),
        }

    def load_state_dict(self, state):
        self.tokens = list(state.get("tokens", []))
        self.hidden_states = list(state.get("hidden", []))
