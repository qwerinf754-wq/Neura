"""
Neura Stopping Criteria
"""


class StoppingCriteria:

    def __init__(
        self,
        eos_token=3,
        max_new_tokens=128
    ):

        self.eos_token = eos_token
        self.max_new_tokens = max_new_tokens

    def should_stop(
        self,
        generated_tokens
    ):

        if len(generated_tokens) >= self.max_new_tokens:
            return True

        if generated_tokens:

            if generated_tokens[-1] == self.eos_token:
                return True

        return False
