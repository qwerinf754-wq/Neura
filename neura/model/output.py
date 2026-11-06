"""
Neura Model Output
"""


class ModelOutput:

    def __init__(
        self,
        token_id,
        scores
    ):

        self.token_id = token_id
        self.scores = scores


    def to_dict(self):

        return {
            "token_id": self.token_id,
            "vocab_size": len(self.scores)
        }
