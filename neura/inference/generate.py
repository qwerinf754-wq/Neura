from .engine import NeuraEngine


ENGINE = NeuraEngine()


def generate(text):

    return ENGINE.generate(text)
