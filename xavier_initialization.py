import math
import numpy as np

# copied from stack overflow

scale = 1 / max(1., (2 + 2) / 2.)
limit = math.sqrt(3.0 * scale)


def init(shape):
    return np.random.uniform(-limit, limit, size=shape)