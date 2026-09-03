"""Five ways a value can be bound, to find where testmon's tracking stops.

Nothing here is about CUDA, GPUs or this project. That is the point: the blind spot
measured here is a property of coverage-based
selection, and it reproduces in thirty lines of ordinary Python.
"""

# 1. plain module-level constant
SIZE = 10

# 2. derived at import time from another constant
LIMIT = 5
DERIVED = LIMIT * 10

# 3. a mutable structure
NAMES = ["a", "b"]

# 4. used as a default argument, evaluated once at def time
FACTOR = 3


def scale(value, factor=FACTOR):
    return value * factor


# 5. read inside a function body, evaluated at call time
def limit_plus(n):
    return LIMIT + n
