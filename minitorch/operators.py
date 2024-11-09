"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    $f(x, y) = x * y$

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    $f(x) = x$

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    $f(x, y) = x + y$

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    $f(x) = -x$

    """
    return -x


def lt(x: float, y: float) -> float:
    """Less than.

    $f(x) =$ 1.0 if x less than y else 0.0

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Equal.

    $f(x) =$ 1.0 if x equals y else 0.0

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Max of two numbers.

    $f(x) =$ x if x is greater than y else y

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Close.

    $f(x) = |x - y| < 1e-2$

    Note:
        math.fabs() can take ints, floats and return floats.
        abs() can take ints, floats and complex numbers.

    """
    return 1.0 if math.fabs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    r"""Sigmoid function.

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Rectified linear unit.

    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    """Logarithm. Add a small constant EPS to avoid log(0).

    $f(x) = log(x)$
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Exponentiation.

    $f(x) = e^x$
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"""Compute the backward pass of the log function.

    If $f = log$ as above, compute $d \times f'(x)$
    """
    assert x > 0, "x must be positive"
    return d / x


def inv(x: float) -> float:
    """Invert a number.

    $f(x) = 1/x$"
    """
    assert x != 0, "x must be non-zero"
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"""Compute the backward pass of the inv function.

    If $f(x) = 1/x$ compute $d \times f'(x)$
    """
    assert x != 0, "x must be non-zero"
    return -d / x**2


def relu_back(x: float, d: float) -> float:
    r"""Compute the backward pass of the relu function.

    If $f(x) = relu(x)$ compute $d \times f'(x)$
    """
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
