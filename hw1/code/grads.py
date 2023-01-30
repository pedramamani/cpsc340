import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 1
    λ = 4
    for x_i in x:
        result += x_i ** λ
    return result


def foo_grad(x):
    return 4 * x ** 3


def bar(x):
    return np.prod(x)


def bar_grad(x):
    grad = np.zeros_like(x)
    for i, _ in enumerate(grad):
        grad[i] = np.prod(x[0:i]) * np.prod(x[i+1:])
    return grad
