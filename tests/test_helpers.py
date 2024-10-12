import numpy as np
import pytest
from rl.helpers import _soft_max_ as soft_max


def test_soft_max_single():
    assert soft_max(np.array([0])) == 1.0
    assert soft_max(np.array([1])) == 1.0


def test_soft_max_two():
    e = np.exp(1)
    is_close = lambda x, y: np.all(np.isclose(soft_max(x), y))

    assert is_close(np.array([1, 1]), np.array([0.5, 0.5]))
    assert is_close(np.array([1, 0]), np.array([e / (e + 1), 1 / (e + 1)]))


def test_soft_max_three():
    e = np.exp(1)
    is_close = lambda x, y: np.all(np.isclose(soft_max(x), y))

    assert is_close(np.array([1, 1, 1]), np.array([1 / 3, 1 / 3, 1 / 3]))
    assert is_close(np.array([1, 0, 0]), np.array([e / (e + 2), 1 / (e + 2), 1 / (e + 2)]))


def test_soft_max_large_input_number():
    e = np.exp(1)
    is_close = lambda x, y: np.all(np.isclose(soft_max(x), y))

    assert is_close(np.array([1e200, 1e200]), np.array([0.5, 0.5]))
