"""
    In the following tests, we will use the deterministic dynamic and policy
    are as follows
       1     2     3   TERM
    ------------------------
    | >r1 | >r2 | >r3 | X0 |
    ------------------------
    where, '>' means move left, X is the terminal state for an episodic
    task, and ri is the reward.

    The Bellman equations for states_values are written as
    V(1) = pi(>|1) P(2, r1| > , 1) (r1 + gamma V(2))
    V(2) = pi(>|2) P(3, r2| > , 2) (r2 + gamma V(3))
    V(3) = pi(>|3) P(T, r3| > , 3) (r3 + gamma V(T)) =
           pi(>|3) P(T, r3| > , 3) (r3)


    Example 1:
      1     2    3  TERM
    ---------------------
    | >0 | >0 | >0 | X0 |
    ---------------------
    The Bellmann equations rewrite as
    V(1) = (0 + gamma V(2))
    V(2) = (0 + gamma V(3))
    V(3) = 0
    So,
    V(1) = 0
    V(2) = 0
    V(3) = 0
    and the policy iteration must be converge to zero for any gamma.


    Example 1:
      1     2    3  TERM
    ---------------------
    | >0 | >0 | >1 | X0 |
    ---------------------
    The Bellmann equations rewrite as
    V(1) = (0 + gamma V(2))
    V(2) = (0 + gamma V(3))
    V(3) = 1
    So,
    V(1) = gamma**2
    V(2) = gamma
    V(3) = 1
"""

from collections.abc import Iterator, Mapping
from math import gamma
from typing import Any

import pytest
from arrow import get
from rl.dynamic_programming import __Bellman_average__ as Bellman_average
from rl.dynamic_programming import policy_evaluation


class Class_Policy(Mapping):

    def __init__(self) -> None:
        self.__states__ = ["1", "2", "3"]
        self.__actions__ = {"left": 1.0}

    def __contains__(self, key: object) -> bool:
        return key in self.__states__

    def __iter__(self) -> Iterator:
        return iter(self.__states__)

    def __len__(self) -> int:
        return len(self.__states__)

    def __getitem__(self, key: Any) -> Any:
        return self.__actions__

    def __repr__(self) -> str:
        return (
            "{\n"
            + "".join([f"'{k}':" + str(self.__actions__) + ",\n" for k in self.__states__])
            + "\n}"
        )


@pytest.fixture
def zero_state_values():
    return {"1": 0, "2": 0, "3": 0, "TERM": 0}


@pytest.fixture
def one_state_values():
    return {"1": 1, "2": 1, "3": 1, "TERM": 0}


@pytest.fixture
def deterministic_policy():
    """
      1   2   3  TERM
    ------------------
    | > | > | > |  X |
    ------------------

    (state, action): probability
    """
    return {"1": {"left": 1.0}, "2": {"left": 1.0}, "3": {"left": 1.0}}


@pytest.fixture
def deterministic_dynamics_zero_reward():
    """
      1     2    3  TERM
    ---------------------
    | >0 | >0 | >0 | X0 |
    ---------------------

    (state, action): {(next_state, reward):probability}
    """
    return {
        ("1", "left"): {("2", 0): 1.0},
        ("2", "left"): {("3", 0): 1.0},
        ("3", "left"): {("TERM", 0): 1.0},
    }


@pytest.fixture
def deterministic_dynamics_one_reward():
    """
      1     2    3  TERM
    ---------------------
    | >0 | >0 | >1 | X0 |
    ---------------------

    (state, action): {(next_state, reward):probability}
    """
    return {
        ("1", "left"): {("2", 0): 1.0},
        ("2", "left"): {("3", 0): 1.0},
        ("3", "left"): {("TERM", 1): 1.0},
    }


def test_Bellman_average_deterministic_zeros(
    deterministic_policy, deterministic_dynamics_zero_reward, zero_state_values
):

    v1 = Bellman_average(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        states_value=zero_state_values,
        gamma=1.0,
        current_state="1",
    )
    assert v1 == 0.0, (
        f"When state-values are zero and rewards are all zeros,"
        f" the Bellman_average deterministic policy must get zero return."
    )

    v2 = Bellman_average(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        states_value=zero_state_values,
        gamma=1.0,
        current_state="2",
    )
    assert v2 == 0.0, (
        f"When state-values are zero and rewards are all zeros,"
        f" the Bellman_average deterministic policy must get zero return."
    )

    v3 = Bellman_average(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        states_value=zero_state_values,
        gamma=1.0,
        current_state="3",
    )
    assert v3 == 0.0, (
        f"When state-values are zero and rewards are all zeros,"
        f" the Bellman_average deterministic policy must get zero return."
    )


def test_Bellman_average_deterministic_zeros_2(
    deterministic_policy, deterministic_dynamics_zero_reward, one_state_values
):

    for gamma in [0.0, 0.5, 1.0]:
        v1 = Bellman_average(
            deterministic_policy,
            deterministic_dynamics_zero_reward,
            states_value=one_state_values,
            gamma=gamma,
            current_state="1",
        )
        assert v1 == gamma, (
            f"When state-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        v2 = Bellman_average(
            deterministic_policy,
            deterministic_dynamics_zero_reward,
            states_value=one_state_values,
            gamma=gamma,
            current_state="2",
        )
        assert v2 == gamma, (
            f"When state-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        v3 = Bellman_average(
            deterministic_policy,
            deterministic_dynamics_zero_reward,
            states_value=one_state_values,
            gamma=gamma,
            current_state="3",
        )
        assert v3 == 0.0, (
            f"For zero reward and the action that ends in termination,"
            f" the Bellman_average deterministic policy must be equal to '0.0'."
        )


def test_Bellman_average_deterministic_ones(
    deterministic_policy, deterministic_dynamics_one_reward, one_state_values
):

    for gamma in [0.0, 0.5, 1.0]:
        v1 = Bellman_average(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            states_value=one_state_values,
            gamma=gamma,
            current_state="1",
        )
        assert v1 == gamma, (
            f"When state-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        v2 = Bellman_average(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            states_value=one_state_values,
            gamma=gamma,
            current_state="2",
        )
        assert v2 == gamma, (
            f"When state-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        v3 = Bellman_average(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            states_value=one_state_values,
            gamma=gamma,
            current_state="3",
        )
        assert v3 == 1.0, (
            f"For one reward and the action that ends in termination,"
            f" the Bellman_average deterministic policy must be equal to '1.0'."
        )


def test_policy_evaluation_zeros(deterministic_policy, deterministic_dynamics_zero_reward):
    states_value = policy_evaluation(
        deterministic_policy, deterministic_dynamics_zero_reward, gamma=0.1
    )
    assert states_value == {"1": 0.0, "2": 0.0, "3": 0.0, "TERM": 0.0}


def test_policy_evaluation_ones(deterministic_policy, deterministic_dynamics_one_reward):
    for gamma in [0.01, 0.1, 0.5, 0.99, 1.0]:
        states_value = policy_evaluation(
            deterministic_policy, deterministic_dynamics_one_reward, gamma=gamma
        )
        assert states_value == {"1": gamma**2, "2": gamma, "3": 1.0, "TERM": 0.0}


def test_class_policy_evaluation_ones(deterministic_dynamics_one_reward):
    policy = Class_Policy()
    for gamma in [0.01, 0.1, 0.5, 0.99, 1.0]:
        states_value = policy_evaluation(policy, deterministic_dynamics_one_reward, gamma=gamma)
        assert states_value == {"1": gamma**2, "2": gamma, "3": 1.0, "TERM": 0.0}
