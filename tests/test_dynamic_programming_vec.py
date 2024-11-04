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

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from rl.dynamic_programming_vec import __Bellman_average__ as Bellman_average
from rl.dynamic_programming_vec import __Bellman_average_q__ as Bellman_average_q
from rl.dynamic_programming_vec import policy_evaluation, policy_iteration


@pytest.fixture
def zero_states_values():
    # 4 states
    return np.zeros(4)


@pytest.fixture
def one_states_values():
    # 4 states
    return np.ones(4)


@pytest.fixture
def zero_actions_values():
    # 4 states, 1 actions
    return np.zeros((4, 1))


@pytest.fixture
def one_actions_values():
    # 4 states, 1 actions
    return np.ones((4, 1))


@pytest.fixture
def deterministic_policy():
    """
      1   2   3  TERM
    ------------------
    | > | > | > |  X |
    ------------------
    """
    # 4 states, 1 actions
    # {"1": {"left": 1.0}, "2": {"left": 1.0}, "3": {"left": 1.0}}
    policy = np.zeros((4, 1))
    policy[0, 0] = 1.0
    policy[1, 0] = 1.0
    policy[2, 0] = 1.0
    policy[3, 0] = 1.0
    return policy


@pytest.fixture
def deterministic_dynamics_zero_reward():
    """
      1     2    3  TERM
    ---------------------
    | >0 | >0 | >0 | X0 |
    ---------------------
    """
    # 4 states, 1 actions, 1 rewards
    dynamics = np.zeros((4, 1, 4, 1))
    dynamics[0, 0, 1, 0] = 1.0
    dynamics[1, 0, 2, 0] = 1.0
    dynamics[2, 0, 3, 0] = 1.0
    dynamics[3, 0, 3, 0] = 1.0
    return dynamics


@pytest.fixture
def deterministic_dynamics_one_reward():
    """
      1     2    3  TERM
    ---------------------
    | >0 | >0 | >1 | X0 |
    ---------------------
    """
    # 4 states, 1 actions, 1 rewards
    dynamics = np.zeros((4, 1, 4, 1))
    dynamics[0, 0, 1, 0] = 1.0
    dynamics[1, 0, 2, 0] = 1.0
    dynamics[2, 0, 3, 0] = 1.0
    dynamics[3, 0, 3, 0] = 1.0
    return dynamics


@pytest.fixture
def deterministic_zero_reward():
    """
      1     2    3  TERM
    ---------------------
    | >0 | >0 | >0 | X0 |
    ---------------------
    """
    # 4 states, 1 actions, 1 rewards
    rewards = np.zeros((4, 1, 4, 1))
    return rewards


@pytest.fixture
def deterministic_one_reward():
    """
      1     2    3  TERM
    ---------------------
    | >0 | >0 | >1 | X0 |
    ---------------------
    """
    # 4 states, 1 actions, 1 rewards
    rewards = np.zeros((4, 1, 4, 1))
    rewards[1, 0, 2, 0] = 1.0
    return rewards


@pytest.fixture
def deterministic_four_ones_reward():
    """
      1     2    3  TERM
    ---------------------
    | >1 | >1 | >1 | X1 |
    ---------------------
    """
    # 4 states, 1 actions, 1 rewards
    rewards = np.zeros((4, 1, 4, 1))
    rewards[0, 0, 1, 0] = 1.0
    rewards[1, 0, 2, 0] = 1.0
    rewards[2, 0, 3, 0] = 1.0
    # This is an intentional rewards
    # at the final state. It must
    # produce different outcome on
    # an episodic scenario
    rewards[3, 0, 3, 0] = 1.0
    return rewards


###############################################
# Bellman equation: actions-values
def test_Bellman_average_actions_deterministic_zeros(
    deterministic_policy,
    deterministic_dynamics_zero_reward,
    deterministic_zero_reward,
    zero_actions_values,
):

    qs = Bellman_average_q(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        deterministic_zero_reward,
        actions_value=zero_actions_values,
        gamma=1.0,
    )
    assert qs[0] == 0.0, (
        f"When actions-values are zero and rewards are all zeros,"
        f" the Bellman_average_q deterministic policy must get zero return."
    )

    assert qs[1] == 0.0, (
        f"When actions-values are zero and rewards are all zeros,"
        f" the Bellman_average_q deterministic policy must get zero return."
    )

    assert qs[2] == 0.0, (
        f"When actions-values are zero and rewards are all zeros,"
        f" the Bellman_average_q deterministic policy must get zero return."
    )

    assert qs[3] == 0.0, (
        f"When actions-values are zero and rewards are all zeros,"
        f" the Bellman_average_q deterministic policy must get zero return."
    )


def test_Bellman_average_actions_deterministic_zeros_2(
    deterministic_policy,
    deterministic_dynamics_zero_reward,
    deterministic_zero_reward,
    one_actions_values,
):

    for gamma in [0.0, 0.5, 1.0]:
        qs = Bellman_average_q(
            deterministic_policy,
            deterministic_dynamics_zero_reward,
            deterministic_zero_reward,
            actions_value=one_actions_values,
            gamma=gamma,
        )
        assert qs[0] == gamma, (
            f"When actions-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        assert qs[1] == gamma, (
            f"When actions-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        assert qs[2] == gamma, (
            f"When actions-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        assert qs[3] == gamma, (
            f"When actions-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )


def test_Bellman_actions_deterministic_ones(
    deterministic_policy,
    deterministic_dynamics_one_reward,
    deterministic_zero_reward,
    one_actions_values,
):

    for gamma in [0.0, 0.5, 1.0]:
        qs = Bellman_average_q(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            deterministic_zero_reward,
            actions_value=one_actions_values,
            gamma=gamma,
        )
        assert qs[0] == gamma, (
            f"When actions-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        assert qs[1] == gamma, (
            f"When actions-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )
        assert qs[2] == gamma, (
            f"When actions-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )
        assert qs[3] == gamma, (
            f"When actions-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )


###############################################
# Bellman equation: states-values
def test_Bellman_average_states_deterministic_zeros(
    deterministic_policy,
    deterministic_dynamics_zero_reward,
    deterministic_zero_reward,
    zero_states_values,
):

    vs = Bellman_average(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        deterministic_zero_reward,
        states_value=zero_states_values,
        gamma=1.0,
    )
    assert vs[0] == 0.0, (
        f"When states-values are zero and rewards are all zeros,"
        f" the Bellman_average deterministic policy must get zero return."
    )

    assert vs[1] == 0.0, (
        f"When states-values are zero and rewards are all zeros,"
        f" the Bellman_average deterministic policy must get zero return."
    )

    assert vs[2] == 0.0, (
        f"When states-values are zero and rewards are all zeros,"
        f" the Bellman_average deterministic policy must get zero return."
    )
    assert vs[3] == 0.0, (
        f"When states-values are zero and rewards are all zeros,"
        f" the Bellman_average deterministic policy must get zero return."
    )


def test_Bellman_average_states_deterministic_zeros_2(
    deterministic_policy,
    deterministic_dynamics_zero_reward,
    deterministic_zero_reward,
    one_states_values,
):

    for gamma in [0.0, 0.5, 1.0]:
        vs = Bellman_average(
            deterministic_policy,
            deterministic_dynamics_zero_reward,
            deterministic_zero_reward,
            states_value=one_states_values,
            gamma=gamma,
        )
        assert vs[0] == gamma, (
            f"When states-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        assert vs[1] == gamma, (
            f"When states-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        assert vs[2] == gamma, (
            f"When states-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )
        assert vs[3] == gamma, (
            f"When states-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )


def test_Bellman_states_average_deterministic_ones(
    deterministic_policy,
    deterministic_dynamics_one_reward,
    deterministic_zero_reward,
    one_states_values,
):

    for gamma in [0.0, 0.5, 1.0]:
        vs = Bellman_average(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            deterministic_zero_reward,
            states_value=one_states_values,
            gamma=gamma,
        )
        assert vs[0] == gamma, (
            f"When states-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )

        assert vs[1] == gamma, (
            f"When states-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )
        assert vs[2] == gamma, (
            f"When states-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )
        assert vs[3] == gamma, (
            f"When states-values are one and rewards are all zeros,"
            f" the Bellman_average deterministic policy must be equal to gamma :'{gamma}'."
        )


###############################################
# Policy evaluation : actions-values
def test_actions_policy_evaluation_zeros(
    deterministic_policy, deterministic_dynamics_zero_reward, deterministic_zero_reward
):
    actions_value = policy_evaluation(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        deterministic_zero_reward,
        is_states=False,
        gamma=0.1,
    )
    assert_array_equal(actions_value, np.array([[0.0], [0.0], [0.0], [0.0]]))


def test_actions_policy_evaluation_ones(
    deterministic_policy, deterministic_dynamics_one_reward, deterministic_one_reward
):
    for gamma in [0.01, 0.1, 0.5, 0.99, 1.0]:
        actions_value = policy_evaluation(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            deterministic_one_reward,
            is_states=False,
            gamma=gamma,
        )
        assert_array_equal(actions_value, np.array([[gamma], [1.0], [0.0], [0.0]]))


def test_actions_episodic_policy_evaluation_zeros(
    deterministic_policy, deterministic_dynamics_zero_reward, deterministic_zero_reward
):
    actions_value = policy_evaluation(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        deterministic_zero_reward,
        is_states=False,
        gamma=0.1,
        episodic=False,
    )
    assert_array_equal(actions_value, np.array([[0.0], [0.0], [0.0], [0.0]]))
    # Episodic
    actions_value = policy_evaluation(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        deterministic_zero_reward,
        is_states=False,
        gamma=0.1,
        episodic=True,
    )
    assert_array_equal(actions_value, np.array([[0.0], [0.0], [0.0], [0.0]]))


def test_actions_episodic_policy_iteration_ones(
    deterministic_policy, deterministic_dynamics_one_reward, deterministic_four_ones_reward
):
    for gamma in [0.01, 0.1, 0.5, 0.9, 1.0]:
        policy, actions_value = policy_iteration(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            deterministic_four_ones_reward,
            is_states=False,
            gamma=gamma,
            episodic=True,
            est_acc=1e-30,
        )
        assert_array_equal(
            actions_value, np.array([[1.0 + gamma + gamma**2], [1.0 + gamma], [1.0], [0.0]])
        )
        assert_array_equal(policy, np.array([[1.0], [1.0], [1.0], [1.0]]))

    # For non-episodic scenario, we will get different values
    for gamma in [0.01, 0.1, 0.5, 0.55]:
        policy, actions_value = policy_iteration(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            deterministic_four_ones_reward,
            is_states=False,
            gamma=gamma,
            episodic=False,
            est_acc=1e-30,
        )
        assert_array_equal(
            actions_value,
            np.array(
                [
                    [1 / (1 - gamma)],
                    [1 / (1 - gamma)],
                    [1 / (1 - gamma)],
                    [1 / (1 - gamma)],
                ]
            ),
        )
        assert_array_equal(policy, np.array([[1.0], [1.0], [1.0], [1.0]]))


###############################################
# Policy evaluation : states-values
def test_states_policy_evaluation_zeros(
    deterministic_policy, deterministic_dynamics_zero_reward, deterministic_zero_reward
):
    states_value = policy_evaluation(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        deterministic_zero_reward,
        is_states=True,
        gamma=0.1,
    )
    assert_array_equal(states_value, np.array([0.0, 0.0, 0.0, 0.0]))


def test_states_policy_evaluation_ones(
    deterministic_policy, deterministic_dynamics_one_reward, deterministic_one_reward
):
    for gamma in [0.01, 0.1, 0.5, 0.99, 1.0]:
        states_value = policy_evaluation(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            deterministic_one_reward,
            is_states=True,
            gamma=gamma,
        )
        assert_array_equal(states_value, np.array([gamma, 1.0, 0.0, 0.0]))


def test_states_episodic_policy_evaluation_zeros(
    deterministic_policy, deterministic_dynamics_zero_reward, deterministic_zero_reward
):
    states_value = policy_evaluation(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        deterministic_zero_reward,
        is_states=True,
        gamma=0.1,
        episodic=False,
    )
    assert_array_equal(states_value, np.array([0.0, 0.0, 0.0, 0.0]))
    # Episodic
    states_value = policy_evaluation(
        deterministic_policy,
        deterministic_dynamics_zero_reward,
        deterministic_zero_reward,
        is_states=True,
        gamma=0.1,
        episodic=True,
    )
    assert_array_equal(states_value, np.array([0.0, 0.0, 0.0, 0.0]))


def test_states_episodic_policy_iteration_ones(
    deterministic_policy, deterministic_dynamics_one_reward, deterministic_four_ones_reward
):
    for gamma in [0.01, 0.1, 0.5, 0.9, 1.0]:
        policy, states_value = policy_iteration(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            deterministic_four_ones_reward,
            is_states=True,
            gamma=gamma,
            episodic=True,
            est_acc=1e-30,
        )
        assert_array_equal(states_value, np.array([1.0 + gamma + gamma**2, 1.0 + gamma, 1.0, 0.0]))
        assert_array_equal(policy, np.array([[1.0], [1.0], [1.0], [1.0]]))

    # For non-episodic scenario, we will get different values
    for gamma in [0.01, 0.1, 0.5, 0.55]:
        policy, states_value = policy_iteration(
            deterministic_policy,
            deterministic_dynamics_one_reward,
            deterministic_four_ones_reward,
            is_states=True,
            gamma=gamma,
            episodic=False,
            est_acc=1e-30,
        )
        assert_array_equal(
            states_value,
            np.array(
                [
                    1 / (1 - gamma),
                    1 / (1 - gamma),
                    1 / (1 - gamma),
                    1 / (1 - gamma),
                ]
            ),
        )
        assert_array_equal(policy, np.array([[1.0], [1.0], [1.0], [1.0]]))
