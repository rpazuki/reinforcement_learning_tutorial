"""
Dynamic programin algorithms:

    1- Policy iteration.

    2- Value iteration.
"""

import copy
import warnings
from collections.abc import Hashable, Mapping

import numpy as np


def __return_expectation__(
    current_state: Hashable,
    gamma: float,
    action: Hashable,
    dynamics: dict,
    states_value: dict,
) -> float:
    """
    sum_{s', r} p(s',r| s, a) [r + gamma v(s')]
    """
    # Expanded list-comprehension
    # ret = 0
    # for (next_state, reward), p_dynamic in p[(current_state, action)].items():
    #     new_value = p_dynamic * (reward + gamma * v[next_state])
    #     ret += new_value

    # return ret
    # return np.sum(
    #     [
    #         p_dynamic * (reward + gamma * v[next_state])  # p(s',r| s,a) [r + \gamma v(s')]
    #         for (next_state, reward), p_dynamic in p[(current_state, action)].items()
    #     ]
    # )
    current_state_dynamics = dynamics[(current_state, action)]
    probs = current_state_dynamics["probs"]
    rewards = current_state_dynamics["rewards"]
    indices = current_state_dynamics["indices"]
    values = states_value["values"]

    return np.sum(probs * (rewards + gamma * values[indices]))  # p(s',r| s,a) [r + \gamma v(s')]


def __Bellman_average__(
    policy: Mapping[Hashable, Mapping[Hashable, float]],
    dynamics: dict,
    states_value: dict,
    gamma: float,
    current_state: Hashable,
) -> Mapping[Hashable, float]:
    """
    sum_{a in A(s)} pi(a|s) sum_{s', r} p(s',r| s,a) [r + gamma v(s')]
    """

    # Expanded list-comprehension
    # ret = 0
    # for action, p_policy in policy[current_state].items():
    #     new_value = p_policy * return_expectation(current_state, action)
    #     ret += new_value
    # return ret
    return np.sum(
        [
            p_policy * __return_expectation__(current_state, gamma, action, dynamics, states_value)
            for action, p_policy in policy[current_state].items()
        ]
    )


def __zero_states_value__(
    policy: Mapping[Hashable, Mapping[Hashable, float]],
    episodic: bool,
    term_key: Hashable = "TERM",
) -> dict:
    if episodic:
        states_value = {
            "states": {key: i for i, key in enumerate(policy)} | {term_key: len(policy)},
            "values": np.array([0.0 for _ in policy] + [0.0]),
        }
    else:
        states_value = {
            "states": {key: i for i, key in enumerate(policy)},
            "values": np.array([0.0 for _ in policy]),
        }
    return states_value


def policy_evaluation(
    policy: Mapping[Hashable, Mapping[Hashable, float]],
    dynamics: Mapping[tuple[Hashable, Hashable], dict],
    states_value: dict = None,
    gamma: float = 0.9,
    episodic: bool = True,
    est_acc: float = 0.001,
    max_iteration: int = 100,
    term_key: Hashable = "TERM",
) -> dict:
    """
    Iterative policy evaluation for estimating V = v_{pi}.

    Given a policy, this function finds the state-value function.

    Parameters
    ----------
    policy: Mapping[Hashable, Mapping[Hashable, float]]
        The policy function, pi(a|s). It is a probability distribution for each action,
        given the state.
        It must be a dictianry of dictionaries {state:{action:probability}}.
        Deterministic polices have one and only one action for each state.

    dynamics: Mapping[tuple[Hashable, Hashable], dict]
        The environment's dynamic p(s', r | s, a). It is a dictionary of dictionaries,
        such that its keys are tuples of (state, action) and its values are dictionaries
        with three elements
        {"indices":ndarray, "rewards":ndarray, "probs":ndarray}.
        The "indices" are the indices of states in state_values.

    states_value: dict, default=None
        The initial values of estimating stat-value function, v(s).
        When it is 'None', the method initialise it. For episodic=True
        inputs, the initialisation create the "TERM" state too.
        It must be a dictionary like {"states":{state:index}, "values":ndarray}.

    gamma: float, default=0.9
        The discount value. It must be in (0,1].

    episodic: bool, default=True
        Episodic tasks. If it is True, there MUST be one state that is called
        term_key.

    est_acc: float, default=0.001
        The accuracy of the estimation. The iteration will stop when the difference
        between the current estimates and the previous one is less than est_acc.

    max_iteration: int, default=100
        The maximum iteration before halting the iterartive algorithm. Raise a warning
        in case it halts the iteration.

    term_key: Hashable, default="TERM"
        For episodic tasks, term_key will be used as the terminal states keys in
        dictionaries.

    Returns
    -------
    states_value: dict
        The updated stat-value function.

    """
    assert gamma > 0.0, f"The discount_gamma='{gamma}' must be greater than zero."
    assert gamma <= 1.0, f"The discount_gamma='{gamma}' must be greater than or equal one."

    if states_value is None:
        states_value = __zero_states_value__(policy, episodic, term_key)

    if episodic and term_key not in states_value["states"]:
        raise ValueError(f"For episodic tasks, there must be a terminal state '{term_key}'.")

    iteration = 0
    while iteration < max_iteration:  # expected update loop
        # The variable for storing the maximum difference of changes per iteration
        max_delta = 0
        for state, i in states_value["states"].items():
            if state == term_key:
                continue
            value = states_value["values"][i]
            # Averaged state-value fro one step using Bellman eq.
            updated_value = __Bellman_average__(policy, dynamics, states_value, gamma, state)
            # Store the total maximum changes per iteration
            max_delta = max(max_delta, abs(updated_value - value))
            # In-place update
            states_value["values"][i] = updated_value
        iteration += 1

        if max_delta <= est_acc:
            return states_value
    warnings.warn(
        f"The policy_evaluation function halted after maximum '{max_iteration}' iteration"
    )
    return states_value


def policy_improvement(
    policy: Mapping[Hashable, Mapping[Hashable, float]],
    dynamics: Mapping[tuple[Hashable, Hashable], dict],
    states_value: dict,
    gamma: float = 0.9,
) -> tuple[Mapping[Hashable, float], bool]:
    """
    Iterative policy improvement for pi(s).

    Given a states-value function, this function finds the greedy improved policy.

    Parameters
    ----------
    policy: Mapping[Hashable, Mapping[Hashable, float]]
        The policy function, pi(a|s). It is a probability distribution for each action,
        given the state.
        It must be a dictianry of dictionaries {state:{action:probability}}.
        Deterministic polices have one and only one action for each state.

    dynamics: Mapping[tuple[Hashable, Hashable], dict]
        The environment's dynamic p(s', r | s, a). It is a dictionary of dictionaries,
        such that its keys are tuples of (state, action) and its values are dictionaries
        with three elements
        {"indices":ndarray, "rewards":ndarray, "probs":ndarray}.
        The "indices" are the indices of states in state_values.

    states_value: dict, default=None
        The initial values of estimating stat-value function, v(s).
        When it is 'None', the method initialise it. For episodic=True
        inputs, the initialisation create the "TERM" state too.
        It must be a dictionary like {"states":{state:index}, "values":ndarray}.

    gamma: float, default=0.9
        The discount value. It must be in (0,1].

    Returns
    -------
    states_value: Mapping[Hashable, float]
        The updated stat-value function.
    is_stable: bool
        Is the new policy the same as the old one.
    """
    assert gamma > 0.0, f"The discount_gamma='{gamma}' must be greater than zero."
    assert gamma <= 1.0, f"The discount_gamma='{gamma}' must be greater than or equal one."

    is_stable = True
    for current_state in policy.keys():
        actions = {action for state, action in dynamics.keys() if state == current_state}
        # Save the previous action with maximum probability for later comparision
        old_action, _ = max(policy[current_state].items(), key=lambda item: item[1])
        # Calculate the expected return, based on the given states_values
        updated_action_values = [
            (action, __return_expectation__(current_state, gamma, action, dynamics, states_value))
            for action in actions
        ]
        # find the action with maximum expeted return
        _, max_return = max(updated_action_values, key=lambda item: item[1])
        # collect all of them, in case there are ties
        all_new_actions = [action for action, r in updated_action_values if r == max_return]
        # update the policy with maximum return action(s)
        policy[current_state] = {action: 1.0 / len(all_new_actions) for action in all_new_actions}
        # Check to see if the policy has changed (unstable) or not (stable)
        if old_action not in all_new_actions:
            is_stable = False
    # if verbose:
    #    print(f"Stability: {is_stable!a:^5}")
    return policy, is_stable


def policy_iteration(
    policy: Mapping[Hashable, Mapping[Hashable, float]],
    dynamics: Mapping[tuple[Hashable, Hashable], dict],
    states_value: dict = None,
    gamma: float = 0.9,
    episodic: bool = True,
    est_acc: float = 0.1,
    l2_acc: float = 0.1,
    max_evaluation_iteration: int = 100,
    max_iteration: int = 100,
    term_key: Hashable = "TERM",
    verbose: bool = False,
) -> Mapping[Hashable, float]:
    """
    Iterative policy evaluation for estimating pi* and V* = v_{pi*}.

    Given a policy, this function finds the optimum policy and state-value functions.

    Parameters
    ----------
    policy: Mapping[Hashable, Mapping[Hashable, float]]
        The policy function, pi(a|s). It is a probability distribution for each action,
        given the state.
        It must be a dictianry of dictionaries {state:{action:probability}}.
        Deterministic polices have one and only one action for each state.

    dynamics: Mapping[tuple[Hashable, Hashable], dict]
        The environment's dynamic p(s', r | s, a). It is a dictionary of dictionaries,
        such that its keys are tuples of (state, action) and its values are dictionaries
        with three elements
        {"indices":ndarray, "rewards":ndarray, "probs":ndarray}.
        The "indices" are the indices of states in state_values.

    states_value: dict, default=None
        The initial values of estimating stat-value function, v(s).
        When it is 'None', the method initialise it. For episodic=True
        inputs, the initialisation create the "TERM" state too.
        It must be a dictionary like {"states":{state:index}, "values":ndarray}.

    gamma: float, default=0.9
        The discount value. It must be in (0,1].

    episodic: bool, default=True
        Episodic tasks. If it is True, there MUST be one state that is called
        term_key.

    est_acc: float, default=0.1
        The accuracy of the estimation. The iteration will stop when the difference
        between the current estimates and the previous one is less than est_acc.

    l2_acc: float, default=0.1
        When the norm-2 difference between two consecutive state_values in a
        policy evaluation is smaller than l2_acc, it stops the iteration and
        assumes a convergence. It is usually a sign that there are two or more
        equally optimal policy that the iteration switch between them.

    max_evaluation_iteration: int, default=100
        The maximum iteration for policy evaluation.
        Raise a warning in case it halts the iteration.

    max_iteration: int, default=100
        The maximum iteration before halting the iterative algorithm of policy evaluation.
        Raise a warning in case it halts the iteration.

    term_key: Hashable, default="TERM"
        For episodic tasks, term_key will be used as the terminal states keys in
        dictionaries.

    Returns
    -------
    policy: Mapping[str, Mapping[Hashable, float]]
        The converged, optimised policy.
    states_value: dict
        The updated stat-value function.

    """
    assert gamma > 0.0, f"The discount_gamma='{gamma}' must be greater than zero."
    assert gamma <= 1.0, f"The discount_gamma='{gamma}' must be greater than or equal one."

    if states_value is None:
        states_value = __zero_states_value__(policy, episodic, term_key)

    if episodic and term_key not in states_value["states"]:
        raise ValueError(f"For episodic tasks, there must be a terminal state '{term_key}'.")

    iteration = 0

    while iteration < max_iteration:  # expected update loop
        # Policy Evaluation
        states_value_new = policy_evaluation(
            policy,
            dynamics,
            copy.deepcopy(states_value),
            gamma,
            episodic,
            est_acc,
            max_evaluation_iteration,
            term_key,
        )
        values_L2 = np.linalg.norm(states_value["values"] - states_value_new["values"])
        states_value = states_value_new
        # Policy Improvement
        policy, is_stable = policy_improvement(policy, dynamics, states_value, gamma)
        # If the policy is stable, stop the iteration
        if is_stable:
            if verbose:
                print(f"Convereged at iteration '{iteration}'.")
            return policy, states_value
        #
        if values_L2 <= l2_acc:
            if verbose:
                print(f"State-values convereged at iteration '{iteration}' for L2:{values_L2}.")
            return policy, states_value
        #
        iteration += 1

    warnings.warn(
        f"The policy_iteration function halted after maximum '{max_iteration}' iteration"
    )
    return policy, states_value


def value_iteration(
    policy: Mapping[Hashable, Mapping[Hashable, float]],
    dynamics: Mapping[tuple[Hashable, Hashable], dict],
    states_value: dict = None,
    gamma: float = 0.9,
    episodic: bool = True,
    est_acc: float = 0.001,
    max_iteration: int = 1000,
    term_key: Hashable = "TERM",
    verbose: bool = False,
) -> tuple[Mapping[Hashable, Mapping[Hashable, float]], Mapping[Hashable, float]]:
    """
    Iterative value iteration for estimating  pi* and V* = v_{pi*}..

    Given a policy, this function finds the optimum policy and state-value functions.

    Parameters
    ----------
    policy: Mapping[Hashable, Mapping[Hashable, float]]
        The policy function, pi(a|s). It is a probability distribution for each action,
        given the state.
        It must be a dictianry of dictionaries {state:{action:probability}}.
        Deterministic polices have one and only one action for each state.

    dynamics: Mapping[tuple[Hashable, Hashable], dict]
        The environment's dynamic p(s', r | s, a). It is a dictionary of dictionaries,
        such that its keys are tuples of (state, action) and its values are dictionaries
        with three elements
        {"indices":ndarray, "rewards":ndarray, "probs":ndarray}.
        The "indices" are the indices of states in state_values.

    states_value: dict, default=None
        The initial values of estimating stat-value function, v(s).
        When it is 'None', the method initialise it. For episodic=True
        inputs, the initialisation create the "TERM" state too.
        It must be a dictionary like {"states":{state:index}, "values":ndarray}.

    gamma: float, default=0.9
        The discount value. It must be in (0,1].

    episodic: bool, default=True
        Episodic tasks. If it is True, there MUST be one state that is called
        "TERM".

    est_acc: float, default=0.001
        The accuracy of the estimation. The iteration will stop when the difference
        between the current estimates and the previous one is less than est_acc.

    max_iteration: int, default=1000
        The maximum iteration before halting the iterartive algorithm. Raise a warning
        in case it halts the iteration.

    term_key: Hashable, default="TERM"
        For episodic tasks, term_key will be used as the terminal states keys in
        dictionaries.

    Returns
    -------
    states_value: dict
        The updated stat-value function.

    """
    assert gamma > 0.0, f"The discount_gamma='{gamma}' must be greater than zero."
    assert gamma <= 1.0, f"The discount_gamma='{gamma}' must be greater than or equal one."

    if states_value is None:
        states_value = __zero_states_value__(policy, episodic, term_key)

    if episodic and term_key not in states_value["states"]:
        raise ValueError(f"For episodic tasks, there must be a terimanl state '{term_key}'.")

    iteration = 0
    while iteration < max_iteration:  # expected update loop
        # The diff
        max_delta = 0
        for current_state, i in states_value["states"].items():
            value = states_value["values"][i]

            if current_state == term_key:
                continue
            actions = policy[current_state]
            # Calculate the expected return, based on the given states_values
            updated_returns = [
                __return_expectation__(current_state, gamma, action, dynamics, states_value)
                for action in actions.keys()
            ]
            # find the maximum expeted return
            new_value = max(updated_returns)
            # Store the changes
            max_delta = max(max_delta, abs(new_value - value))
            # In-place update
            states_value["values"][i] = new_value

        if max_delta <= est_acc:
            if verbose:
                print(f"Convereged at iteration '{iteration}'. Delta:{max_delta}")
            break
        iteration += 1
    if iteration >= max_iteration:
        warnings.warn(
            f"The value_iteration function halted after maximum '{max_iteration}' iteration"
        )
    #
    for current_state in policy.keys():
        actions = policy[current_state]
        # Calculate the expected return, based on the given states_values
        updated_action_values = [
            (action, __return_expectation__(current_state, gamma, action, dynamics, states_value))
            for action in actions
        ]
        # find the action with maximum expeted return
        _, max_return = max(updated_action_values, key=lambda item: item[1])
        # collect all of them, in case there are ties
        all_new_actions = [action for action, r in updated_action_values if r == max_return]
        # update the policy with maximum return action(s)
        policy[current_state] = {action: 1.0 / len(all_new_actions) for action in all_new_actions}

    return policy, states_value
