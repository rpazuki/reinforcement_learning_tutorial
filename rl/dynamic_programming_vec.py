"""
Vectorised version of policy and value iterations for actions and states values.

For
# states: n
# action: m
# rewards: k
the followings are the ndarrays and thier dimensions that we will use:

 - states_value: v(states)  ndarray(n x 1)
 - actions_value: q(states, action)  ndarray(n x m)
 - policy: pi(states, action)  ndarray(n x m)
 - dynamics: p(states, action, new_state, reward)  ndarray(n x m x n x k)
 - rewards: r(state, reward) ndarray(n x k)

 When the task is episodic, the last element of states_value and the last row
 of actions_value are assumed to correspond to the 'Termination' state and are
 kept as zero.
"""

import warnings

import numpy as np
from numpy import ndarray


def __return_expectation__(dynamics, rewards, states_value, gamma):
    """
    sum_{s', r} p(s',r| s,a) [r + gamma v(s')]
    """
    #  dot (.) is a element-wise multiplication

    #  sum_{s', r} p(s',r| s,a) [r + gamma v(s')]
    #  sum_{n1, k} (n x m x n1 x k) . [ (n x m x n1 x k) + (n1 x 1) ] = (n x m)
    return np.sum(dynamics * (rewards + gamma * states_value.reshape((1, 1, -1, 1))), axis=(2, 3))


def __Bellman_average__(
    policy: ndarray,
    dynamics: ndarray,
    rewards: ndarray,
    states_value: ndarray,
    gamma: float,
) -> float:
    """
    sum_{a in A(s)} pi(a|s) sum_{s', r} p(s',r| s,a) [r + gamma v(s')]
    """
    #  dot (.) is a element-wise multiplication

    #  sum_{s', r} p(s',r| s,a) [r + gamma v(s')]
    #  sum_{n1, k} (n x m x n1 x k) . [ (n x m x n1 x k) + (n1 x 1) ] = (n x m)
    states_update = __return_expectation__(dynamics, rewards, states_value, gamma)
    #  sum_{a in A(s)} pi(a|s) states_update
    #  sum_m (n x m) . (n x m) = (n)
    return np.sum(policy * states_update, axis=1)


def __Bellman_average_q__(
    policy: ndarray,
    dynamics: ndarray,
    rewards: ndarray,
    actions_value: ndarray,
    gamma: float,
) -> float:
    """
    sum_{s', r}  p(s',r|s,a) [r + gamma sum_{a' in A(s')} pi(a'|s') q(s', a')]
    """
    #  dot (.) is a element-wise multiplication

    # sum_{a' in A(s')} pi(a'|s') q(s', a')]
    #  sum_m (n x m) . (n x m) = (n)
    actions_update = np.sum(policy * actions_value, axis=1)[:, np.newaxis]
    # sum_{s', r}  p(s',r|s,a) [r + gamma actions_update]
    #  sum_{n1, k} (n x m x n1 x k) . [ (n x k) + (n) ] = (n x m)
    return np.sum(dynamics * (rewards + gamma * actions_update), axis=(2, 3))


def __zero_states_value__(policy: ndarray, episodic: bool) -> ndarray:
    # Create a zero states value with the same number of states in the policy
    # For episodic tasks, it assumes the last row of the policy is the
    # "Termination" state.
    states_value = np.zeros((policy.shape[0], 1))
    return states_value


def __zero_actions_value__(policy: ndarray, episodic: bool) -> ndarray:
    # Create a zero actions value with the same shape as
    # policy (row -> states, columns -> actions)
    # For episodic tasks, it assumes the last row of the policy is the
    # "Termination" state.
    actions_value = np.zeros_like(policy)
    return actions_value


def policy_evaluation(
    policy: ndarray,
    dynamics: ndarray,
    rewards: ndarray,
    actions_value: ndarray = None,
    states_value: ndarray = None,
    is_states=False,
    gamma: float = 0.9,
    episodic: bool = True,
    est_acc: float = 0.001,
    max_iteration: int = 100,
) -> ndarray:
    """
    Iterative policy evaluation for estimating V = v_{pi}.

    Given a policy, this function finds the state-value function.

    Parameters
    ----------
    policy: ndarray(n x m)
        The policy function, pi(a|s). It is a probability distribution for each action,
        given the state.
        It must be an ndarray(n x m) with n rows for states and m columns for actions.
        Deterministic polices have one and only one action for each state (n x 1).
        For episodic=True, the last row's elements are assumed as 'Termination'
        state.

    dynamics: ndarray(n x m x n x k)
        The environment's dynamic p(s', r | s, a). Or the probability of a new state and
        a reward, given the current state and action.
        It must be an ndarray(n x m x n x k) with axis=0 with n states, axis=1 with m actions,
        axis=2 with n next statesm, and axis=3 with k rewards.
        Deterministic dynamics have one and only one action and one rewards (n x 1 x n x 1).

    actions_value: ndarray(n x m), default=None
        The actions-value function, q(s, a).
        It must be an ndarray(n x m) with n rows for states and m columns for actions.
        When it is 'None', the method initialise it by using the policy.
        For episodic=True, the last row is assumed as 'Termination' and its elements
        remain zero.

    states_value: ndarray(n x 1), default=None
        The actions-value function, v(s). It must be an ndarray(n x 1) with n rows for states.
        When it is 'None', the method initialise it by using the policy.
        For episodic=True, the last element is assumed as 'Termination'
        state and remains zero.

    is_states: bool, default=False
        States_value or actions_value are provided.

    gamma: float, default=0.9
        The discount value. It must be in (0,1].

    episodic: bool, default=True
        Episodic tasks. If it is True, the last element of states_value or the last row of
        actions_value are assumed as 'Termination' state and remains zero.

    est_acc: float, default=0.001
        The accuracy of the estimation. The iteration will stop when the difference
        between the current estimates and the previous one is less than est_acc.

    max_iteration: int, default=100
        The maximum iteration before halting the iterartive algorithm. Raise a warning
        in case it halts the iteration.

    Returns
    -------
    values: ndarray
        The updated stats-value or actions-value.

    """
    assert gamma > 0.0, f"The discount_gamma='{gamma}' must be greater than zero."
    assert gamma <= 1.0, f"The discount_gamma='{gamma}' must be greater than or equal one."

    if is_states:
        # Initilise the None states_values
        if states_value is None:
            values = __zero_states_value__(policy, episodic)
        else:  # Used the provided states_values
            values = states_value
        # Bellman average for states_values
        Bellman_average = __Bellman_average__

        # Termination states for episodic tasks
        def termination_terms(mat):
            if episodic:
                mat[-1] = 0.0
            return mat

        #
        termination = termination_terms

    else:
        # Initilise the None actions_value
        if actions_value is None:
            values = __zero_actions_value__(policy, episodic)
        else:  # Used the provided actions_value
            values = actions_value
        # Bellman average for actions_value
        Bellman_average = __Bellman_average_q__

        # Termination states for episodic tasks
        def termination_terms(mat):
            if episodic:
                mat[-1, :] = 0.0
            return mat

        #
        termination = termination_terms
    ##############
    iteration = 0
    while iteration < max_iteration:  # Expected update loop
        #
        updated_values = Bellman_average(policy, dynamics, rewards, values, gamma)
        updated_values = termination(updated_values)
        max_delta = np.max(np.abs(updated_values - values))
        values = updated_values
        iteration += 1
        # Stops when the desired accuracy has reached
        if max_delta <= est_acc:
            return values
    # End of the while loop
    ###############
    warnings.warn(
        f"The policy_evaluation function halted after maximum '{max_iteration}' iteration"
    )
    return values


def policy_improvement(
    policy: ndarray,
    dynamics: ndarray,
    rewards: ndarray,
    actions_value: ndarray = None,
    states_value: ndarray = None,
    is_states=False,
    gamma: float = 0.9,
) -> ndarray:
    """
    Iterative policy improvement for pi(s) based on q(s,a) or v(s).

    Given the actions-value or states-values, this function finds
    the greedy improved policy.

    Parameters
    ----------
    policy: ndarray(n x m)
        The policy function, pi(a|s). It is a probability distribution for each action,
        given the state.
        It must be an ndarray(n x m) with n rows for states and m columns for actions.
        Deterministic polices have one and only one action for each state (n x 1).
        For episodic=True, the last row's elements are assumed as 'Termination'
        state.

    dynamics: ndarray(n x m x n x k)
        The environment's dynamic p(s', r | s, a). Or the probability of a new state and
        a reward, given the current state and action.
        It must be an ndarray(n x m x n x k) with axis=0 with n states, axis=1 with m actions,
        axis=2 with n next statesm, and axis=3 with k rewards.
        Deterministic dynamics have one and only one action and one rewards (n x 1 x n x 1).

    actions_value: ndarray(n x m), default=None
        The actions-value function, q(s, a).
        It must be an ndarray(n x m) with n rows for states and m columns for actions.

    states_value: ndarray(n x 1), default=None
        The actions-value function, v(s). It must be an ndarray(n x 1) with n rows for states.

    is_states: bool, default=False
        States_value or actions_value are provided.

    gamma: float, default=0.9
        The discount value. It must be in (0,1].

    Returns
    -------
    states_value: ndarray
        The updated stat-value function.
    is_stable: bool
        Is the new policy is the same as the old one.
    """
    if is_states:
        #  sum_{s', r} p(s',r| s,a) [r + gamma v(s')]
        actions_value = __return_expectation__(dynamics, rewards, states_value, gamma)
    # The greedy policy
    policy_not_normalised = actions_value == np.max(actions_value, axis=1)[:, np.newaxis]
    # Normalised greedy policy
    policy_new = policy_not_normalised / np.sum(policy_not_normalised, axis=1)[:, np.newaxis]
    # Check its stability (it is stable when it does not change by new updates)
    if np.all(np.isclose(policy, policy_new)):
        return policy_new, True
    else:
        return policy_new, False


def policy_iteration(
    policy: ndarray,
    dynamics: ndarray,
    rewards: ndarray,
    actions_value: ndarray = None,
    states_value: ndarray = None,
    is_states=False,
    gamma: float = 0.9,
    episodic: bool = True,
    est_acc: float = 0.1,
    l2_acc: float = 0.1,
    max_evaluation_iteration: int = 100,
    max_iteration: int = 100,
    verbose: bool = False,
) -> tuple[ndarray, ndarray]:
    """
    Iterative policy evaluation for estimating pi* and q* = q_{pi*} or v* = v_{pi}.

    Given a policy, this function finds the optimum policy and actions-value or
    states-values functions.

    Parameters
    ----------
    policy: ndarray(n x m)
        The policy function, pi(a|s). It is a probability distribution for each action,
        given the state.
        It must be an ndarray(n x m) with n rows for states and m columns for actions.
        Deterministic polices have one and only one action for each state (n x 1).
        For episodic=True, the last row's elements are assumed as 'Termination'
        state.

    dynamics: ndarray(n x m x n x k)
        The environment's dynamic p(s', r | s, a). Or the probability of a new state and
        a reward, given the current state and action.
        It must be an ndarray(n x m x n x k) with axis=0 with n states, axis=1 with m actions,
        axis=2 with n next statesm, and axis=3 with k rewards.
        Deterministic dynamics have one and only one action and one rewards (n x 1 x n x 1).

    actions_value: ndarray(n x m), default=None
        The initial values of estimating actions-value function, q(s, a).
        It must be an ndarray(n x m) with n rows for states and m columns for actions.
        When it is 'None', the method initialise it by using the policy.
        For episodic=True, the last row is assumed as 'Termination' and its elements
        remain zero.

    states_value: ndarray(n x 1), default=None
        The initial values of estimating actions-value function, v(s).
        The actions-value function, v(s). It must be an ndarray(n x 1) with n rows for states.
        When it is 'None', the method initialise it by using the policy.
        For episodic=True, the last element is assumed as 'Termination'
        state and remains zero.

    is_states: bool, default=False
        States_value or actions_value are provided.

    gamma: float, default=0.9
        The discount value. It must be in (0,1].

    episodic: bool, default=True
        Episodic tasks. If it is True, the last element of states_value or the last row of
        actions_value are assumed as 'Termination' state and remains zero.

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

    Returns
    -------
    policy: ndarray
        The converged, optimised policy.
    actions_value or states_value: ndarray
        The updated actions_value or states_value.

    """
    ###################
    # Iteration counter
    iteration = 0
    # The norm of updates store in this variable
    diff_L2 = np.inf
    while iteration < max_iteration:  # expected update loop
        # Policy Evaluation
        value_new = policy_evaluation(
            policy,
            dynamics,
            rewards,
            actions_value,
            states_value,
            is_states,
            gamma,
            episodic,
            est_acc,
            max_evaluation_iteration,
        )
        if is_states:
            # Save the norm of the update from the second iteration
            if states_value is not None:
                diff_L2 = np.linalg.norm(value_new - states_value)
            states_value = value_new
        else:
            # Save the norm of the update from the second iteration
            if actions_value is not None:
                diff_L2 = np.linalg.norm(value_new - actions_value)
            actions_value = value_new
        # Policy Improvement
        policy, is_stable = policy_improvement(
            policy, dynamics, rewards, actions_value, states_value, is_states, gamma
        )
        # If the policy is stable, stop the iteration
        if is_stable:
            if verbose:
                print(f"Convereged at iteration '{iteration}'.")
            return (policy, states_value if is_states else actions_value)
        #
        if diff_L2 <= l2_acc:
            if verbose:
                print(f"Convereged at iteration '{iteration}' for L2:{diff_L2}.")
            return (policy, states_value if is_states else actions_value)
        #
        iteration += 1
        # End of the while loop
        #######################

    warnings.warn(
        f"The policy_iteration function halted after maximum '{max_iteration}' iteration"
    )
    return (policy, states_value if is_states else actions_value)


def value_iteration(
    policy: ndarray,
    dynamics: ndarray,
    rewards: ndarray,
    states_value: ndarray = None,
    gamma: float = 0.9,
    episodic: bool = True,
    est_acc: float = 0.001,
    max_iteration: int = 1000,
    verbose: bool = False,
) -> tuple[ndarray, ndarray]:
    """
    Iterative policy evaluation for estimating pi* and q* = q_{pi*}.

    Given a policy, this function finds the optimum policy and action-value functions.

    Parameters
    ----------
    policy: ndarray(n x m)
        The policy function, pi(a|s). It is a probability distribution for each action,
        given the state.
        It must be an ndarray(n x m) with n rows for states and m columns for actions.
        Deterministic polices have one and only one action for each state (n x 1).
        For episodic=True, the last row's elements are assumed as 'Termination'
        state.

    dynamics: ndarray(n x m x n x k)
        The environment's dynamic p(s', r | s, a). Or the probability of a new state and
        a reward, given the current state and action.
        It must be an ndarray(n x m x n x k) with axis=0 with n states, axis=1 with m actions,
        axis=2 with n next statesm, and axis=3 with k rewards.
        Deterministic dynamics have one and only one action and one rewards (n x 1 x n x 1).

    states_value: ndarray(n x 1), default=None
        The initial values of estimating actions-value function, v(s).
        The actions-value function, v(s). It must be an ndarray(n x 1) with n rows for states.
        When it is 'None', the method initialise it by using the policy.
        For episodic=True, the last element is assumed as 'Termination'
        state and remains zero.

    gamma: float, default=0.9
        The discount value. It must be in (0,1].

    episodic: bool, default=True
        Episodic tasks. If it is True, the last element of states_value is assumed
        as 'Termination' state and remains zero.

    est_acc: float, default=0.1
        The accuracy of the estimation. The iteration will stop when the difference
        between the current estimates and the previous one is less than est_acc.

    max_iteration: int, default=1000
        The maximum iteration before halting the iterative algorithm of policy evaluation.
        Raise a warning in case it halts the iteration.

    Returns
    -------
    policy: ndarray
        The converged, optimised policy.
    actions_value: ndarray
        The updated actions-value function.

    """
    assert gamma > 0.0, f"The discount_gamma='{gamma}' must be greater than zero."
    assert gamma <= 1.0, f"The discount_gamma='{gamma}' must be greater than or equal one."

    states_value = __zero_states_value__(policy, episodic)
    #############################
    iteration = 0
    while iteration < max_iteration:  # expected update loop
        # sum_{s', r} p(s',r| s, a) [r + gamma v(s')]
        updated_returns = __return_expectation__(dynamics, rewards, states_value, gamma)
        # find the maximum expected return
        new_values = np.max(updated_returns, axis=1)
        # Store the changes
        max_delta = np.max(np.abs(new_values - states_value))
        #
        states_value = new_values
        #
        if max_delta <= est_acc:
            if verbose:
                print(f"Convereged at iteration '{iteration}'. Delta:{max_delta}")
            break
        iteration += 1
    # End of while
    #############################
    if iteration >= max_iteration:
        warnings.warn(
            f"The value_iteration function halted after maximum '{max_iteration}' iteration. Delta:{max_delta}"
        )
    policy, _ = policy_improvement(policy, dynamics, rewards, None, states_value, True, gamma)
    return policy, states_value
