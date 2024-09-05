"""
K-Arms bandit algorithms.
"""

import numpy as np


def k_arms_reward_creator(k: int = 10, mean: float = 0.0, std: float = 1.0):
    """
    Create a normal random reward model for k actions.

    q_i*, pr K actions averages, are selected from a normal
    distribution N(mu, sigma^2). Each i arm will be sampled
    like a normal distribution N(q_i*, 1) for query.

    Parameters
    ----------
    k : int, default=10
        number of arms or actions.
    mean : float, default=0.0
        The mean value of random q_i*.
    std : float, default=1.0
        The standard deviation value of random q_i*.

    Returns
    -------
    q_stars : ndarray
        One-D array of k q_i*.
    reward : func
        A function with signuture reward_i that samples from N(q_i*, 1).
    """
    q_stars = np.random.normal(loc=mean, scale=std, size=k)

    def reward(i):
        assert i <= k, f"The maximum arm number is '{k}'. The requested number is '{i}'."
        return np.random.normal(loc=q_stars[i], scale=1)

    return q_stars, reward


def greedy_ee(
    k,
    bandits,
    epsilon: float = 0.1,
    max_try: int = 10000,
    Qs_init: float = 0.0,
    record_stat: bool = True,
):
    """
    A greedy exploration/explotation algorithm.

    Parameters
    ----------
    bandits :
        A k-arms bandits that returns rewards for actions in {0, 1, ..., k-1}.
    epsilon : float, default=.1
        Probability of exploration.
    max_try : int, default=10000
        Maximum number of try.
    Qs_init : int, default=0.0
        Initial value of Q estimates. For optimistic initialisation,
        use positive values.
    record_stat : bool, default=True
        Records the statistics and then returns it as stat dictionary content (see Returns).

    Returns
    -------
    Qs : ndarray
        A one-D array of size 'k' that contains the estimated expected action-value.
    stats : dict
        A dictionary of recorded statistics. It will be empty when 'record_stat' is False,
        returns empty. The followings are the keys of the recorded stats:

            'rewards' : ndarray
                A one-D array of size 'max_try' that contains each action's reward in order.
            'Ns' : ndarray
                A one-D array of size 'k' that contains the number of time
                each action were selected.
            'actions' : ndarray
                A one-D array of size 'max_try' that contains each actions in order.
    """
    # Estimated action-value expectation
    Qs = np.zeros(k)
    # Set the intial Qs
    Qs[:] = Qs_init
    # Action's execution numbers
    Ns = np.zeros(k)
    iteration_num = 0
    stats = {}
    if record_stat:
        stats["rewards"] = np.zeros(max_try)
        stats["actions"] = np.zeros(max_try)
        stats["Ns"] = Ns

    while iteration_num < max_try:
        # Sample a from uniform distribution
        p = np.random.rand()
        if p <= epsilon:  # Exploration
            # Randomly select one action
            next_action = np.random.choice(k)
        else:  # Explotation
            # Select the action with maximum expected reward
            next_action = np.argmax(Qs)
        # Query the next reward
        reward = bandits(next_action)
        # Update the stats
        Ns[next_action] += 1
        Qs[next_action] += (reward - Qs[next_action]) / Ns[next_action]
        if record_stat:
            stats["rewards"][iteration_num] = reward
            stats["actions"][iteration_num] = next_action
        #
        iteration_num += 1

    return Qs, stats


def greedy_ucb(
    k,
    bandits,
    c: float = 2.0,
    max_try: int = 10000,
    Qs_init: float = 0.0,
    record_stat: bool = True,
):
    """
    An upper-confidence-bound exploration/explotation algorithm.

    Parameters
    ----------
    bandits :
        A k-arms bandits that returns rewards for actions in {0, 1, ..., k-1}.
    c : float, default=2.0
        A positive value for controling exploration degree.
    max_try : int, default=10000
        Maximum number of try.
    Qs_init : int, default=0.0
        Initial value of Q estimates. For optimistic initialisation,
        use positive values.
    record_stat : bool, default=True
        Records the statistics and then returns it as stat dictionary content (see Returns).

    Returns
    -------
    Qs : ndarray
        A one-D array of size 'k' that contains the estimated expected action-value.
    stats : dict
        A dictionary of recorded statistics. It will be empty when 'record_stat' is False,
        returns empty. The followings are the keys of the recorded stats:

            'rewards' : ndarray
                A one-D array of size 'max_try' that contains each action's reward in order.
            'Ns' : ndarray
                A one-D array of size 'k' that contains the number of time
                each action were selected.
            'actions' : ndarray
                A one-D array of size 'max_try' that contains each actions in order.
    """
    # Estimated action-value expectation
    Qs = np.zeros(k)
    # Set the intial Qs
    Qs[:] = Qs_init
    # Action's execution numbers
    Ns = np.zeros(k)
    iteration_num = 0
    stats = {}
    if record_stat:
        stats["rewards"] = np.zeros(max_try)
        stats["actions"] = np.zeros(max_try)
        stats["Ns"] = Ns

    while iteration_num < max_try:
        # Sample a from uniform distribution
        p = np.random.rand()
        # Exploration and Explotation
        t = iteration_num
        # Estimate Qs based on iteration number and previous selections
        Qs_UCB = Qs + c * np.sqrt(np.abs(np.log(t + 1e-20) / (Ns + 1e-20)))
        next_action = np.argmax(Qs_UCB)
        # Query the next reward
        reward = bandits(next_action)
        # Update the stats
        Ns[next_action] += 1
        Qs[next_action] += (reward - Qs[next_action]) / Ns[next_action]
        if record_stat:
            stats["rewards"][iteration_num] = reward
            stats["actions"][iteration_num] = next_action
        #
        iteration_num += 1

    return Qs, stats


def simulations_decorators(func):
    """
    A decorator that stacks the returns of all parallel simulations.

    It assumes the 'func' rturns (Qs, stats) format that bandit algorithms follow.
    """

    def unbox_returns(*args, **kwargs):
        # Call the function. The result will be contains
        res = func(*args, **kwargs)
        # Unbox Qs and stats as two different tuples
        Qs, stats = zip(*res)
        # Qs can be directly be turned to 2d ndarray
        Qs = np.array(Qs)
        #
        unboxed_stats = {}
        # If stats have already be recorded, it unboxes them too
        if len(stats) > 0:
            # Extract the keys from the first dict item
            keys = [key for key in stats[0].keys()]
            # For each key, turns the corresponding statistics to
            # a 2d ndarray, and store it with the same key
            unboxed_stats = {key: np.array([item[key] for item in stats]) for key in keys}
        return Qs, unboxed_stats

    return unbox_returns
