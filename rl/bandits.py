"""
K-Arms bandit algorithms.
"""

import numpy as np

from .helpers import _soft_max_ as soft_max


def k_arms_reward_creator(
    k: int = 10,
    mean: float = 0.0,
    std: float = 1.0,
    non_stationary: bool = False,
    non_stationary_std: float = 0.01,
):
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
    non_stationary : bool, default=False
        If True, the bandit probability distribution will be non-stationary.
        The non-stationary distributions have a drift parameter that randomly
        selected from a Gaussian N(0, non_stationary_std). And on calling the
        bandit, it accepts a 'time' as the second parameter that shifts the
        mean value of the distribution based on the drift.
    non_stationary_std : float, default=0.01
        The standard deviation of drifts for non-stationary bandits.

    Returns
    -------
    q_stars : ndarray
        One-D array of k q_i*.
    reward : func
        A function with signuture reward_i that samples from N(q_i*, 1).
    """
    q_stars = np.random.normal(loc=mean, scale=std, size=k)
    drifts = np.random.normal(loc=0.0, scale=non_stationary_std, size=k)

    def reward(i):
        assert i <= k, f"The maximum arm number is '{k}'. The requested number is '{i}'."
        return np.random.normal(loc=q_stars[i], scale=1)

    def reward_non_stationary(i, t):
        assert i <= k, f"The maximum arm number is '{k}'. The requested number is '{i}'."
        return np.random.normal(loc=t * drifts[i] + q_stars[i], scale=1)

    if non_stationary:
        return q_stars, reward_non_stationary
    else:
        return q_stars, reward


def greedy_exploration(
    k: int,
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
    k : int
        Number of arms.
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
        # Update the action-value estimate
        Ns[next_action] += 1
        Qs[next_action] += (reward - Qs[next_action]) / Ns[next_action]
        # Update the stats
        if record_stat:
            stats["rewards"][iteration_num] = reward
            stats["actions"][iteration_num] = next_action
        #
        iteration_num += 1

    return Qs, stats


def greedy_upper_confidence_bound(
    k: int,
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
    k : int
        Number of arms.
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
        # Exploration and Explotation
        t = iteration_num
        # Estimate Qs based on iteration number and previous selections
        Qs_UCB = Qs + c * np.sqrt(np.abs(np.log(t + 1e-20) / (Ns + 1e-20)))
        next_action = np.argmax(Qs_UCB)
        # Query the next reward
        reward = bandits(next_action)
        # Update the action-value estimate
        Ns[next_action] += 1
        Qs[next_action] += (reward - Qs[next_action]) / Ns[next_action]
        # Update the stats
        if record_stat:
            stats["rewards"][iteration_num] = reward
            stats["actions"][iteration_num] = next_action
        #
        iteration_num += 1

    return Qs, stats


def greedy_unbiased_constant_step_size(
    k: int,
    bandits,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    max_try: int = 10000,
    Qs_init: float = 0.0,
    record_stat: bool = True,
):
    """
    An unbiased constant step exploration/explotation algorithm.

    Since sample averaged action-value's estimates are poorly performs
    on nonstationary problems, while do not produce initil bias, in this
    algorithm we update the step size by using a trace of rewards during
    the run.

    Parameters
    ----------
    k : int
        Number of arms.
    bandits :
        A k-arms bandits that returns rewards for actions in {0, 1, ..., k-1}.
    epsilon : float, default=.1
        Probability of exploration.
    alpha : float, default=0.1
        A positive value constant step size.
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
    iteration_num = 0
    reward_trace = 0
    beta = alpha
    stats = {}
    if record_stat:
        stats["rewards"] = np.zeros(max_try)
        stats["actions"] = np.zeros(max_try)
        stats["Ns"] = np.zeros(k)

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
        # Update the action-value estimate
        Qs[next_action] += beta * (reward - Qs[next_action])
        # Update the reward's trace
        reward_trace += alpha * (1 - reward_trace)
        # Update the step size
        beta = alpha / reward_trace
        # Update the stats
        if record_stat:
            stats["Ns"][next_action] += 1
            stats["rewards"][iteration_num] = reward
            stats["actions"][iteration_num] = next_action
        #
        iteration_num += 1

    return Qs, stats


def gradient_bandit(
    k: int,
    bandits,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    max_try: int = 10000,
    record_stat: bool = True,
):
    """
    The gradient bandit (Energy) algorithm.

    Parameters
    ----------
    k : int
        Number of arms.
    bandits :
        A k-arms bandits that returns rewards for actions in {0, 1, ..., k-1}.
    epsilon : float, default=.1
        Probability of exploration.
    alpha : float, default=0.1
        A positive value constant step size.
    max_try : int, default=10000
        Maximum number of try.
    record_stat : bool, default=True
        Records the statistics and then returns it as stat dictionary content (see Returns).

    Returns
    -------
    Hs : ndarray
        A one-D array of size 'k' that contains the energy of action-value.
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
    Hs = np.zeros(k)
    # Action's execution numbers
    iteration_num = 0
    #
    average_reward = 0.0
    stats = {}
    if record_stat:
        stats["rewards"] = np.zeros(max_try)
        stats["actions"] = np.zeros(max_try)
        stats["Ns"] = np.zeros(k)

    while iteration_num < max_try:
        # The probabilites of Hs by soft-max
        Ps = soft_max(Hs)
        # Sample a from uniform distribution
        if np.random.rand() <= epsilon:  # Exploration
            # Randomly select one action
            next_action = np.random.choice(k)
        else:  # Explotation
            # Select the action with maximum energy
            next_action = np.argmax(Ps)
        # Query the next reward
        reward = bandits(next_action)
        # Update the average reward
        n = iteration_num + 1
        average_reward = ((n - 1) * average_reward + reward) / n
        # Update the energies of action-value
        for i in range(k):
            if i == next_action:
                Hs[i] += alpha * (reward - average_reward) * (1 - Ps[i])
            else:
                Hs[i] += -alpha * (reward - average_reward) * (Ps[i])
        # Update the stats
        if record_stat:
            stats["Ns"][next_action] += 1
            stats["rewards"][iteration_num] = reward
            stats["actions"][iteration_num] = next_action
        #
        iteration_num += 1

    return Hs, stats


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
