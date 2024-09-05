"""
Simulation methods for examining Reinforcment Learning algorithms.
"""
from multiprocess import Pool


def parallel(
    func,
    simulations: int = 32,
    concurrent: int = 32,
    func_uniqe_index: bool = True,
    func_args: tuple = None,
):
    """
    Execute the func simulteniously on parrallel processes.

    Parameters
    ----------
    func : function
        A function that will be called simulteniously.
    simulations : int, default=32
        The number of total function calls.
    concurrent : int, default=32
        The number of concurrency of parallel processes.
    func_uniqe_index : bool, default=True
        If True, the iteration number will be send as the first
        argument to the function.
    func_args : tuple(obj), default=None
        The argument(s) on calling func.

    Returns
    -------
    list
        The list of returns of all func calls.
    """
    # Create the function calls' arguments as an iterator
    if func_uniqe_index:
        # Set the iteration number as first argument
        args_iter = ((i,) for i in range(simulations))
    else:
        # Otherwise, an empty argument is fine
        args_iter = (() for _i in range(simulations))
    # Append the func_args to the iterator's items
    if func_args is not None:
        args_iter = (items + func_args for items in args_iter)

    with Pool(concurrent) as pool:
        # call starmap to pass tuples as arguments
        ret = pool.starmap(func, args_iter)
    return ret
