import pytest

from rl.simulations import parallel

def no_arg_func():
    return "A"

def add(a,b):
    return a+b

def index(i):
    return i

def add_index(i, b):
    return i + b

def list_return():
    return [1]*5

def test_parallel_no_args():
    simulations=8
    rets = parallel(no_arg_func,
                    simulations=simulations,
                    concurrent=4,
                    func_uniqe_index=False,
                    func_args=None)

    assert rets == ["A"]*simulations

def test_parallel_fix_args():
    simulations=8
    rets = parallel(add,
                    simulations=simulations,
                    concurrent=4,
                    func_uniqe_index=False,
                    func_args=(1,2))

    assert rets == [3]*simulations

def test_parallel_index_args():
    simulations=8
    rets = parallel(index,
                    simulations=simulations,
                    concurrent=4,
                    func_uniqe_index=True,
                    func_args=None)

    assert rets == [i for i in range(simulations)]

def test_parallel_index_and_args():
    simulations=8
    fix_num = 20
    rets = parallel(add_index,
                    simulations=simulations,
                    concurrent=4,
                    func_uniqe_index=True,
                    func_args=(fix_num, ))

    assert rets == [fix_num + i for i in range(simulations)]

def test_parallel_no_args_list_return():
    simulations=8
    rets = parallel(list_return,
                    simulations=simulations,
                    concurrent=4,
                    func_uniqe_index=False,
                    func_args=None)

    assert rets == [[1]*5]*simulations


