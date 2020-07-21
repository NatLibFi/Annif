"""Unit tests for parallel processing functionality in Annif"""

import multiprocessing
import multiprocessing.dummy
import annif.parallel


def test_get_pool_0():
    n_jobs, pool_class = annif.parallel.get_pool(0)
    assert n_jobs is None
    assert pool_class is multiprocessing.Pool


def test_get_pool_1():
    n_jobs, pool_class = annif.parallel.get_pool(1)
    assert n_jobs == 1
    assert pool_class is multiprocessing.dummy.Pool


def test_get_pool_2():
    n_jobs, pool_class = annif.parallel.get_pool(2)
    assert n_jobs == 2
    assert pool_class is multiprocessing.Pool
