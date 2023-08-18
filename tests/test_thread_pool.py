import time
from concurrent.futures import ThreadPoolExecutor
import random


def test_thread_pool_always_returns_same_order():

    pool = ThreadPoolExecutor(4)

    def func(x):
        # ensure that threads all finish at different times
        time.sleep(random.uniform(0, 0.01))
        return x

    items = list(range(10))

    for i in range(4):
        assert list(pool.map(func, items)) == items
