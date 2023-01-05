from typing import Callable

from joblib import Parallel, cpu_count, delayed
from threading import Thread
from typing import Iterable


def batch_processing(fn: Callable, data: list, **kwargs) -> list:
    n_jobs = kwargs.get("n_jobs", cpu_count() - 1)
    return Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(fn)(data=i, **kwargs) for i in data)


def threading(fn: Callable, wait: bool = False, args: Iterable = ()) -> None:
    t = Thread(target=fn, args=args)
    t.start()

    if wait:
        t.join()
