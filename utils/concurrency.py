from typing import Callable

from joblib import Parallel, delayed


def batch_processing(fn: Callable, data: list, **kwargs) -> list:
    return Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(fn)(data=i, **kwargs) for i in data)
