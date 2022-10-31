from typing import Callable

from joblib import Parallel, delayed
from threading import Thread
from typing import Iterable

def batch_processing(fn: Callable, data: list, **kwargs) -> list:
    return Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(fn)(data=i, **kwargs) for i in data)

def threading(fn: Callable, wait: bool = False, args: Iterable = ()) -> None:
	t = Thread(target=fn, args=args)
	t.start()
	
	if wait:
		t.join()
