from itertools import islice, tee
from typing import Any, Generator, Iterable, List
from numpy.random import RandomState


class SubscriptableGenerator:
    def __init__(self, it: Iterable[Any]):
        self.__iterable = iter(it)

        self.__len = sum(1 for _ in self.__copy())

    def __iter__(self) -> Generator:
        for i in self.__iterable:
            yield i

    def __getitem__(self, indexer: int | slice) -> Any | Iterable[Any]:
        if isinstance(indexer, int):
            if indexer >= self.__len:
                raise IndexError(f"Generator index {indexer} is out of range")

            index = (self.__len + indexer) % self.__len
            return next(islice(self.__copy(), index, self.__len))
        elif isinstance(indexer, slice):
            start, stop = None, None
            if indexer.start:
                start = (self.__len + indexer.start) % self.__len
            if indexer.stop:
                stop = (self.__len + indexer.stop) % self.__len

            return iter(
                it for index, it in enumerate(self.__copy())
                if start and start <= index or stop and stop >= index)
        else:
            raise KeyError(f"Key '{indexer}' is not a valid indexer.")

    def __len__(self) -> int:
        return self.__len

    def __copy(self) -> Generator:
        self.__iterable, it = tee(self.__iterable)
        return it

    def to_list(self) -> List[Any]:
        return [*self.__copy()]


def chunks(lst: Iterable[Any], n: int) -> Generator:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def sample(it: Iterable[Any], size: int | float, random_state: int = None) -> Iterable[Any]:
    if type(size) not in [int, float]:
        raise TypeError("Parameter 'size' must be float or int")

    it, it2 = tee(it)

    it_len = sum(1 for _ in it2)

    indexes = RandomState(random_state).choice(
        it_len,
        size if type(size) == int else int(it_len * size),
        replace=False)

    return SubscriptableGenerator(
        item for index, item in enumerate(it) if index in indexes)
