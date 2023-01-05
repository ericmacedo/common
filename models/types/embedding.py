from numbers import Number
from struct import pack, unpack
from typing import Iterable

from sqlalchemy.types import LargeBinary, TypeDecorator


class Embedding(TypeDecorator):
    cache_ok = True
    impl = LargeBinary

    def process_bind_param(self, value: Iterable[float], dialec) -> bytes:
        if value is None:
            return None

        if not isinstance(value, Iterable) or isinstance(value, str):
            raise TypeError("Value must be an iterable of floats")
        if any(not isinstance(i, Number) for i in value):
            raise TypeError("List items must be numbers (integer or float)")

        lst = [float(item) for item in value]
        return pack(f"{len(lst)}f", *lst)

    def process_result_value(self, value: bytes, dialec) -> Iterable[float]:
        if value is None:
            return None
        return [*unpack(f"{int(len(value)/4)}f", value)]

    def copy(self, **kwargs):
        return self.__class__(self.impl.length)
