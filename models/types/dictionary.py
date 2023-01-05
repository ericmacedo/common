import gzip
import json
from typing import Dict, Iterable

from sqlalchemy.types import LargeBinary, TypeDecorator

from ...utils.miscellaneous import are_instances


class Dictionary(TypeDecorator):
    cache_ok = True
    impl = LargeBinary

    def process_bind_param(self, value: Dict[str, int], dialec) -> bytes:
        if value is None:
            return None

        if not isinstance(value, dict):
            raise TypeError("Value must be a Dictionary[str, int]")
        if not are_instances(value.keys(), str):
            raise TypeError("Dict keys must be strings")
        if not are_instances(value.values(), int):
            raise TypeError("Dict values must be integers")

        return gzip.compress(json.dumps(value).encode("utf-8"))

    def process_result_value(self, value: bytes, dialec) -> Iterable[float]:
        if value is None:
            return None
        return json.loads(gzip.decompress(value))

    def copy(self, **kwargs):
        return self.__class__(self.impl.length)
