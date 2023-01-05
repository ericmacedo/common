from __future__ import annotations
from abc import abstractmethod

from typing import Any, Dict, Iterable

from sqlalchemy import update
from sqlalchemy.exc import DatabaseError, OperationalError, ResourceClosedError

from ..helpers.orm import session_scope
import time


class MixinORM:
    # required in order to access columns with server defaults
    # or SQL expression defaults, subsequent to a flush, without
    # triggering an expired load
    __mapper_args__ = {"eager_defaults": True}

    @classmethod
    def __class_getitem__(cls, indexer: str):
        if indexer not in cls.FIELDS:
            raise ValueError(f"Indexer {indexer} is not a valid column")
        return getattr(cls, indexer)

    @abstractmethod
    def as_dict(self) -> Dict:
        pass

    def save(self) -> None:
        with session_scope() as s:
            s.add(self)

    @classmethod
    def save_all(cls, data: Iterable[MixinORM]):
        with session_scope() as s:
            s.add_all(data)

    def update(self, new_values: Dict):
        with session_scope() as s:
            s.execute(
                update(self.__class__)
                .where(self.__class__.id == self.id)
                .values(**new_values))

    def diff(self, other: MixinORM) -> Dict[str, Any]:
        return {
            key: other[key] for key in self.FIELDS
            if key != "id" and other[key] != self[key]}

    def __getitem__(self, index: str) -> Any:
        if isinstance(index, str) and index in self.FIELDS:
            return getattr(self, index)
        return None
