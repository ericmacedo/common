from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict
from typing import Any, Dict, Iterable, Tuple

from sqlalchemy import select, update

from ..helpers.orm import session_scope


class MixinORM:
    @classmethod
    def __class_getitem__(cls, indexer: str):
        if indexer not in cls.FIELDS:
            raise ValueError(f"Indexer {indexer} is not a valid column")
        return getattr(cls, indexer)

    def asdict(self) -> Dict:
        return asdict(self)

    def save(self) -> None:
        with session_scope() as s:
            s.add(self)

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
