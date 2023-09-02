from __future__ import annotations

import hashlib
from abc import abstractmethod
from typing import Any, Dict, Iterable

from sqlalchemy import update

from common.database.connector import DriverDB


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

    @classmethod
    def save_all(cls, data: Iterable[MixinORM], **kwargs):
        with DriverDB.session_scope(**kwargs) as s:
            s.add_all(data)

    def save(self, **kwargs) -> None:
        with DriverDB.session_scope(**kwargs) as s:
            s.add(self)

    def update(self, new_values: Dict, **kwargs):
        with DriverDB.session_scope(**kwargs) as s:
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

    @classmethod
    def hash(cls, data: str) -> str:
        return hashlib.md5(data.encode("utf-8")).hexdigest()


class MultitonMixin:
    _instances = {}

    def __new__(cls, DB_NAME=None):
        if DB_NAME and DB_NAME not in cls._instances:
            cls._instances[DB_NAME] = super().__new__(cls)
        return cls._instances[DB_NAME]
