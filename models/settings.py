from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List

from sqlalchemy import JSON, Column, String, select

from ..database import MapperRegistry
from ..database.connector import DriverDB
from ..mixins.db import MixinORM


@MapperRegistry.mapped
@dataclass
class Settings(MixinORM):
    __tablename__ = "database_settings"
    __sa_dataclass_metadata_key__ = "sa"

    key: str = field(metadata={"sa": Column(String, primary_key=True)})
    value: Any = field(metadata={"sa": Column(JSON, nullable=False)})

    FIELDS: ClassVar[List[str]] = ["key", "value"]

    DEFAULTS: ClassVar[Dict[str, Any]] = {
        "last_document_processed": 0,
    }

    @classmethod
    def reset_defaults(cls, **kwargs):
        with DriverDB.session_scope(**kwargs) as session:
            items = session.execute(select(cls)).all()
            if items:
                for item in items:
                    item.value = cls.DEFAULTS[item.key]
            else:
                session.add_all([
                    cls(key=key, value=value)
                    for key, value in cls.DEFAULTS.items()])

    @classmethod
    def get(cls, key: str, **kwargs):
        if key not in cls.DEFAULTS.keys():
            raise ValueError(f"Key {key} is not a valid setting key")

        with DriverDB.session_scope(**kwargs) as session:
            return session.get(cls, key)

    @classmethod
    def set(cls, key: str, value: Any, **kwargs):
        if key not in cls.DEFAULTS.keys():
            raise ValueError(f"Key {key} is not a valid setting key")

        with DriverDB.session_scope(**kwargs) as session:
            item = session.get(cls, key)
            if not item:
                session.add(cls(key=key, value=value))
            else:
                item.value = value


# Base.metadata.create_all(Engine)
