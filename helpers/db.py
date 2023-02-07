from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections import namedtuple
from contextlib import contextmanager
from os import getenv
from typing import Any, Dict, Iterable

from sqlalchemy import create_engine, delete, desc, func, select, update, and_
from sqlalchemy.orm import close_all_sessions, registry, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy_utils import create_database, database_exists

from ..utils.itertools import SubscriptableGenerator

MapperRegistry = registry()
Base = MapperRegistry.generate_base()


class DriverDB:
    def __init__(self, **kwargs):
        def get_val(key, fallback):
            return kwargs.get(key, getenv(key, fallback))

        self.__user = get_val("DB_USER", "postgres")
        self.__password = get_val("DB_PASWORD", "123456")
        self.__host = get_val("DB_HOST", "localhost")
        self.__port = get_val("DB_PORT", "5432")
        self.__db = get_val("DB_NAME", "postgres")

        if not database_exists(self.engine.url):
            create_database(self.engine.url)

    @property
    def db_uri(self) -> str:
        return "postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}".format(
            user=self.__user,
            password=self.__password,
            host=self.__host,
            port=self.__port,
            db=self.__db)

    @property
    def engine(self):
        if getattr(self, "__engine", None) is None:
            self.__engine = create_engine(self.db_uri, poolclass=NullPool)
        return self.__engine

    @property
    def session(self):
        if getattr(self, "__session", None) is None:
            self.__session = sessionmaker(bind=self.engine,
                                          expire_on_commit=False,
                                          autoflush=True)
        return self.__session

    def create_all(self):
        Base.metadata.create_all(bind=self.engine, checkfirst=True)

    @classmethod
    @contextmanager
    def session_scope(cls, **kwargs):
        db_driver = cls(**kwargs)
        session = db_driver.session()
        try:
            yield session
            session.commit()
            session.expunge_all()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            del db_driver, session


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


class DB:
    def __init__(self, orm_model: MixinORM, index: Iterable[int] = None, **kwargs):
        if orm_model and isinstance(orm_model, MixinORM):
            raise ValueError(
                f"Class {orm_model.__class__} is not a subclass of MixinORM")

        self.__driver = DriverDB(**kwargs)
        self.__session = self.__driver.session()
        self.__engine = self.__driver.engine
        self.__model = orm_model

        if index == None or len(self.index) == len(index):
            self._index = None
        elif len(index) == 0:
            self._index = []
        else:
            self._index = index

        self.create_table()

    @property
    def session(self):
        if self.__session and self.__session.is_active:
            session = self.__session
        else:
            session = self.__session()
        try:
            yield session
            session.commit()
            session.expunge_all()
        except Exception:
            session.rollback()
            raise

    def __count(self, statement) -> int:
        return next(self.session).execute(
            select(func.count("*")).select_from(statement)
        ).scalar()

    def __len__(self) -> int:
        return len(self._index) if (
            self.is_custom_index()
        ) else self.__count(select(self.__model.id))

    @property
    def index(self) -> Iterable:
        if self.is_custom_index():
            return SubscriptableGenerator(
                (i for i in self._index),
                len(self._index))

        statement = self.__build_statement(select=self.__model.id)

        query = next(self.session).execute(
            statement.order_by(self.__model.id)
        ).yield_per(1000)

        lenght = self.__count(statement)
        return SubscriptableGenerator((row[0] for row in query), lenght)

    def rows(self) -> Iterable:
        statement = self.__build_statement(select=self.__model)

        query = next(self.session).execute(
            statement.order_by(self.__model.id)
        ).yield_per(1000)

        lenght = self.__count(statement)
        return SubscriptableGenerator((row[0] for row in query), lenght)

    def drop_table(self):
        close_all_sessions()
        self.__model.__table__.drop(bind=self.__engine, checkfirst=True)

    def create_table(self):
        close_all_sessions()
        self.__model.__table__.create(bind=self.__engine, checkfirst=True)

    def filter_by(self, **kwargs):
        if any([1 for key in kwargs.keys() if key not in self.__model.FIELDS]):
            raise Exception("Table {0} doesn't have one of {1} fields".format(
                self.__model.__tablename__, [*kwargs]))

        query = next(self.session).execute(
            self.__build_statement(
                select=self.__model,
                filter_by=kwargs)
        )
        return query

    def select_columns(self, columns: Iterable[str]) -> Iterable:
        if isinstance(columns, str):
            columns = [columns]

        statement = self.__build_statement(
            select=[self.__model[column] for column in columns])

        query = next(self.session).execute(
            statement.order_by(self.__model["id"])
        ).yield_per(1000)

        Row = namedtuple("Row", [*columns])
        lenght = len(self)
        return SubscriptableGenerator(
            (Row(*row) if len(row) > 1 else row[0] for row in query),
            lenght)

    def bulk_update(self, items: Iterable[MixinORM]):
        old_items = [
            *filter(lambda it: getattr(it, "id", None) != None, items)]
        new_items = [
            *filter(lambda it: getattr(it, "id", None) == None, items)]

        session = next(self.session)
        if new_items:
            session.add_all(new_items)
        if old_items:
            session.bulk_save_objects(old_items)

        session.commit()

    def find(self, id: str | Iterable[str]) -> MixinORM:
        if isinstance(id, str):
            return None if (
                self._index and id not in self._index
            ) else next(self.session).get(self.__model, id)
        elif isinstance(id, Iterable):
            if self.is_custom_index():
                index = self.index
                id = [i for i in id if i in index]
                del index
            return next(self.session).execute(
                select(self.__model)
                .where(self.__model.id.in_(id)))
        else:
            return None

    def find_by_index(self, index: int) -> MixinORM:
        statement = self.__build_statement(
            select=self.__model,
            order_by=desc(self.__model.id) if index < 0 else self.__model.id,
            offset=abs(index),
            limit=1
        )
        match = next(self.session).execute(statement).first()
        return match[0] if match else None

    def find_by_slice(self, indexer: slice) -> Iterable[MixinORM]:
        statement0 = self.__build_statement(select=self.__model)

        statement = statement0.order_by(
            desc(self.__model.id) if (
                indexer.step and indexer.step < 0
            ) else self.__model.id)

        db_size = len(self)

        if indexer.start:
            offset = (db_size + indexer.start) % db_size
            statement = statement.offset(offset)

        if indexer.stop and indexer.stop <= db_size:
            limit = (db_size + indexer.stop) % db_size
            if indexer.start:
                limit -= offset
            statement = statement.limit(limit if limit >= 0 else 0)

        query = next(self.session).execute(statement).yield_per(1000)
        lenght = self.__count(statement0)
        return SubscriptableGenerator((row[0] for row in query), lenght)

    def find_by_match(self, **kwargs) -> MixinORM:
        match = next(self.session).execute(
            self.__build_statement(
                select=self.__model,
                filter_by=kwargs,
                limit=1)
        ).first()
        return match[0] if match else None

    def delete_where(self, *args):
        if not args:
            return

        statement = delete(self.__model).where(and_(*args))

        session = next(self.session)
        session.execute(statement)
        session.commit()

    def __build_statement(self, **kwargs):
        if "select" not in kwargs:
            return None

        statement = select(kwargs.pop("select"))
        statement = statement.where(
            self.__model.id.in_([*self.index])
        ) if self.is_custom_index() else statement

        for operation, stat in kwargs.items():
            statement = getattr(statement, operation)
            if isinstance(stat, dict):
                statement = statement(**stat)
            else:
                statement = statement(stat)

        return statement

    def is_custom_index(self):
        return hasattr(self, "_index") and self._index != None
