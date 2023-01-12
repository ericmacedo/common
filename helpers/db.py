from __future__ import annotations

from collections import namedtuple
from typing import Iterable

from sqlalchemy import desc, func, select
from sqlalchemy.orm import close_all_sessions

from ..helpers.orm import Engine, Session
from ..mixins.orm import MixinORM
from ..utils.itertools import SubscriptableGenerator


class DB:
    def __init__(self, orm_model: MixinORM, index: Iterable[int] = None):
        if orm_model and isinstance(orm_model, MixinORM):
            raise ValueError(
                f"Class {orm_model.__class__} is not a subclass of MixinORM")

        self.__session = Session()
        self.__model = orm_model

        if index == None or len(self.index) == len(index):
            self._index = None
        elif len(index) == 0:
            self._index = []
        else:
            self._index = index

    @property
    def session(self):
        if self.__session and self.__session.is_active:
            session = self.__session
        else:
            session = Session()
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

    def rows(self, index: Iterable) -> Iterable:
        statement = self.__build_statement(
            select=self.__model,
            filter=self.__model.id.in_(index))

        query = next(self.session).execute(
            statement.order_by(self.__model.id)
        ).yield_per(1000)

        lenght = self.__count(statement)
        return SubscriptableGenerator((row[0] for row in query), lenght)

    def drop_table(self):
        close_all_sessions()
        self.__model.__table__.drop(bind=Engine, checkfirst=True)

    def create_table(self):
        close_all_sessions()
        self.__model.__table__.create(bind=Engine, checkfirst=True)

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

    def find(self, id: int) -> MixinORM:
        return None if (
            self._index and id not in self._index
        ) else next(self.session).get(self.__model, id)

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
        statement = statement0.order_by(self.__model.id)

        if indexer.start:
            offset = (len(self) + indexer.start) % len(self)
            statement = statement.offset(offset)

        if indexer.stop:
            limit = (len(self) + indexer.stop) % len(self)
            statement = statement.limit(limit)

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
