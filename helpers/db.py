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

        self.__model = orm_model
        self.__index = index
        self.__session = Session()

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
        statement = select(self.__model.id.in_(
            self.__index
        ) if self.__index else self.__model.id)
        return self.__count(statement)

    @property
    def index(self) -> Iterable:
        statement = select(self.__model.id)

        query = next(self.session).execute(
            statement.order_by(self.__model.id)
        ).yield_per(1000)

        lenght = self.__count(statement)
        return SubscriptableGenerator((row[0] for row in query), lenght)

    def rows(self, index: Iterable) -> Iterable:
        statement = select(self.__model).filter(self.__model.id.in_(index))

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
            select(self.__model)
            .filter_by(**kwargs))
        return query

    def select_columns(self, columns: Iterable[str]) -> Iterable:
        if isinstance(columns, str):
            columns = [columns]

        statement = select([self.__model[column] for column in columns])

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
        return next(self.session).get(self.__model, id)

    def find_by_index(self, index: int) -> MixinORM:
        match = next(self.session).execute(
            select(self.__model)
            .order_by(
                desc(self.__model.id) if index < 0 else self.__model.id
            ).offset(abs(index))
            .limit(1)
        ).first()
        return match[0] if match else None

    def find_by_slice(self, indexer: slice) -> Iterable[MixinORM]:
        statement0 = select(self.__model)
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
            select(self.__model)
            .filter_by(**kwargs)
            .limit(1)
        ).first()
        return match[0] if match else None
