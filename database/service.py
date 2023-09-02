from collections import namedtuple
from typing import Any, Iterable, List

from sqlalchemy import and_, delete, desc, func, select
from sqlalchemy.orm import close_all_sessions

from common.database.connector import DriverDB
from common.mixins.db import MixinORM
from common.utils.itertools import SubscriptableGenerator


class ServiceDB:
    def __init__(self, orm_model: MixinORM, **kwargs):
        if orm_model and isinstance(orm_model, MixinORM):
            raise ValueError(
                f"Class {orm_model.__class__} is not a subclass of MixinORM")

        self.__driver = DriverDB(**kwargs)
        self.__model = orm_model

        if not self.__driver.engine.has_table(orm_model.__tablename__):
            print("Creating table {0}".format(orm_model.__tablename__))
            self.create_table()

    @property
    def session(self):
        session = self.__driver.session()

        try:
            yield session
            session.commit()
            session.expunge_all()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def __count(self, statement) -> int:
        return next(self.session).execute(
            select(func.count("*")).select_from(statement)
        ).scalar()

    def drop_table(self):
        close_all_sessions()
        self.__model.__table__.drop(bind=self.__driver.engine, checkfirst=True)

    def create_table(self):
        close_all_sessions()
        self.__model.__table__.create(
            bind=self.__driver.engine, checkfirst=True)

    def min(self, column: str) -> Any:
        if column not in self.__model.FIELDS:
            raise Exception("Table {0} doesn't have a {1} field".format(
                self.__model.__tablename__, column))

        return next(self.session).execute(
            func.min(self.__model[column])
        ).scalar()

    def max(self, column: str) -> Any:
        if column not in self.__model.FIELDS:
            raise Exception("Table {0} doesn't have a {1} field".format(
                self.__model.__tablename__, column))

        return next(self.session).execute(
            func.max(self.__model[column])
        ).scalar()

    def len(self, filters: List = []):
        statement = select(self.__model)
        if filters:
            statement = statement.where(and_(*filters))
        return self.__count(statement)

    def filter_by(self, **kwargs):
        if any([1 for key in kwargs.keys() if key not in self.__model.FIELDS]):
            raise Exception("Table {0} doesn't have one of {1} fields".format(
                self.__model.__tablename__, [*kwargs]))

        query = next(self.session).execute(
            self.__build_statement(
                select=self.__model,
                filter_by=kwargs))
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
            return next(self.session).get(self.__model, id)
        elif isinstance(id, Iterable):
            return next(self.session).execute(
                select(self.__model)
                .where(self.__model.id.in_(id)))
        else:
            return None

    def find_where(self,
                   filters: List = [],
                   sort: List = [],
                   page: int = None,
                   page_size: int = None) -> Iterable[MixinORM]:

        statement = select(self.__model)

        if filters:
            statement = statement.where(and_(*filters))

        if len(sort) == 0:
            sort = [self.__model.id]
        statement = statement.order_by(*sort)

        if page != None and page_size != None:
            statement = statement.offset(
                page * page_size).limit(page_size)

        session = next(self.session)
        query = session.execute(statement)

        lenght = self.__count(statement)
        return SubscriptableGenerator((row[0] for row in query), lenght)

    def find_by_index(self, index: int) -> MixinORM:
        statement = self.__build_statement(
            select=self.__model,
            order_by=desc(self.__model.id) if index < 0 else self.__model.id,
            offset=abs(index),
            limit=1)

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

        for operation, stat in kwargs.items():
            statement = getattr(statement, operation)
            if isinstance(stat, dict):
                statement = statement(**stat)
            else:
                statement = statement(stat)

        return statement
