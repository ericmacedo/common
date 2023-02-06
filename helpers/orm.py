from contextlib import contextmanager
from os import getenv
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import registry, sessionmaker
from sqlalchemy.pool import NullPool

from sqlalchemy_utils import database_exists, create_database

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
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def session_scope(self):
        session = self.session()
        try:
            yield session
            session.commit()
            session.expunge_all()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def TransactionORM(self, fn: Callable):
        def inner(*args, **kwargs):
            with self.session_scope() as session:
                fn(session, *args, **kwargs)

        return inner


driver: DriverDB = DriverDB()

# exports
session_scope = driver.session_scope
Session = driver.session
Engine = driver.engine

driver.create_all()
