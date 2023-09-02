import asyncio
from contextlib import contextmanager
from os import getenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
from sqlalchemy_utils import create_database, database_exists

from common.database import Base


class DriverDB:
    _instances = {}
    DB_PROTOCOL = "postgresql+psycopg2"
    ASYNC_DB_PROTOCOL = "postgresql+asyncpg"

    def __new__(cls, DB_NAME=None):
        # Implements Multition pattern
        if DB_NAME is None:
            raise ValueError("DB_NAME cannot be None")
        if DB_NAME not in cls._instances:
            cls._instances[DB_NAME] = super().__new__(cls)
        return cls._instances[DB_NAME]

    def __init__(self, **kwargs):
        def get_val(key, fallback):
            return kwargs.get(key, getenv(key, fallback))

        self.__user = get_val("DB_USER", "postgres")
        self.__password = get_val("DB_PASWORD", "postgres")
        self.__host = get_val("DB_HOST", "localhost")
        self.__port = get_val("DB_PORT", "5432")
        self.__db = get_val("DB_NAME", "postgres")

        if not database_exists(self.engine.url):
            create_database(self.engine.url)

    @property
    def db_uri(self) -> str:
        return "{protocol}://{user}:{password}@{host}:{port}/{db}".format(
            protocol=self.DB_PROTOCOL,
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
            self.__session_factory = sessionmaker(bind=self.engine,
                                                  expire_on_commit=False,
                                                  autocommit=False,
                                                  autoflush=False)

        return self.__session_factory

    def create_all(self):
        Base.metadata.create_all(bind=self.engine, checkfirst=True)

    async def get_session(self):
        while self.__locked:
            await asyncio.sleep(0.1)

        self.__locked = True
        session = scoped_session(self.session)

        try:
            yield session
            session.commit()
            session.expunge_all()
        except Exception:
            session.rollback()
            raise
        finally:
            self.__locked = False
            session.close()
