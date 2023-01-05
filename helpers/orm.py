from contextlib import contextmanager
from os import getenv
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import registry, sessionmaker
from sqlalchemy.pool import NullPool


# --------------------
# 		Settings
# --------------------
DATABASE_URI = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}".format(
    user=getenv("DB_USER", "postgres"),
    password=getenv("DB_PASSWORD", "123456"),
    host=getenv("DB_HOST", "localhost"),
    port=getenv("DB_PORT", "5432"),
    db=getenv("DB_NAME", "postgres")
)


# --------------------
# 		Constructors
# --------------------
Engine = create_engine(DATABASE_URI, poolclass=NullPool)

MapperRegistry = registry()

Base = MapperRegistry.generate_base()
Base.metadata.create_all(Engine)

Session = sessionmaker(bind=Engine, expire_on_commit=False, autoflush=True)


# --------------------
# 		Context
# --------------------
@contextmanager
def session_scope():
    session = Session()
    try:
        yield session
        session.commit()
        session.expunge_all()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def TransactionORM(fn: Callable):
    def inner(*args, **kwargs):
        with session_scope() as session:
            fn(session, *args, **kwargs)

    return inner
