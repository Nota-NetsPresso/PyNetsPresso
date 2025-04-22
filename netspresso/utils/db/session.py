from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import create_database, database_exists

from app.configs.settings import settings

DB_URL = settings.DATABASE_URL

engine = create_engine(
    f"{DB_URL}",
    pool_pre_ping=True,
    pool_use_lifo=True,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)

Base = declarative_base()


def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise e
    finally:
        db.close()


def check_database(engine: Engine):
    if not database_exists(engine.url):
        logger.info("The database did not exist, so it has been created.")
        create_database(engine.url)
    else:
        logger.info("The database has already been created.")


check_database(engine=engine)
