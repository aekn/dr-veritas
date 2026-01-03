from contextlib import contextmanager

from sqlmodel import Session, create_engine

from vera.config.settings import settings


engine = create_engine(str(settings.db.pg_dsn), pool_pre_ping=True)


@contextmanager
def session_scope() -> Session:
    with Session(engine) as session:
        yield session
