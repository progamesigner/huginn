from lib.database import DatabaseDeclarativeBase


class AppRecord(DatabaseDeclarativeBase):
    __abstract__ = True
