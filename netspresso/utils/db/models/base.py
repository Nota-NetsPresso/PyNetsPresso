from sqlalchemy import Boolean, Column

from netspresso.utils.db.mixins import TimestampMixin
from netspresso.utils.db.session import Base


class BaseModel(Base, TimestampMixin):
    """모든 모델의 기본 모델"""
    __abstract__ = True

    is_deleted = Column(Boolean, nullable=False, default=False)
