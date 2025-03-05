from sqlalchemy import Column, Integer, String

from netspresso.utils.db.models.base import BaseModel


class User(BaseModel):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    email = Column(String(36), nullable=False)
    password = Column(String(36), nullable=False)
    api_key = Column(String(36), nullable=False)
    user_id = Column(String(36), nullable=False)
