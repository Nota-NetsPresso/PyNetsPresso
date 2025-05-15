from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.models.base import BaseModel, generate_uuid


class Model(BaseModel):
    __tablename__ = "model"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    model_id = Column(
        String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="model")
    )
    name = Column(String(255), nullable=False)
    type = Column(String(30), nullable=False)
    is_retrainable = Column(Boolean, nullable=False, default=False)
    object_path = Column(String(255), nullable=True)

    project_id = Column(String(36), ForeignKey("project.project_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), nullable=False)

    # Back-reference to Project
    project = relationship("Project", back_populates="models")
