from sqlalchemy import JSON, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.models.base import BaseModel, generate_uuid


class ConversionTask(BaseModel):
    __tablename__ = "conversion_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))

    # Conversion settings
    framework = Column(String(30), nullable=False)
    device_name = Column(String(100), nullable=False)
    software_version = Column(String(30), nullable=True)
    precision = Column(String(30), nullable=False)

    # Task information
    convert_task_uuid = Column(String(36), nullable=True)
    status = Column(String(30), nullable=False)
    error_detail = Column(JSON, nullable=True)

    user_id = Column(String(36), nullable=False)

    # Relationship to Model (source model)
    input_model_id = Column(String(36), ForeignKey("model.model_id"), nullable=True)
    input_model = relationship(
        "Model",
        uselist=False,
        lazy="joined",
        foreign_keys=[input_model_id],
    )

    # Relationship to Model (converted model)
    model_id = Column(String(36), ForeignKey("model.model_id"), nullable=True)
    model = relationship(
        "Model",
        uselist=False,
        lazy="joined",
        foreign_keys=[model_id],
    )
