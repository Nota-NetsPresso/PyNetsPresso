from sqlalchemy import JSON, Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.models.base import BaseModel, generate_uuid


class Evaluation(BaseModel):
    __tablename__ = "evaluation_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))

    dataset_id = Column(String(36), nullable=True)
    is_dataset_deleted = Column(Boolean, nullable=False, default=False)
    metric_unit = Column(String(30), nullable=True)
    metric_value = Column(Float, nullable=True)
    results_path = Column(String(255), nullable=True)

    status = Column(String(30), nullable=False)
    error_detail = Column(JSON, nullable=True)

    # Relationship to Model (source model)
    input_model_id = Column(String(36), ForeignKey("model.model_id"), nullable=True)
    input_model = relationship(
        "Model",
        uselist=False,
        lazy="joined",
        foreign_keys=[input_model_id],
    )

    # Relationship to TrainingTask
    training_task_id = Column(String(36), nullable=True)

    # Reference to conversion task if a converted model was used
    conversion_task_id = Column(String(36), nullable=True)

