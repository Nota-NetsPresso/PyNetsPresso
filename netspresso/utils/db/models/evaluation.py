from sqlalchemy import JSON, Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.models.base import BaseModel, generate_uuid


class EvaluationTask(BaseModel):
    __tablename__ = "evaluation_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))

    dataset_id = Column(String(36), nullable=True)
    is_dataset_deleted = Column(Boolean, nullable=False, default=False)

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

    # 평가 결과들의 관계
    results = relationship("EvaluationResult", back_populates="evaluation_task", cascade="all, delete-orphan")


class EvaluationResult(BaseModel):
    __tablename__ = "evaluation_result"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    result_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="result"))

    evaluation_task_id = Column(String(36), ForeignKey("evaluation_task.task_id"), nullable=False)
    evaluation_task = relationship("EvaluationTask", back_populates="results")

    confidence_score = Column(Float, nullable=False)
    metric_unit = Column(String(30), nullable=True)
    metric_value = Column(Float, nullable=True)
    results_path = Column(String(255), nullable=True)

    status = Column(String(30), nullable=False)
    error_detail = Column(JSON, nullable=True)

