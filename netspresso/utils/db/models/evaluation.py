from sqlalchemy import JSON, Boolean, Column, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.models.base import BaseModel, generate_uuid


class EvaluationTask(BaseModel):
    __tablename__ = "evaluation_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))

    dataset_id = Column(String(36), nullable=True)
    is_dataset_deleted = Column(Boolean, nullable=False, default=False)

    # 평가 설정
    confidence_score = Column(Numeric(precision=2, scale=1), nullable=True)
    metrics = Column(JSON, nullable=True)
    metrics_names = Column(JSON, nullable=True)
    primary_metric = Column(String(30), nullable=True)
    results_path = Column(String(255), nullable=True)

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

    # Relationship to TrainingTask
    training_task_id = Column(String(36), nullable=True)

    # Reference to conversion task if a converted model was used
    conversion_task_id = Column(String(36), nullable=True)
