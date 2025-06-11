from sqlalchemy import JSON, Boolean, Column, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.models.base import Base, BaseModel, generate_uuid


class EvaluationDataset(BaseModel):
    """Dataset for evaluation tasks."""
    __tablename__ = "evaluation_dataset"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    dataset_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="dataset"))
    name = Column(String(100), nullable=False)
    path = Column(String(255), nullable=False)
    id_mapping = Column(JSON, nullable=True)
    palette = Column(JSON, nullable=True)
    task_type = Column(String(30), nullable=False)
    mime_type = Column(String(30), default="image")
    class_count = Column(Integer, nullable=False)
    count = Column(Integer, nullable=False)

    storage_location = Column(String(50), nullable=False)
    storage_info = Column(JSON, nullable=True)

    # Updated: Changed to support multiple tasks per dataset
    tasks = relationship("EvaluationTask", back_populates="dataset")


class EvaluationTask(BaseModel):
    __tablename__ = "evaluation_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))

    # Updated: Changed to support many-to-one relationship with dataset
    dataset_id = Column(String(36), ForeignKey("evaluation_dataset.dataset_id"), nullable=True)
    dataset = relationship("EvaluationDataset", back_populates="tasks", uselist=False)
    is_dataset_deleted = Column(Boolean, nullable=False, default=False)

    # 평가 설정
    confidence_score = Column(Numeric(precision=2, scale=1), nullable=False)
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
