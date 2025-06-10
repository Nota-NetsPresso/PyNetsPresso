from sqlalchemy import JSON, BigInteger, Column, Float, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.models.base import BaseModel, generate_uuid


class CompressionModelResult(BaseModel):
    """Entity to store model metrics before and after compression."""
    __tablename__ = "compression_model_result"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)

    # Model metrics
    size = Column(BigInteger, nullable=False, default=0)
    flops = Column(BigInteger, nullable=False, default=0)
    number_of_parameters = Column(BigInteger, nullable=False, default=0)
    trainable_parameters = Column(BigInteger, nullable=False, default=0)
    non_trainable_parameters = Column(BigInteger, nullable=False, default=0)
    number_of_layers = Column(Integer, nullable=True)

    # Relationship to CompressionTask
    compression_task_id = Column(String(36), ForeignKey("compression_task.task_id"), nullable=False)
    compression_task = relationship("CompressionTask", back_populates="model_results")

    # Type of result (original or compressed)
    result_type = Column(String(20), nullable=False)  # 'original' or 'compressed'


class CompressionTask(BaseModel):
    __tablename__ = "compression_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))

    # Compression settings
    method = Column(String(30), nullable=False)
    ratio = Column(Numeric(precision=5, scale=4), nullable=False)
    options = Column(JSON, nullable=True)
    layers = Column(JSON, nullable=True)

    # Task information
    compression_task_uuid = Column(String(36), nullable=True)
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

    # Relationship to CompressionModelResult
    model_results = relationship(
        "CompressionModelResult",
        back_populates="compression_task",
        lazy="joined",
    )
