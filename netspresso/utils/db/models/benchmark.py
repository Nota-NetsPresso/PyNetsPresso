from sqlalchemy import JSON, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.models.base import Base, BaseModel, generate_uuid


class BenchmarkResult(Base):
    __tablename__ = "benchmark_result"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)

    # Performance metrics
    processor = Column(String(30), nullable=True)
    memory_footprint_gpu = Column(Float, server_default="0", nullable=True)
    memory_footprint_cpu = Column(Float, server_default="0", nullable=True)
    power_consumption = Column(Float, server_default="0", nullable=True)
    ram_size = Column(Float, server_default="0", nullable=True)
    latency = Column(Float, server_default="0", nullable=True)
    file_size = Column(Float, server_default="0", nullable=True)

    # Relationship to BenchmarkTask
    task_id = Column(String(36), ForeignKey("benchmark_task.task_id"), nullable=False)
    task = relationship("BenchmarkTask", back_populates="result")


class BenchmarkTask(BaseModel):
    __tablename__ = "benchmark_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))

    # Benchmark settings
    framework = Column(String(30), nullable=True)
    device_name = Column(String(100), nullable=False)
    software_version = Column(String(30), nullable=True)
    precision = Column(String(30), nullable=False)
    hardware_type = Column(String(30), nullable=True)

    # Task information
    benchmark_task_id = Column(String(36), nullable=True)
    status = Column(String(30), nullable=False)
    error_detail = Column(JSON, nullable=True)

    user_id = Column(String(36), nullable=False)

    # Relationships
    result = relationship("BenchmarkResult", uselist=False, back_populates="task", cascade="all, delete-orphan")

    # Relationship to Model (source model)
    input_model_id = Column(String(36), ForeignKey("model.model_id"), nullable=True)
    input_model = relationship(
        "Model",
        uselist=False,
        lazy="joined",
        foreign_keys=[input_model_id],
    )

    # Relationship to Model (benchmark model)
    model_id = Column(String(36), ForeignKey("model.model_id"), nullable=True)
    model = relationship(
        "Model",
        uselist=False,
        lazy="joined",
        foreign_keys=[model_id],
    )
