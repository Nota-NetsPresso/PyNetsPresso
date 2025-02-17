from sqlalchemy import JSON, Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from netspresso.utils.db.generate_uuid import generate_uuid
from netspresso.utils.db.mixins import TimestampMixin
from netspresso.utils.db.session import Base


class BenchmarkResult(Base):
    __tablename__ = "benchmark_result"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)

    # Performance metrics
    processor = Column(String(30), nullable=True)
    memory_footprint_gpu = Column(Integer, nullable=True)
    memory_footprint_cpu = Column(Integer, nullable=True)
    power_consumption = Column(Integer, nullable=True)
    ram_size = Column(Integer, nullable=True)
    latency = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)

    # Relationship to BenchmarkTask
    task_id = Column(String(36), ForeignKey("benchmark_task.task_id"), nullable=False, unique=True)
    task = relationship("BenchmarkTask", back_populates="result")


class BenchmarkTask(Base, TimestampMixin):
    __tablename__ = "benchmark_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))

    # Benchmark settings
    framework = Column(String(30), nullable=True)
    device_name = Column(String(30), nullable=False)
    software_version = Column(String(30), nullable=True)
    precision = Column(String(30), nullable=False)
    hardware_type = Column(String(30), nullable=True)

    # Task information
    benchmark_task_id = Column(String(36), nullable=True)
    status = Column(String(30), nullable=False)
    error_detail = Column(JSON, nullable=True)

    is_deleted = Column(Boolean, nullable=False, default=False)

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
