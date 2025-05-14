from sqlalchemy import JSON, Column, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from netspresso.utils.db.models.base import Base, BaseModel, generate_uuid


class Augmentation(Base):
    __tablename__ = "augmentation"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    name = Column(String(50), nullable=False)
    parameters = Column(JSON, nullable=False)
    phase = Column(String(30), nullable=False)  # train, inference

    hyperparameter_id = Column(Integer, ForeignKey("hyperparameter.id"), nullable=False)
    hyperparameter = relationship("Hyperparameter", back_populates="augmentations", lazy="joined")


class TrainingTask(BaseModel):
    __tablename__ = "training_task"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    task_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="task"))
    pretrained_model = Column(String(100), nullable=False)
    task = Column(String(30), nullable=False)
    framework = Column(String(30), nullable=False)
    input_shapes = Column(JSON, nullable=False)
    status = Column(String(30), nullable=False)
    error_detail = Column(JSON, nullable=True)

    user_id = Column(String(36), nullable=False)

    # Relationships (1:1 Mapping)
    dataset = relationship("Dataset", back_populates="task", uselist=False, cascade="all, delete-orphan", lazy="joined")
    hyperparameter = relationship(
        "Hyperparameter", back_populates="task", uselist=False, cascade="all, delete-orphan", lazy="joined"
    )
    environment = relationship(
        "Environment", back_populates="task", uselist=False, cascade="all, delete-orphan", lazy="joined"
    )
    performance = relationship(
        "Performance", back_populates="task", uselist=False, cascade="all, delete-orphan", lazy="joined"
    )

    # Relationship to Model
    model_id = Column(String(36), ForeignKey("model.model_id"), nullable=True)
    model = relationship(
        "Model",
        uselist=False,
        lazy="joined",
    )


class Dataset(Base):
    __tablename__ = "training_dataset"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    dataset_id = Column(String(36), index=True, unique=True, nullable=False, default=lambda: generate_uuid(entity="dataset"))
    name = Column(String(100), nullable=False)
    path = Column(String(255), nullable=False)
    id_mapping = Column(JSON, nullable=True)
    palette = Column(JSON, nullable=True)
    task_type = Column(String(30), nullable=False)
    mime_type = Column(String(30), default="image")
    class_count = Column(Integer, nullable=False)

    valid_split_ratio = Column(Float, default=0.1)
    random_seed = Column(Integer, default=0)

    storage_location = Column(String(50), nullable=False)
    storage_info = Column(JSON, nullable=True)

    # Relationship to TrainingTask
    task_id = Column(String(36), ForeignKey("training_task.task_id", ondelete="CASCADE"), unique=True, nullable=False)
    task = relationship("TrainingTask", back_populates="dataset")


class Hyperparameter(Base):
    __tablename__ = "hyperparameter"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    epochs = Column(Integer, nullable=False, default=0)
    batch_size = Column(Integer, nullable=False)
    optimizer = Column(JSON, nullable=True)
    scheduler = Column(JSON, nullable=True)

    augmentations = relationship(
        "Augmentation", back_populates="hyperparameter", cascade="all, delete-orphan", lazy="joined"
    )

    # Relationship to TrainingTask
    task_id = Column(String(36), ForeignKey("training_task.task_id", ondelete="CASCADE"), unique=True, nullable=False)
    task = relationship("TrainingTask", back_populates="hyperparameter")


class Environment(Base):
    __tablename__ = "environment"

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    seed = Column(Integer, nullable=False)
    num_workers = Column(Integer, nullable=False)
    gpus = Column(String(30), nullable=False)  # GPUs (예: "1, 0")

    # Relationship to TrainingTask
    task_id = Column(String(36), ForeignKey("training_task.task_id", ondelete="CASCADE"), unique=True, nullable=False)
    task = relationship("TrainingTask", back_populates="environment")


class Performance(Base):
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, index=True, unique=True, autoincrement=True, nullable=False)
    train_losses = Column(JSON, nullable=False)
    valid_losses = Column(JSON, nullable=False)
    train_metrics = Column(JSON, nullable=False)
    valid_metrics = Column(JSON, nullable=False)
    metrics_list = Column(JSON, nullable=False)
    primary_metric = Column(String(36), nullable=False)
    flops = Column(String(50), nullable=False, default=0)
    params = Column(String(50), nullable=False, default=0)
    total_train_time = Column(Float, nullable=False, default=0)
    best_epoch = Column(Integer, nullable=False, default=0)
    last_epoch = Column(Integer, nullable=False, default=0)
    total_epoch = Column(Integer, nullable=False, default=0)
    status = Column(String(36), nullable=True)

    # Relationship to TrainTask
    task_id = Column(String(36), ForeignKey("training_task.task_id", ondelete="CASCADE"), unique=True, nullable=False)
    task = relationship("TrainingTask", back_populates="performance")
