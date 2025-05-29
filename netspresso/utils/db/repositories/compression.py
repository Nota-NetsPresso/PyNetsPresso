from netspresso.utils.db.models.compression import CompressionTask
from netspresso.utils.db.repositories.base import BaseRepository


class CompressionTaskRepository(BaseRepository[CompressionTask]):
    pass


compression_task_repository = CompressionTaskRepository(CompressionTask)
