from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.exceptions.model import ModelIsDeletedException, ModelNotFoundException
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.repositories.base import BaseRepository, Order, TimeSort


class ModelRepository(BaseRepository[Model]):
    def __is_available(self, model: Optional[Model]) -> Model:
        if model is None:
            raise ModelNotFoundException()

        if model.is_deleted:
            raise ModelIsDeletedException(model_id=model.model_id)

        return model

    def get_by_model_id(self, db: Session, model_id: str) -> Model:
        conditions = [self.model.model_id == model_id]
        model = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(model=model)

    def get_all_by_user_id(
        self,
        db: Session,
        user_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> List[Model]:
        conditions = [self.model.user_id == user_id]
        models = self.find_all(
            db=db,
            conditions=conditions,
            start=start,
            size=size,
            order=order,
            time_sort=time_sort,
        )

        return models

    def get_all_by_project_id(
        self,
        db: Session,
        project_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> List[Model]:
        conditions = [self.model.project_id == project_id]
        models = self.find_all(
            db=db,
            conditions=conditions,
            start=start,
            size=size,
            order=order,
            time_sort=time_sort,
        )

        return models

    def count_by_user_id(self, db: Session, user_id: str) -> int:
        count_field = self.model.user_id
        conditions = [self.model.user_id == user_id]

        count = self.count_by_field(db=db, count_field=count_field, conditions=conditions)

        return count


model_repository = ModelRepository(Model)
