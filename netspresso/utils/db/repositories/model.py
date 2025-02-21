from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from netspresso.exceptions.model import ModelIsDeletedException, ModelNotFoundException
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.repositories.base import BaseRepository, Order


class ModelRepository(BaseRepository[Model]):
    def __is_available(self, model: Optional[Model]) -> Model:
        if model is None:
            raise ModelNotFoundException()

        if model.is_deleted:
            raise ModelIsDeletedException(model_id=model.model_id)

        return model

    def get_by_model_id(self, db: Session, model_id: str) -> Optional[Model]:
        model = db.query(self.model).filter(self.model.model_id == model_id).first()

        return self.__is_available(model=model)

    def _get_models(
        self,
        db: Session,
        condition,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
    ) -> Optional[List[Model]]:
        ordering_func = self.choose_order_func(order)
        query = db.query(self.model).filter(*condition)

        if order:
            query = query.order_by(ordering_func(self.model.created_at))

        if start is not None and size is not None:
            query = query.offset(start).limit(size)

        models = query.all()

        return models

    def get_all_by_user_id(
        self,
        db: Session,
        user_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
    ) -> Optional[List[Model]]:
        return self._get_models(
            db=db,
            condition=[self.model.user_id == user_id, self.model.is_deleted.is_(False)],
            start=start,
            size=size,
            order=order,
        )

    def get_all_by_project_id(
        self,
        db: Session,
        project_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = Order.DESC,
    ) -> Optional[List[Model]]:
        return self._get_models(
            db=db,
            condition=[self.model.project_id == project_id, self.model.is_deleted.is_(False)],
            start=start,
            size=size,
            order=order,
        )

    def count_by_user_id(self, db: Session, user_id: str) -> int:
        return (
            db.query(func.count(self.model.user_id))
            .filter(self.model.user_id == user_id, self.model.is_deleted.is_(False))
            .scalar()
        )

    def delete_by_model_id(self, db: Session, model_id: str) -> Model:
        model = self.get_by_model_id(db, model_id)
        model.is_deleted = True
        model = self.update(db, model)

        return model


model_repository = ModelRepository(Model)
