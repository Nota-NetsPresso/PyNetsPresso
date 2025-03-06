from enum import Enum
from typing import Any, Generic, List, Optional, Type, TypeVar

from sqlalchemy import UnaryExpression, and_, asc, desc, func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Query, Session

from netspresso.exceptions.database import DatabaseOperationError
from netspresso.utils.db.models.base import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)  # type: ignore


class Order(str, Enum):
    DESC = "desc"
    ASC = "asc"


class TimeSort(str, Enum):
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class BaseRepository(Generic[ModelType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

    def choose_order_func(self, order: Optional[Order]) -> UnaryExpression:
        if order is None or order == Order.DESC:
            return desc
        return asc

    def choose_time_sort(self, time_sort: Optional[TimeSort]) -> UnaryExpression:
        if time_sort is None or time_sort == TimeSort.CREATED_AT:
            return self.model.created_at
        elif time_sort == TimeSort.UPDATED_AT:
            return self.model.updated_at
        else:
            raise ValueError(f"Invalid time sort: {time_sort}")

    def _apply_pagination(self, query: Query, start: Optional[int], size: Optional[int]) -> Query:
        """Apply pagination to query.

        Args:
            query: The query to modify
            start: Pagination start index
            size: Page size

        Returns:
            Modified query with pagination applied
        """
        if start is not None and size is not None:
            query = query.offset(start).limit(size)
        return query

    def _apply_sorting(self, query: Query, order: Optional[Order], time_sort: Optional[TimeSort]) -> Query:
        """Apply sorting to query based on order and time_sort parameters.

        Args:
            query: The query to modify
            order: Sort direction (DESC or ASC)
            time_sort: Field to sort by (CREATED_AT or UPDATED_AT)

        Returns:
            Modified query with sorting applied
        """
        ordering_func = self.choose_order_func(order)
        time_sort_field = self.choose_time_sort(time_sort)
        return query.order_by(ordering_func(time_sort_field))

    def _build_base_query(self, db: Session, conditions: Optional[List[Any]] = None) -> Query:
        query = db.query(self.model)
        if conditions:
            query = query.filter(and_(*conditions))
        return query

    def _apply_group_by(self, query: Query, group_fields: List[Any]) -> Query:
        """Apply GROUP BY to query.

        Args:
            query: The query to modify
            group_fields: List of fields to group by

        Returns:
            Modified query with GROUP BY applied
        """
        if group_fields:
            query = query.group_by(*group_fields)
        return query

    def _prepare_conditions(self, conditions: Optional[List[Any]] = None) -> List[Any]:
        """Prepare query conditions by adding default conditions.

        Args:
            conditions: Additional filter conditions

        Returns:
            List of combined filter conditions
        """
        base_conditions = [self.model.is_deleted.is_(False)]
        if conditions:
            base_conditions.extend(conditions)
        return base_conditions

    def find_first(
        self,
        db: Session,
        conditions: Optional[List[Any]] = None,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> Optional[ModelType]:
        """Find first entity matching given conditions with optional sorting.

        Args:
            db: Database session
            conditions: List of filter conditions
            order: Sort order (DESC/ASC)
            time_sort: Time field to sort by (CREATED_AT/UPDATED_AT)

        Returns:
            Optional[ModelType]: First matching entity or None if not found
        """
        base_conditions = self._prepare_conditions(conditions)

        query = self._build_base_query(db, base_conditions)
        query = self._apply_sorting(query, order, time_sort)

        return query.first()

    def find_all(
        self,
        db: Session,
        conditions: Optional[List[Any]] = None,
        group_fields: Optional[List[Any]] = None,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> List[ModelType]:
        """Find all entities matching given conditions with optional sorting, pagination, and grouping.

        Args:
            db: Database session
            conditions: Filter conditions
            start: Pagination start index
            size: Page size
            order: Sort direction
            time_sort: Time field to sort by
            group_fields: Fields to group by

        Returns:
            List of matching entities
        """
        base_conditions = self._prepare_conditions(conditions)

        query = self._build_base_query(db, base_conditions)
        query = self._apply_group_by(query, group_fields)
        query = self._apply_sorting(query, order, time_sort)
        query = self._apply_pagination(query, start, size)

        return query.all()

    def save(self, db: Session, model: ModelType, refresh: bool = True) -> ModelType:
        """Save model to database with comprehensive error handling.

        Args:
            db: Database session
            model: Model instance to save
            refresh: Whether to refresh the model after commit (default: True)

        Returns:
            ModelType: Saved and refreshed model instance

        Raises:
            EntityDuplicateError: If entity with unique constraint already exists
            EntityRelationError: If related entity reference is invalid
            DatabaseOperationError: For other database errors
        """
        try:
            db.add(model)
            db.flush()  # Flush to detect errors before commit

            if refresh:
                db.refresh(model)

            db.commit()
            return model

        except IntegrityError as e:
            db.rollback()

            # Default integrity error
            raise DatabaseOperationError(
                error_log=f"Database integrity error: {str(e)}", error_code="DB_INTEGRITY_ERROR"
            )

        except SQLAlchemyError as e:
            db.rollback()
            raise DatabaseOperationError(error_log=f"SQLAlchemy error: {str(e)}", error_code="DB_SQLALCHEMY_ERROR")

        except Exception as e:
            db.rollback()
            raise DatabaseOperationError(
                error_log=f"Unexpected error during save operation: {str(e)}", error_code="DB_UNEXPECTED_ERROR"
            )

    def update(self, db: Session, model: ModelType) -> ModelType:
        return self.save(db, model)

    def soft_delete(self, db: Session, model: ModelType) -> ModelType:
        model.is_deleted = True
        return self.update(db, model)

    def count_by_field(
        self,
        db: Session,
        count_field: Any,
        conditions: List[Any],
    ) -> int:
        """Count entities by field value with optional additional conditions.

        Args:
            db: Database session
            count_field: Field to count (defaults to model.id if None)
            conditions: Additional filter conditions

        Returns:
            int: Count of matching entities
        """
        base_conditions = self._prepare_conditions(conditions)

        return db.query(func.count(count_field)).filter(and_(*base_conditions)).scalar()
