from typing import Optional

from sqlalchemy.orm import Session

from netspresso.exceptions.user import UserIsDeletedException, UserNotFoundException
from netspresso.utils.db.models.user import User
from netspresso.utils.db.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def __is_available(self, user: Optional[User]) -> User:
        if user is None:
            raise UserNotFoundException()

        if user.is_deleted:
            raise UserIsDeletedException(user_id=user.user_id)

        return user

    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        conditions = [self.model.email == email]
        user = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(user=user)

    def get_by_user_id(self, db: Session, user_id: str) -> Optional[User]:
        conditions = [self.model.user_id == user_id]
        user = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(user=user)

    def get_by_api_key(self, db: Session, api_key: str) -> Optional[User]:
        conditions = [self.model.api_key == api_key]
        user = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(user=user)


user_repository = UserRepository(User)
