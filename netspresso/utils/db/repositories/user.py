from typing import Optional

from sqlalchemy.orm import Session

from netspresso.utils.db.models.user import User
from netspresso.utils.db.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        conditions = [self.model.email == email]
        user = self.find_first(
            db=db,
            conditions=conditions,
        )

        return user

    def get_by_user_id(self, db: Session, user_id: str) -> Optional[User]:
        conditions = [self.model.user_id == user_id]
        user = self.find_first(
            db=db,
            conditions=conditions,
        )

        return user

    def get_by_api_key(self, db: Session, api_key: str) -> Optional[User]:
        conditions = [self.model.api_key == api_key]
        user = self.find_first(
            db=db,
            conditions=conditions,
        )

        return user


user_repository = UserRepository(User)
