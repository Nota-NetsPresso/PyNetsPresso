from sqlalchemy.orm import Session

from app.api.v1.schemas.user import ApiKeyPayload, CreditInfo, DetailData, UserPayload
from netspresso.netspresso import NetsPresso
from netspresso.utils.db.models.base import generate_uuid
from netspresso.utils.db.models.user import User
from netspresso.utils.db.repositories.user import user_repository


class UserService:
    def create_user(self, db: Session, email: str, password: str, api_key: str, user_id: str) -> User:
        user = User(
            email=email,
            password=password,
            api_key=api_key,
            user_id=user_id,
        )
        user = user_repository.save(db=db, model=user)

        return user

    def generate_api_key(self, db: Session, email: str, password: str) -> ApiKeyPayload:
        generated_id = generate_uuid(entity="user")

        user = user_repository.get_by_email(db=db, email=email)

        if user:
            if user.password != password:
                user.password = password
                user.api_key = generated_id
                user = user_repository.save(db=db, model=user)
        else:
            netspresso = NetsPresso(email=email, password=password)
            user = self.create_user(
                db=db,
                email=email,
                password=password,
                api_key=generated_id,
                user_id=netspresso.user_info.user_id,
            )

        api_key = ApiKeyPayload(api_key=user.api_key)

        return api_key

    def get_user_by_api_key(self, db: Session, api_key: str) -> User:
        user = user_repository.get_by_api_key(db=db, api_key=api_key)

        return user

    def build_netspresso_with_api_key(self, db: Session, api_key: str) -> NetsPresso:
        user = user_repository.get_by_api_key(db=db, api_key=api_key)
        netspresso = NetsPresso(email=user.email, password=user.password)

        return netspresso

    def get_user_info(self, db: Session, api_key: str) -> UserPayload:
        netspresso = self.build_netspresso_with_api_key(db=db, api_key=api_key)

        user = UserPayload(
            user_id=netspresso.user_info.user_id,
            email=netspresso.user_info.email,
            detail_data=DetailData(
                first_name=netspresso.user_info.detail_data.first_name,
                last_name=netspresso.user_info.detail_data.last_name,
                company=netspresso.user_info.detail_data.company,
            ),
            credit_info=CreditInfo(
                free=netspresso.user_info.credit_info.free,
                reward=netspresso.user_info.credit_info.reward,
                contract=netspresso.user_info.credit_info.contract,
                paid=netspresso.user_info.credit_info.paid,
                total=netspresso.user_info.credit_info.total,
            ),
        )

        return user


user_service = UserService()
