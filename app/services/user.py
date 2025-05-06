from app.api.v1.schemas.user import CreditInfo, DetailData, UserPayload
from netspresso.netspresso import NetsPresso


class UserService:
    def get_user_info(self, api_key: str) -> UserPayload:
        netspresso = NetsPresso(api_key=api_key)

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
