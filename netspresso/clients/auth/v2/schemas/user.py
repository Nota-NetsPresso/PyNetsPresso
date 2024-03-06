from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from netspresso.clients.auth import response_body
from netspresso.clients.auth.response_body import (
    UserDetailResponse
)
from netspresso.clients.auth.v2.schemas.common import (
    AbstractResponse,
    MembershipType,
)
from netspresso.clients.auth.v2.schemas.credit import SummarizedCreditResponse
from netspresso.clients.auth.v2.schemas.user_agreement import UserAgreementBase


class UserBase(BaseModel):
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    region: str


class UserPayload(UserBase):
    user_id: str
    is_active: bool
    is_deleted: bool
    is_admin: bool
    is_reset: bool
    last_login_time: datetime
    membership_type: MembershipType

    user_agreement: UserAgreementBase

    class Config:
        from_attributes = True


class UserResponse(AbstractResponse):
    data: UserPayload

    def to(
        self, summarized_credit_response: SummarizedCreditResponse
    ) -> response_body.UserResponse:
        return response_body.UserResponse(
            **{
                "user_id": self.data.user_id,
                "email": self.data.email,
                "detail_data": UserDetailResponse(
                    **{
                        "first_name": self.data.first_name,
                        "last_name": self.data.last_name,
                        "company": self.data.company,
                    }
                ),
                "credit_info": summarized_credit_response.to(),
            }
        )
