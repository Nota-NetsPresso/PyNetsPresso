from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from netspresso.clients.auth import response_body
from netspresso.clients.auth.response_body import UserDetailResponse
from netspresso.clients.auth.v2.schemas.common import AbstractResponse, MembershipType
from netspresso.clients.auth.v2.schemas.credit import SummarizedCreditResponse
from netspresso.clients.auth.v2.schemas.user_agreement import UserAgreementBase


@dataclass
class UserBase:
    username: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    company: Optional[str]
    region: str


@dataclass
class UserPayload(UserBase):
    user_id: str
    is_active: bool
    is_deleted: bool
    is_admin: bool
    is_reset: bool
    last_login_time: datetime
    membership_type: MembershipType

    user_agreement: UserAgreementBase = field(default_factory=UserAgreementBase)

    def __post_init__(self):
        self.user_agreement = UserAgreementBase(**self.user_agreement)


@dataclass
class UserResponse(AbstractResponse):
    data: UserPayload = field(default_factory=UserPayload)

    def __post_init__(self):
        self.data = UserPayload(**self.data)

    def to(self, summarized_credit_response: SummarizedCreditResponse) -> response_body.UserResponse:
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
