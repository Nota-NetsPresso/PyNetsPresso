from pydantic import BaseModel, EmailStr, Field

from netspresso.clients.auth.request_body import Tokens
from netspresso.clients.auth.response_body import (
    UserResponse,
    UserDetailResponse,
    CreditResponse,
)


class LoginResponse(BaseModel):
    current_time: str = Field(..., description="Login Time")
    region: str = Field(..., description="User Resion")
    tokens: Tokens = Field(default_factory=Tokens, description="Session Token")


class UserInfo(BaseModel):
    user_id: str = Field(..., description="User ID")
    email: EmailStr = Field(..., description="Email")
    username: str = Field(..., description="User Name")
    detail_data: UserDetailResponse = Field(..., description="User Detail")
    is_active: bool = Field(..., description="Active Status")
    is_admin: bool = Field(..., description="Admin Status")
    current_time: str = Field(..., description="Current Time")
    created_time: str = Field(..., description="Created Time")
    last_login_time: str = Field(..., description="Last Login Time")
    free: int = Field(0, description="Free Credit")
    reward: int = Field(0, description="Reward Credit")
    contract: int = Field(0, description="Contract Credit")
    paid: int = Field(0, description="Paid Credit")
    total: int = Field(0, description="Total Credit")

    def to(self) -> UserResponse:
        return UserResponse(
            **{
                "user_id": self.user_id,
                "email": self.email,
                "detail_data": self.detail_data,
                "credit_info": CreditResponse(
                    **{
                        "free": self.free,
                        "reward": self.reward,
                        "contract": self.contract,
                        "paid": self.paid,
                        "total": self.total,
                    }
                ),
            }
        )
