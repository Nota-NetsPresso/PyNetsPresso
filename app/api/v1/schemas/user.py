from pydantic import BaseModel, Field

from app.api.v1.schemas.base import ResponseItem


class CreditInfo(BaseModel):
    free: int = Field(default=0, description="Free credits available.")
    reward: int = Field(default=0, description="Reward credits available.")
    contract: int = Field(default=0, description="Contract-based credits available.")
    paid: int = Field(default=0, description="Paid credits available.")
    total: int = Field(default=0, description="Total available credits.")


class DetailData(BaseModel):
    first_name: str = Field(..., description="First name of the user.")
    last_name: str = Field(..., description="Last name of the user.")
    company: str = Field(..., description="Company name of the user.")


class UserPayload(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    email: str = Field(..., description="The email address of the user.")
    detail_data: DetailData = Field(..., description="Detailed information of the user.")
    credit_info: CreditInfo = Field(..., description="Credit information of the user.")


class UserResponse(ResponseItem):
    data: UserPayload
