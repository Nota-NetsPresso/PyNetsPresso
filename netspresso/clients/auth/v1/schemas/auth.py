from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    username: str = Field(..., description="User Name")
    password: str = Field(..., description="Password")


class Tokens(BaseModel):
    access_token: str = Field(..., description="Access Token")
    refresh_token: str = Field(..., description="Refresh Token")


class LoginResponse(BaseModel):
    current_time: str = Field(..., description="Login Time")
    region: str = Field(..., description="User Resion")
    tokens: Tokens = Field(default_factory=Tokens, description="Session Token")


class CreditResponse(BaseModel):
    free: int = Field(..., description="Free Credit")
    reward: int = Field(..., description="Reward Credit")
    contract: int = Field(..., description="Contract Credit")
    paid: int = Field(..., description="Paid Credit")
    total: int = Field(..., description="Total Credit")


class UserDetailResponse(BaseModel):
    first_name: str = Field(..., description="First Name")
    last_name: str = Field(..., description="Last Name")
    company: str = Field(..., description="Company")


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
