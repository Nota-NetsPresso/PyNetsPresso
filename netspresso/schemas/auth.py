from uuid import UUID

from pydantic import BaseModel, Field, EmailStr


class LoginRequest(BaseModel):
    username: str = Field(..., description="User Name")
    password: str = Field(..., description="Password")


class LoginResponse(BaseModel):
    email: EmailStr = Field(..., description="Email")
    username: str = Field(..., description="User Name")
    user_id: UUID = Field(..., description="User ID")
    access_token: str = Field(..., description="Access Token")
    refresh_token: str = Field(..., description="Refresh Token")
    # session_id: UUID = Field(..., description="Session ID")
    # current_time: str = Field(..., description="Current Time")


class RefreshTokenRequest(BaseModel):
    access_token: str = Field(..., description="Access Token")
    refresh_token: str = Field(..., description="Refresh Token")


class RefreshTokenResponse(BaseModel):
    access_token: str = Field(..., description="Access Token")
    refresh_token: str = Field(..., description="Refresh Token")


class CreditResponse(BaseModel):
    free: int = Field(..., description="Free Credit")
    reward: int = Field(..., description="Reward Credit")
    contract: int = Field(..., description="Contract Credit")
    paid: int = Field(..., description="Paid Credit")
    total: int = Field(..., description="Total Credit")


class Authorities(BaseModel):
    compressor: bool = Field(False, description="Compressor Authority")
    searcher: bool = Field(False, description="Searcher Authority")
    launcher: bool = Field(False, description="Launcher Authority")


class UserRespone(BaseModel):
    user_id: UUID = Field(..., description="User ID")
    email: EmailStr = Field(..., description="Email")
    username: str = Field(..., description="User Name")
    first_name: str = Field(..., description="First Name")
    last_name: str = Field(..., description="Last Name")
    company: str = Field(..., description="Company")
    # credit: int = Field(..., description="Credit")
    # is_active: bool = Field(..., description="Active Status")
    # is_admin: bool = Field(..., description="Admin Status")
    # privacy_policy_agreement: bool = Field(..., description="Privacy Policy Agreement")
    # marketing_agreement: bool = Field(..., description="Marketing Agreement")
    # current_time: str = Field(..., description="Current Time")
    # created_time: str = Field(..., description="Created Time")
    # last_login_time: str = Field(..., description="Last Login Time")
    # authorities: Authorities = Field(default_factory=Authorities, description="NP Module Authorities")
