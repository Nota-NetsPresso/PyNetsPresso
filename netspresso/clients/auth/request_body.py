from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., description="User Name")
    password: str = Field(..., description="Password")


class Tokens(BaseModel):
    access_token: str = Field(..., description="Access Token")
    refresh_token: str = Field(..., description="Refresh Token")


