from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., description="User Name")
    password: str = Field(..., description="Password")



