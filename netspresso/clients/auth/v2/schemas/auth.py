from pydantic import BaseModel


class TokenRefreshRequest(BaseModel):
    refresh_token: str
