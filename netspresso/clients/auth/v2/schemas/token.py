from pydantic import BaseModel

from netspresso.clients.auth.request_body import Tokens
from netspresso.clients.auth.v2.schemas.common import AbstractResponse


class TokenPayload(BaseModel):
    access_token: str
    refresh_token: str


class TokenResponse(AbstractResponse):
    data: TokenPayload
    access_token: str = None
    refresh_token: str = None

    def to(self) -> Tokens:
        return Tokens(
            access_token=self.data.access_token, refresh_token=self.data.refresh_token
        )
