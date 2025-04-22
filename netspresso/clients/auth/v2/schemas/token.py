from dataclasses import dataclass, field

from netspresso.clients.auth import response_body
from netspresso.clients.auth.v2.schemas.common import AbstractResponse


@dataclass
class TokenPayload:
    access_token: str
    refresh_token: str


@dataclass
class TokenResponse(AbstractResponse):
    access_token: str
    refresh_token: str

    data: TokenPayload = field(default_factory=TokenPayload)

    def __post_init__(self):
        self.data = TokenPayload(**self.data)

    def to(self) -> response_body.TokenResponse:
        return response_body.TokenResponse(access_token=self.data.access_token, refresh_token=self.data.refresh_token)
