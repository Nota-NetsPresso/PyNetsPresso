from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from netspresso.clients.auth.v2.schemas.common import ApiKeyStatus


@dataclass
class TokenRefreshRequest:
    refresh_token: str


@dataclass
class ApiKeyPayload:
    access_key: str
    status: ApiKeyStatus
    deactivated_at: Optional[datetime] = None
    created_time: Optional[datetime] = None


@dataclass
class ApiKeyResponse:
    data: Optional[ApiKeyPayload] = field(default_factory=ApiKeyPayload)

    def __post_init__(self):
        self.data = ApiKeyPayload(**self.data)
