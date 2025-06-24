from dataclasses import dataclass

from netspresso.enums.base import StrEnum


@dataclass
class AbstractResponse:
    pass


@dataclass
class PagingResponse(AbstractResponse):
    total_count: int
    result_count: int


class MembershipType(StrEnum):
    BASIC = "BASIC"
    PRO = "PRO"
    PREMIUM = "PREMIUM"


class CreditType(StrEnum):
    FREE = "FREE"
    PAID = "PAID"


class ApiKeyStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DEACTIVATED = "DEACTIVATED"
