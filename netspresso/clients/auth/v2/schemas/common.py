from enum import Enum

from pydantic import BaseModel


class AbstractResponse(BaseModel):
    data: BaseModel


class PagingResponse(AbstractResponse):
    total_count: int
    result_count: int


class MembershipType(str, Enum):
    BASIC = "BASIC"
    PRO = "PRO"
    PREMIUM = "PREMIUM"


class CreditType(str, Enum):
    FREE = "FREE"
    PAID = "PAID"
