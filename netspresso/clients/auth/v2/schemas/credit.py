from pydantic import BaseModel

from netspresso.clients.auth.response_body import CreditResponse
from netspresso.clients.auth.v2.schemas.common import AbstractResponse


class CreditWithType(BaseModel):
    user_id: str
    free_credit: int
    paid_credit: int
    total_credit: int


class SummarizedCreditResponse(AbstractResponse):
    data: CreditWithType

    def to(self) -> CreditResponse:
        return CreditResponse(
            free=self.data.free_credit,
            reward=0,
            contract=0,
            paid=self.data.paid_credit,
            total=self.data.total_credit,
        )
