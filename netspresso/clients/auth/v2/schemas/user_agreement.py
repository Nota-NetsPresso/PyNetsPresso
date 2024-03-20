from pydantic import BaseModel


class UserAgreementBase(BaseModel):
    privacy_policy_agreement: bool = False
    marketing_agreement: bool = False
    personal_information_agreement: bool = False
    accessing_age_agreement: bool = False
    terms_of_service_agreement: bool = False
