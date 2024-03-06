from pydantic import BaseModel


class UserAgreementBase(BaseModel):
    privacy_policy_agreement: bool = False
    marketing_agreement: bool = False
    personal_information_agreement: bool = False
    accessing_age_agreement: bool = False
    terms_of_service_agreement: bool = False

    def to_model(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} class must implement this method."
        )

    class Config:
        from_attributes = True
