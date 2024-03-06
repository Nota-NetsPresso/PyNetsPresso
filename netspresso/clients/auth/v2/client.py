from netspresso.clients.auth import response_body
from netspresso.clients.auth.request_body import Tokens, LoginRequest
from netspresso.clients.auth.v2.schemas.auth import TokenRefreshRequest
from netspresso.clients.auth.v2.schemas.credit import SummarizedCreditResponse
from netspresso.clients.auth.v2.schemas.token import TokenResponse
from netspresso.clients.auth.v2.schemas.user import UserResponse
from netspresso.clients.config import Config, Module
from netspresso.clients.utils.requester import Requester


class AuthClientV2:
    def __init__(self, config: Config = Module.GENERAL):
        """Initialize the UserSession.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
        """

        self.host = "http://10.169.1.62"
        self.port = 40003
        self.uri_prefix = "/api/v3"
        self.base_url = f"{self.host}:{self.port}{self.uri_prefix}"

    def login(self, email, password, verify_ssl: bool = True) -> Tokens:
        url = f"{self.base_url}/auth/login"
        request_body = LoginRequest(username=email, password=password)
        response = Requester.post_as_form(url=url, request_body=request_body.__dict__)
        response_body = TokenResponse(**response.json())
        return response_body.to()

    def get_user_info(
        self, access_token, verify_ssl: bool = True
    ) -> response_body.UserResponse:
        user_response = self.__get_user_info(
            access_token=access_token, verify_ssl=verify_ssl
        )
        summarized_credit_response = self.__get_credit(
            access_token=access_token, verify_ssl=verify_ssl
        )
        return user_response.to(summarized_credit_response=summarized_credit_response)

    def __get_user_info(self, access_token, verify_ssl: bool = True) -> UserResponse:
        url = f"{self.base_url}/users/me"
        headers = self.__make_bearer_header(token=access_token)

        response = self.requester.get(url=url, headers=headers)
        return UserResponse(**response.json())

    def get_credit(self, access_token, verify_ssl: bool = True) -> int:
        summarized_credit_response = self.__get_credit(
            access_token=access_token, verify_ssl=verify_ssl
        )
        return summarized_credit_response.data.total_credit

    def __get_credit(
        self, access_token, user_id: str = None, verify_ssl: bool = True
    ) -> SummarizedCreditResponse:
        if user_id is None:
            user_response = self.__get_user_info(
                access_token=access_token, verify_ssl=verify_ssl
            )
            user_id = user_response.data.user_id

        url = f"{self.base_url}/api/v3/users/{user_id}/credit/summarized"
        headers = self.__make_bearer_header(token=access_token)

        response = Requester.get(url=url, headers=headers)
        return SummarizedCreditResponse(**response.json())

    def reissue_token(
        self, access_token, refresh_token, verify_ssl: bool = True
    ) -> Tokens:
        request_body = TokenRefreshRequest(**{"refresh_token": refresh_token})
        url = f"{self.base_url}/auth/login_by_refresh_token"
        response = Requester.post_as_json(
            url=url, request_body=request_body.model_dump()
        )
        return TokenResponse(**response.json()).to()

    def __make_bearer_header(self, token: str):
        return {"Authorization": f"Bearer {token}"}


auth_client_v2 = AuthClientV2()
