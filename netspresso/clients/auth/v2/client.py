from dataclasses import asdict

from loguru import logger

from netspresso.clients.auth import response_body
from netspresso.clients.auth.request_body import LoginRequest
from netspresso.clients.auth.v2.schemas.auth import TokenRefreshRequest
from netspresso.clients.auth.v2.schemas.credit import SummarizedCreditResponse
from netspresso.clients.auth.v2.schemas.token import TokenResponse
from netspresso.clients.auth.v2.schemas.user import UserResponse
from netspresso.clients.config import Config
from netspresso.clients.utils.requester import Requester


class AuthClientV2:
    def __init__(self, config: Config):
        """Initialize the UserSession.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
        """
        self.config = config
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX
        self.base_url = f"{self.host}:{self.port}{self.prefix}"

    def login(self, email, password, verify_ssl: bool = True) -> response_body.TokenResponse:
        try:
            url = f"{self.base_url}/auth/login"
            request_body = LoginRequest(username=email, password=password)
            response = Requester.post_as_form(url=url, request_body=asdict(request_body))
            token_response = TokenResponse(**response.json())
            logger.info("Login successfully with email and password")
            return token_response.to()
        except Exception as e:
            logger.error(f"Login failed. Error: {e}")
            raise e

    def login_by_api_key(self, api_key: str, verify_ssl: bool = True) -> response_body.TokenResponse:
        try:
            url = f"{self.base_url}/auth/login_by_api_key"
            headers = self.__make_api_key_header(api_key=api_key)
            response = Requester.post_as_json(url=url, headers=headers)
            token_response = TokenResponse(**response.json())
            logger.info("Login successfully with api_key")
            return token_response.to()
        except Exception as e:
            logger.error(f"Login failed. Error: {e}")
            raise e

    def get_user_info(self, access_token, verify_ssl: bool = True) -> response_body.UserResponse:
        user_response = self.__get_user_info(access_token=access_token, verify_ssl=verify_ssl)
        summarized_credit_response = self.__get_credit(access_token=access_token, verify_ssl=verify_ssl)
        logger.info("Successfully got user information")
        return user_response.to(summarized_credit_response=summarized_credit_response)

    def __get_user_info(self, access_token, verify_ssl: bool = True) -> UserResponse:
        try:
            url = f"{self.base_url}/users/me"
            headers = self.__make_bearer_header(token=access_token)

            response = Requester.get(url=url, headers=headers)
            return UserResponse(**response.json())
        except Exception as e:
            logger.error(f"Failed to get user information. Error: {e}")
            raise e

    def get_credit(self, access_token, verify_ssl: bool = True) -> int:
        summarized_credit_response = self.__get_credit(access_token=access_token, verify_ssl=verify_ssl)
        logger.info("Successfully got user credit")
        return summarized_credit_response.data.total_credit

    def __get_credit(self, access_token, user_id: str = None, verify_ssl: bool = True) -> SummarizedCreditResponse:
        try:
            if user_id is None:
                user_response = self.__get_user_info(access_token=access_token, verify_ssl=verify_ssl)
                user_id = user_response.data.user_id

            url = f"{self.base_url}/users/{user_id}/credit/summarized"
            headers = self.__make_bearer_header(token=access_token)

            response = Requester.get(url=url, headers=headers)
            return SummarizedCreditResponse(**response.json())
        except Exception as e:
            logger.error(f"Failed to get user credit. Error: {e}")
            raise e

    def reissue_token(self, access_token, refresh_token, verify_ssl: bool = True) -> response_body.TokenResponse:
        try:
            request_body = TokenRefreshRequest(**{"refresh_token": refresh_token})
            url = f"{self.base_url}/auth/login_by_refresh_token"
            response = Requester.post_as_json(url=url, request_body=asdict(request_body))
            logger.info("Successfully reissued token")
            return TokenResponse(**response.json()).to()
        except Exception as e:
            logger.info(f"Failed to reissue token. Error: {e}")
            raise e

    def __make_bearer_header(self, token: str):
        return {"Authorization": f"Bearer {token}"}

    def __make_api_key_header(self, api_key: str):
        return {"api-key": api_key}
