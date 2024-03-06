import json

import requests
from loguru import logger

from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.auth.v1.schemas import (
    LoginRequest,
    LoginResponse,
    Tokens,
    UserInfo,
)
from netspresso.clients.config import Config, Module
from netspresso.clients.utils import get_headers


class AuthClientV1:
    def __init__(self, config: Config = Module.GENERAL):
        """Initialize the UserSession.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
        """

        self.config = Config(config)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.uri_prefix = self.config.URI_PREFIX
        self.base_url = f"{self.host}:{self.port}{self.uri_prefix}"

    def login(self, email, password, verify_ssl: bool = True) -> Tokens:
        try:
            url = f"{self.base_url}/auth/local/login"
            data = LoginRequest(username=email, password=password)
            response = requests.post(
                url, json=data.dict(), headers=get_headers(), verify=verify_ssl
            )
            response_body = json.loads(response.text)

            if response.status_code == 200 or response.status_code == 201:
                session = LoginResponse(**response_body)
                logger.info("Login successfully")
                return session.tokens
            else:
                raise Exception(response_body["detail"])

        except Exception as e:
            logger.error(f"Login failed. Error: {e}")
            raise e

    def get_user_info(self, access_token, verify_ssl: bool = True) -> UserResponse:
        try:
            url = f"{self.base_url}/user"
            response = requests.get(
                url, headers=get_headers(access_token=access_token), verify=verify_ssl
            )
            response_body = json.loads(response.text)

            if response.status_code == 200 or response.status_code == 201:
                user_info = UserInfo(**response_body)
                logger.info("Successfully got user information")
                return user_info.to()
            else:
                raise Exception(response_body["detail"])

        except Exception as e:
            logger.error(f"Failed to get user information. Error: {e}")
            raise e

    def get_credit(self, access_token, verify_ssl: bool = True) -> int:
        user_info = self.get_user_info(access_token, verify_ssl)

        return user_info.credit_info.total

    def reissue_token(
        self, access_token, refresh_token, verify_ssl: bool = True
    ) -> Tokens:
        try:
            url = f"{self.base_url}/auth/token"
            data = Tokens(access_token=access_token, refresh_token=refresh_token)
            response = requests.post(
                url,
                data=data.json(),
                headers=get_headers(json_type=True),
                verify=verify_ssl,
            )
            response_body = json.loads(response.text)

            if response.status_code == 200 or response.status_code == 201:
                tokens = Tokens(**response_body["tokens"])
                logger.info("Successfully reissued token")
                return tokens
            else:
                raise Exception(response_body["detail"])

        except Exception as e:
            raise e


auth_client_v1 = AuthClientV1()
