import json
from functools import wraps

import requests
from loguru import logger

from netspresso.clients.auth.schemas import (
    LoginRequest,
    LoginResponse,
    Tokens,
    UserResponse,
)
from netspresso.clients.config import Config, Module
from netspresso.clients.utils import check_jwt_exp, get_headers


def validate_token(func) -> None:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not check_jwt_exp(self.auth.access_token):
            self.reissue_token()
        return func(self, *args, **kwargs)

    return wrapper


class AuthClient:
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

    def login(self, email, password) -> None:
        try:
            url = f"{self.base_url}/auth/local/login"
            data = LoginRequest(username=email, password=password)
            response = requests.post(url, json=data.dict(), headers=get_headers())
            response_body = json.loads(response.text)

            if response.status_code == 200 or response.status_code == 201:
                session = LoginResponse(**response_body)
                self.access_token = session.tokens.access_token
                self.refresh_token = session.tokens.refresh_token
                logger.info("Login successfully")
            else:
                raise Exception(response_body["detail"])

        except Exception as e:
            logger.error(f"Login failed. Error: {e}")
            raise e

    def get_user_info(self) -> UserResponse:
        try:
            url = f"{self.base_url}/user"
            response = requests.get(
                url, headers=get_headers(access_token=self.access_token)
            )
            response_body = json.loads(response.text)

            if response.status_code == 200 or response.status_code == 201:
                user_info = UserResponse(**response_body)
                logger.info("Successfully got user information")
                return user_info
            else:
                raise Exception(response_body["detail"])

        except Exception as e:
            logger.error(f"Failed to get user information. Error: {e}")
            raise e

    def get_credit(self) -> int:
        user_info = self.get_user_info()
        
        return user_info.total

    def reissue_token(self) -> None:
        try:
            url = f"{self.base_url}/auth/token"
            data = Tokens(
                access_token=self.access_token, refresh_token=self.refresh_token
            )
            response = requests.post(
                url, data=data.json(), headers=get_headers(json_type=True)
            )
            response_body = json.loads(response.text)

            if response.status_code == 200 or response.status_code == 201:
                tokens = Tokens(**response_body["tokens"])
                self.access_token = tokens.access_token
                self.refresh_token = tokens.refresh_token
            else:
                raise Exception(response_body["detail"])

        except Exception as e:
            raise e
