import json
from datetime import datetime

import requests
from loguru import logger
import jwt
import pytz

from netspresso.clients.auth.schemas import (
    LoginRequest,
    LoginResponse,
    Tokens,
    UserResponse,
)
from netspresso.clients.config import Config, Module
from netspresso.clients.utils import get_headers


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

    def login(self, email, password) -> Tokens:
        try:
            url = f"{self.base_url}/auth/local/login"
            data = LoginRequest(username=email, password=password)
            response = requests.post(url, json=data.dict(), headers=get_headers())
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

    def get_user_info(self, access_token) -> UserResponse:
        try:
            url = f"{self.base_url}/user"
            response = requests.get(
                url, headers=get_headers(access_token=access_token)
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

    def get_credit(self, access_token) -> int:
        user_info = self.get_user_info(access_token)
        
        return user_info.total

    def reissue_token(self, access_token, refresh_token) -> Tokens:
        try:
            url = f"{self.base_url}/auth/token"
            data = Tokens(
                access_token=access_token, refresh_token=refresh_token
            )
            response = requests.post(
                url, data=data.json(), headers=get_headers(json_type=True)
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


class TokenHandler:
    def __init__(self, tokens) -> None:
        self.tokens = tokens

    def check_jwt_exp(self):
        payload = jwt.decode(self.tokens.access_token, options={"verify_signature": False})
        return datetime.now(pytz.utc).timestamp() <= payload["exp"]

    def validate_token(self):
        if not self.check_jwt_exp():
            self.tokens = auth_client.reissue_token(self.tokens.access_token, self.tokens.refresh_token)


auth_client = AuthClient()
