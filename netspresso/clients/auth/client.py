import warnings
from datetime import datetime

import jwt
import pytz
from loguru import logger

from netspresso.clients.auth.response_body import TokenResponse, UserResponse
from netspresso.clients.auth.v2.client import AuthClientV2
from netspresso.clients.config import Config, ServiceModule, ServiceName


class AuthClient:
    def __init__(self, config: Config = Config(ServiceName.NP, ServiceModule.AUTH)):
        """
        Initialize the UserSession.
        """

        self.config = config
        self.api_client = AuthClientV2(config=config)

    def is_cloud(self) -> bool:
        # TODO
        return self.config.is_cloud()

    def login(self, api_key: str = None, email: str = None, password: str = None, verify_ssl: bool = True) -> TokenResponse:
        if api_key is not None:
            return self.api_client.login_by_api_key(api_key=api_key, verify_ssl=verify_ssl)
        elif email is not None and password is not None:
            logger.warning("[DEPRECATED] Email/password login is deprecated and will be removed in a future release. Please use API Key login.")
            return self.api_client.login(email=email, password=password, verify_ssl=verify_ssl)
        else:
            raise ValueError("You must provide either api_key or both email and password.")

    def get_user_info(self, access_token, verify_ssl: bool = True) -> UserResponse:
        return self.api_client.get_user_info(access_token=access_token, verify_ssl=verify_ssl)

    def get_credit(self, access_token, verify_ssl: bool = True) -> int:
        return self.api_client.get_credit(access_token=access_token, verify_ssl=verify_ssl)

    def reissue_token(self, access_token, refresh_token, verify_ssl: bool = True) -> TokenResponse:
        return self.api_client.reissue_token(
            access_token=access_token,
            refresh_token=refresh_token,
            verify_ssl=verify_ssl,
        )


class TokenHandler:
    def __init__(self, api_key: str = None, email: str = None, password: str = None, verify_ssl: bool = True) -> None:
        if api_key is not None:
            self.tokens = auth_client.login(api_key=api_key, verify_ssl=verify_ssl)
            self.api_key = api_key
            self.email = None
            self.password = None
        elif email is not None and password is not None:
            self.tokens = auth_client.login(email=email, password=password, verify_ssl=verify_ssl)
            self.api_key = None
            self.email = email
            self.password = password
        else:
            raise ValueError("You must provide either api_key or both email and password.")
        self.verify_ssl = verify_ssl

    def check_jwt_exp(self):
        payload = jwt.decode(self.tokens.access_token, options={"verify_signature": False})
        return datetime.now(pytz.utc).timestamp() + 60 <= payload["exp"]

    def validate_token(self):
        if not self.check_jwt_exp():
            if self.api_key is not None:
                self.tokens = auth_client.login(api_key=self.api_key, verify_ssl=self.verify_ssl)
            elif self.email is not None and self.password is not None:
                self.tokens = auth_client.login(email=self.email, password=self.password, verify_ssl=self.verify_ssl)
            logger.info("The token has expired. the token has been reissued.")


auth_client = AuthClient()
