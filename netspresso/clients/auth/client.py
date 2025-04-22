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

    def login(self, email, password, verify_ssl: bool = True) -> TokenResponse:
        return self.api_client.login(email=email, password=password, verify_ssl=verify_ssl)

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
    def __init__(self, email, password, verify_ssl: bool = True) -> None:
        self.tokens = auth_client.login(email=email, password=password, verify_ssl=verify_ssl)
        self.email = email
        self.password = password
        self.verify_ssl = verify_ssl

    def check_jwt_exp(self):
        payload = jwt.decode(self.tokens.access_token, options={"verify_signature": False})
        return datetime.now(pytz.utc).timestamp() + 60 <= payload["exp"]

    def validate_token(self):
        if not self.check_jwt_exp():
            self.tokens = auth_client.login(email=self.email, password=self.password, verify_ssl=self.verify_ssl)
            logger.info("The token has expired. the token has been reissued.")


auth_client = AuthClient()
