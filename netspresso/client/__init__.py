import requests, json
from functools import wraps
from loguru import logger

from netspresso.schemas.auth import LoginRequest, RefreshTokenRequest, LoginResponse, RefreshTokenResponse, UserResponse
from netspresso.client.utils.common import get_headers
from netspresso.client.utils.token import check_jwt_exp
from netspresso.client.config import Config, EndPoint

def validate_token(func) -> None:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not check_jwt_exp(self.user_session.access_token):
            self.user_session.__reissue_token()
        return func(self, *args, **kwargs)

    return wrapper

class SessionClient:
    def __init__(self, email: str, password: str, config: Config = None):
        """Initialize the UserSession.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
        """

        self.email = email
        self.password = password
        self.config = config if config is not None else Config(EndPoint.GENRAL)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.uri_prefix = self.config.URI_PREFIX
        self.base_url = f"{self.host}:{self.port}{self.uri_prefix}"
        self.user_id = None
        self.__login()
        self.__get_user_info()

    def __login(self) -> None:
        try:
            url = f"{self.base_url}/auth/local/login"
            data = LoginRequest(username=self.email, password=self.password)
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

    def __get_user_info(self):
        try:
            url = f"{self.base_url}/user"
            response = requests.get(url, headers=get_headers(access_token=self.access_token))
            response_body = json.loads(response.text)

            if response.status_code == 200 or response.status_code == 201:
                user_info = UserResponse(**response_body)
                self.user_id = user_info.user_id
                logger.info("successfully got user information")
            else:
                raise Exception(response_body["detail"])

        except Exception as e:
            logger.error(f"Failed to get user information. Error: {e}")
            raise e

    def __reissue_token(self) -> None:
        try:
            url = f"{self.base_url}/token"
            data = RefreshTokenRequest(access_token=self.access_token, refresh_token=self.refresh_token)
            response = requests.post(url, data=data.json(), headers=get_headers(json_type=True))
            response_body = json.loads(response.text)

            if response.status_code == 200 or response.status_code == 201:
                session = RefreshTokenResponse(**response_body)
                self.access_token = session.access_token
                self.refresh_token = session.refresh_token
            else:
                raise Exception(response_body["detail"])

        except Exception as e:
            raise e

class BaseClient:
    user_session: SessionClient = None
    def __init__(self, email=None, password=None, user_session=None):
        """Initialize the Model Compressor.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
            user_session (SessionClient): The SessionClient object.

        Available constructors: 
            BaseClient(email='USER_EMAIL',password='PASSWORD')
            BaseClient(user_session=SessionClient(email='USER_EMAIL',password='PASSWORD')
        """

        if user_session:
            # Case 1: Creating from a user_session
            self.user_session = user_session
        elif email and password:
            # Case 2: Creating from email and password
            self.user_session = SessionClient(email=email, password=password)
        else:
            raise NotImplementedError("There is no avaliable constructors for given paremeters.")


    @validate_token
    def get_credit(self) -> int:
        """Get the available NetsPresso credits.

        Raises:
            e: If an error occurs while getting credit information.

        Returns:
            int: The total amount of available NetsPresso credits.
        """

        try:
            credit = self.client.get_credit(access_token=self.access_token)
            logger.info(f"Get Credit successfully. Credit: {credit.total}")

            return credit.total

        except Exception as e:
            logger.error(f"Get Credit failed. Error: {e}")
            raise e